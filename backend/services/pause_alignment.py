"""Intra-segment pause detection and sub-clause alignment for natural dubbing."""
from __future__ import annotations

import math
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from backend import config
from backend.schemas import ProjectScript, Segment
from backend.services.media_mixer import apply_gentle_tempo
from workers.prosody_worker import read_wav

_MIN_SEGMENT_FOR_PAUSES_SEC = 3.6
_MIN_PAUSE_SEC = 0.35
_MIN_ISLAND_SEC = 0.65
_MAX_ISLANDS = 3
_CLAUSE_RE = re.compile(r"[^，。！？；…,.!?;\n]+[，。！？；…,.!?;\n]*")


@dataclass(frozen=True)
class PauseAlignedClause:
    text: str
    rel_start: float
    rel_end: float


def _active_threshold_for_samples(samples: list[float]) -> float:
    rms = math.sqrt(sum(v * v for v in samples) / len(samples)) if samples else 0.0
    return max(10 ** (-43.0 / 20.0), rms * 0.14)


def detect_speech_islands(
    vocals_path: Path,
    seg_start: float,
    seg_end: float,
    *,
    min_pause_sec: float = _MIN_PAUSE_SEC,
    min_island_sec: float = _MIN_ISLAND_SEC,
    max_islands: int = _MAX_ISLANDS,
) -> list[tuple[float, float]]:
    """Detect >= 2 separated speech-activity islands within one segment (relative seconds)."""
    dur = seg_end - seg_start
    if dur < _MIN_SEGMENT_FOR_PAUSES_SEC or not vocals_path.is_file():
        return []
    try:
        with wave.open(str(vocals_path), "rb") as wf:
            wav_dur = wf.getnframes() / float(max(1, wf.getframerate()))
        clipped_end = min(seg_end, wav_dur)
        if clipped_end - seg_start < _MIN_SEGMENT_FOR_PAUSES_SEC:
            return []
        samples, rate, _ = read_wav(vocals_path, start=seg_start, end=clipped_end)
    except (OSError, ValueError, wave.Error, EOFError):
        return []

    hop = max(1, round(rate * 0.02))
    frame_sec = hop / float(rate)
    gate = _active_threshold_for_samples(samples)
    active_flags: list[bool] = []
    for offset in range(0, len(samples), hop):
        frame = samples[offset:offset + hop]
        frame_rms = math.sqrt(sum(v * v for v in frame) / max(1, len(frame)))
        active_flags.append(frame_rms >= gate)

    raw_islands: list[list[float]] = []
    cur_start: int | None = None
    for idx, flag in enumerate(active_flags):
        if flag and cur_start is None:
            cur_start = idx
        elif not flag and cur_start is not None:
            raw_islands.append([round(cur_start * frame_sec, 3), round(idx * frame_sec, 3)])
            cur_start = None
    if cur_start is not None:
        raw_islands.append([round(cur_start * frame_sec, 3), round(len(active_flags) * frame_sec, 3)])

    if not raw_islands:
        return []

    # Bridge micro-gaps shorter than min_pause_sec so normal stops don't fragment speech.
    bridged: list[list[float]] = [raw_islands[0]]
    for start, end in raw_islands[1:]:
        if start - bridged[-1][1] < min_pause_sec:
            bridged[-1][1] = end
        else:
            bridged.append([start, end])

    # Drop tiny noise clicks unless only one remains; then absorb/filter to substantial islands.
    significant = [(s, e) for s, e in bridged if (e - s) >= min_island_sec]
    if len(significant) < 2:
        return []

    islands = list(significant)
    while len(islands) > max_islands:
        gaps = [islands[i + 1][0] - islands[i][1] for i in range(len(islands) - 1)]
        merge_idx = min(range(len(gaps)), key=lambda i: gaps[i])
        merged = (islands[merge_idx][0], islands[merge_idx + 1][1])
        islands[merge_idx:merge_idx + 2] = [merged]

    return islands if len(islands) >= 2 else []


def _clause_units(text: str, cjk: bool) -> int:
    cleaned = re.sub(r"[，。！？；…,.!?;\s\"'“”‘’（）()]+", "", text)
    if cjk:
        return len(cleaned)
    return len(cleaned.split()) if cleaned else 0


def align_clauses_to_islands(
    target_text: str,
    target_lang: str,
    islands: list[tuple[float, float]],
) -> list[PauseAlignedClause] | None:
    """Partition target_text clauses across acoustic speech islands proportionally."""
    if len(islands) < 2:
        return None
    raw_clauses = [m.group(0).strip() for m in _CLAUSE_RE.finditer(target_text or "") if m.group(0).strip()]
    cjk = target_lang.removeprefix("all_").split("-")[0] in {"zh", "ja", "yue", "ko"}
    clauses = [c for c in raw_clauses if _clause_units(c, cjk) >= 1]
    if len(clauses) < 2:
        return None

    working_islands = list(islands)
    while len(working_islands) > len(clauses) and len(working_islands) >= 2:
        gaps = [working_islands[i + 1][0] - working_islands[i][1] for i in range(len(working_islands) - 1)]
        idx = min(range(len(gaps)), key=lambda i: gaps[i])
        working_islands[idx:idx + 2] = [(working_islands[idx][0], working_islands[idx + 1][1])]
    if len(working_islands) < 2:
        return None

    k = len(working_islands)
    island_durs = [max(0.2, end - start) for start, end in working_islands]
    total_dur = sum(island_durs)
    dur_shares = [d / total_dur for d in island_durs]

    weights = [max(1, _clause_units(c, cjk)) for c in clauses]
    total_weight = sum(weights)
    min_group_units = 2 if cjk else 2

    best_score = float("inf")
    best_cuts: tuple[int, ...] | None = None

    def _search(start_idx: int, group_idx: int, cuts: list[int], cost: float) -> None:
        nonlocal best_score, best_cuts
        if group_idx == k - 1:
            group_w = sum(weights[start_idx:])
            if group_w < min_group_units:
                return
            score = cost + abs((group_w / total_weight) - dur_shares[group_idx])
            if score < best_score:
                best_score = score
                best_cuts = tuple(cuts)
            return
        max_end = len(clauses) - (k - 1 - group_idx)
        running_w = 0
        for end_idx in range(start_idx + 1, max_end + 1):
            running_w += weights[end_idx - 1]
            if running_w < min_group_units:
                continue
            step_cost = cost + abs((running_w / total_weight) - dur_shares[group_idx])
            if step_cost < best_score:
                _search(end_idx, group_idx + 1, [*cuts, end_idx], step_cost)

    _search(0, 0, [], 0.0)
    if best_cuts is None:
        return None

    boundaries = [0, *best_cuts, len(clauses)]
    sep = "" if cjk else " "
    aligned: list[PauseAlignedClause] = []
    for idx in range(k):
        chunk_text = sep.join(clauses[boundaries[idx]:boundaries[idx + 1]]).strip()
        if not chunk_text:
            return None
        aligned.append(PauseAlignedClause(
            text=chunk_text,
            rel_start=working_islands[idx][0],
            rel_end=working_islands[idx][1],
        ))
    return aligned


def plan_segment_pause_synthesis(
    script: ProjectScript,
    seg: Segment,
    vocals_path: Path | None = None,
) -> list[PauseAlignedClause] | None:
    """Plan sub-clause synthesis when a segment contains genuine multi-island dramatic pauses."""
    if vocals_path is None:
        if not script.media.vocals_wav:
            return None
        try:
            vocals_path = config.resolve_media_path(script.media.vocals_wav)
        except ValueError:
            return None
    if not vocals_path.is_file():
        return None
    islands = detect_speech_islands(vocals_path, seg.start, seg.end)
    if not islands:
        return None
    return align_clauses_to_islands(seg.target_text, seg.target_lang, islands)


def _trim_edge_silence(clip: AudioSegment, *, silence_thresh_db: float = -46.0, pad_ms: int = 25) -> AudioSegment:
    """Trim leading/trailing digital silence emitted by TTS while keeping a short breath cushion."""
    if len(clip) <= 80 or clip.rms == 0:
        return clip
    chunk_ms = 10
    threshold_rms = clip.max_possible_amplitude * (10 ** (silence_thresh_db / 20.0))
    first_ms = 0
    last_ms = len(clip)
    for pos in range(0, len(clip), chunk_ms):
        if clip[pos:pos + chunk_ms].rms >= threshold_rms:
            first_ms = pos
            break
    for pos in range(len(clip), 0, -chunk_ms):
        start_pos = max(0, pos - chunk_ms)
        if clip[start_pos:pos].rms >= threshold_rms:
            last_ms = pos
            break
    if last_ms <= first_ms:
        return clip
    trimmed = clip[max(0, first_ms - pad_ms):min(len(clip), last_ms + pad_ms)]
    fade_ms = min(8, len(trimmed) // 4)
    return trimmed.fade_in(fade_ms).fade_out(fade_ms) if fade_ms >= 2 else trimmed


def assemble_island_wavs(
    clause_wavs: list[Path],
    clauses: list[PauseAlignedClause],
    slot_duration: float,
    out_wav: Path,
    tmp_dir: Path,
) -> None:
    """Assemble clause WAVs onto their original dramatic speech-island onset positions."""
    if len(clause_wavs) != len(clauses) or not clause_wavs:
        raise ValueError("clause_wavs and clauses must be aligned and non-empty")

    loaded: list[AudioSegment] = []
    for wav_path in clause_wavs:
        with wav_path.open("rb") as fp:
            loaded.append(_trim_edge_silence(AudioSegment.from_wav(fp)))

    target_rate = loaded[0].frame_rate or 32000
    target_channels = 1
    placed_clips: list[tuple[int, AudioSegment]] = []
    cursor_ms = 0

    for idx, (clip, clause) in enumerate(zip(loaded, clauses)):
        clip = clip.set_frame_rate(target_rate).set_channels(target_channels).set_sample_width(2)
        desired_ms = round(clause.rel_start * 1000)
        if idx == 0:
            pos_ms = min(180, desired_ms)
        else:
            pos_ms = max(cursor_ms + 40, desired_ms)

        if idx < len(clauses) - 1:
            next_island_ms = round(clauses[idx + 1].rel_start * 1000)
            orig_pause_ms = max(0, next_island_ms - round(clause.rel_end * 1000))
            preserve_pause_ms = max(220, round(orig_pause_ms * 0.55))
            allowed_ms = max(350, next_island_ms - pos_ms - preserve_pause_ms)
            if len(clip) > allowed_ms and allowed_ms > 200:
                speed = min(1.20, len(clip) / float(allowed_ms))
                if speed >= 1.03:
                    src = tmp_dir / f"island_{idx}.wav"
                    with src.open("wb") as fp:
                        clip.export(fp, format="wav")
                    adjusted = apply_gentle_tempo(src, speed, tmp_dir)
                    with adjusted.open("rb") as fp:
                        clip = AudioSegment.from_wav(fp).set_frame_rate(target_rate).set_channels(1).set_sample_width(2)

        placed_clips.append((pos_ms, clip))
        cursor_ms = pos_ms + len(clip)

    total_ms = max(cursor_ms, round(slot_duration * 1000))
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=target_rate).set_channels(1).set_sample_width(2)
    for pos_ms, clip in placed_clips:
        canvas = canvas.overlay(clip, position=pos_ms)
    # Trim any trailing dead silence past the actual speech end while preserving minimum natural tail.
    used_ms = max(pos + len(clip) for pos, clip in placed_clips)
    canvas = canvas[:min(len(canvas), max(used_ms, min(total_ms, used_ms + 30)))]
    with out_wav.open("wb") as out_fp:
        canvas.export(out_fp, format="wav")
