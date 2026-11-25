"""Acoustic-guided segment snapping, long-turn splitting, and tiny-fragment merging for MOSS output."""

from __future__ import annotations

import math
import re
import wave
from array import array
from pathlib import Path
from typing import Any

MAX_MOSS_SEGMENT_SECONDS = 11.5
MIN_CHILD_SEGMENT_SECONDS = 2.0
MIN_CHILD_CHARS = 4
MERGE_MAX_GAP_SECONDS = 0.65
MERGE_MAX_TOTAL_SECONDS = 10.5
TINY_DURATION_SECONDS = 1.25
TINY_CHAR_THRESHOLD = 5

_STRONG_PUNCT_RE = re.compile(r"[。！？；.!?;]")
_SOFT_PUNCT_RE = re.compile(r"[、，,:：]")
_TERMINAL_PUNCT_RE = re.compile(r"""[.!?。！？…]["'”’）)\]]*$""")


def _compact_len(text: str) -> int:
    return len("".join((text or "").split()))


def _join_text(left: str, right: str) -> str:
    left = (left or "").rstrip()
    right = (right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if bool(re.search(r"[A-Za-z0-9]$", left)) and bool(re.match(r"[A-Za-z0-9]", right)):
        return f"{left} {right}"
    if left[-1] in ",;:" and bool(re.match(r"[A-Za-z0-9]", right)):
        return f"{left} {right}"
    return f"{left}{right}"


def _read_mono_envelope(wav_path: Path, hop_sec: float = 0.01) -> tuple[list[float], float, float]:
    """Compute a fast normalized RMS envelope (0..1) from PCM16 WAV."""
    if not wav_path.is_file():
        return [], hop_sec, 0.0
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            nframes = wf.getnframes()
            if rate <= 0 or channels <= 0 or width != 2 or nframes <= 0:
                return [], hop_sec, 0.0
            raw = wf.readframes(nframes)
    except (OSError, wave.Error, EOFError):
        return [], hop_sec, 0.0

    wav_dur = nframes / float(rate)
    hop = max(1, round(rate * hop_sec))
    actual_hop_sec = hop / float(rate)

    try:
        import numpy as np

        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples[: (len(samples) // channels) * channels].reshape(-1, channels).mean(axis=1)
        usable = (len(samples) // hop) * hop
        if usable <= 0:
            return [], actual_hop_sec, wav_dur
        frames = samples[:usable].reshape(-1, hop)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        return rms.tolist(), actual_hop_sec, wav_dur
    except ImportError:
        samples_arr = array("h")
        samples_arr.frombytes(raw)
        envelope: list[float] = []
        stride = hop * channels
        for pos in range(0, len(samples_arr) - stride + 1, stride):
            sq = 0.0
            for idx in range(pos, pos + stride, channels):
                v = sum(samples_arr[idx + ch] for ch in range(channels)) / (channels * 32768.0)
                sq += v * v
            envelope.append(math.sqrt(sq / hop))
        return envelope, actual_hop_sec, wav_dur


def _speech_threshold(envelope: list[float]) -> float | None:
    if not envelope:
        return None
    peak = max(envelope)
    if peak < 1e-4:
        return None
    sorted_env = sorted(envelope)
    noise_floor = sorted_env[max(0, int(len(sorted_env) * 0.15))]
    noise_cap = min(peak * 0.18, noise_floor * 2.5)
    return max(peak * 0.04, noise_cap, 0.005)


def _snap_segment_boundaries(
    segments: list[dict[str, Any]],
    envelope: list[float],
    hop_sec: float,
    wav_dur: float,
    thresh: float,
) -> list[dict[str, Any]]:
    """Expand clipped MOSS endpoints into neighboring silence valleys without overlapping neighbors."""
    if not segments or not envelope:
        return segments
    n_frames = len(envelope)
    out = [dict(seg) for seg in segments]

    def frame_at(t: float) -> int:
        return max(0, min(n_frames - 1, int(round(t / hop_sec))))

    for i, seg in enumerate(out):
        start = float(seg["start"])
        end = float(seg["end"])
        prev_end = float(out[i - 1]["end"]) if i > 0 else 0.0
        next_start = float(out[i + 1]["start"]) if i + 1 < len(out) else wav_dur

        min_start = max(0.0, prev_end + 0.02) if i > 0 and prev_end <= start else 0.0
        max_end = min(wav_dur, next_start - 0.02) if i + 1 < len(out) and next_start >= end else wav_dur

        # 1. If start cuts active speech, look slightly earlier (<= 0.25s) for silence onset.
        s_idx = frame_at(start)
        if envelope[s_idx] >= thresh and start - min_start >= 0.05:
            look_left = frame_at(max(min_start, start - 0.25))
            for k in range(s_idx, look_left - 1, -1):
                if envelope[k] < thresh:
                    candidate = round(max(min_start, k * hop_sec - 0.02), 3)
                    if candidate < start:
                        start = candidate
                    break
        else:
            # Trim excess leading silence (> 0.28s) inside the segment itself.
            e_limit = frame_at(max(start, end - 0.45))
            first_active = None
            for k in range(s_idx, e_limit + 1):
                if envelope[k] >= thresh:
                    first_active = k * hop_sec
                    break
            if first_active is not None and (first_active - start) > 0.28:
                start = round(max(min_start, first_active - 0.06), 3)

        # 2. If end clips active speech, extend slightly (<= 0.35s) to reach silence valley.
        e_idx = frame_at(end)
        if envelope[e_idx] >= thresh and max_end - end >= 0.05:
            look_right = frame_at(min(max_end, end + 0.35))
            for k in range(e_idx, look_right + 1):
                if envelope[k] < thresh:
                    candidate = round(min(max_end, (k + 1) * hop_sec + 0.04), 3)
                    if candidate > end:
                        end = candidate
                    break

        if end > start:
            seg["start"] = round(start, 3)
            seg["end"] = round(end, 3)
    return out


def _find_silence_valleys(
    envelope: list[float],
    hop_sec: float,
    start: float,
    end: float,
    thresh: float,
    *,
    min_silence_sec: float = 0.22,
) -> list[tuple[float, float]]:
    """Return candidate internal pause midpoints (time_sec, quality_score)."""
    lo = max(0, int(math.ceil((start + MIN_CHILD_SEGMENT_SECONDS) / hop_sec)))
    hi = min(len(envelope), int(math.floor((end - MIN_CHILD_SEGMENT_SECONDS) / hop_sec)))
    if hi <= lo:
        return []

    def scan(limit_thresh: float, min_dur: float) -> list[tuple[float, float]]:
        valleys: list[tuple[float, float]] = []
        run_start: int | None = None
        for idx in range(lo, hi):
            if envelope[idx] < limit_thresh:
                if run_start is None:
                    run_start = idx
            elif run_start is not None:
                dur = (idx - run_start) * hop_sec
                if dur >= min_dur:
                    valleys.append((round(((run_start + idx) * 0.5) * hop_sec, 3), dur))
                run_start = None
        if run_start is not None:
            dur = (hi - run_start) * hop_sec
            if dur >= min_dur:
                valleys.append((round(((run_start + hi) * 0.5) * hop_sec, 3), dur))
        return valleys

    valleys = scan(thresh, min_silence_sec)
    if not valleys and (end - start) >= 13.0:
        valleys = scan(thresh * 1.45, 0.16)
    return valleys


def _clause_split_points(text: str) -> list[tuple[int, float]]:
    """Return candidate character split offsets with punctuation strength weight."""
    points: list[tuple[int, float]] = []
    for match in re.finditer(r"[。！？；.!?;、，,:：]", text):
        pos = match.end()
        while pos < len(text) and text[pos] in """"'”’）)]""":
            pos += 1
        left = text[:pos].strip()
        right = text[pos:].strip()
        if _compact_len(left) < MIN_CHILD_CHARS or _compact_len(right) < MIN_CHILD_CHARS:
            continue
        punct = match.group(0)
        strength = 1.0 if _STRONG_PUNCT_RE.match(punct) else 0.45
        points.append((pos, strength))
    return points


def _active_speech_integral(
    envelope: list[float],
    hop_sec: float,
    start: float,
    end: float,
    thresh: float,
) -> float:
    lo = max(0, int(round(start / hop_sec)))
    hi = min(len(envelope), int(round(end / hop_sec)))
    if hi <= lo:
        return max(0.1, end - start)
    active = sum(1 for v in envelope[lo:hi] if v >= thresh)
    return max(0.1, active * hop_sec)


def _split_overlong_segment(
    seg: dict[str, Any],
    envelope: list[float],
    hop_sec: float,
    thresh: float,
) -> list[dict[str, Any]]:
    """Split a long (>11.5s) MOSS segment into 2-3 coherent segments at acoustic pause valleys."""
    start = float(seg["start"])
    end = float(seg["end"])
    dur = end - start
    text = str(seg.get("orig_text") or "").strip()
    if dur <= MAX_MOSS_SEGMENT_SECONDS or _compact_len(text) < MIN_CHILD_CHARS * 2:
        return [seg]

    valleys = _find_silence_valleys(envelope, hop_sec, start, end, thresh)
    if not valleys:
        return [seg]
    text_points = _clause_split_points(text)
    if not text_points:
        return [seg]

    total_chars = max(1, _compact_len(text))
    total_active = _active_speech_integral(envelope, hop_sec, start, end, thresh)

    def build_children(cut_times: list[float], cut_offsets: list[int]) -> list[dict[str, Any]]:
        times = [start, *cut_times, end]
        offsets = [0, *cut_offsets, len(text)]
        children: list[dict[str, Any]] = []
        for idx in range(len(cut_times) + 1):
            c_start, c_end = round(times[idx], 3), round(times[idx + 1], 3)
            c_text = text[offsets[idx]:offsets[idx + 1]].strip()
            if c_end - c_start < MIN_CHILD_SEGMENT_SECONDS or _compact_len(c_text) < MIN_CHILD_CHARS:
                return []
            child = dict(seg)
            child["start"] = c_start
            child["end"] = c_end
            child["orig_text"] = c_text
            child["words"] = []
            children.append(child)
        return children

    best_score = float("inf")
    best_children: list[dict[str, Any]] | None = None

    # Try 2-way splits (and 3-way splits when duration > 16s).
    for v_time, v_dur in valleys:
        left_active = _active_speech_integral(envelope, hop_sec, start, v_time, thresh)
        audio_share = left_active / total_active
        max_part_dur = max(v_time - start, end - v_time)
        dur_penalty = max(0.0, max_part_dur - MAX_MOSS_SEGMENT_SECONDS) * 0.22
        for pos, strength in text_points:
            char_share = _compact_len(text[:pos]) / total_chars
            mismatch = abs(char_share - audio_share)
            if mismatch > 0.24:
                continue
            score = mismatch * 2.0 + dur_penalty - strength * 0.28 - min(0.35, v_dur) * 0.25
            if score < best_score:
                children = build_children([v_time], [pos])
                if children:
                    best_score = score
                    best_children = children

    if dur > 15.5 and len(valleys) >= 2 and len(text_points) >= 2:
        for i in range(len(valleys)):
            t1, d1 = valleys[i]
            if t1 - start < MIN_CHILD_SEGMENT_SECONDS:
                continue
            for j in range(i + 1, len(valleys)):
                t2, d2 = valleys[j]
                if t2 - t1 < MIN_CHILD_SEGMENT_SECONDS or end - t2 < MIN_CHILD_SEGMENT_SECONDS:
                    continue
                if max(t1 - start, t2 - t1, end - t2) > 12.5:
                    continue
                a1 = _active_speech_integral(envelope, hop_sec, start, t1, thresh) / total_active
                a2 = _active_speech_integral(envelope, hop_sec, t1, t2, thresh) / total_active
                a3 = _active_speech_integral(envelope, hop_sec, t2, end, thresh) / total_active
                for p_idx in range(len(text_points)):
                    pos1, s1 = text_points[p_idx]
                    c1 = _compact_len(text[:pos1]) / total_chars
                    if abs(c1 - a1) > 0.22:
                        continue
                    for q_idx in range(p_idx + 1, len(text_points)):
                        pos2, s2 = text_points[q_idx]
                        c2 = _compact_len(text[pos1:pos2]) / total_chars
                        c3 = _compact_len(text[pos2:]) / total_chars
                        if abs(c2 - a2) > 0.22 or abs(c3 - a3) > 0.22:
                            continue
                        score = (
                            (abs(c1 - a1) + abs(c2 - a2) + abs(c3 - a3)) * 1.4
                            - (s1 + s2) * 0.18
                            - min(0.6, d1 + d2) * 0.15
                        )
                        if score < best_score:
                            children = build_children([t1, t2], [pos1, pos2])
                            if children:
                                best_score = score
                                best_children = children

    return best_children or [seg]


def _merge_tiny_fragments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge ultra-short (<1.25s or <=5 chars) fragments into adjacent same-speaker segments."""
    if len(segments) <= 1:
        return segments
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        gap = round(float(seg["start"]) - float(prev["end"]), 3)
        prev_dur = float(prev["end"]) - float(prev["start"])
        cur_dur = float(seg["end"]) - float(seg["start"])
        prev_text = str(prev.get("orig_text") or "").strip()
        cur_text = str(seg.get("orig_text") or "").strip()
        prev_chars = _compact_len(prev_text)
        cur_chars = _compact_len(cur_text)

        same_speaker = (
            prev.get("speaker") == seg.get("speaker")
            and prev.get("speaker") not in {None, "", "UNKNOWN"}
        )
        one_is_tiny = (
            prev_dur < TINY_DURATION_SECONDS
            or prev_chars <= TINY_CHAR_THRESHOLD
            or cur_dur < TINY_DURATION_SECONDS
            or cur_chars <= TINY_CHAR_THRESHOLD
        )
        non_terminal = not _TERMINAL_PUNCT_RE.search(prev_text.rstrip())
        tiny_interjection = min(prev_chars, cur_chars) <= 4 and gap <= 0.55

        can_merge = (
            same_speaker
            and 0.0 <= gap <= MERGE_MAX_GAP_SECONDS
            and (float(seg["end"]) - float(prev["start"])) <= MERGE_MAX_TOTAL_SECONDS
            and one_is_tiny
            and (non_terminal or tiny_interjection)
        )
        if not can_merge:
            merged.append(dict(seg))
            continue

        prev["end"] = round(float(seg["end"]), 3)
        prev["orig_text"] = _join_text(prev_text, cur_text)
        merged[-1] = prev
    return merged


def refine_moss_segments(
    raw_segments: list[dict[str, Any]],
    vocals_wav: Path | None,
    *,
    language: str | None = None,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """Snap MOSS boundaries to silence valleys, split >11.5s turns, and merge tiny fragments."""
    del language
    if not raw_segments:
        return []
    ordered = sorted((dict(seg) for seg in raw_segments), key=lambda item: (float(item["start"]), float(item["end"])))
    envelope, hop_sec, wav_dur = _read_mono_envelope(vocals_wav) if vocals_wav is not None else ([], 0.01, 0.0)
    thresh = _speech_threshold(envelope)
    if not envelope or thresh is None or wav_dur <= 0:
        return [dict(seg, id=idx) for idx, seg in enumerate(ordered)]

    total_dur = max(wav_dur, duration or 0.0)
    snapped = _snap_segment_boundaries(ordered, envelope, hop_sec, total_dur, thresh)
    split_segments: list[dict[str, Any]] = []
    for seg in snapped:
        split_segments.extend(_split_overlong_segment(seg, envelope, hop_sec, thresh))
    final = _merge_tiny_fragments(split_segments)
    return [dict(seg, id=idx) for idx, seg in enumerate(final)]
