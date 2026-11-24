"""Store manual references and cut automatic clips from reviewed source ranges."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import BinaryIO

from backend import config
from backend.schemas import ProjectScript, ReferencePreview, Segment
from backend.services.text_normalization import normalize_tts_lang, prepare_tts_text

MIN_REFERENCE_SECONDS = 3
MAX_REFERENCE_SECONDS = 10
MAX_REFERENCE_BYTES = 32 * 1024 * 1024
SUPPORTED_SOURCE_LANGS = {"zh", "en", "ja", "ko", "yue"}


def _copy_bounded(source: BinaryIO, target: BinaryIO, max_bytes: int) -> None:
    total = 0
    while chunk := source.read(65536):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("参考音文件不能超过 32 MB")
        target.write(chunk)


def store_reference_audio(project_id: str, source: BinaryIO) -> str:
    dest = config.WORKSPACE_DIR / project_id / "references"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.wav"
    with tempfile.TemporaryDirectory(dir=dest) as temp:
        original, converted = Path(temp) / "input", Path(temp) / name
        with original.open("wb") as target:
            _copy_bounded(source, target, MAX_REFERENCE_BYTES)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                    "-format_whitelist", "wav,mp3,flac,ogg,mov,aac,matroska,webm",
                    "-i", str(original), "-map", "0:a:0", "-t", "10.1", "-ac", "1",
                    "-ar", "32000", "-c:a", "pcm_s16le", str(converted),
                ],
                check=True, capture_output=True, timeout=30,
            )
        except FileNotFoundError as exc:
            raise ValueError("上传参考音需要安装 FFmpeg") from exc
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("无法读取参考音，请上传可播放的音频文件") from exc
        with wave.open(str(converted), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
            raise ValueError("GPT-SoVITS 参考音长度须在 3–10 秒之间")
        converted.replace(dest / name)
    return f"/workspace/{project_id}/references/{name}"


def _whole_timing_confirmed(seg: Segment) -> bool:
    # Old reviewed JSON keeps its historical eligibility. New segment-only
    # confirmation permits whole audio, never invented word-level crops.
    return (not seg.segment_timing_needs_review
            or (not seg.alignment_needs_review and _verified_word_offsets(seg) is not None))


def _candidate_clips(script: ProjectScript, speaker: str) -> list[list[Segment]]:
    ordered = sorted(script.segments, key=lambda seg: (seg.start, seg.id))
    candidates = []
    for index, seg in enumerate(ordered):
        if seg.speaker != speaker or not _whole_timing_confirmed(seg) or seg.speaker_needs_review or not seg.orig_text.strip():
            continue
        clip = []
        for following in range(index, len(ordered)):
            other = ordered[following]
            if (other.speaker != speaker or not _whole_timing_confirmed(other) or other.speaker_needs_review
                    or not other.orig_text.strip() or other.end - seg.start > MAX_REFERENCE_SECONDS
                    or (clip and not 0 <= round(other.start - clip[-1].end, 3) <= 0.35)):
                break
            clip.append(other)
            duration = other.end - seg.start
            if MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
                if not any(
                    item.start < other.end and item.end > seg.start
                    and item.id not in {part.id for part in clip}
                    for item in ordered
                ):
                    transcript = " ".join(part.orig_text.strip() for part in clip)
                    if len(re.sub(r"\s+", "", transcript)) <= duration * 150:
                        candidates.append(clip)
                    break
    return sorted(candidates, key=lambda clip: (
        len(clip),
        abs(clip[-1].end - clip[0].start - 5),
        len("".join(part.orig_text for part in clip)),
    ))


def _verified_word_offsets(seg: Segment) -> list[int] | None:
    """Match every aligned token to the reviewed text without guessing text offsets."""
    letters = [(index, char) for index, char in enumerate(seg.orig_text) if not char.isspace()]
    compact = "".join(char for _, char in letters)
    cursor = 0
    previous_end = seg.start
    offsets = []
    for word in seg.words:
        token = "".join(word.word.split())
        if (not token or not compact.startswith(token, cursor)
                or word.start is None or word.end is None
                or not seg.start <= word.start < word.end <= seg.end
                or word.start < previous_end or word.speaker != seg.speaker):
            return None
        offsets.append(letters[cursor][0])
        cursor += len(token)
        previous_end = word.end
    return offsets if offsets and cursor == len(compact) else None


def _trimmed_transcript(seg: Segment, start: float, end: float) -> str | None:
    offsets = _verified_word_offsets(seg)
    if offsets is None:
        return None
    included = [index for index, word in enumerate(seg.words)
                if word.start >= start and word.end <= end]
    if not included or included != list(range(included[0], included[-1] + 1)):
        return None
    left, right = included[0], included[-1] + 1
    return seg.orig_text[offsets[left]:offsets[right] if right < len(offsets) else len(seg.orig_text)].strip()


def _long_reference_window(script: ProjectScript, seg: Segment) -> tuple[float, float, str, list[int]] | None:
    if (seg.alignment_needs_review or seg.speaker_needs_review
            or seg.end - seg.start <= MAX_REFERENCE_SECONDS
            or _verified_word_offsets(seg) is None):
        return None
    choices = []
    for index, word in enumerate(seg.words[:-1]):
        duration = word.end - seg.start
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
            continue
        if any(other.id != seg.id and other.start < word.end and other.end > seg.start
               for other in script.segments):
            continue
        transcript = _trimmed_transcript(seg, seg.start, word.end)
        if transcript:
            choices.append((seg.start, word.end, transcript, [seg.id]))
    return min(choices, key=lambda window: abs(window[1] - window[0] - 5)) if choices else None


def reference_unavailable_reason(script: ProjectScript, speaker: str) -> str:
    """Explain missing automatic reference without changing review assertions."""
    segments = [seg for seg in script.segments if seg.speaker == speaker]
    prefix = f"角色 {speaker} 缺少可用参考音。"
    if not segments or speaker == "UNKNOWN":
        return prefix + "请先确认说话人，或在角色设置中上传 3–10 秒参考音并填写对应原文逐字稿和语种。"
    whole = [seg for seg in segments if MIN_REFERENCE_SECONDS <= seg.end - seg.start <= MAX_REFERENCE_SECONDS]
    if whole:
        ids = "、".join(f"#{seg.id}" for seg in whole)
        return prefix + f"可检查片段 {ids}：试听原声、校对原文后，在台词检查器确认「句段时间与原文」及「当前说话人」，保存后重试。也可在角色设置上传参考音。"
    return prefix + "该角色没有 3–10 秒的完整句段，且没有可验证的词级时间，不能直接截取长句前几秒。请在角色设置上传 3–10 秒参考音，填写与音频一致的原文逐字稿和语种；或先将源对白切分、校对为合适句段。"


def automatic_reference(
    script: ProjectScript,
    speaker: str,
    emotion: str | None = None,
) -> ReferencePreview | None:
    reviewed = _reviewed_automatic_reference(script, speaker, emotion=emotion)
    if reviewed is not None:
        return reviewed
    from backend.services.automatic_moss_reference import automatic_moss_reference
    return automatic_moss_reference(script, speaker, emotion=emotion)


def _resolve_valid_vocals_source(script: ProjectScript) -> tuple[Path, Path] | None:
    if not script.media.vocals_wav:
        return None
    source = config.resolve_media_path(script.media.vocals_wav)
    project_dir = (config.WORKSPACE_DIR / script.project_id).resolve()
    is_banquet_dialogue = (
        script.media.cinematic_stems is not None
        and script.media.vocals_wav == script.media.cinematic_stems.dialogue
        and source.name == "dialogue.wav"
        and source.parent.name.startswith("cinematic-")
        and source.parent.parent == project_dir
    )
    if source != project_dir / "vocals.wav" and not is_banquet_dialogue:
        return None
    return source, project_dir


def _slice_window_to_ref(
    script: ProjectScript,
    source: Path,
    project_dir: Path,
    vocals: wave.Wave_read,
    lang: str,
    window_start: float,
    window_end: float,
    transcript: str,
    segment_ids: list[int],
) -> ReferencePreview | None:
    rate, channels, width = vocals.getframerate(), vocals.getnchannels(), vocals.getsampwidth()
    start = max(0, round(window_start * rate))
    requested_end = round(window_end * rate)
    end = min(requested_end, vocals.getnframes())
    if requested_end > end:
        # Only discard a small, proven word-free alignment margin.
        source_end = end / rate
        related = [seg for seg in script.segments if seg.id in segment_ids]
        if (requested_end - end > round(0.05 * rate)
                or not related
                or any(_verified_word_offsets(seg) is None for seg in related)
                or any(word.end > source_end for seg in related for word in seg.words
                       if word.start < window_end and word.end > window_start)):
            return None
    if end - start < MIN_REFERENCE_SECONDS * rate or end - start > MAX_REFERENCE_SECONDS * rate:
        return None
    key = (str(source), source.stat().st_mtime_ns, start, end, channels, width)
    name = f"auto_{hashlib.sha256(repr(key).encode()).hexdigest()[:24]}.wav"
    path = project_dir / "references" / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            draft = Path(temp) / name
            vocals.setpos(start)
            frames = vocals.readframes(end - start)
            if channels == 1 and width == 2 and rate == 32000:
                with wave.open(str(draft), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(32000)
                    output.writeframes(frames)
            else:
                raw = Path(temp) / "raw.wav"
                with wave.open(str(raw), "wb") as source_wav:
                    source_wav.setnchannels(channels)
                    source_wav.setsampwidth(width)
                    source_wav.setframerate(rate)
                    source_wav.writeframes(frames)
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw), "-ac", "1", "-ar", "32000",
                     "-c:a", "pcm_s16le", str(draft)],
                    check=True, capture_output=True, timeout=30,
                )
            draft.replace(path)
    text = prepare_tts_text(unicodedata.normalize("NFKC", transcript), lang)
    if not text:
        return None
    return ReferencePreview(
        wav=f"/workspace/{script.project_id}/references/{name}",
        text=text,
        lang=lang,
        segment_ids=segment_ids,
    )


def extract_confirmed_segment_reference(script: ProjectScript, seg: Segment) -> ReferencePreview | None:
    """Slice a single confirmed segment from project vocals into references/."""
    if seg.speaker == "UNKNOWN" or not _whole_timing_confirmed(seg) or seg.speaker_needs_review or not seg.orig_text.strip():
        return None
    lang = normalize_tts_lang(script.source_lang, default="")
    if lang not in SUPPORTED_SOURCE_LANGS:
        return None
    resolved = _resolve_valid_vocals_source(script)
    if resolved is None:
        return None
    source, project_dir = resolved
    try:
        with wave.open(str(source), "rb") as vocals:
            return _slice_window_to_ref(
                script, source, project_dir, vocals, lang,
                seg.start, seg.end, seg.orig_text.strip(), [seg.id],
            )
    except (ValueError, OSError, EOFError, wave.Error, subprocess.SubprocessError):
        return None


def _reviewed_automatic_reference(
    script: ProjectScript,
    speaker: str,
    *,
    emotion: str | None = None,
) -> ReferencePreview | None:
    if speaker == "UNKNOWN":
        return None
    lang = normalize_tts_lang(script.source_lang, default="")
    if lang not in SUPPORTED_SOURCE_LANGS or not script.media.vocals_wav:
        return None
    candidates = _candidate_clips(script, speaker)
    windows: list[tuple[float, float, str, list[int]]] = []
    seen_segment_tuples: set[tuple[int, ...]] = set()
    if emotion:
        from backend.services.emotion_routing import select_emotion_reference_candidates

        for seg in select_emotion_reference_candidates(script, speaker, emotion):
            if _whole_timing_confirmed(seg) and not seg.speaker_needs_review and MIN_REFERENCE_SECONDS <= seg.end - seg.start <= MAX_REFERENCE_SECONDS:
                key = (seg.id,)
                seen_segment_tuples.add(key)
                windows.append((seg.start, seg.end, seg.orig_text.strip(), [seg.id]))
    for clip in candidates:
        key = tuple(part.id for part in clip)
        if key in seen_segment_tuples:
            continue
        windows.append((
            clip[0].start,
            clip[-1].end,
            " ".join(part.orig_text.strip() for part in clip),
            [part.id for part in clip],
        ))
    for seg in sorted(script.segments, key=lambda item: (item.start, item.id)):
        if seg.speaker == speaker:
            window = _long_reference_window(script, seg)
            if window is not None:
                windows.append(window)
    if not windows:
        return None

    try:
        resolved = _resolve_valid_vocals_source(script)
        if resolved is None:
            return None
        source, project_dir = resolved
        with wave.open(str(source), "rb") as vocals:
            for window_start, window_end, transcript, segment_ids in windows:
                preview = _slice_window_to_ref(
                    script, source, project_dir, vocals, lang,
                    window_start, window_end, transcript, segment_ids,
                )
                if preview is not None:
                    return preview
    except (ValueError, OSError, EOFError, wave.Error, subprocess.SubprocessError):
        return None
    return None
