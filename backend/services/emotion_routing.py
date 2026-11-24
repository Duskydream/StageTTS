"""Acoustic emotion/prosody hints and multi-emotion reference routing for GPT-SoVITS."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

from backend import config
from backend.schemas import EmotionRef, EmotionType, ProjectScript, ReferencePreview, Segment
from backend.services.text_normalization import normalize_tts_lang
from workers.prosody_worker import measure_clip, read_wav

SUPPORTED_EMOTIONS: tuple[EmotionType, ...] = ("neutral", "happy", "angry", "sad", "whisper")
_MIN_ACOUSTIC_SEC = 0.35
_MAX_SER_SEC = 30.0
_CACHE_FILENAME = "_acoustic_hints_v1.json"


def _resolve_project_vocals(script: ProjectScript) -> Path | None:
    if not script.media.vocals_wav:
        return None
    try:
        path = config.resolve_media_path(script.media.vocals_wav)
    except ValueError:
        return None
    if not path.is_file():
        return None
    ws_root = config.WORKSPACE_DIR.resolve()
    assets_root = config.ASSETS_DIR.resolve()
    if ws_root not in path.parents and assets_root not in path.parents:
        return None
    return path


def _non_overlapping_ids(segments: list[Segment]) -> set[int]:
    ordered = sorted(segments, key=lambda item: (item.start, item.id))
    overlapping: set[int] = set()
    for idx, seg in enumerate(ordered):
        for nxt in range(idx + 1, len(ordered)):
            other = ordered[nxt]
            if other.start >= seg.end:
                break
            overlapping.add(seg.id)
            overlapping.add(other.id)
    return {seg.id for seg in ordered if seg.id not in overlapping}


def _ser_ready() -> bool:
    if not (config.ENV_ANALYSIS_PYTHON and config.EMOTION_MODEL_PATH and config.EMOTION_USER_AUTHORIZED):
        return False
    python_path = Path(config.ENV_ANALYSIS_PYTHON)
    model_path = config.EMOTION_MODEL_PATH
    required = ("config.yaml", "configuration.json", "model.pt", "tokens.txt")
    return (
        python_path.is_file()
        and model_path.is_dir()
        and all((model_path / name).is_file() for name in required)
    )


def _run_ser_batch(
    vocals_path: Path,
    spans: list[dict[str, Any]],
    *,
    allow_gpu_ser: bool,
) -> dict[int, dict[str, Any]]:
    if not spans or not _ser_ready():
        return {}
    worker_path = config.ROOT_DIR / "workers" / "acoustic_hint_worker.py"
    if not worker_path.is_file():
        return {}
    device = config.EMOTION_DEVICE if allow_gpu_ser else "cpu"
    with tempfile.TemporaryDirectory(prefix="stagetts_ser_") as temp_dir:
        tmp = Path(temp_dir)
        audio_16k = tmp / "vocals_16k.wav"
        req_path = tmp / "request.json"
        out_path = tmp / "result.json"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(vocals_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_16k),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            req_path.write_text(
                json.dumps(
                    {
                        "audio_16k_wav": str(audio_16k),
                        "model_path": str(config.EMOTION_MODEL_PATH),
                        "device": device,
                        "spans": spans,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONUTF8": "1"}
            proc = subprocess.run(
                [
                    config.ENV_ANALYSIS_PYTHON,
                    str(worker_path),
                    "--input",
                    str(req_path),
                    "--output",
                    str(out_path),
                ],
                env=env,
                cwd=config.ROOT_DIR,
                capture_output=True,
                timeout=min(config.PARSE_TIMEOUT_SEC, 300),
                check=False,
            )
            if proc.returncode != 0 or not out_path.is_file():
                return {}
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, subprocess.SubprocessError):
            return {}
    if data.get("status") != "ok":
        return {}
    mapped: dict[int, dict[str, Any]] = {}
    for row in data.get("results") or []:
        if isinstance(row, dict) and isinstance(row.get("id"), int):
            mapped[row["id"]] = row
    return mapped


def _infer_suggested_emotion(
    *,
    rms_dbfs: float | None,
    rms_z: float | None,
    f0_range_z: float | None,
    energy_active_ratio: float | None,
    ser_top: str | None,
    ser_score: float | None,
) -> EmotionType:
    active = energy_active_ratio if energy_active_ratio is not None else 0.0
    score = ser_score if ser_score is not None else 0.0
    is_quiet = (rms_z is not None and rms_z <= -1.1) or (
        rms_dbfs is not None and rms_dbfs <= -29.0 and (rms_z is None or rms_z <= -0.5)
    )
    if is_quiet and active >= 0.35 and ser_top not in {"angry", "happy"}:
        return "whisper"
    if (ser_top == "angry" and score >= 0.42) or (
        rms_z is not None and rms_z >= 1.35 and ser_top in {"angry", "disgusted"}
    ):
        return "angry"
    if (ser_top in {"happy", "surprised"} and score >= 0.42) or (
        rms_z is not None
        and rms_z >= 0.85
        and (f0_range_z or 0.0) >= 0.75
        and ser_top != "angry"
    ):
        return "happy"
    if ser_top in {"sad", "fearful"} and score >= 0.42:
        return "sad"
    return "neutral"


def _compute_z_scores(values_by_id: dict[int, float | None]) -> dict[int, float | None]:
    valid = [val for val in values_by_id.values() if val is not None]
    if len(valid) < 2:
        return {seg_id: None for seg_id in values_by_id}
    dev = statistics.pstdev(valid)
    if dev <= 1e-12:
        return {seg_id: None for seg_id in values_by_id}
    mean = statistics.mean(valid)
    return {
        seg_id: (round((val - mean) / dev, 3) if val is not None else None)
        for seg_id, val in values_by_id.items()
    }


def _cache_file(script: ProjectScript) -> Path | None:
    project_dir = (config.WORKSPACE_DIR / script.project_id).resolve()
    ws_root = config.WORKSPACE_DIR.resolve()
    if ws_root not in project_dir.parents:
        return None
    return project_dir / "references" / _CACHE_FILENAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".acoustic-hints-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2))
            stream.flush()
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def extract_segment_acoustics(
    script: ProjectScript,
    *,
    use_ser: bool = True,
    allow_gpu_ser: bool = False,
) -> dict[int, dict[str, Any]]:
    """Measure per-segment RMS/activity prosody and optional offline emotion2vec+ hints."""
    vocals_path = _resolve_project_vocals(script)
    if vocals_path is None:
        return {}

    try:
        stat = vocals_path.stat()
        with wave.open(str(vocals_path), "rb") as stream:
            rate = stream.getframerate()
            wav_duration = (stream.getnframes() / float(rate)) if rate > 0 else 0.0
    except (OSError, wave.Error, EOFError):
        return {}
    if wav_duration <= 0:
        return {}

    ser_active = bool(use_ser and _ser_ready())
    key_payload = {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "segments": [[s.id, round(s.start, 3), round(s.end, 3), s.speaker] for s in script.segments],
        "use_ser": bool(use_ser),
        "ser_active": ser_active,
        "revision": config.EMOTION_REVISION,
    }
    cache_digest = hashlib.sha256(json.dumps(key_payload, separators=(",", ":")).encode("utf-8")).hexdigest()
    cache_path = _cache_file(script)
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_digest and isinstance(cached.get("hints"), dict):
                return {int(k): v for k, v in cached["hints"].items() if isinstance(v, dict)}
        except (OSError, ValueError, TypeError):
            pass

    clean_ids = _non_overlapping_ids(script.segments)
    raw_metrics: dict[int, dict[str, Any]] = {}
    ser_spans: list[dict[str, Any]] = []
    by_speaker: dict[str, list[Segment]] = {}

    for seg in script.segments:
        if seg.id not in clean_ids or seg.speaker == "UNKNOWN":
            continue
        clip_end = min(seg.end, wav_duration)
        dur = clip_end - seg.start
        if dur < _MIN_ACOUSTIC_SEC:
            continue
        try:
            samples, sample_rate, _ = read_wav(vocals_path, start=seg.start, end=clip_end)
            measurements = measure_clip(samples, sample_rate, enable_f0=False)
        except (OSError, ValueError, wave.Error, EOFError):
            continue
        metric_map = {item["name"]: item.get("value") for item in measurements}
        raw_metrics[seg.id] = {
            "speaker": seg.speaker,
            "rms_dbfs": round(float(metric_map["rms_dbfs"]), 2) if metric_map.get("rms_dbfs") is not None else None,
            "f0_median_hz": (
                round(float(metric_map["f0_median_hz"]), 1)
                if metric_map.get("f0_median_hz") is not None
                else None
            ),
            "f0_range_hz": (
                round(float(metric_map["f0_range_hz"]), 1)
                if metric_map.get("f0_range_hz") is not None
                else None
            ),
            "energy_active_ratio": (
                round(float(metric_map["energy_active_ratio"]), 3)
                if metric_map.get("energy_active_ratio") is not None
                else None
            ),
        }
        by_speaker.setdefault(seg.speaker, []).append(seg)
        if ser_active and 0.4 <= dur <= _MAX_SER_SEC:
            ser_spans.append({"id": seg.id, "start": round(seg.start, 4), "end": round(clip_end, 4)})

    rms_z_map: dict[int, float | None] = {}
    f0_range_z_map: dict[int, float | None] = {}
    for segs in by_speaker.values():
        rms_vals = {seg.id: raw_metrics[seg.id]["rms_dbfs"] for seg in segs}
        f0_vals = {seg.id: raw_metrics[seg.id]["f0_range_hz"] for seg in segs}
        rms_z_map.update(_compute_z_scores(rms_vals))
        f0_range_z_map.update(_compute_z_scores(f0_vals))

    ser_results = _run_ser_batch(vocals_path, ser_spans, allow_gpu_ser=allow_gpu_ser) if ser_spans else {}

    hints: dict[int, dict[str, Any]] = {}
    for seg_id, base in raw_metrics.items():
        rms_z = rms_z_map.get(seg_id)
        f0_range_z = f0_range_z_map.get(seg_id)
        ser = ser_results.get(seg_id) or {}
        ser_top = ser.get("top_label")
        ser_score = ser.get("top_score")
        suggested = _infer_suggested_emotion(
            rms_dbfs=base["rms_dbfs"],
            rms_z=rms_z,
            f0_range_z=f0_range_z,
            energy_active_ratio=base["energy_active_ratio"],
            ser_top=ser_top,
            ser_score=ser_score,
        )
        hints[seg_id] = {
            "rms_dbfs": base["rms_dbfs"],
            "rms_z": rms_z,
            "f0_median_hz": base["f0_median_hz"],
            "f0_range_hz": base["f0_range_hz"],
            "f0_range_z": f0_range_z,
            "energy_active_ratio": base["energy_active_ratio"],
            "ser_top": ser_top,
            "ser_score": ser_score,
            "scores": ser.get("scores") or {},
            "suggested_emotion": suggested,
        }

    if cache_path is not None:
        try:
            _atomic_write_json(
                cache_path,
                {"cache_key": cache_digest, "hints": {str(k): v for k, v in hints.items()}},
            )
        except OSError:
            pass
    return hints


def _is_eligible_ref_segment(seg: Segment, clean_ids: set[int], speaker: str) -> bool:
    if (
        seg.speaker != speaker
        or speaker == "UNKNOWN"
        or seg.keep_original
        or not seg.orig_text.strip()
        or seg.id not in clean_ids
    ):
        return False
    dur = seg.end - seg.start
    if seg.timing_source == "moss_joint_generated" and seg.alignment_granularity == "segment":
        return dur >= 3.2
    return not seg.segment_timing_needs_review and not seg.speaker_needs_review and 3.0 <= dur <= 10.0


def select_emotion_reference_candidates(
    script: ProjectScript,
    speaker: str,
    target_emotion: str,
    acoustics: dict[int, dict[str, Any]] | None = None,
) -> list[Segment]:
    """Rank clean non-overlapping reference candidates for a speaker and emotion."""
    clean_ids = _non_overlapping_ids(script.segments)
    hints = acoustics if acoustics is not None else extract_segment_acoustics(script, use_ser=True, allow_gpu_ser=False)

    eligible: list[Segment] = []
    for seg in script.segments:
        if not _is_eligible_ref_segment(seg, clean_ids, speaker):
            continue
        hint = hints.get(seg.id)
        if hint is not None:
            active = hint.get("energy_active_ratio")
            rms_dbfs = hint.get("rms_dbfs")
            if (active is not None and active < 0.28) or (rms_dbfs is not None and rms_dbfs < -42.0):
                continue
        eligible.append(seg)

    if not eligible:
        return []

    def _dur_penalty(seg: Segment) -> float:
        dur = seg.end - seg.start
        return 0.12 * abs(min(8.0, dur) - 5.5) + (0.5 if dur > 10.0 else 0.0)

    if target_emotion == "neutral":
        return sorted(
            eligible,
            key=lambda seg: (
                abs((hints.get(seg.id) or {}).get("rms_z") or 0.0)
                + 0.4 * abs((hints.get(seg.id) or {}).get("f0_range_z") or 0.0)
                + _dur_penalty(seg)
                - (0.6 if (hints.get(seg.id) or {}).get("ser_top") == "neutral" else 0.0)
                - (0.25 if seg.emotion == "neutral" else 0.0)
            ),
        )

    scored: list[tuple[float, Segment]] = []
    for seg in eligible:
        hint = hints.get(seg.id) or {}
        scores = hint.get("scores") or {}
        ser_top = hint.get("ser_top")
        rms_z = hint.get("rms_z")
        rms_dbfs = hint.get("rms_dbfs")
        f0_range_z = hint.get("f0_range_z")

        matched = seg.emotion == target_emotion
        evidence_score = 0.55 if seg.emotion == target_emotion else 0.0

        if target_emotion == "happy":
            ser_val = max(float(scores.get("happy") or 0.0), float(scores.get("surprised") or 0.0) * 0.85)
            if ser_top in {"happy", "surprised"} or ser_val >= 0.30:
                matched = True
                evidence_score += ser_val * 1.4
            if (rms_z is not None and rms_z >= 0.35) or (f0_range_z is not None and f0_range_z >= 0.35):
                matched = True
                evidence_score += 0.35 * max(0.0, rms_z or 0.0) + 0.35 * max(0.0, f0_range_z or 0.0)
        elif target_emotion == "angry":
            ser_val = float(scores.get("angry") or 0.0)
            if ser_top == "angry" or ser_val >= 0.30:
                matched = True
                evidence_score += ser_val * 1.4
            if rms_z is not None and rms_z >= 0.45:
                matched = True
                evidence_score += 0.45 * rms_z
        elif target_emotion == "sad":
            ser_val = max(float(scores.get("sad") or 0.0), float(scores.get("fearful") or 0.0) * 0.8)
            if ser_top in {"sad", "fearful"} or ser_val >= 0.30:
                matched = True
                evidence_score += ser_val * 1.4
            if rms_z is not None and rms_z <= -0.25:
                matched = True
                evidence_score += 0.4 * abs(rms_z)
        elif target_emotion == "whisper":
            if (rms_z is not None and rms_z <= -0.65) or (
                rms_dbfs is not None and rms_dbfs <= -26.5 and (rms_z is None or rms_z <= -0.25)
            ):
                matched = True
                evidence_score += 0.5 * abs(rms_z if rms_z is not None else -0.7)

        if matched:
            scored.append((evidence_score - 0.08 * _dur_penalty(seg), seg))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [seg for _, seg in scored]


def _has_valid_emotion_ref(ref: EmotionRef | None) -> bool:
    if not ref or not ref.wav or not ref.text.strip() or not ref.lang.strip():
        return False
    try:
        return config.resolve_media_path(ref.wav).is_file()
    except ValueError:
        return False


def _materialize_segment_reference(script: ProjectScript, seg: Segment) -> EmotionRef | None:
    preview: ReferencePreview | None = None
    if seg.timing_source == "moss_joint_generated" and seg.alignment_granularity == "segment":
        from backend.services.automatic_moss_reference import extract_moss_segment_reference

        preview = extract_moss_segment_reference(script, seg)
    else:
        from backend.services.references import extract_confirmed_segment_reference

        preview = extract_confirmed_segment_reference(script, seg)
    if preview is None:
        return None
    return EmotionRef(wav=preview.wav, text=preview.text, lang=preview.lang)


def build_character_emotion_refs(
    script: ProjectScript,
    speaker: str | None = None,
    *,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, list[str]]:
    """Populate per-character emotion_refs using acoustics and clean source segments."""
    if not normalize_tts_lang(script.source_lang, default=""):
        return {}
    if _resolve_project_vocals(script) is None:
        return {}

    target_speakers = [speaker] if speaker is not None else [key for key in script.characters if key != "UNKNOWN"]
    target_speakers = [spk for spk in target_speakers if spk in script.characters and spk != "UNKNOWN"]
    if not target_speakers:
        return {}

    if log:
        log(f"正在分析声学特征并为 {len(target_speakers)} 个角色匹配多情绪参考音…")
    acoustics = extract_segment_acoustics(script, use_ser=True, allow_gpu_ser=False)
    resolved_by_seg_id: dict[int, EmotionRef] = {}
    populated: dict[str, list[str]] = {}

    # Plan candidate orders per (speaker, emotion) and prewarm top MOSS clips in one GPU batch.
    planned_candidates: dict[tuple[str, EmotionType], list[Segment]] = {}
    moss_batch_segs: list[Segment] = []
    for spk in target_speakers:
        character = script.characters[spk]
        used_preview_ids: set[int] = set()
        for emotion in SUPPORTED_EMOTIONS:
            if not overwrite and _has_valid_emotion_ref(character.emotion_refs.get(emotion)):
                continue
            candidates = select_emotion_reference_candidates(script, spk, emotion, acoustics=acoustics)
            if not candidates:
                continue
            preferred = [seg for seg in candidates if seg.id not in used_preview_ids]
            ordered = [*preferred, *[seg for seg in candidates if seg.id in used_preview_ids]]
            planned_candidates[(spk, emotion)] = ordered
            top_seg = ordered[0]
            used_preview_ids.add(top_seg.id)
            if top_seg.timing_source == "moss_joint_generated" and top_seg.alignment_granularity == "segment":
                moss_batch_segs.append(top_seg)

    if moss_batch_segs:
        if log:
            log(f"正在使用 MOSS GPU 批量校对 {len({s.id for s in moss_batch_segs})} 条参考音片段…")
        try:
            from backend.services.automatic_moss_reference import prewarm_moss_segment_references

            prewarmed = prewarm_moss_segment_references(script, moss_batch_segs)
            for seg_id, preview in prewarmed.items():
                resolved_by_seg_id[seg_id] = EmotionRef(
                    wav=preview.wav,
                    text=preview.text,
                    lang=preview.lang,
                )
        except (ValueError, OSError, subprocess.SubprocessError):
            pass

    for spk in target_speakers:
        character = script.characters[spk]
        added: list[str] = []
        used_seg_ids: set[int] = set()

        for emotion in SUPPORTED_EMOTIONS:
            ordered = planned_candidates.get((spk, emotion))
            if not ordered:
                continue
            preferred = [seg for seg in ordered if seg.id not in used_seg_ids]
            candidates = [*preferred, *[seg for seg in ordered if seg.id in used_seg_ids]]

            for seg in candidates:
                ref = resolved_by_seg_id.get(seg.id)
                if ref is None:
                    try:
                        ref = _materialize_segment_reference(script, seg)
                    except (ValueError, OSError, subprocess.SubprocessError):
                        ref = None
                    if ref is None:
                        continue
                    resolved_by_seg_id[seg.id] = ref
                character.emotion_refs[emotion] = ref
                used_seg_ids.add(seg.id)
                added.append(emotion)
                break
        if added:
            populated[spk] = added
            if log:
                log(f"角色 {character.display_name or spk} 已配置参考音：{'、'.join(added)}")

    return populated
