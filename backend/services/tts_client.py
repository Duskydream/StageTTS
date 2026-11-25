"""GPT-SoVITS 推理客户端：按说话人分组批量合成 + 动态情绪参考音频路由。"""

from __future__ import annotations

import hashlib
import inspect
import io
import re
import tempfile
import wave
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import yaml
from pydub import AudioSegment

from backend import config
from backend.schemas import (
    CharacterConfig,
    EmotionRef,
    ProjectScript,
    Segment,
)
from backend.services.emotion_routing import (
    _non_overlapping_ids,
    _resolve_project_vocals,
    extract_segment_acoustics,
)
from backend.services.media_mixer import apply_gentle_tempo
from backend.services.pause_alignment import assemble_island_wavs, plan_segment_pause_synthesis
from backend.services.references import automatic_reference, reference_unavailable_reason
from backend.services.task_runner import ensure_tts_running, gpu_task_guard
from backend.services.text_normalization import (
    deterministic_seed,
    normalize_tts_lang,
    prepare_tts_text,
)

_EMOTION_SAMPLING: dict[str, dict[str, float | int]] = {
    "whisper": {"top_k": 12, "top_p": 0.88, "temperature": 0.80},
    "sad": {"top_k": 15, "top_p": 0.90, "temperature": 0.85},
    "neutral": {"top_k": 15, "top_p": 0.92, "temperature": 0.88},
    "happy": {"top_k": 18, "top_p": 0.95, "temperature": 0.95},
    "angry": {"top_k": 18, "top_p": 0.95, "temperature": 0.95},
}


class TTSServiceUnavailableError(RuntimeError):
    """GPT-SoVITS 推理服务不可达或响应超时。"""


# Only track components that StageTTS changed. None marks an uncertain switch.
_active_weights: dict[str, str | None] = {}
_restore_weights: dict[str, str] = {}


def reset_active_weights_cache() -> None:
    """GPT-SoVITS 按需重启后清空模块级权重缓存。"""
    _active_weights.clear()
    _restore_weights.clear()


def _active_tts_engine() -> str:
    from backend.services.tts_settings import read_tts_engine

    return read_tts_engine()


def _tts_base_url(engine: str | None = None) -> str:
    selected = engine or _active_tts_engine()
    return config.COSYVOICE3_API_URL if selected == "cosyvoice3" else config.TTS_API_URL


def check_tts_service(timeout: float = 1.5) -> bool:
    """快速探测当前已选配音推理服务是否在线。"""
    try:
        resp = httpx.get(f"{_tts_base_url()}/", timeout=timeout)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


def _resolve_ref_for_segment(
    script: ProjectScript, character: CharacterConfig | None, segment: Segment,
) -> tuple[EmotionRef, Path] | None:
    has_any_valid_manual = False
    if character:
        refs = character.emotion_refs
        # 1. Exact emotion ref configured on character takes top priority.
        exact = refs.get(segment.emotion)
        if exact and exact.text.strip() and exact.lang.strip():
            try:
                path = config.resolve_media_path(exact.wav)
                if path.is_file():
                    return exact, path
            except ValueError:
                pass
        # Check whether the character has any valid manual/populated ref at all.
        for emotion in dict.fromkeys(("neutral", *refs)):
            ref = refs.get(emotion)
            if ref and ref.text.strip() and ref.lang.strip():
                try:
                    path = config.resolve_media_path(ref.wav)
                    if path.is_file():
                        has_any_valid_manual = True
                        break
                except ValueError:
                    continue
        # 2. If character has no configured refs, try emotion-matched automatic clip before neutral fallback.
        if not has_any_valid_manual:
            auto_emotion = automatic_reference(script, segment.speaker, emotion=segment.emotion)
            if auto_emotion is not None:
                return auto_emotion, config.resolve_media_path(auto_emotion.wav)
        # 3. Fall back to character's configured neutral/other ref.
        for emotion in dict.fromkeys(("neutral", *refs)):
            ref = refs.get(emotion)
            if ref and ref.text.strip() and ref.lang.strip():
                try:
                    path = config.resolve_media_path(ref.wav)
                except ValueError:
                    continue
                if path.is_file():
                    return ref, path
    ref = automatic_reference(script, segment.speaker, emotion=segment.emotion)
    return (ref, config.resolve_media_path(ref.wav)) if ref else None


def _default_weights() -> dict[str, str | None]:
    weights = {}
    if config.GPT_SOVITS_CONFIG:
        try:
            settings = yaml.safe_load(Path(config.GPT_SOVITS_CONFIG).read_text(encoding="utf-8")) or {}
            custom = settings.get("custom") or {}
            version = settings.get(custom.get("version", "v2")) or {}
            weights = {**version, **{key: value for key, value in custom.items() if value}}
        except (OSError, yaml.YAMLError, AttributeError, TypeError):
            pass
    return {
        "gpt": config.TTS_DEFAULT_GPT_WEIGHTS or str(weights.get("t2s_weights_path") or "") or None,
        "sovits": config.TTS_DEFAULT_SOVITS_WEIGHTS or str(weights.get("vits_weights_path") or "") or None,
    }


def _set_weight(client: httpx.Client, kind: str, custom: str, default: str | None) -> None:
    if not custom and kind not in _active_weights:
        return
    baseline = _restore_weights.get(kind) or config.resolve_weights_path(default or "")
    desired = config.resolve_weights_path(custom) if custom else baseline
    if kind not in _active_weights and desired == baseline:
        return
    if not baseline:
        raise RuntimeError(f"缺少默认 {kind.upper()} 权重")
    if _active_weights.get(kind) != desired:
        _restore_weights[kind] = baseline
        _active_weights[kind] = None
        try:
            response = client.get(
                f"{config.TTS_API_URL}/set_{kind}_weights",
                params={"weights_path": desired}, timeout=config.TTS_TIMEOUT_SEC,
            )
        except httpx.HTTPError as exc:
            raise TTSServiceUnavailableError(f"无法连接 GPT-SoVITS {kind.upper()} 权重接口") from exc
        if response.status_code != 200:
            raise RuntimeError(f"切换 {kind.upper()} 权重失败 ({response.status_code}): {response.text[:300]}")
    if desired == baseline:
        _active_weights.pop(kind, None)
        _restore_weights.pop(kind, None)
    else:
        _active_weights[kind] = desired


def _switch_weights(
    client: httpx.Client, defaults: dict[str, str | None], gpt: str | None, sovits: str | None,
) -> None:
    requested = {"sovits": (sovits or "").strip(), "gpt": (gpt or "").strip()}
    missing = [kind.upper() for kind, path in requested.items() if path and not defaults[kind]]
    if missing:
        raise RuntimeError(f"请先配置 GPT-SoVITS 的默认 {' / '.join(missing)} 权重，以便切换后恢复")
    # Loading SoVITS can change the active model version.
    for kind, path in requested.items():
        _set_weight(client, kind, path, defaults[kind])


@contextmanager
def _weight_scope(client: httpx.Client, errors: list[str]) -> Iterator[dict[str, str | None]]:
    defaults = _default_weights()
    try:
        yield defaults
    finally:
        for kind in ("sovits", "gpt"):
            try:
                _set_weight(client, kind, "", defaults[kind])
            except TTSServiceUnavailableError:
                break
            except RuntimeError as exc:
                errors.append(f"恢复默认权重失败：{exc}")


def measure_wav_duration(wav_path: Path) -> float:
    """精确测量合成 WAV 文件时长（秒）。"""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return round(wf.getnframes() / float(rate), 3)
    except (wave.Error, EOFError):
        pass
    with wav_path.open("rb") as source:
        audio = AudioSegment.from_wav(source)
    return round(len(audio) / 1000.0, 3)


def _slice_aux_wav_cached(
    vocals_path: Path,
    project_id: str,
    seg: Segment,
) -> Path | None:
    """Slice and cache a clean 32kHz mono clip for multi-reference timbre fusion."""
    refs_dir = (config.WORKSPACE_DIR / project_id / "references").resolve()
    ws_root = config.WORKSPACE_DIR.resolve()
    if ws_root not in refs_dir.parents:
        return None
    try:
        stat = vocals_path.stat()
        with wave.open(str(vocals_path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            total_frames = wf.getnframes()
            start_frame = max(0, round(seg.start * rate))
            end_frame = min(total_frames, round(seg.end * rate))
            if end_frame - start_frame < round(3.0 * rate) or end_frame - start_frame > round(10.0 * rate):
                return None
            key = (str(vocals_path), stat.st_mtime_ns, start_frame, end_frame)
            name = f"aux_{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.wav"
            out_path = refs_dir / name
            if out_path.is_file():
                return out_path
            refs_dir.mkdir(parents=True, exist_ok=True)
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)
    except (OSError, wave.Error, EOFError, ValueError):
        return None

    try:
        audio = AudioSegment(
            data=frames,
            sample_width=width,
            frame_rate=rate,
            channels=channels,
        ).set_channels(1).set_frame_rate(32000).set_sample_width(2)
        with tempfile.TemporaryDirectory(dir=refs_dir) as tmp_dir:
            tmp_file = Path(tmp_dir) / name
            with tmp_file.open("wb") as fp:
                audio.export(fp, format="wav")
            tmp_file.replace(out_path)
        return out_path
    except (OSError, ValueError):
        return None


def _collect_aux_ref_paths(
    script: ProjectScript,
    character: CharacterConfig | None,
    speaker: str,
    primary_ref_path: Path | None,
    *,
    acoustics: dict[int, dict] | None = None,
    max_aux: int = 3,
) -> list[Path]:
    """Gather up to `max_aux` distinct clean speaker clips for GPT-SoVITS timbre fusion."""
    if primary_ref_path is None or speaker == "UNKNOWN":
        return []
    primary_resolved = primary_ref_path.resolve()
    chosen: list[Path] = []
    seen: set[Path] = {primary_resolved}

    if character is not None:
        for ref in character.emotion_refs.values():
            if not ref or not ref.wav:
                continue
            try:
                candidate = config.resolve_media_path(ref.wav).resolve()
            except ValueError:
                continue
            if candidate not in seen and candidate.is_file():
                seen.add(candidate)
                chosen.append(candidate)
                if len(chosen) >= max_aux:
                    return chosen

    vocals_path = _resolve_project_vocals(script)
    if vocals_path is None:
        return chosen
    hints = acoustics if acoustics is not None else extract_segment_acoustics(script, use_ser=False, allow_gpu_ser=False)
    clean_ids = _non_overlapping_ids(script.segments)
    pool = [
        seg for seg in script.segments
        if (
            seg.speaker == speaker
            and seg.id in clean_ids
            and not seg.keep_original
            and seg.orig_text.strip()
            and not seg.speaker_needs_review
            and 3.0 <= (seg.end - seg.start) <= 10.0
        )
    ]
    pool.sort(
        key=lambda seg: (
            -(float((hints.get(seg.id) or {}).get("clarity_score") or 0.5)),
            abs((seg.end - seg.start) - 5.5),
            seg.id,
        )
    )
    for seg in pool:
        if len(chosen) >= max_aux:
            break
        sliced = _slice_aux_wav_cached(vocals_path, script.project_id, seg)
        if sliced is not None:
            resolved = sliced.resolve()
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                chosen.append(resolved)
    return chosen


def _leading_silence_seconds(wav_bytes: bytes, *, silence_thresh_db: float = -46.0) -> float:
    try:
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    except Exception:
        return 0.0
    if len(audio) <= 100 or audio.rms == 0:
        return 0.0
    thresh = audio.max_possible_amplitude * (10.0 ** (silence_thresh_db / 20.0))
    for pos in range(0, len(audio), 20):
        if audio[pos:pos + 20].rms >= thresh:
            return pos / 1000.0
    return len(audio) / 1000.0


def _wav_duration_bytes(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            return (wf.getnframes() / float(rate)) if rate > 0 else 0.0
    except Exception:
        return 0.0


def _expected_max_speech_seconds(text: str, text_lang: str) -> float:
    base_lang = normalize_tts_lang(text_lang).removeprefix("all_").split("-")[0]
    cjk = base_lang in {"zh", "ja", "yue", "ko"}
    cleaned = re.sub(r"[\s，。！？；…,.!?;\"'“”‘’（）()]+", "", text or "")
    units = len(cleaned) if cjk else max(1, len(cleaned.split()))
    return max(5.0, units * (0.48 if cjk else 0.75) + 2.5)


def _trim_runaway_edge_silence(wav_bytes: bytes, *, silence_thresh_db: float = -46.0) -> bytes:
    """Trim >350ms dead leading/trailing silence emitted by TTS while keeping 30ms breath cushion."""
    try:
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    except Exception:
        return wav_bytes
    if len(audio) <= 150 or audio.rms == 0:
        return wav_bytes
    thresh = audio.max_possible_amplitude * (10.0 ** (silence_thresh_db / 20.0))
    first_ms, last_ms = 0, len(audio)
    for pos in range(0, len(audio), 20):
        if audio[pos:pos + 20].rms >= thresh:
            first_ms = pos
            break
    for pos in range(len(audio), 0, -20):
        if audio[max(0, pos - 20):pos].rms >= thresh:
            last_ms = pos
            break
    if last_ms <= first_ms:
        return wav_bytes
    if first_ms <= 350 and (len(audio) - last_ms) <= 350:
        return wav_bytes
    trimmed = audio[max(0, first_ms - 30):min(len(audio), last_ms + 30)]
    buf = io.BytesIO()
    trimmed.export(buf, format="wav")
    return buf.getvalue()


def _synthesize_single_line(
    client: httpx.Client,
    seg: Segment,
    ref: EmotionRef,
    ref_path: Path,
    out_wav: Path,
    glossary: dict[str, str] | None = None,
    *,
    api_url: str | None = None,
    engine: str = "gpt_sovits",
    aux_ref_paths: list[Path] | None = None,
) -> None:
    text_lang = normalize_tts_lang(seg.target_lang)
    prompt_lang = normalize_tts_lang(ref.lang)
    base_url = (api_url or _tts_base_url(engine)).rstrip("/")
    label = "CosyVoice 3" if engine == "cosyvoice3" else "GPT-SoVITS"
    sampling = _EMOTION_SAMPLING.get(seg.emotion, _EMOTION_SAMPLING["neutral"])
    prepared_text = prepare_tts_text(seg.target_text, text_lang, glossary)
    base_seed = deterministic_seed(seg.id, seg.target_text, ref.text, ref_path.name)
    base_payload = {
        "text": prepared_text,
        "text_lang": text_lang,
        "ref_audio_path": str(ref_path),
        "prompt_text": prepare_tts_text(ref.text, prompt_lang),
        "prompt_lang": prompt_lang,
        "speed_factor": 1.0,
        "media_type": "wav",
        "text_split_method": "cut0",
        "fragment_interval": 0.0,
        "batch_size": 1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        **sampling,
    }
    if engine == "gpt_sovits" and aux_ref_paths:
        primary_resolved = ref_path.resolve()
        aux_list = [str(p) for p in aux_ref_paths if p.resolve() != primary_resolved and p.is_file()]
        if aux_list:
            base_payload["aux_ref_audio_paths"] = aux_list

    max_reasonable = _expected_max_speech_seconds(prepared_text, text_lang)
    best_bytes: bytes | None = None
    best_dur: float = float("inf")

    for attempt in range(3):
        payload = {**base_payload, "seed": (base_seed + attempt) % (2**31 - 1)}
        try:
            resp = client.post(
                f"{base_url}/tts",
                json=payload,
                timeout=config.TTS_TIMEOUT_SEC,
            )
        except httpx.HTTPError as exc:
            raise TTSServiceUnavailableError(
                f"无法连接 {label} 服务 ({base_url})，请确认推理服务已启动: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"片段 #{seg.id} TTS 推理失败 ({resp.status_code}): {resp.text[:300]}"
            )
        if not resp.content:
            raise RuntimeError(f"片段 #{seg.id} TTS 返回空音频流")

        audio_bytes = resp.content
        dur = _wav_duration_bytes(audio_bytes)
        lead_silence = _leading_silence_seconds(audio_bytes)
        if dur > 0 and dur < best_dur:
            best_bytes = audio_bytes
            best_dur = dur
        elif best_bytes is None:
            best_bytes = audio_bytes
        if dur <= max_reasonable and lead_silence < 1.2:
            break

    assert best_bytes is not None
    out_wav.write_bytes(_trim_runaway_edge_silence(best_bytes))


def synthesize_segments_grouped(
    script: ProjectScript,
    segment_ids: Optional[List[int]] = None,
    *,
    require_shorter: bool = False,
    require_longer: bool = False,
) -> Tuple[List[int], List[str]]:
    """按 speaker 分组批量调用 GPT-SoVITS (api_v2.py) 合成单句 WAV。

    严格遵守 ARCHITECTURE.md 铁律：
    1. 过滤 keep_original == True 的片段；segment_ids 为 None 时合成全部非 ready 片段。
    2. 先按 speaker 分组，每个角色最多切换一次 GPT/SoVITS 权重。
    3. 动态情绪参考音频路由 + 缺失时零样本截取 vocals.wav 兜底。
    4. 测量调速后的成品，成功后替换试听 WAV。
    """
    target_set = set(segment_ids) if segment_ids is not None else None
    grouped: Dict[str, List[Segment]] = defaultdict(list)

    for seg in script.segments:
        if seg.keep_original:
            continue
        if target_set is not None and seg.id not in target_set:
            continue
        if target_set is None and seg.status == "ready":
            continue
        grouped[seg.speaker].append(seg)

    if not grouped:
        return [], []

    tts_dir = config.WORKSPACE_DIR / script.project_id / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    synthesized_ids: List[int] = []
    errors: List[str] = []

    # Validate references before loading GPU models. Missing references are
    # actionable project errors, not failures of the inference service.
    resolved_refs = {}
    eligible_groups = defaultdict(list)
    for speaker, segments in grouped.items():
        character = script.characters.get(speaker)
        for seg in segments:
            if not seg.target_text.strip():
                seg.status = "error"
                errors.append(f"片段 #{seg.id} 目标台词为空，跳过合成")
                continue
            key = (speaker, seg.emotion)
            try:
                if key not in resolved_refs:
                    resolved_refs[key] = _resolve_ref_for_segment(script, character, seg)
                if resolved_refs[key] is None:
                    raise ValueError(reference_unavailable_reason(script, speaker))
            except Exception as exc:
                seg.status = "error"
                errors.append(f"片段 #{seg.id} 合成前检查失败：{exc}")
                continue
            eligible_groups[speaker].append(seg)
    if not eligible_groups:
        return [], errors

    engine = _active_tts_engine()
    api_url = _tts_base_url(engine)
    accepts_extended_args = "aux_ref_paths" in inspect.signature(_synthesize_single_line).parameters
    acoustics = (
        extract_segment_acoustics(script, use_ser=False, allow_gpu_ser=False)
        if accepts_extended_args and _resolve_project_vocals(script) is not None
        else {}
    )
    aux_cache: dict[tuple[str, Path | None], list[Path]] = {}

    with gpu_task_guard():
        try:
            ensure_tts_running(engine=engine)
        except RuntimeError as exc:
            raise TTSServiceUnavailableError(str(exc)) from exc
        weight_scope = _weight_scope if engine == "gpt_sovits" else (lambda *_args: nullcontext({}))
        with (
            httpx.Client() as session,
            tempfile.TemporaryDirectory(dir=tts_dir) as tmp_name,
            weight_scope(session, errors) as default_weights,
        ):
            tmp_dir = Path(tmp_name)
            for speaker, segs in eligible_groups.items():
                character = script.characters.get(speaker)
                if engine == "gpt_sovits":
                    try:
                        _switch_weights(
                            session, default_weights,
                            character.gpt_weights if character else None,
                            character.sovits_weights if character else None,
                        )
                    except TTSServiceUnavailableError:
                        for seg in segs:
                            seg.status = "error"
                        raise
                    except RuntimeError as exc:
                        for seg in segs:
                            seg.status = "error"
                            errors.append(f"片段 #{seg.id} 加载角色权重失败：{exc}")
                        continue

                for seg in segs:
                    if not seg.target_text.strip():
                        seg.status = "error"
                        errors.append(f"片段 #{seg.id} 目标台词为空，跳过合成")
                        continue

                    out_name = f"seg_{seg.id:04d}.wav"
                    try:
                        ref_obj, ref_path = resolved_refs[(speaker, seg.emotion)]
                        raw_wav = tmp_dir / out_name
                        if not accepts_extended_args:
                            _synthesize_single_line(
                                session,
                                seg,
                                ref_obj,
                                ref_path,
                                raw_wav,
                                script.glossary,
                            )
                        else:
                            aux_key = (speaker, ref_path)
                            if aux_key not in aux_cache:
                                aux_cache[aux_key] = (
                                    _collect_aux_ref_paths(
                                        script,
                                        character,
                                        speaker,
                                        ref_path,
                                        acoustics=acoustics,
                                    )
                                    if engine == "gpt_sovits"
                                    else []
                                )
                            aux_paths = aux_cache[aux_key]
                            planned_clauses = (
                                plan_segment_pause_synthesis(script, seg)
                                if ref_path is not None
                                else None
                            )
                            if planned_clauses and len(planned_clauses) >= 2:
                                try:
                                    clause_wavs: list[Path] = []
                                    for c_idx, clause in enumerate(planned_clauses):
                                        clause_wav = tmp_dir / f"seg_{seg.id:04d}_part{c_idx}.wav"
                                        clause_seg = seg.model_copy(update={"target_text": clause.text})
                                        _synthesize_single_line(
                                            session,
                                            clause_seg,
                                            ref_obj,
                                            ref_path,
                                            clause_wav,
                                            script.glossary,
                                            api_url=api_url,
                                            engine=engine,
                                            aux_ref_paths=aux_paths,
                                        )
                                        clause_wavs.append(clause_wav)
                                    assemble_island_wavs(
                                        clause_wavs,
                                        planned_clauses,
                                        seg.end - seg.start,
                                        raw_wav,
                                        tmp_dir,
                                    )
                                except TTSServiceUnavailableError:
                                    raise
                                except Exception:
                                    _synthesize_single_line(
                                        session,
                                        seg,
                                        ref_obj,
                                        ref_path,
                                        raw_wav,
                                        script.glossary,
                                        api_url=api_url,
                                        engine=engine,
                                        aux_ref_paths=aux_paths,
                                    )
                            else:
                                _synthesize_single_line(
                                    session,
                                    seg,
                                    ref_obj,
                                    ref_path,
                                    raw_wav,
                                    script.glossary,
                                    api_url=api_url,
                                    engine=engine,
                                    aux_ref_paths=aux_paths,
                                )
                        finished_wav = apply_gentle_tempo(raw_wav, seg.speed_factor, tmp_dir)
                        actual_duration = measure_wav_duration(finished_wav)
                        if actual_duration <= 0:
                            raise RuntimeError("合成音频没有有效时长")
                        if require_shorter and seg.actual_duration is not None and actual_duration >= seg.actual_duration:
                            raise RuntimeError("改写后的配音未缩短，保留上一版")
                        if require_longer and seg.actual_duration is not None and actual_duration <= seg.actual_duration:
                            raise RuntimeError("改写后的配音未变长，保留上一版")
                        finished_wav.replace(tts_dir / out_name)
                    except TTSServiceUnavailableError:
                        seg.status = "error"
                        raise
                    except Exception as exc:
                        seg.status = "error"
                        errors.append(f"片段 #{seg.id} 合成失败：{exc}")
                        continue

                    seg.tts_wav = f"/workspace/{script.project_id}/tts/{out_name}"
                    seg.actual_duration = actual_duration
                    seg.applied_speed_factor = seg.speed_factor
                    seg.status = "ready"
                    synthesized_ids.append(seg.id)

    return synthesized_ids, errors
