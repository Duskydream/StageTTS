"""GPT-SoVITS 推理客户端：按说话人分组批量合成 + 动态情绪参考音频路由。"""

from __future__ import annotations

import tempfile
import wave
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
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
from backend.services.media_mixer import apply_gentle_tempo
from backend.services.references import automatic_reference, reference_unavailable_reason
from backend.services.task_runner import ensure_tts_running, gpu_task_guard
from backend.services.text_normalization import (
    deterministic_seed,
    normalize_tts_lang,
    prepare_tts_text,
)


class TTSServiceUnavailableError(RuntimeError):
    """GPT-SoVITS 推理服务不可达或响应超时。"""


# Only track components that StageTTS changed. None marks an uncertain switch.
_active_weights: dict[str, str | None] = {}
_restore_weights: dict[str, str] = {}


def reset_active_weights_cache() -> None:
    """GPT-SoVITS 按需重启后清空模块级权重缓存。"""
    _active_weights.clear()
    _restore_weights.clear()


def check_tts_service(timeout: float = 1.5) -> bool:
    """快速探测 GPT-SoVITS api_v2.py 是否在线。"""
    try:
        resp = httpx.get(f"{config.TTS_API_URL}/", timeout=timeout)
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


def _synthesize_single_line(
    client: httpx.Client,
    seg: Segment,
    ref: EmotionRef,
    ref_path: Path,
    out_wav: Path,
    glossary: dict[str, str] | None = None,
) -> None:
    text_lang = normalize_tts_lang(seg.target_lang)
    prompt_lang = normalize_tts_lang(ref.lang)
    payload = {
        "text": prepare_tts_text(seg.target_text, text_lang, glossary),
        "text_lang": text_lang,
        "ref_audio_path": str(ref_path),
        "prompt_text": prepare_tts_text(ref.text, prompt_lang),
        "prompt_lang": prompt_lang,
        "speed_factor": 1.0,
        "media_type": "wav",
        "text_split_method": "cut0",
        "fragment_interval": 0.0,
        "batch_size": 1,
        "seed": deterministic_seed(seg.id, seg.target_text, ref.text, ref_path.name),
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    try:
        resp = client.post(
            f"{config.TTS_API_URL}/tts",
            json=payload,
            timeout=config.TTS_TIMEOUT_SEC,
        )
    except httpx.HTTPError as exc:
        raise TTSServiceUnavailableError(
            f"无法连接 GPT-SoVITS 服务 ({config.TTS_API_URL})，请确认已启动 api_v2.py: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"片段 #{seg.id} TTS 推理失败 ({resp.status_code}): {resp.text[:300]}"
        )
    if not resp.content:
        raise RuntimeError(f"片段 #{seg.id} TTS 返回空音频流")

    out_wav.write_bytes(resp.content)


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

    with gpu_task_guard():
        try:
            ensure_tts_running()
        except RuntimeError as exc:
            raise TTSServiceUnavailableError(str(exc)) from exc
        with (
            httpx.Client() as session,
            tempfile.TemporaryDirectory(dir=tts_dir) as tmp_name,
            _weight_scope(session, errors) as default_weights,
        ):
            tmp_dir = Path(tmp_name)
            for speaker, segs in eligible_groups.items():
                character = script.characters.get(speaker)
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
                        _synthesize_single_line(session, seg, ref_obj, ref_path, raw_wav, script.glossary)
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
