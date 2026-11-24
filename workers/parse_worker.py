"""独立感知解析 Worker（运行在 env_parse：PyTorch cu128 + Banquet + WhisperX + pyannote）。

由 backend/services/task_runner.py 通过 subprocess.run 启动，进程退出时由 OS 100% 释放显存。
严禁在 env_backend 中直接导入本文件或依赖。

四阶段流水线：
  Stage 0: FFmpeg 抽取标准 WAV 并探测媒体精确时长
  Stage 1: Banquet 分离对白、歌声、伴奏和音效
  Stage 2: WhisperX 语音识别 + 词级时间戳对齐（含 CUDA OOM 自动降级 int8 / cpu）
  Stage 3: pyannote 3.x 说话人聚类（跳过、缺失 Token 或失败时标记 UNKNOWN 待人工确认）
  Stage 4: LLM 导演口语化翻译、情绪/动作/原声叹词标注，校验并落盘 script.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 阶段 0 前置：在导入任何 torch/whisperx 前加载 backend/.env，
# 确保 HF_HOME / HUGGINGFACE_HUB_CACHE / TORCH_HOME 严格落盘到配置的 F 盘缓存目录。
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    _env_file = ROOT_DIR / "backend" / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

# PyTorch 2.6+ 默认 weights_only=True 会拦截 pyannote/lightning 的 omegaconf/Task 权重反序列化。
# 本 worker 仅在受控子进程中加载官方 WhisperX/pyannote 检查点，显式开启兼容模式。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from backend import config  # noqa: E402
from backend.schemas import (  # noqa: E402
    AvatarConfig,
    CharacterConfig,
    CinematicStems,
    EvidenceProvenance,
    MediaInfo,
    ProjectScript,
    Segment,
)
from backend.services.evidence import build_source_evidence  # noqa: E402
from backend.services.translation import create_openai_client, translate_script  # noqa: E402
from workers.aligned_segments import normalize_segments  # noqa: E402

SPEAKER_PALETTE = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ec4899",
    "#8b5cf6",
    "#06b6d4",
    "#ef4444",
    "#84cc16",
]
SPEAKER_LABELS = ["角色 A", "角色 B", "角色 C", "角色 D", "角色 E", "角色 F"]
AUDIO_ONLY_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}


def _log(stage: str, message: str) -> None:
    print(f"[parse_worker][{stage}] {message}", flush=True)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cublas_status_alloc_failed" in text
        or "no kernel image" in text
    )


def _release_cuda_memory() -> None:
    """阶段间主动释放 Python 引用与 CUDA 缓存，防止 8GB 显卡阶段叠加 OOM。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def _resolve_device(requested: str) -> str:
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_PYANNOTE_COMPAT_PATCHED = False


def _enable_pyannote_torch_load_compat() -> None:
    """兼容 PyTorch 2.6+ 与 pyannote 3.x / lightning / speechbrain 在 Python 3.12 下的运行时问题。

    1. 部分 lightning 版本会显式传入 weights_only=True，绕过 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD；
       此处在独立解析 worker 内对 torch.load 做一次性降级包装。
    2. speechbrain 1.0 的 LazyModule.__getattr__ 在 Python 3.12 被 inspect.stack()/getmodule()
       访问 __file__ 时会触发 inspect.getframeinfo 无限递归；对内置元属性直接抛出 AttributeError。
    """
    global _PYANNOTE_COMPAT_PATCHED
    if _PYANNOTE_COMPAT_PATCHED:
        return
    try:
        import torch

        orig_load = torch.load

        def _compat_torch_load(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("weights_only", None) is not False:
                kwargs["weights_only"] = False
            return orig_load(*args, **kwargs)

        torch.load = _compat_torch_load  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import speechbrain.utils.importutils as sb_iu

        orig_getattr = sb_iu.LazyModule.__getattr__

        def _safe_lazy_getattr(self: Any, attr: str) -> Any:
            if attr in ("__file__", "__path__", "__spec__", "__wrapped__"):
                raise AttributeError(attr)
            return orig_getattr(self, attr)

        sb_iu.LazyModule.__getattr__ = _safe_lazy_getattr  # type: ignore[method-assign]
    except Exception:
        pass

    _PYANNOTE_COMPAT_PATCHED = True


def _probe_duration(media_path: Path) -> float:
    if not media_path.is_file() or shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        raw = json.loads(res.stdout or "{}").get("format", {}).get("duration")
        return max(0.0, round(float(raw), 3)) if raw else 0.0
    except Exception:
        return 0.0


def _extract_wav_and_probe(input_path: Path, work_wav: Path) -> float:
    """用 FFmpeg 将任意音视频统一抽取为 48kHz stereo WAV（与 Banquet 采样率一致）。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未在系统 PATH 中找到 ffmpeg，无法抽取音频。")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(work_wav),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0 or not work_wav.is_file():
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"FFmpeg 抽取音频失败 (code={proc.returncode}): {tail}")

    duration = _probe_duration(input_path) or _probe_duration(work_wav)
    _log("Stage0", f"音频抽取完成 ({work_wav.name})，媒体时长 {duration:.2f}s")
    return duration


_ACTIVE_BANQUET_PROC: Optional[subprocess.Popen[str]] = None


def _kill_banquet_tree(proc_obj: subprocess.Popen[str]) -> None:
    pid = getattr(proc_obj, "pid", None)
    real_pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None
    if real_pid is None:
        proc_obj.kill()
        proc_obj.wait()
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(real_pid)],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc_obj.poll() is None:
            proc_obj.kill()
    else:
        try:
            os.killpg(real_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc_obj.kill()
    proc_obj.wait()


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    proc_obj = _ACTIVE_BANQUET_PROC
    if proc_obj is not None and proc_obj.poll() is None:
        _kill_banquet_tree(proc_obj)
    raise SystemExit(143)


def _stage_banquet(extracted_wav: Path, output_dir: Path, device: str) -> None:
    source_root, checkpoint = config.resolve_banquet_paths()

    hop_raw = os.getenv("BANQUET_HOP_SECONDS", "").strip()
    hop_args: List[str] = []
    if hop_raw:
        try:
            hop_seconds = float(hop_raw)
        except ValueError as exc:
            raise RuntimeError(f"BANQUET_HOP_SECONDS 配置非法：{hop_raw}") from exc
        if not (0 < hop_seconds <= 4.0):
            raise RuntimeError(f"BANQUET_HOP_SECONDS 必须在 (0, 4.0] 秒范围内，当前为 {hop_raw}")
        hop_args = ["--hop-seconds", str(hop_seconds)]

    env = os.environ.copy()
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    env["PYTHONUTF8"] = "1"
    devices = ("cuda", "cpu") if device == "cuda" else ("cpu",)
    deadline = time.monotonic() + 900

    def _drain_banquet(stream: Any, lines: deque[str]) -> None:
        if stream is None:
            return
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[banquet] "):
                _log("Stage1", line.removeprefix("[banquet] ").strip())
            else:
                lines.append(line)

    for target_device in devices:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Banquet ({target_device}) 分离超时（>900s）。")
        _log("Stage1", f"Banquet ({target_device}) 直接分离对白、歌声、音乐与音效")
        cmd = [
            sys.executable,
            str(ROOT_DIR / "workers/banquet_worker.py"),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(extracted_wav),
            "--output-dir",
            str(output_dir),
            "--device",
            target_device,
            *hop_args,
        ]
        global _ACTIVE_BANQUET_PROC
        output_lines: deque[str] = deque(maxlen=200)
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        _ACTIVE_BANQUET_PROC = proc
        reader = threading.Thread(target=_drain_banquet, args=(proc.stdout, output_lines), daemon=True)
        reader.start()
        try:
            try:
                returncode = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                _kill_banquet_tree(proc)
                reader.join(timeout=3)
                raise RuntimeError(f"Banquet ({target_device}) 分离超时（>900s）。") from exc
            except BaseException:
                if proc.poll() is None:
                    _kill_banquet_tree(proc)
                reader.join(timeout=3)
                raise
            reader.join(timeout=5)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            _ACTIVE_BANQUET_PROC = None
        if returncode == 0:
            return
        detail = "\n".join(output_lines).strip()
        if target_device == "cpu" or not _is_cuda_oom(RuntimeError(detail)):
            suffix = detail[-600:] if detail else "无错误输出"
            raise RuntimeError(f"Banquet ({target_device}) 分离失败 (code={returncode})：{suffix}")
        _log("Stage1", "Banquet CUDA 显存不足，改用 CPU 重试")


def _ffmpeg_filter_help(filter_name: str) -> str:
    """Return `ffmpeg -h filter=<name>` output, or empty text if unavailable."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"filter={filter_name}"],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    text = proc.stdout + proc.stderr
    if "Unknown filter" in text or f"Filter {filter_name}" not in text:
        return ""
    return text


def _ffmpeg_filter_has_option(help_text: str, option: str) -> bool:
    for line in help_text.splitlines():
        name = line.strip().split("<", 1)[0].strip()
        if name == option:
            return True
    return False


def _build_bgm_filter(inputs: int = 3) -> str:
    """Build a BGM mix filter compatible with both old and new FFmpeg builds."""
    filters: List[str] = []
    amix_help = _ffmpeg_filter_help("amix")
    if _ffmpeg_filter_has_option(amix_help, "normalize"):
        filters.append(f"amix=inputs={inputs}:duration=first:normalize=0")
    elif _ffmpeg_filter_has_option(amix_help, "weights"):
        filters.append(f"amix=inputs={inputs}:duration=first:weights={' '.join(['1'] * inputs)}")
    else:
        # Legacy amix always renormalizes; compensate approximately by input count.
        filters.append(f"amix=inputs={inputs}:duration=first")
        filters.append(f"volume={inputs}")

    limiter_help = _ffmpeg_filter_help("alimiter")
    if not limiter_help:
        filters.append("volume=0.98")
        return ",".join(filters)

    limiter_options = ["limit=0.98"]
    if _ffmpeg_filter_has_option(limiter_help, "level"):
        limiter_options.append("level=0")
    if _ffmpeg_filter_has_option(limiter_help, "latency"):
        limiter_options.append("latency=1")
    filters.append("alimiter=" + ":".join(limiter_options))
    return ",".join(filters)


def _finish_four_stems(output_dir: Path, project_id: str) -> Tuple[Path, Path, CinematicStems]:
    tracks = {stem: output_dir / f"{stem}.wav" for stem in ("dialogue", "singing", "music", "effects")}
    if not all(path.is_file() and path.stat().st_size > 44 for path in tracks.values()):
        raise RuntimeError("影视四轨分离未生成完整的对白、歌声、音乐、音效 WAV 文件。")

    bgm_path = output_dir / "bgm.wav"
    amix_filter = _build_bgm_filter(3)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-i", str(tracks["effects"]), "-i", str(tracks["music"]), "-i", str(tracks["singing"]),
         "-filter_complex", amix_filter,
         "-c:a", "pcm_s16le", str(bgm_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"影视背景轨合成失败：{proc.stderr[-600:]}")
    if not bgm_path.is_file() or bgm_path.stat().st_size <= 44:
        raise RuntimeError("影视背景轨合成结果为空，请检查分离音轨后重试。")

    stems = CinematicStems(**{
        stem: _project_media_url(path, output_dir.parent, project_id) for stem, path in tracks.items()
    })
    _log("Stage1", "四轨分离完成，交接对白轨至解析后端")
    return tracks["dialogue"], bgm_path, stems


def _stage_whisperx_asr_align(
    vocals_wav: Path,
    compute_type: str,
    whisper_model: str,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    """阶段 2：WhisperX 语音识别与词级时间戳对齐（含 CUDA OOM 自动降级 int8 / cpu）。"""
    _enable_pyannote_torch_load_compat()
    import whisperx

    audio = whisperx.load_audio(str(vocals_wav))

    candidates: List[Tuple[str, str, int]] = []
    if device == "cuda":
        if compute_type == "float16":
            candidates.append(("cuda", "float16", 8))
        candidates.append(("cuda", "int8", 4))
    candidates.append(("cpu", "int8", 4))

    result: Optional[Dict[str, Any]] = None
    used_device = "cpu"
    last_exc: Optional[BaseException] = None

    for dev, ctype, batch_size in candidates:
        model = None
        try:
            _log("Stage2", f"加载 WhisperX 模型 {whisper_model} (device={dev}, compute_type={ctype})...")
            model = whisperx.load_model(
                whisper_model,
                device=dev,
                compute_type=ctype,
            )
            result = model.transcribe(audio, batch_size=batch_size)
            used_device = dev
            _log(
                "Stage2",
                f"ASR 完成：识别语言={result.get('language', 'unknown')}，初稿片段数={len(result.get('segments', []))}",
            )
            break
        except Exception as exc:
            last_exc = exc
            if dev == "cuda" and (_is_cuda_oom(exc) or "float16" in str(exc).lower()):
                _log("Stage2", f"CUDA ({ctype}) 失败，准备降级重试：{str(exc)[:180]}")
            else:
                _log("Stage2", f"WhisperX ({dev}/{ctype}) 异常，尝试下一方案：{str(exc)[:180]}")
        finally:
            if model is not None:
                del model
            _release_cuda_memory()

    if result is None:
        raise RuntimeError(f"WhisperX 语音识别全部方案失败: {last_exc}")

    detected_lang = str(result.get("language") or "")
    if detected_lang and result.get("segments"):
        align_attempts = [used_device, "cpu"] if used_device == "cuda" else ["cpu"]
        for align_dev in align_attempts:
            align_model = None
            try:
                _log("Stage2", f"加载词级对齐模型 (lang={detected_lang}, device={align_dev})...")
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=align_dev,
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    align_dev,
                    return_char_alignments=False,
                )
                _log("Stage2", f"词级时间戳对齐完成 ({align_dev})")
                break
            except Exception as exc:
                _log("Stage2", f"词级对齐 ({align_dev}) 跳过/降级：{str(exc)[:180]}")
            finally:
                if align_model is not None:
                    del align_model
                _release_cuda_memory()

    return audio, result, detected_lang


def _unknown_speaker_segments(segments: List[Dict[str, Any]], reason: str,
                              language: str | None = None) -> List[Dict[str, Any]]:
    """UNKNOWN 是未确认归属的占位符；分别保留源片段，不能据此合并为同一人。"""
    normalized = []
    for raw in segments:
        for part in normalize_segments([raw], diarized=False, language=language):
            part["speaker"] = "UNKNOWN"
            part["speaker_needs_review"] = True
            part["speaker_fallback_reason"] = reason
            for word in part.get("words") or []:
                word["speaker"] = None
            normalized.append(part)
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return [dict(part, id=index) for index, part in enumerate(normalized)]


def _review_assigned_speakers(segments: List[Dict[str, Any]], language: str | None = None) -> List[Dict[str, Any]]:
    """保留成功聚类的标签，但缺失的词级证据不能被片段多数标签掩盖。"""
    normalized = []
    prepared = []
    for raw in segments:
        prepared.append({**raw, "speaker": str(raw.get("speaker") or "").strip() or "UNKNOWN"})
    for part in normalize_segments(prepared, diarized=True, language=language):
        words = part.get("words") or []
        labels = [str(word.get("speaker") or "").strip() for word in words]
        has_speaker = any(label and label != "UNKNOWN" for label in labels)
        has_segment_speaker = part["speaker"] not in {"", "UNKNOWN"}
        if not has_speaker and not has_segment_speaker:
            part["speaker"] = "UNKNOWN"
            part["speaker_needs_review"] = True
            part["speaker_fallback_reason"] = "speaker_assignment_missing"
        elif not words or any(label in {"", "UNKNOWN"} for label in labels):
            part["speaker_needs_review"] = True
            part["speaker_fallback_reason"] = "word_speaker_missing"
        normalized.append(part)
    return normalized


def _diarization_activity_rows(diarize_df: Any) -> List[Dict[str, Any]]:
    """Copy original dataframe rows without collapsing concurrent speaker tracks."""
    rows = diarize_df.to_dict("records")
    return [{"start": row.get("start"), "end": row.get("end"), "speaker": row.get("speaker")}
            for row in rows]


def _mark_overlapping_activity(segments: List[Dict[str, Any]], activity_rows: List[Dict[str, Any]]) -> None:
    """Legacy transcript ownership cannot silently confirm simultaneous speakers."""
    valid = []
    for row in activity_rows:
        try:
            valid.append((float(row["start"]), float(row["end"]), row.get("speaker")))
        except (KeyError, TypeError, ValueError):
            continue
    valid.sort(key=lambda row: row[0])
    active = []
    overlaps = []
    for start, end, speaker in valid:
        active = [row for row in active if row[1] > start]
        for left_start, left_end, left_speaker in active:
            if speaker and left_speaker and speaker != left_speaker and min(end, left_end) > start:
                overlaps.append((start, min(end, left_end)))
        active.append((start, end, speaker))
    for seg in segments:
        if any(start < seg["end"] and end > seg["start"] for start, end in overlaps):
            seg["speaker_needs_review"] = True
            seg["speaker_fallback_reason"] = seg.get("speaker_fallback_reason") or "overlapping_speaker_activity"


def _stage_diarize(
    audio: Any,
    aligned_result: Dict[str, Any],
    device: str,
    skip_diarization: bool,
    evidence_context: Optional[Dict[str, Any]] = None,
    language: str | None = None,
) -> List[Dict[str, Any]]:
    """阶段 3：pyannote 说话人聚类与词级分配；无证据时明确保留待确认状态。"""
    original_result = copy.deepcopy(aligned_result)
    segments = original_result.get("segments") or []
    if evidence_context is not None:
        evidence_context.update(raw_segments=copy.deepcopy(segments), diarization_spans=[],
                                exclusive_spans=[], fallback_reason=None)

    def fallback(reason: str) -> List[Dict[str, Any]]:
        if evidence_context is not None:
            evidence_context["fallback_reason"] = reason
        return _unknown_speaker_segments(segments, reason, language=language)

    if not segments:
        return []

    if skip_diarization:
        _log("Stage3", "已通过参数跳过说话人聚类，全部片段标记 UNKNOWN 待确认")
        return fallback("diarization_skipped")
    if not config.is_hf_token_configured():
        _log("Stage3", "未配置有效 HF_TOKEN，全部片段标记 UNKNOWN 待确认")
        return fallback("hf_token_missing")

    _enable_pyannote_torch_load_compat()
    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except Exception as exc:
        _log("Stage3", f"无法加载说话人聚类，待人工校对：{str(exc)[:220]}")
        return fallback("diarization_import_failed")

    attempts = [device, "cpu"] if device == "cuda" else ["cpu"]
    for diar_dev in attempts:
        diarize_model = None
        try:
            _log("Stage3", f"启动 pyannote/speaker-diarization-3.1 (device={diar_dev})...")
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=config.HF_TOKEN,
                device=diar_dev,
            )
            diarize_df = diarize_model(audio)
            activity_rows = _diarization_activity_rows(diarize_df) if evidence_context is not None else []
            if evidence_context is not None:
                evidence_context["diarization_spans"] = activity_rows
                evidence_context["device"] = diar_dev
            # WhisperX mutates its argument: isolate speaker assignment from the
            # immutable source ASR snapshot and from a possible device retry.
            assigned = assign_word_speakers(diarize_df, copy.deepcopy(original_result), fill_nearest=False)
            assigned_segments = assigned.get("segments") or []
            if not assigned_segments:
                return fallback("speaker_assignment_missing")
            _log("Stage3", f"说话人聚类完成 ({diar_dev})")
            normalized = _review_assigned_speakers(assigned_segments, language=language)
            _mark_overlapping_activity(normalized, activity_rows)
            return normalized
        except Exception as exc:
            _log("Stage3", f"说话人聚类 ({diar_dev}) 失败：{str(exc)[:220]}")
        finally:
            if diarize_model is not None:
                del diarize_model
            _release_cuda_memory()

    _log("Stage3", "pyannote 聚类不可用，全部片段标记 UNKNOWN 待确认")
    return fallback("diarization_failed")


def _moss_model_request() -> Dict[str, Any]:
    if not config.MOSS_MODEL_PATH or not config.MOSS_REVISION or not config.MOSS_CUSTOM_CODE_APPROVED:
        raise RuntimeError("MOSS backend is not configured: local model/revision/custom-code approval required; no legacy fallback")
    return {"id": "OpenMOSS-Team/MOSS-Transcribe-Diarize", "local_path": str(config.MOSS_MODEL_PATH),
            "revision": config.MOSS_REVISION, "transformers_version": config.MOSS_TRANSFORMERS_VERSION,
            "torch_version": config.MOSS_TORCH_VERSION, "custom_code_approved": config.MOSS_CUSTOM_CODE_APPROVED}


def _preflight_moss(device: str) -> None:
    from workers.moss_worker import validate_request
    try:
        validate_request({"version": "1.0", "model": _moss_model_request(),
                          "source": {"id": "preflight", "kind": "dialogue"},
                          "device": "cuda:0" if device == "cuda" else "cpu",
                          "max_audio_seconds": config.MOSS_MAX_AUDIO_SECONDS,
                          "max_new_tokens": config.MOSS_MAX_NEW_TOKENS}, model_only=True)
    except (OSError, ValueError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(f"MOSS preflight failed before separation: {exc}; no legacy fallback") from exc


def _stage_moss_transcribe(vocals_wav: Path, device: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Joint ASR/speaker generation with source segment timestamps, not word alignment."""
    from workers.moss_worker import run_request as run_moss

    if not config.MOSS_MODEL_PATH or not config.MOSS_REVISION or not config.MOSS_CUSTOM_CODE_APPROVED:
        raise RuntimeError("MOSS backend is not configured: local model/revision/custom-code approval required; no legacy fallback")
    source_digest = hashlib.sha256()
    with vocals_wav.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            source_digest.update(chunk)
    source_id = f"sha256:{source_digest.hexdigest()}"
    with tempfile.TemporaryDirectory(prefix="moss_input_", dir=vocals_wav.parent) as temporary:
        mono = Path(temporary) / "dialogue_16k.wav"
        conversion = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(vocals_wav),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(mono)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if conversion.returncode != 0 or not mono.is_file():
            raise RuntimeError(f"MOSS explicit 16k mono conversion failed: {conversion.stderr[-500:]}")
        request = {
            "version": "1.0", "model": _moss_model_request(),
            "source": {"id": source_id, "kind": "dialogue", "audio_path": str(mono.resolve()),
                       "window_id": "whole_recording"},
            "device": "cuda:0" if device == "cuda" else "cpu",
            "max_audio_seconds": config.MOSS_MAX_AUDIO_SECONDS,
            "max_new_tokens": config.MOSS_MAX_NEW_TOKENS,
        }
        try:
            result = run_moss(request)
        finally:
            _release_cuda_memory()
    if result.get("status") not in {"ok", "partial"} or not result.get("segments"):
        raise RuntimeError(f"MOSS transcription did not complete: {result.get('error') or result.get('review_reasons')}; no legacy fallback")
    # A token-limit/invalid stream must be reviewed before publishing a source script.
    blocking_reasons = [reason for reason in result.get("review_reasons", [])
                        if reason != "source_wave_tail_within_explicit_0.05s_budget_tolerance"]
    if blocking_reasons:
        raise RuntimeError(f"MOSS generation requires review before import: {blocking_reasons}")
    scope_hash = hashlib.sha256(f"{source_id}:whole_recording".encode()).hexdigest()[:12]
    raw_segments, normalized, spans = [], [], []
    for index, row in enumerate(result["segments"]):
        label = str(row.get("speaker") or "UNKNOWN")
        speaker = f"MOSS_{scope_hash}_{label}" if label != "UNKNOWN" else "UNKNOWN"
        raw_segments.append({"start": row["start"], "end": row["end"], "text": row["text"], "words": [],
                             "generated_start": row.get("generated_start", row["start"]),
                             "generated_end": row.get("generated_end", row["end"]),
                             "review_reasons": [*row.get("review_reasons", []), *result.get("review_reasons", [])]})
        normalized.append({"id": index, "start": row["start"], "end": row["end"], "speaker": speaker,
                           "orig_text": row["text"], "words": [], "alignment_needs_review": True,
                           "alignment_granularity": "segment", "timing_source": "moss_joint_generated",
                           "segment_timing_needs_review": True,
                           "speaker_needs_review": True,
                           "speaker_fallback_reason": "moss_joint_generated_segment_requires_review"})
        if speaker != "UNKNOWN":
            spans.append({"start": row["start"], "end": row["end"], "speaker": speaker,
                          "review_reasons": ["moss_joint_generated_activity_unvalidated", *row.get("review_reasons", [])]})
    context = {"raw_segments": raw_segments, "diarization_spans": spans, "exclusive_spans": [],
               "fallback_reason": "moss_joint_generated_segment_requires_review", "backend": "moss",
               "raw_generation": result["raw_text"], "speaker_scope": f"{source_id}:whole_recording",
               "model_revision": config.MOSS_REVISION, "device": request["device"],
               "source_audio_id": source_id,
               "source_wave_duration": result.get("duration_seconds"),
               "input_audio_budget_seconds": result.get("audio_budget_seconds"),
               "input_media_tail_tolerance_seconds": result.get("media_tail_tolerance_seconds"),
               "preprocessing": "ffmpeg_pcm16_mono_16000_whole_recording"}
    return normalized, context, ""


def _project_media_url(path: Path, project_dir: Path, project_id: str) -> str:
    return f"/workspace/{project_id}/{path.relative_to(project_dir).as_posix()}"


def _build_and_save_script(
    args: argparse.Namespace,
    duration: float,
    vocals_wav: Path,
    bgm_wav: Optional[Path],
    raw_segments: List[Dict[str, Any]],
    detected_lang: str,
    cinematic_stems: Optional[CinematicStems] = None,
    evidence_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """组装并通过 Pydantic v2 强校验后落盘 workspace/{project_id}/script.json。"""
    project_id = args.project_id
    script_out = args.output_dir / "script.json"
    existing = ProjectScript.model_validate_json(script_out.read_text(encoding="utf-8")) if script_out.is_file() else None
    input_ext = args.input.suffix.lower()
    is_audio_only = input_ext in AUDIO_ONLY_EXTS

    source_video_url: Optional[str] = None
    if not is_audio_only:
        # 若输入文件位于 output_dir 内，直接映射标准 /workspace/{id}/{filename}
        if args.input.resolve().parent == args.output_dir.resolve():
            source_video_url = f"/workspace/{project_id}/{args.input.name}"
        else:
            copied_input = args.output_dir / f"input{input_ext or '.mp4'}"
            if not copied_input.exists():
                shutil.copyfile(args.input, copied_input)
            source_video_url = f"/workspace/{project_id}/{copied_input.name}"

    discovered_speakers = list(dict.fromkeys(seg["speaker"] for seg in raw_segments))
    characters = dict(existing.characters) if existing else {}
    for idx, spk in enumerate(discovered_speakers):
        if spk in characters:
            if spk == "UNKNOWN":
                characters[spk] = characters[spk].model_copy(update={"display_name": "待确认说话人"})
            continue
        label = "待确认说话人" if spk == "UNKNOWN" else (
            SPEAKER_LABELS[idx] if idx < len(SPEAKER_LABELS) else f"角色 {idx + 1}"
        )
        color = SPEAKER_PALETTE[idx % len(SPEAKER_PALETTE)]
        avatar_slot = "host_a" if idx % 2 == 0 else "host_b"
        characters[spk] = CharacterConfig(
            display_name=label,
            color=color,
            gpt_weights=None,
            sovits_weights=None,
            emotion_refs={},
            avatar=AvatarConfig(
                type="live2d",
                model_url=f"/assets/models/{avatar_slot}/model3.json",
            ),
        )

    for seg in raw_segments:
        if "alignment_granularity" not in seg:
            has_word_alignment = bool(seg.get("words")) and not seg.get("alignment_needs_review", True)
            seg["alignment_granularity"] = "word" if has_word_alignment else "unknown"
            seg["timing_source"] = "forced_alignment" if has_word_alignment else "unknown"
    segments = [
        Segment(**seg, target_text=seg["orig_text"], target_lang=args.target_lang, action="idle")
        for seg in raw_segments
    ]

    max_seg_end = max((s.end for s in segments), default=0.0)
    final_duration = round(max(duration, max_seg_end), 3)

    evidence = None
    if evidence_context is not None:
        digest = hashlib.sha256()
        with args.input.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        source_id = f"sha256:{digest.hexdigest()}"
        def package_version(name: str) -> str | None:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None
        backend = evidence_context.get("backend", "legacy")
        provenance = [
            EvidenceProvenance(stage="asr", model=("OpenMOSS-Team/MOSS-Transcribe-Diarize" if backend == "moss"
                                                   else getattr(args, "whisper_model", None)),
                               model_version=(evidence_context.get("model_revision") if backend == "moss" else package_version("whisperx")),
                               source_id=source_id,
                               parameters={"audio_source": "dialogue", "language": getattr(args, "source_lang", None) or detected_lang or None,
                                           "timestamp_granularity": "segment" if backend == "moss" else "word_or_unknown",
                                           "preprocessing": evidence_context.get("preprocessing"),
                                           "source_audio_id": evidence_context.get("source_audio_id"),
                                           "source_wave_duration": evidence_context.get("source_wave_duration"),
                                           "input_audio_budget_seconds": evidence_context.get("input_audio_budget_seconds"),
                                           "input_media_tail_tolerance_seconds": evidence_context.get("input_media_tail_tolerance_seconds"),
                                           "source_language_kind": "user_declared" if getattr(args, "source_lang", None) else (
                                               "model_detected" if detected_lang else "unknown")}),
            EvidenceProvenance(stage="diarization", model=("OpenMOSS-Team/MOSS-Transcribe-Diarize" if backend == "moss"
                                                           else "pyannote/speaker-diarization-3.1"),
                               model_version=(evidence_context.get("model_revision") if backend == "moss" else package_version("pyannote.audio")),
                               source_id=source_id,
                               parameters={"audio_source": "dialogue", "device": evidence_context.get("device"),
                                           "fallback_reason": evidence_context.get("fallback_reason"),
                                           "speaker_scope": evidence_context.get("speaker_scope"),
                                           "timestamp_correction_policy": "clamp_source_bounds_tolerance_0.05s" if backend == "moss" else None,
                                           "activity_kind": "joint_generated_derived_timestamps" if backend == "moss" else "diarizer_activity"}),
        ]
        evidence = build_source_evidence(
            evidence_context.get("raw_segments") or [], evidence_context.get("diarization_spans") or [],
            segments, source_id=source_id, provenance=provenance,
            fallback_reason=evidence_context.get("fallback_reason"),
            exclusive_spans=evidence_context.get("exclusive_spans") or [],
            raw_generation=evidence_context.get("raw_generation"),
            speaker_scope=evidence_context.get("speaker_scope"),
        )

    script = ProjectScript(
        project_id=project_id,
        mode=args.mode,
        source_lang=getattr(args, "source_lang", None) or (existing.source_lang if existing else None) or detected_lang or None,
        glossary=existing.glossary if existing else {},
        media=MediaInfo(
            source_video=source_video_url,
            vocals_wav=_project_media_url(vocals_wav, args.output_dir, project_id),
            bgm_wav=_project_media_url(bgm_wav, args.output_dir, project_id) if bgm_wav else None,
            cinematic_stems=cinematic_stems,
            output_video=None,
            duration=final_duration,
            bgm_volume=0.75,
        ),
        characters=characters,
        segments=segments,
        evidence=evidence,
    )

    if evidence_context and evidence_context.get("backend") == "moss" and not script.source_lang:
        _log("Stage3", "MOSS未检测源语种，请到翻译设置确认原声语言；确认前自动参考不可用，不能用目标语种代填")
    if args.skip_llm or not config.is_llm_configured():
        _log("Stage4", "跳过翻译或未配置 LLM，保留识别原文")
    elif segments:
        translated = translate_script(script, create_openai_client(), lambda message: _log("Stage4", message))
        _log("Stage4", f"已翻译 {len(translated)}/{len(segments)} 条台词")
        if script.source_lang:
            try:
                from backend.services.emotion_routing import build_character_emotion_refs

                populated = build_character_emotion_refs(
                    script,
                    overwrite=False,
                    log=lambda message: _log("Stage4", message),
                )
                if populated:
                    summary = "、".join(
                        f"{spk}({len(emos)})" for spk, emos in populated.items()
                    )
                    _log("Stage4", f"已自动匹配角色多情绪参考音：{summary}")
            except Exception as exc:
                _log("Stage4", f"自动匹配多情绪参考音跳过：{exc}")

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
        draft = Path(temp) / script_out.name
        draft.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        draft.replace(script_out)
    _log(
        "Done",
        f"script.json 生成并校验通过：{len(segments)} 条片段，{len(characters)} 个角色 -> {script_out}",
    )
    return script_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StageTTS 上游感知解析流水线 (env_parse)")
    parser.add_argument("--project-id", required=True, help="项目唯一 ID")
    parser.add_argument("--input", required=True, type=Path, help="源视频或源音频路径")
    parser.add_argument("--output-dir", required=True, type=Path, help="项目输出目录 workspace/{project_id}")
    parser.add_argument("--target-lang", default="zh", help="目标配音语种（默认 zh）")
    parser.add_argument("--source-lang", choices=["zh", "en", "ja", "ko", "yue"], default=None,
                        help="人工声明源音频语种；MOSS不输出语言检测，留空待确认，不使用目标语种代填")
    parser.add_argument(
        "--mode",
        default="video_dubbing",
        choices=["video_dubbing", "virtual_stage"],
        help="项目初始化模式",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备（默认 auto 自动探测 CUDA 并支持 OOM 降级）",
    )
    parser.add_argument("--parse-backend", choices=["moss", "legacy"], default=None,
                        help="解析后端，默认 PARSE_BACKEND=moss；legacy 为显式旧链回滚")
    parser.add_argument("--skip-diarization", action="store_true", help="仅 legacy：跳过 pyannote 说话人聚类")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 翻译与情绪标注")
    return parser.parse_args()


def _cleanup_stale_cinematic_dirs(output_dir: Path, keep: Path) -> None:
    keep_resolved = keep.resolve()
    for candidate in output_dir.iterdir():
        if candidate.is_dir() and candidate.name.startswith("cinematic-") and candidate.resolve() != keep_resolved:
            shutil.rmtree(candidate, ignore_errors=True)


def run_pipeline(args: argparse.Namespace) -> Path:
    if not args.input.is_file():
        raise FileNotFoundError(f"输入媒体文件不存在: {args.input}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    backend = getattr(args, "parse_backend", None) or config.PARSE_BACKEND
    if backend not in {"moss", "legacy"}:
        raise RuntimeError(f"Unknown parse backend: {backend}; no automatic fallback")
    model_summary = (f"MOSS revision={config.MOSS_REVISION or '(not configured)'}, timestamps=segment"
                     if backend == "moss" else "legacy WhisperX + pyannote")
    _log(
        "Init",
        f"project={args.project_id}, backend={backend}, device={device}, {model_summary}, "
        f"HF_HOME={os.getenv('HF_HOME', '(default)')}, TORCH_HOME={os.getenv('TORCH_HOME', '(default)')}",
    )
    if backend == "moss":
        _preflight_moss(device)

    # Publish new audio separately so a failed reparse preserves the previous script's tracks.
    cache_key = {"input": str(args.input.resolve()), "size": args.input.stat().st_size,
                 "mtime_ns": args.input.stat().st_mtime_ns}
    cached = None
    for folder in sorted(args.output_dir.glob("cinematic-*")):
        marker = folder / "separation_complete.json"
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == cache_key and all((folder / (name + ".wav")).is_file() for name in ("dialogue", "singing", "music", "effects", "bgm")):
                cached = folder
                break
        except (OSError, ValueError):
            pass
    cinematic_dir = cached or Path(tempfile.mkdtemp(prefix="cinematic-", dir=args.output_dir))
    try:
        with tempfile.TemporaryDirectory(prefix="stagetts_work_") as work_tmp:
            extracted_wav = Path(work_tmp) / "extracted_audio.wav"
            duration = _extract_wav_and_probe(args.input, extracted_wav)
            if backend == "moss":
                with wave.open(str(extracted_wav), "rb") as extracted_audio:
                    waveform_duration = extracted_audio.getnframes() / extracted_audio.getframerate()
                if not 0 < waveform_duration <= min(config.MOSS_MAX_AUDIO_SECONDS, 5400.0) + 0.05:
                    raise RuntimeError(f"MOSS initial audio budget exceeded or unknown waveform duration: {waveform_duration:.6f}s; "
                                       "audio exceeds configured MOSS input limit (maximum 90 minutes)")
            if cached is None:
                _stage_banquet(extracted_wav, cinematic_dir, device)
            else:
                _log("Stage1", "复用已完成且输入未改变的 Banquet 分轨")
            vocals_wav, bgm_wav, cinematic_stems = _finish_four_stems(cinematic_dir, args.project_id)
            (cinematic_dir / "separation_complete.json").write_text(json.dumps(cache_key), encoding="utf-8")

        if backend == "moss":
            _log("Stage2", "MOSS joint ASR + speaker segment timestamps (no word alignment)")
            raw_segments, evidence_context, detected_lang = _stage_moss_transcribe(vocals_wav, device)
        elif backend == "legacy":
            audio, aligned_result, detected_lang = _stage_whisperx_asr_align(
                vocals_wav=vocals_wav,
                compute_type=getattr(args, "compute_type", "float16"),
                whisper_model=getattr(args, "whisper_model", "large-v3"),
                device=device,
            )
            evidence_context: Dict[str, Any] = {}
            raw_segments = _stage_diarize(
                audio=audio,
                aligned_result=aligned_result,
                device=device,
                skip_diarization=args.skip_diarization,
                evidence_context=evidence_context,
                language=detected_lang,
            )
        else:
            raise RuntimeError(f"Unknown parse backend: {backend}; no automatic fallback")
        review_count = sum(seg["alignment_needs_review"] or seg["speaker_needs_review"] for seg in raw_segments)
        if review_count:
            _log("Stage3", f"{review_count} 条片段的原文范围或说话人待人工校对")
        script_out = _build_and_save_script(
            args=args,
            duration=duration,
            vocals_wav=vocals_wav,
            bgm_wav=bgm_wav,
            raw_segments=raw_segments,
            detected_lang=detected_lang,
            cinematic_stems=cinematic_stems,
            evidence_context=evidence_context,
        )
    except BaseException:
        if not (cinematic_dir / "separation_complete.json").is_file():
            shutil.rmtree(cinematic_dir, ignore_errors=True)
        raise

    _cleanup_stale_cinematic_dirs(args.output_dir, keep=cinematic_dir)
    return script_out


def main() -> None:
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigterm)
    args = parse_args()
    try:
        run_pipeline(args)
    finally:
        _release_cuda_memory()


if __name__ == "__main__":
    main()
