"""GPT-SoVITS one-click character fine-tuning and VoicePreset registration."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from backend import config
from backend.agent.tools import load_project_script, project_task_guard, save_project_script
from backend.schemas import ProjectScript, Segment, VoicePresetCreateRequest
from backend.services.emotion_routing import _non_overlapping_ids, build_character_emotion_refs
from backend.services.references import _trimmed_transcript, _verified_word_offsets
from backend.services.task_runner import GpuBusyError, ensure_tts_stopped, gpu_task_guard
from backend.services.text_normalization import normalize_tts_lang
from backend.services.training_settings import (
    BASE_MODELS,
    TrainingSettings,
    _missing_model_files,
    read_training_settings,
)
from backend.services.tts_client import check_tts_service
from backend.services.voice_presets import create_voice_preset
from workers.prosody_worker import read_wav

MIN_CLIP_SECONDS = 1.0
MAX_CLIP_SECONDS = 20.0
MIN_CLIPS = 2
MIN_TOTAL_SECONDS = 6.0
FEW_SHOT_SECONDS_THRESHOLD = 20.0
FEW_SHOT_CLIPS_THRESHOLD = 6
LOW_DATA_WARNING_SECONDS = 60.0
_CLAUSE_SPLIT_RE = re.compile(r"[^。！？!?；;…]+[。！？!?；;…]+|[^。！？!?；;…]+$")

_jobs_lock = threading.RLock()
_launch_lock = threading.Lock()
_jobs_by_project: Dict[str, "FinetuneJobStatus"] = {}


class FinetuneStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset_name: str | None = Field(default=None, max_length=80)
    allow_unreviewed: bool = False
    bind_to_character: bool = True
    save_as_preset: bool = True
    mode: Literal["auto", "standard", "few_shot"] = "auto"


class FinetuneJobStatus(BaseModel):
    job_id: str | None = None
    active: bool = False
    project_id: str | None = None
    speaker: str | None = None
    training_mode: Literal["standard", "few_shot"] = "standard"
    base_model: str | None = None
    gpt_epochs: int | None = None
    sovits_epochs: int | None = None
    stage: str = "idle"
    progress_pct: int = 0
    clip_count: int = 0
    usable_duration_sec: float = 0.0
    gpt_weights: str | None = None
    sovits_weights: str | None = None
    preset_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    updated_at: float = 0.0


@dataclass
class TrainingSlice:
    clip_id: str
    start: float
    end: float
    text: str
    speaker_needs_review: bool


@dataclass
class ValidatedFinetunePlan:
    script: ProjectScript
    speaker: str
    lang: str
    vocals_path: Path
    eligible: list[TrainingSlice]
    total_seconds: float
    warnings: list[str]
    recipe: TrainingSettings
    training_mode: Literal["standard", "few_shot"]
    effective_gpt_epochs: int
    effective_sovits_epochs: int
    root: Path
    python_exe: str


def _save_every_epoch(total_epochs: int) -> int:
    return next(step for step in (5, 4, 3, 2, 1) if total_epochs % step == 0)


def _append_log(project_id: str, line: str, *, stage: str | None = None, progress_pct: int | None = None) -> None:
    cleaned = line.strip()
    with _jobs_lock:
        job = _jobs_by_project.get(project_id)
        if job is None:
            return
        if stage is not None:
            job.stage = stage
        if progress_pct is not None:
            job.progress_pct = progress_pct
        if cleaned:
            job.logs = [*job.logs, cleaned[:240]][-80:]
        job.updated_at = time.time()


def get_finetune_status(project_id: str, speaker: str | None = None) -> FinetuneJobStatus:
    with _jobs_lock:
        job = _jobs_by_project.get(project_id)
        if job is None:
            return FinetuneJobStatus(project_id=project_id, speaker=speaker, stage="idle")
        if speaker is not None and job.speaker != speaker and not job.active:
            return FinetuneJobStatus(project_id=project_id, speaker=speaker, stage="idle")
        return job.model_copy(deep=True)


def _resolve_gsv_python() -> str:
    python_cmd = (config.GPT_SOVITS_PYTHON or "").strip()
    if python_cmd and (Path(python_cmd).is_file() or shutil.which(python_cmd) is not None):
        return python_cmd
    raise ValueError("未找到可用的 GPT-SoVITS Python 运行时，请检查 GPT_SOVITS_PYTHON 配置")


def _build_training_slices(seg: Segment, vocals_path: Path) -> list[TrainingSlice]:
    """Keep 1.0-20.0s clips directly and split >20s monologues at verified boundaries or pauses."""
    text = re.sub(r"[\r\n|]+", " ", seg.orig_text).strip()
    dur = seg.end - seg.start
    if not text or dur < MIN_CLIP_SECONDS:
        return []
    if dur <= MAX_CLIP_SECONDS:
        return [TrainingSlice(f"seg_{seg.id:04d}", seg.start, seg.end, text, seg.speaker_needs_review)]

    # Strategy A: use verified word-level timestamps when available.
    if _verified_word_offsets(seg) is not None and len(seg.words) >= 2:
        slices: list[TrainingSlice] = []
        chunk_start = seg.start
        idx = 1
        for pos, word in enumerate(seg.words):
            if word.end is None:
                continue
            chunk_dur = word.end - chunk_start
            is_last = pos == len(seg.words) - 1
            if chunk_dur >= 6.0 or is_last:
                if 3.0 <= chunk_dur <= 15.0 or (is_last and chunk_dur >= 1.5 and chunk_dur <= 18.0):
                    sub_text = _trimmed_transcript(seg, chunk_start, word.end)
                    if sub_text:
                        slices.append(TrainingSlice(
                            f"seg_{seg.id:04d}_{idx:02d}",
                            chunk_start,
                            word.end,
                            sub_text,
                            seg.speaker_needs_review,
                        ))
                        idx += 1
                        chunk_start = seg.words[pos + 1].start if pos + 1 < len(seg.words) and seg.words[pos + 1].start is not None else word.end
        if slices:
            return slices

    # Strategy B: align major punctuation clauses with >= 0.25s low-energy acoustic valleys.
    clauses = [m.group(0).strip() for m in _CLAUSE_SPLIT_RE.finditer(text) if m.group(0).strip()]
    if len(clauses) < 2:
        return []
    try:
        samples, rate, _ = read_wav(vocals_path, start=seg.start, end=seg.end)
    except (OSError, ValueError, wave.Error, EOFError):
        return []
    hop = max(1, round(rate * 0.02))
    clip_rms = (sum(v * v for v in samples) / max(1, len(samples))) ** 0.5
    gate = max(10 ** (-42.0 / 20.0), clip_rms * 0.15)
    valleys: list[float] = []
    silent_start: int | None = None
    frame_sec = hop / float(rate)
    for fi, offset in enumerate(range(0, len(samples), hop)):
        frame = samples[offset:offset + hop]
        rms = (sum(v * v for v in frame) / max(1, len(frame))) ** 0.5
        if rms < gate:
            if silent_start is None:
                silent_start = fi
        else:
            if silent_start is not None:
                silent_dur = (fi - silent_start) * frame_sec
                if silent_dur >= 0.25 and silent_start > 0:
                    valleys.append(round(seg.start + ((silent_start + fi) * 0.5) * frame_sec, 3))
                silent_start = None
    if not valleys:
        return []

    total_chars = sum(len(re.sub(r"\s+", "", c)) for c in clauses)
    if total_chars <= 0:
        return []
    slices = []
    anchor_t = seg.start
    clause_idx = 0
    slice_num = 1
    cum_chars = 0
    for ci, clause in enumerate(clauses[:-1]):
        cum_chars += len(re.sub(r"\s+", "", clause))
        target_ratio = cum_chars / total_chars
        est_t = seg.start + dur * target_ratio
        nearest = min(valleys, key=lambda vt: abs(vt - est_t))
        if abs((nearest - seg.start) / dur - target_ratio) > 0.22:
            continue
        chunk_dur = nearest - anchor_t
        if 3.0 <= chunk_dur <= 15.0:
            sub_text = " ".join(clauses[clause_idx:ci + 1]).strip()
            if sub_text:
                slices.append(TrainingSlice(
                    f"seg_{seg.id:04d}_{slice_num:02d}",
                    anchor_t,
                    nearest,
                    sub_text,
                    seg.speaker_needs_review,
                ))
                slice_num += 1
                anchor_t = nearest
                clause_idx = ci + 1
    tail_dur = seg.end - anchor_t
    if clause_idx < len(clauses) and 2.5 <= tail_dur <= 16.0:
        tail_text = " ".join(clauses[clause_idx:]).strip()
        if tail_text:
            slices.append(TrainingSlice(
                f"seg_{seg.id:04d}_{slice_num:02d}",
                anchor_t,
                seg.end,
                tail_text,
                seg.speaker_needs_review,
            ))
    return slices if len(slices) >= 2 else []


def validate_finetune_request(
    script: ProjectScript,
    speaker: str,
    req: FinetuneStartRequest,
) -> ValidatedFinetunePlan:
    if speaker not in script.characters:
        raise FileNotFoundError("角色不存在")
    if speaker == "UNKNOWN":
        raise ValueError("请先在剧本中确认说话人，再开始角色微调")

    lang = normalize_tts_lang(script.source_lang, default="")
    if lang not in {"zh", "ja", "en", "ko", "yue"}:
        raise ValueError("请先在翻译设置中确认原声语种（zh/ja/en/ko/yue），再开始角色微调")

    if not script.media.vocals_wav:
        raise ValueError("项目缺少可用人声干声文件，请先完成音频解析")
    vocals_path = config.resolve_media_path(script.media.vocals_wav)
    ws_root = config.WORKSPACE_DIR.resolve()
    if not vocals_path.is_file() or (
        ws_root not in vocals_path.parents and config.ASSETS_DIR.resolve() not in vocals_path.parents
    ):
        raise ValueError("找不到有效的项目人声文件，请先完成音频解析")

    root = config.GPT_SOVITS_ROOT
    if root is None or not root.is_dir():
        raise ValueError("未配置或找不到 GPT_SOVITS_ROOT 目录，无法执行角色微调")
    python_exe = _resolve_gsv_python()

    recipe = read_training_settings()
    missing = _missing_model_files(recipe.base_model, include_discriminator=True)
    pretrained_root = root / "GPT_SoVITS" / "pretrained_models"
    for shared_dir in ("chinese-roberta-wwm-ext-large", "chinese-hubert-base"):
        if not (pretrained_root / shared_dir).exists():
            missing.append(shared_dir)
    if missing:
        raise ValueError(f"微调底模 {recipe.base_model} 缺少必要模型文件：{', '.join(missing)}")

    clean_ids = _non_overlapping_ids(script.segments)
    clean_speaker_segs = [
        seg
        for seg in sorted(script.segments, key=lambda item: (item.start, item.id))
        if (
            seg.speaker == speaker
            and not seg.keep_original
            and seg.orig_text.strip()
            and seg.id in clean_ids
            and (seg.end - seg.start) >= MIN_CLIP_SECONDS
        )
    ]

    if not req.allow_unreviewed:
        selected_segs = [seg for seg in clean_speaker_segs if not seg.speaker_needs_review]
        if not selected_segs and clean_speaker_segs:
            raise ValueError(
                "该角色尚无已人工确认的干净台词片段。请先在下方『说话人校对』确认台词归属，或勾选允许使用未校对片段后再开始微调。"
            )
    else:
        selected_segs = clean_speaker_segs

    eligible: list[TrainingSlice] = []
    for seg in selected_segs:
        eligible.extend(_build_training_slices(seg, vocals_path))

    total_seconds = round(sum(sl.end - sl.start for sl in eligible), 1)
    if len(eligible) < MIN_CLIPS or total_seconds < MIN_TOTAL_SECONDS:
        raise ValueError(
            f"微调至少需要 {MIN_CLIPS} 条且累计不少于 {int(MIN_TOTAL_SECONDS)} 秒的无重叠干声片段"
            f"（当前仅 {len(eligible)} 条 / {total_seconds}s）"
        )

    is_few_shot = req.mode == "few_shot" or (
        req.mode == "auto"
        and (total_seconds < FEW_SHOT_SECONDS_THRESHOLD or len(eligible) < FEW_SHOT_CLIPS_THRESHOLD)
    )
    training_mode: Literal["standard", "few_shot"] = "few_shot" if is_few_shot else "standard"
    if training_mode == "few_shot":
        effective_sovits_epochs = min(recipe.sovits_epochs, 6 if total_seconds < 12.0 else 8)
        effective_gpt_epochs = 0
    else:
        effective_sovits_epochs = recipe.sovits_epochs
        effective_gpt_epochs = recipe.gpt_epochs

    warnings: list[str] = []
    if training_mode == "few_shot":
        warnings.append(
            f"已自动启用少样本防过拟合模式（{len(eligible)} 条 / {total_seconds}s）："
            f"仅微调 SoVITS 音色（{effective_sovits_epochs} epochs），冻结预训练 GPT 防复读吞字"
        )
    elif total_seconds < LOW_DATA_WARNING_SECONDS:
        warnings.append("当前有效干声不足 60 秒，建议结合多参考音融合增强音色稳定度")
    if req.allow_unreviewed and any(sl.speaker_needs_review for sl in eligible):
        warnings.append("已包含未逐句人工校对的识别片段，若存在串角将写入微调权重")

    return ValidatedFinetunePlan(
        script=script,
        speaker=speaker,
        lang=lang,
        vocals_path=vocals_path,
        eligible=eligible,
        total_seconds=total_seconds,
        warnings=warnings,
        recipe=recipe,
        training_mode=training_mode,
        effective_gpt_epochs=effective_gpt_epochs,
        effective_sovits_epochs=effective_sovits_epochs,
        root=root,
        python_exe=python_exe,
    )


def _slice_training_wav(vocals_path: Path, start: float, duration: float, out_wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(vocals_path),
            "-ac",
            "1",
            "-ar",
            "32000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _run_gsv_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    on_line: Callable[[str], None],
) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if line:
            tail = [*tail, line][-20:]
            on_line(line)
    rc = proc.wait()
    if rc != 0:
        detail = tail[-1] if tail else f"exit code {rc}"
        raise RuntimeError(f"GPT-SoVITS 训练步骤退出异常 (code={rc}): {detail}")


def _ensure_local_tts_released() -> None:
    if config.TTS_AUTO_MANAGE:
        stopped = ensure_tts_stopped()
        if stopped:
            time.sleep(0.5)
    endpoint = urlparse(config.TTS_API_URL)
    host = endpoint.hostname or "localhost"
    try:
        address = ipaddress.ip_address(host)
        is_local = address.is_loopback or address.is_unspecified
        probe_host = "127.0.0.1" if address.is_unspecified and address.version == 4 else (
            "::1" if address.is_unspecified else host
        )
    except ValueError:
        is_local = host.lower() == "localhost"
        probe_host = host
    if is_local:
        port = endpoint.port or 9880
        listening = False
        try:
            with socket.create_connection((probe_host, port), timeout=0.4):
                listening = True
        except OSError:
            listening = False
        if listening or check_tts_service(timeout=0.5):
            raise GpuBusyError(
                f"本机 GPT-SoVITS 端口 {port} 仍在占用显存，请先停止外部 TTS 服务后再开始微调。"
            )


def _merge_text_outputs(opt_dir: Path) -> Path:
    part_path = opt_dir / "2-name2text-0.txt"
    merged_path = opt_dir / "2-name2text.txt"
    if not part_path.is_file():
        raise RuntimeError(f"缺少文本特征输出文件: {part_path}")
    lines = [line for line in part_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    part_path.unlink(missing_ok=True)
    if not lines:
        raise RuntimeError("文本特征提取结果为空 (2-name2text.txt)")
    merged_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return merged_path


def _merge_semantic_outputs(opt_dir: Path) -> Path:
    part_path = opt_dir / "6-name2semantic-0.tsv"
    merged_path = opt_dir / "6-name2semantic.tsv"
    if not part_path.is_file():
        raise RuntimeError(f"缺少语义特征输出文件: {part_path}")
    lines = [line for line in part_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    part_path.unlink(missing_ok=True)
    if not lines:
        raise RuntimeError("语义特征提取结果为空 (6-name2semantic.tsv)")
    merged_path.write_text("\n".join(["item_name\tsemantic_audio", *lines]) + "\n", encoding="utf-8")
    return merged_path


def _locate_trained_weight(directory: Path, primary_glob: str, fallback_glob: str) -> Path:
    exact = sorted(directory.glob(primary_glob), key=lambda item: item.stat().st_mtime, reverse=True)
    if exact:
        return exact[0]
    fallback = sorted(directory.glob(fallback_glob), key=lambda item: item.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]
    raise RuntimeError(f"训练完成后未在 {directory} 找到匹配 {fallback_glob} 的模型权重")


def _gsv_train_cmd(python_exe: str, script_rel: str, script_args: list[str]) -> list[str]:
    """Avoid Windows DataLoader spawn MemoryError by forcing num_workers=0 in training subprocesses."""
    if os.name == "nt":
        launcher = (config.WORKERS_DIR / "gsv_train_launcher.py").resolve()
        return [python_exe, "-s", str(launcher), script_rel, *script_args]
    return [python_exe, "-s", script_rel, *script_args]


def _execute_pipeline(plan: ValidatedFinetunePlan, req: FinetuneStartRequest) -> None:
    project_id = plan.script.project_id
    speaker = plan.speaker
    recipe = plan.recipe
    root = plan.root
    base_model = recipe.base_model
    proj_slug = re.sub(r"[^a-zA-Z0-9]", "", project_id)[:12] or "proj"
    spk_hash = hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:6]
    exp_name = f"st_{proj_slug}_{spk_hash}_{int(time.time()) % 100000}"

    dataset_dir = config.WORKSPACE_DIR / project_id / "finetune" / exp_name
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = root / "logs" / exp_name
    opt_dir.mkdir(parents=True, exist_ok=True)

    _ensure_local_tts_released()
    _append_log(
        project_id,
        f"开始导出 {len(plan.eligible)} 条干声切片（共 {plan.total_seconds}s），底模={base_model}",
        stage="preparing",
        progress_pct=10,
    )

    list_lines: list[str] = []
    for sl in plan.eligible:
        clip_name = f"{sl.clip_id}.wav"
        _slice_training_wav(plan.vocals_path, sl.start, sl.end - sl.start, wav_dir / clip_name)
        clean_text = re.sub(r"[\r\n|]+", " ", sl.text).strip()
        list_lines.append(f"{clip_name}|{exp_name}|{plan.lang.upper()}|{clean_text}")
    list_path = dataset_dir / "train.list"
    list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")

    opt_dir_posix = opt_dir.resolve().as_posix()
    wav_dir_posix = wav_dir.resolve().as_posix()
    list_posix = list_path.resolve().as_posix()

    base_env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONFAULTHANDLER": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "inp_text": list_posix,
        "inp_wav_dir": wav_dir_posix,
        "exp_name": exp_name,
        "opt_dir": opt_dir_posix,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "version": base_model,
        "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    }

    on_line = lambda msg: _append_log(project_id, msg)

    # Step 1A: Text & BERT features
    _append_log(project_id, "[Step 1A] 提取音素与 BERT 文本特征…", stage="step_1a_text", progress_pct=25)
    _run_gsv_command(
        [plan.python_exe, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        cwd=root,
        env=base_env,
        on_line=on_line,
    )
    _merge_text_outputs(opt_dir)

    # Step 1B: HuBERT & ERes2NetV2 speaker embedding (for Pro)
    _append_log(project_id, "[Step 1B] 提取 HuBERT 特征与 32k 音频…", stage="step_1b_ssl_sv", progress_pct=40)
    _run_gsv_command(
        [plan.python_exe, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        cwd=root,
        env=base_env,
        on_line=on_line,
    )
    if "Pro" in base_model:
        _append_log(project_id, "[Step 1B+] 提取 v2Pro 说话人声纹特征 (ERes2NetV2)…")
        _run_gsv_command(
            [plan.python_exe, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"],
            cwd=root,
            env=base_env,
            on_line=on_line,
        )
        if not list((opt_dir / "7-sv_cn").glob("*.pt")):
            raise RuntimeError("未生成 v2Pro 说话人声纹特征 (7-sv_cn)")

    # Step 1C: Semantic tokens
    _append_log(project_id, "[Step 1C] 提取语义编码 tokens…", stage="step_1c_semantic", progress_pct=55)
    s2_cfg_rel = (
        "GPT_SoVITS/configs/s2.json"
        if base_model not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{base_model}.json"
    )
    t2s_rel, s2g_rel, s2d_rel = BASE_MODELS[base_model]
    env_1c = {
        **base_env,
        "pretrained_s2G": f"GPT_SoVITS/pretrained_models/{s2g_rel}",
        "s2config_path": s2_cfg_rel,
    }
    _run_gsv_command(
        [plan.python_exe, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
        cwd=root,
        env=env_1c,
        on_line=on_line,
    )
    _merge_semantic_outputs(opt_dir)

    if plan.training_mode == "few_shot":
        batch_size = min(2, max(1, len(plan.eligible) // 2))
    else:
        batch_size = min(4, max(1, len(plan.eligible) // 2))

    # Step 2: SoVITS training
    sovits_dir_rel = f"SoVITS_weights_{base_model}"
    sovits_weights_dir = root / sovits_dir_rel
    sovits_weights_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / f"logs_s2_{base_model}").mkdir(parents=True, exist_ok=True)

    s2_data = json.loads((root / s2_cfg_rel).read_text(encoding="utf-8"))
    s2_data["train"]["batch_size"] = batch_size
    s2_data["train"]["epochs"] = plan.effective_sovits_epochs
    s2_data["train"]["text_low_lr_rate"] = 0.3 if plan.training_mode == "few_shot" else 0.4
    s2_data["train"]["pretrained_s2G"] = f"GPT_SoVITS/pretrained_models/{s2g_rel}"
    s2_data["train"]["pretrained_s2D"] = f"GPT_SoVITS/pretrained_models/{s2d_rel}"
    s2_data["train"]["if_save_latest"] = True
    s2_data["train"]["if_save_every_weights"] = True
    s2_data["train"]["save_every_epoch"] = _save_every_epoch(plan.effective_sovits_epochs)
    s2_data["train"]["gpu_numbers"] = "0"
    s2_data["train"]["grad_ckpt"] = False
    s2_data["model"]["version"] = base_model
    s2_data["data"]["exp_dir"] = s2_data["s2_ckpt_dir"] = opt_dir_posix
    s2_data["save_weight_dir"] = sovits_dir_rel
    s2_data["name"] = exp_name
    s2_data["version"] = base_model
    tmp_s2 = dataset_dir / "tmp_s2.json"
    tmp_s2.write_text(json.dumps(s2_data, ensure_ascii=False, indent=2), encoding="utf-8")

    _append_log(
        project_id,
        f"[Step 2] 开始微调 SoVITS（{plan.effective_sovits_epochs} epochs, batch={batch_size}）…",
        stage="step_2_sovits",
        progress_pct=70,
    )
    _run_gsv_command(
        _gsv_train_cmd(plan.python_exe, "GPT_SoVITS/s2_train.py", ["--config", tmp_s2.resolve().as_posix()]),
        cwd=root,
        env=base_env,
        on_line=on_line,
    )
    sovits_file = _locate_trained_weight(
        sovits_weights_dir,
        f"{exp_name}_e{plan.effective_sovits_epochs}_s*.pth",
        f"{exp_name}_e*.pth",
    )
    sovits_rel_path = f"{sovits_dir_rel}/{sovits_file.name}"
    time.sleep(1.5)

    # Step 3: GPT training (skipped in few-shot mode to freeze base GPT weights against overfitting)
    gpt_rel_path: str | None = None
    if plan.effective_gpt_epochs > 0:
        gpt_dir_rel = f"GPT_weights_{base_model}"
        gpt_weights_dir = root / gpt_dir_rel
        gpt_weights_dir.mkdir(parents=True, exist_ok=True)
        logs_s1_dir = opt_dir / f"logs_s1_{base_model}"
        logs_s1_dir.mkdir(parents=True, exist_ok=True)

        s1_data = yaml.safe_load((root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").read_text(encoding="utf-8"))
        s1_data["train"]["batch_size"] = batch_size
        s1_data["train"]["epochs"] = plan.effective_gpt_epochs
        s1_data["train"]["save_every_n_epoch"] = _save_every_epoch(plan.effective_gpt_epochs)
        s1_data["train"]["if_save_every_weights"] = True
        s1_data["train"]["if_save_latest"] = True
        s1_data["train"]["if_dpo"] = False
        s1_data["train"]["half_weights_save_dir"] = gpt_dir_rel
        s1_data["train"]["exp_name"] = exp_name
        s1_data["pretrained_s1"] = f"GPT_SoVITS/pretrained_models/{t2s_rel}"
        s1_data["train_semantic_path"] = f"{opt_dir_posix}/6-name2semantic.tsv"
        s1_data["train_phoneme_path"] = f"{opt_dir_posix}/2-name2text.txt"
        s1_data["output_dir"] = logs_s1_dir.resolve().as_posix()
        if os.name == "nt":
            s1_data.setdefault("data", {})["num_workers"] = 0
        tmp_s1 = dataset_dir / "tmp_s1.yaml"
        tmp_s1.write_text(yaml.safe_dump(s1_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        _append_log(
            project_id,
            f"[Step 3] 开始微调 GPT（{plan.effective_gpt_epochs} epochs, batch={batch_size}）…",
            stage="step_3_gpt",
            progress_pct=85,
        )
        env_s1 = {**base_env, "hz": "25hz", "_CUDA_VISIBLE_DEVICES": "0"}
        _run_gsv_command(
            _gsv_train_cmd(plan.python_exe, "GPT_SoVITS/s1_train.py", ["--config_file", tmp_s1.resolve().as_posix()]),
            cwd=root,
            env=env_s1,
            on_line=on_line,
        )
        gpt_file = _locate_trained_weight(
            gpt_weights_dir,
            f"{exp_name}-e{plan.effective_gpt_epochs}.ckpt",
            f"{exp_name}-e*.ckpt",
        )
        gpt_rel_path = f"{gpt_dir_rel}/{gpt_file.name}"
    else:
        _append_log(
            project_id,
            "[Step 3] 少样本防过拟合模式：冻结预训练 GPT 底模，跳过 s1_train 避免跨语种复读/吞字",
            stage="step_3_gpt",
            progress_pct=90,
        )

    # Step 4: Bind weights and register VoicePreset
    _append_log(project_id, "[Step 4] 绑定角色权重并注册跨项目音色预设…", stage="registering_preset", progress_pct=95)
    preset_id: str | None = None
    with project_task_guard(project_id):
        latest_script = load_project_script(project_id)
        character = latest_script.characters.get(speaker)
        if character is None:
            raise ValueError(f"角色 {speaker} 已不存在于当前剧本中")
        refs_populated = False
        if not character.emotion_refs:
            populated = build_character_emotion_refs(latest_script, speaker=speaker, overwrite=False)
            refs_populated = bool(populated.get(speaker))
        if req.bind_to_character:
            character.gpt_weights = gpt_rel_path
            character.sovits_weights = sovits_rel_path
        if req.bind_to_character or refs_populated:
            save_project_script(latest_script)
        if req.save_as_preset:
            preset_name = (req.preset_name or "").strip() or f"{character.display_name or speaker} ({base_model})"
            preset = create_voice_preset(
                VoicePresetCreateRequest(
                    name=preset_name,
                    speaking_style=character.speaking_style or "",
                    gpt_weights=gpt_rel_path,
                    sovits_weights=sovits_rel_path,
                    emotion_refs=character.emotion_refs,
                )
            )
            preset_id = preset.preset_id

    with _jobs_lock:
        job = _jobs_by_project.get(project_id)
        if job is not None:
            job.active = False
            job.stage = "completed"
            job.progress_pct = 100
            job.gpt_weights = gpt_rel_path
            job.sovits_weights = sovits_rel_path
            job.preset_id = preset_id
            gpt_label = gpt_rel_path or "(base_frozen)"
            job.logs = [*job.logs, f"微调完成：GPT={gpt_label}，SoVITS={sovits_rel_path}"][-80:]
            job.updated_at = time.time()


def start_finetune_job(
    project_id: str,
    speaker: str,
    req: FinetuneStartRequest,
) -> FinetuneJobStatus:
    """Validate training prerequisites, acquire GPU lock, and launch background fine-tuning."""
    with _launch_lock:
        with project_task_guard(project_id):
            script = load_project_script(project_id)
            plan = validate_finetune_request(script, speaker, req)

        job_id = f"ft_{int(time.time())}"
        handshake_done = threading.Event()
        handshake_error: list[BaseException] = []

        def _runner() -> None:
            acquired = False
            try:
                with gpu_task_guard():
                    acquired = True
                    with _jobs_lock:
                        _jobs_by_project[project_id] = FinetuneJobStatus(
                            job_id=job_id,
                            active=True,
                            project_id=project_id,
                            speaker=speaker,
                            training_mode=plan.training_mode,
                            base_model=plan.recipe.base_model,
                            gpt_epochs=plan.effective_gpt_epochs,
                            sovits_epochs=plan.effective_sovits_epochs,
                            stage="preparing",
                            progress_pct=5,
                            clip_count=len(plan.eligible),
                            usable_duration_sec=plan.total_seconds,
                            warnings=list(plan.warnings),
                            logs=[f"已获得 GPU 队列锁，准备微调 {speaker}"],
                            updated_at=time.time(),
                        )
                    handshake_done.set()
                    _execute_pipeline(plan, req)
            except BaseException as exc:
                if not acquired:
                    handshake_error.append(exc)
                    handshake_done.set()
                else:
                    with _jobs_lock:
                        job = _jobs_by_project.get(project_id)
                        if job is not None:
                            job.active = False
                            job.stage = "error"
                            job.error = str(exc)
                            job.logs = [*job.logs, f"微调失败：{exc}"][-80:]
                            job.updated_at = time.time()

        worker = threading.Thread(target=_runner, name=f"gsv-finetune-{project_id}", daemon=True)
        worker.start()
        if not handshake_done.wait(timeout=5.0):
            raise GpuBusyError("等待 GPU 训练任务队列超时，请稍后重试")
        if handshake_error:
            err = handshake_error[0]
            if isinstance(err, RuntimeError):
                raise err
            raise RuntimeError(str(err)) from err

        return get_finetune_status(project_id, speaker=speaker)
