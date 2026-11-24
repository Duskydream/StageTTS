"""StageTTS 主后端 FastAPI 入口（运行在 env_backend，纯 CPU 0 显存占用）。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import AsyncIterator, Dict, List
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services.llm_settings import (
    LLMSettingsResponse, LLMSettingsUpdate, get_llm_snapshot, save_llm_settings,
)
from backend.services.training_settings import (
    InferenceModelUpdate, TrainingSettings, TrainingSettingsResponse,
    apply_inference_base_model, training_settings_response, save_training_settings,
)
from backend.services.tts_settings import (
    TtsSettings,
    TtsSettingsResponse,
    read_tts_engine,
    save_tts_engine,
    tts_settings_response,
)
from backend.services.voice_presets import (
    create_voice_preset, delete_voice_preset, list_voice_presets,
)
from backend import config
from backend.agent.orchestrator import run_agent_chat
from backend.agent.tools import (
    load_project_script,
    project_task_guard,
    render_final_video,
    save_project_script,
    synthesize_project,
)
from backend.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    GpuStatusResponse,
    HealthResponse,
    MediaInfo,
    ParseRequest,
    ParseStatusResponse,
    ProjectListResponse,
    ProjectMode,
    ProjectScript,
    ReferencePreview,
    ReferenceUploadResponse,
    RenderRequest,
    RenderResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranslationRequest,
    TranslationResponse,
    UploadResponse,
    VoicePreset,
    VoicePresetCreateRequest,
    VoicePresetDeleteResponse,
    VoicePresetListResponse,
)
from backend.services.media_mixer import check_ffmpeg_available
from backend.services.emotion_routing import build_character_emotion_refs
from backend.services.gsv_finetune import (
    FinetuneJobStatus,
    FinetuneStartRequest,
    get_finetune_status,
    start_finetune_job,
)
from backend.services.references import automatic_reference, store_reference_audio
from backend.services.text_normalization import normalize_tts_lang
from backend.services.task_runner import (
    GpuBusyError,
    ParseWorkerError,
    get_parse_progress,
    gpu_task_guard,
    ensure_tts_stopped,
    is_gpu_busy,
    probe_media_duration,
    read_gpu_memory,
    run_parse_worker,
)
from backend.services.translation import (
    LLMNotConfiguredError,
    LLMServiceError,
    create_openai_client,
    translate_script,
)
from backend.services.tts_client import (
    TTSServiceUnavailableError,
    check_tts_service,
)

# 确保挂载目录存在，避免 StaticFiles 初始化报错
config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

app = FastAPI(
    title="StageTTS Backend API",
    version="0.2.0",
    description="基于剧本中间表示 script.json 与 Agent 协同的多角色配音及虚拟演播平台后端服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件挂载：供前端加载视频、纯人声/BGM、单句 TTS 音频以及 Live2D/VRM 模型
app.mount("/workspace", StaticFiles(directory=str(config.WORKSPACE_DIR)), name="workspace")
app.mount("/assets", StaticFiles(directory=str(config.ASSETS_DIR)), name="assets")


# ---------------------------------------------------------------------------
# 统一业务异常映射（保持接口错误码语义清晰）
# ---------------------------------------------------------------------------


def _err_detail(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


@app.exception_handler(GpuBusyError)
def _gpu_busy_handler(_: Request, exc: GpuBusyError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": _err_detail(exc)})


@app.exception_handler(ParseWorkerError)
def _parse_worker_handler(_: Request, exc: ParseWorkerError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": _err_detail(exc)})


@app.exception_handler(TTSServiceUnavailableError)
def _tts_unavailable_handler(_: Request, exc: TTSServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": _err_detail(exc)})


@app.exception_handler(LLMNotConfiguredError)
def _llm_unconfigured_handler(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": _err_detail(exc)})


@app.exception_handler(LLMServiceError)
def _llm_service_error_handler(_: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": _err_detail(exc)})


@app.exception_handler(FileNotFoundError)
def _not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _err_detail(exc)})


@app.exception_handler(ValueError)
def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": _err_detail(exc)})


@app.exception_handler(RuntimeError)
def _runtime_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    msg = _err_detail(exc)
    status_code = 400 if "导出待审核" in msg else 500
    return JSONResponse(status_code=status_code, content={"detail": msg})


@app.exception_handler(KeyError)
def _key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": _err_detail(exc)})


@app.exception_handler(NotImplementedError)
def _not_implemented_handler(_: Request, exc: NotImplementedError) -> JSONResponse:
    return JSONResponse(status_code=501, content={"detail": _err_detail(exc)})


def _validate_project_id(raw_id: str) -> str:
    cleaned = raw_id.strip()
    if not _PROJECT_ID_RE.match(cleaned):
        raise ValueError("project_id 仅允许字母、数字、下划线或连字符（1~64 位）")
    return cleaned


def _find_input_media(project_id: str, script: ProjectScript | None) -> Path:
    if script and script.media.source_video:
        candidate = config.resolve_media_path(script.media.source_video)
        if candidate.is_file():
            return candidate

    project_dir = config.WORKSPACE_DIR / project_id
    if project_dir.is_dir():
        for match in sorted(project_dir.glob("input.*")):
            if match.is_file():
                return match
    raise FileNotFoundError(f"项目 {project_id} 尚未上传源视频或音频文件")


# ---------------------------------------------------------------------------
# 健康检查与项目剧本读写接口
# ---------------------------------------------------------------------------


@app.get("/api/settings/training", response_model=TrainingSettingsResponse)
def get_training_settings() -> TrainingSettingsResponse:
    return training_settings_response()


@app.put("/api/settings/training", response_model=TrainingSettingsResponse)
def put_training_settings(req: TrainingSettings) -> TrainingSettingsResponse:
    return save_training_settings(req)


@app.put("/api/settings/training/inference", response_model=TrainingSettingsResponse)
def put_inference_base_model(req: InferenceModelUpdate) -> TrainingSettingsResponse:
    return apply_inference_base_model(req.base_model)


@app.get("/api/settings/tts", response_model=TtsSettingsResponse)
def get_tts_settings() -> TtsSettingsResponse:
    return tts_settings_response()


@app.put("/api/settings/tts", response_model=TtsSettingsResponse)
def put_tts_settings(req: TtsSettings) -> TtsSettingsResponse:
    return save_tts_engine(req.engine)


@app.get("/api/voice-presets", response_model=VoicePresetListResponse)
def get_voice_presets() -> VoicePresetListResponse:
    return VoicePresetListResponse(presets=list_voice_presets())


@app.post("/api/voice-presets", response_model=VoicePreset)
def post_voice_preset(req: VoicePresetCreateRequest) -> VoicePreset:
    return create_voice_preset(req)


@app.delete("/api/voice-presets/{preset_id}", response_model=VoicePresetDeleteResponse)
def remove_voice_preset(preset_id: str) -> VoicePresetDeleteResponse:
    return VoicePresetDeleteResponse(deleted_id=delete_voice_preset(preset_id))


@app.get("/api/settings/llm", response_model=LLMSettingsResponse)
def read_llm_settings() -> LLMSettingsResponse:
    return get_llm_snapshot().public()


@app.put("/api/settings/llm", response_model=LLMSettingsResponse)
def update_llm_settings(req: LLMSettingsUpdate) -> LLMSettingsResponse:
    return save_llm_settings(req)


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> Dict[str, object]:
    """服务健康检查、GPU 单任务锁状态与外部环境就绪度探测。"""
    active_tts_url = (
        config.COSYVOICE3_API_URL if read_tts_engine() == "cosyvoice3" else config.TTS_API_URL
    )
    return {
        "status": "ok",
        "busy": is_gpu_busy(),
        "workspace_dir": str(config.WORKSPACE_DIR),
        "assets_dir": str(config.ASSETS_DIR),
        "readiness": {
            "ffmpeg": check_ffmpeg_available(),
            "tts_service": check_tts_service(),
            "tts_auto_manage": config.TTS_AUTO_MANAGE,
            "tts_url": active_tts_url,
            "llm_configured": config.is_llm_configured(),
            "hf_token_configured": config.is_hf_token_configured(),
            "banquet_configured": config.is_banquet_configured(),
            "env_parse_python": config.ENV_PARSE_PYTHON,
        },
    }


@app.get("/api/gpu", response_model=GpuStatusResponse)
def gpu_status() -> Dict[str, object]:
    return {"devices": read_gpu_memory()}


@app.get("/api/projects", response_model=ProjectListResponse)
def list_projects() -> Dict[str, List[str]]:
    """列出 workspace/ 下所有包含 script.json 的有效项目 ID。"""
    projects: List[str] = []
    if config.WORKSPACE_DIR.exists():
        for item in sorted(config.WORKSPACE_DIR.iterdir()):
            if item.is_dir() and (item / "script.json").is_file():
                projects.append(item.name)
    return {"projects": projects}


@app.get("/api/projects/{project_id}/script", response_model=ProjectScript)
def get_project_script(project_id: str) -> ProjectScript:
    """读取指定项目的唯一真相源 script.json。"""
    safe_id = _validate_project_id(project_id)
    return load_project_script(safe_id)


@app.put("/api/projects/{project_id}/script", response_model=ProjectScript)
def put_project_script(project_id: str, payload: ProjectScript) -> ProjectScript:
    """校验并保存前端修改后的完整 ProjectScript。"""
    safe_id = _validate_project_id(project_id)
    if payload.project_id != safe_id:
        raise ValueError(
            f"路径 project_id ({safe_id}) 与请求体 project_id ({payload.project_id}) 不一致"
        )
    with project_task_guard(safe_id):
        try:
            previous = load_project_script(safe_id)
        except FileNotFoundError:
            previous = None
        if previous is not None:
            originals = {segment.id: segment for segment in previous.segments}
            for segment in payload.segments:
                before = originals.get(segment.id)
                if before is None:
                    segment.alignment_granularity = "unknown"
                    segment.timing_source = "unknown"
                    segment.segment_timing_needs_review = True
                    continue
                # Source capabilities come from parsing, not a client-side claim.
                segment.alignment_granularity = before.alignment_granularity
                segment.timing_source = before.timing_source
                if (segment.orig_text, segment.start, segment.end, segment.speaker) != (
                    before.orig_text, before.start, before.end, before.speaker
                ) or payload.media.vocals_wav != previous.media.vocals_wav:
                    segment.segment_timing_needs_review = True
                if (segment.orig_text, segment.start, segment.end) != (before.orig_text, before.start, before.end):
                    segment.alignment_needs_review = True
        save_project_script(payload)
    return payload


@app.post("/api/projects/{project_id}/references", response_model=ReferenceUploadResponse)
def upload_project_reference(project_id: str, file: UploadFile = File(...)) -> dict[str, str]:
    safe_id = _validate_project_id(project_id)
    load_project_script(safe_id)
    if file.size is not None and file.size > 32 * 1024 * 1024:
        raise ValueError("参考音文件不能超过 32 MB")
    return {"wav": store_reference_audio(safe_id, file.file)}


class AutoEmotionRefsRequest(BaseModel):
    overwrite: bool = False


class AutoEmotionRefsResponse(BaseModel):
    script: ProjectScript
    populated_emotions: List[str] = Field(default_factory=list)


@app.post("/api/projects/{project_id}/characters/{speaker}/reference-preview", response_model=ReferencePreview)
def preview_automatic_reference(project_id: str, speaker: str) -> ReferencePreview:
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id), gpu_task_guard():
        script = load_project_script(safe_id)
        if speaker not in script.characters:
            raise FileNotFoundError("角色不存在")
        ref = automatic_reference(script, speaker)
        if ref is None:
            raise ValueError("找不到 3–10 秒的同角色参考音。请确认台词原文、时间范围、说话人和源语种，或上传参考音")
        return ref


@app.post("/api/projects/{project_id}/characters/{speaker}/emotion-refs", response_model=AutoEmotionRefsResponse)
def auto_match_character_emotion_refs(
    project_id: str,
    speaker: str,
    req: AutoEmotionRefsRequest | None = None,
) -> AutoEmotionRefsResponse:
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id), gpu_task_guard():
        script = load_project_script(safe_id)
        if speaker not in script.characters:
            raise FileNotFoundError("角色不存在")
        if speaker == "UNKNOWN":
            raise ValueError("请先在剧本或角色面板确认说话人，再自动匹配多情绪参考音")
        if not normalize_tts_lang(script.source_lang, default=""):
            raise ValueError("请先在翻译设置中确认原声语种，再自动匹配多情绪参考音")
        if not script.media.vocals_wav:
            raise ValueError("项目缺少源人声音轨，无法自动匹配参考音")
        overwrite = bool(req.overwrite) if req is not None else False
        populated = build_character_emotion_refs(script, speaker=speaker, overwrite=overwrite)
        emotions = populated.get(speaker, [])
        if emotions:
            save_project_script(script)
        return AutoEmotionRefsResponse(script=script, populated_emotions=emotions)


@app.post("/api/projects/{project_id}/characters/{speaker}/finetune", response_model=FinetuneJobStatus)
def start_character_finetune(
    project_id: str,
    speaker: str,
    req: FinetuneStartRequest | None = None,
) -> FinetuneJobStatus:
    safe_id = _validate_project_id(project_id)
    return start_finetune_job(safe_id, speaker, req or FinetuneStartRequest())


@app.get("/api/projects/{project_id}/finetune/status", response_model=FinetuneJobStatus)
def read_project_finetune_status(project_id: str, speaker: str | None = None) -> FinetuneJobStatus:
    safe_id = _validate_project_id(project_id)
    return get_finetune_status(safe_id, speaker=speaker)


@app.post("/api/projects/{project_id}/translate", response_model=TranslationResponse)
def retranslate_project(project_id: str, req: TranslationRequest | None = None) -> TranslationResponse:
    """Resume from saved source text; never invoke the media parsing worker."""
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        script = load_project_script(safe_id)
        if not script.segments:
            raise ValueError("没有可翻译片段。请先解析媒体或导入含原文台词的剧本，再从翻译步骤开始。")
        eligible = {seg.id for seg in script.segments if seg.orig_text.strip()}
        if not eligible:
            raise ValueError("片段原文 orig_text 为空。请先补全原文台词，再从翻译步骤开始。")

        # Translate an isolated candidate. A language override must not relabel
        # failed rows or leave old TTS audio attached to the wrong language.
        candidate = script.model_copy(deep=True)
        target_lang = req.target_lang.lower() if req is not None and req.target_lang is not None else None
        if target_lang is not None:
            for seg in candidate.segments:
                if seg.id in eligible:
                    seg.target_lang = target_lang
        batch_errors: list[str] = []
        natural_polish = bool(req.natural_polish) if req is not None else False
        web_search = bool(req.web_search) if req is not None else False
        translated = translate_script(
            candidate,
            create_openai_client(),
            errors=batch_errors,
            enable_analysis=natural_polish or web_search,
            enable_polish=natural_polish,
            enable_web_search=web_search,
        )
        successful = set(translated) & eligible
        if not successful:
            detail = batch_errors[0] if batch_errors else "LLM 未返回任何合法译文，请检查模型配置后重试"
            raise LLMServiceError(f"台词重译失败：{detail}")

        for spk, candidate_char in candidate.characters.items():
            if spk in script.characters and not (script.characters[spk].speaking_style or "").strip():
                if (candidate_char.speaking_style or "").strip():
                    script.characters[spk].speaking_style = candidate_char.speaking_style.strip()
        for term, mapped in candidate.glossary.items():
            if isinstance(term, str) and isinstance(mapped, str) and term.strip() and mapped.strip():
                if term not in script.glossary:
                    script.glossary[term] = mapped

        candidates = {seg.id: seg for seg in candidate.segments}
        for seg in script.segments:
            if seg.id not in successful:
                continue
            result = candidates[seg.id]
            before_audio = (seg.target_text, seg.target_lang, seg.emotion, seg.keep_original)
            seg.target_text = result.target_text
            seg.target_lang = result.target_lang
            seg.emotion = result.emotion
            seg.action = result.action
            seg.keep_original = result.keep_original
            if (seg.target_text, seg.target_lang, seg.emotion, seg.keep_original) != before_audio:
                seg.invalidate_audio()
        missing = eligible - successful
        if missing and not batch_errors:
            batch_errors.append("以下片段未返回合法译文，已保留旧文本/语种/音频：" + ", ".join(f"#{item}" for item in sorted(missing)))
        save_project_script(script)
        return TranslationResponse(
            script=script,
            translated_ids=sorted(successful),
            skipped_count=len(script.segments) - len(successful),
            errors=batch_errors,
        )


# ---------------------------------------------------------------------------
# 上传、解析、分组 TTS 合成、Agent 协同与后期导出接口
# ---------------------------------------------------------------------------


@app.post("/api/projects/upload", response_model=UploadResponse)
def upload_project_media(
    file: UploadFile = File(...),
    mode: ProjectMode = Form("video_dubbing"),
    project_id: str = Form(""),
) -> Dict[str, object]:
    """接收用户上传视频/音频，初始化 workspace/{project_id}/ 与初始 script.json。"""
    safe_id = (
        _validate_project_id(project_id)
        if project_id.strip()
        else f"proj_{int(time.time())}"
    )

    with project_task_guard(safe_id):
        project_dir = config.WORKSPACE_DIR / safe_id
        project_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
        input_name = f"input{ext}"
        input_path = project_dir / input_name

        with input_path.open("wb") as out_fp:
            shutil.copyfileobj(file.file, out_fp)

        duration = probe_media_duration(input_path)
        media_url = f"/workspace/{safe_id}/{input_name}"

        initial_script = ProjectScript(
            project_id=safe_id,
            mode=mode,
            media=MediaInfo(
                source_video=media_url,
                duration=duration,
                bgm_volume=0.75,
            ),
            characters={},
            segments=[],
        )
        save_project_script(initial_script)

    return {
        "project_id": safe_id,
        "media_url": media_url,
        "duration": duration,
    }


@app.get("/api/projects/{project_id}/parse/status", response_model=ParseStatusResponse)
def get_project_parse_status(project_id: str) -> Dict[str, object]:
    """查询指定项目的离线解析实时阶段与日志（供前端节点流轮询）。"""
    safe_id = _validate_project_id(project_id)
    return get_parse_progress(safe_id)


@app.get("/api/projects/{project_id}/parse/stream")
async def stream_project_parse_status(project_id: str, request: Request) -> StreamingResponse:
    """SSE 实时推送指定项目的离线解析阶段与日志。"""
    safe_id = _validate_project_id(project_id)

    async def event_generator() -> AsyncIterator[str]:
        last_key: tuple[object, ...] | None = None
        saw_active = False
        idle_ticks = 0
        while not await request.is_disconnected():
            status = get_parse_progress(safe_id)
            active = bool(status.get("active"))
            if active:
                saw_active = True
            key = (active, status.get("stage"), status.get("log_count"), status.get("error"))
            if key != last_key:
                last_key = key
                yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"
            if not active and (saw_active or idle_ticks >= 15):
                break
            idle_ticks += 1
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/projects/{project_id}/parse", response_model=ProjectScript)
def parse_project_media(project_id: str, req: ParseRequest) -> ProjectScript:
    """启动 env_parse 子进程执行离线解析并生成 script.json（受 GPU 单任务锁保护）。"""
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        existing_script: ProjectScript | None = None
        try:
            existing_script = load_project_script(safe_id)
        except FileNotFoundError:
            existing_script = None

        input_path = _find_input_media(safe_id, existing_script)
        mode = existing_script.mode if existing_script else "video_dubbing"

        run_parse_worker(
            project_id=safe_id,
            input_path=input_path,
            target_lang=req.target_lang,
            source_lang=req.source_lang or (existing_script.source_lang if existing_script else None),
            mode=mode,
            natural_polish=req.natural_polish,
            web_search=req.web_search,
        )
        return load_project_script(safe_id)


class PerformanceAnalysisRequest(BaseModel):
    segment_ids: List[int] | None = None
    enable_emotion: bool = False
    enable_prosody: bool = True


class PerformanceAnalysisResponse(BaseModel):
    project_id: str
    status: str
    stage_status: Dict[str, str] = Field(default_factory=dict)
    review_reasons: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    script: ProjectScript


@app.post("/api/projects/{project_id}/analyze-performance", response_model=PerformanceAnalysisResponse)
def analyze_project_source(project_id: str, req: PerformanceAnalysisRequest) -> PerformanceAnalysisResponse:
    """Explicit local source analysis; no downloads or inferred neutral defaults."""
    from backend.services.performance_analysis import analyze_project_performance, apply_analysis_to_evidence

    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        script = load_project_script(safe_id)
        known_ids = {seg.id for seg in script.segments}
        if req.segment_ids is not None and not set(req.segment_ids) <= known_ids:
            raise ValueError("分析请求包含不存在的片段 ID")
        if not req.enable_emotion and not req.enable_prosody:
            raise ValueError("请至少选择一项源声分析")
        source_path = config.resolve_media_path(script.media.vocals_wav) if script.media.vocals_wav else None
        with gpu_task_guard() if req.enable_emotion else nullcontext():
            ser_configured = all((config.ENV_ANALYSIS_PYTHON, config.EMOTION_MODEL_PATH,
                                  config.EMOTION_MODEL_ID, config.EMOTION_REVISION,
                                  config.EMOTION_FUNASR_VERSION, config.EMOTION_USER_AUTHORIZED))
            if req.enable_emotion and ser_configured:
                # Serialize managed GPU stages; never terminate an external TTS process.
                if config.TTS_AUTO_MANAGE:
                    ensure_tts_stopped()
                tts_host = urlparse(config.TTS_API_URL).hostname or "localhost"
                try:
                    local_tts = ipaddress.ip_address(tts_host).is_loopback or ipaddress.ip_address(tts_host).is_unspecified
                except ValueError:
                    local_tts = tts_host.lower() == "localhost"
                if local_tts and check_tts_service():
                    raise GpuBusyError("源声情绪分析前请释放本机外部 GPT-SoVITS 服务，避免同时占用显存。")
            report = analyze_project_performance(
                script,
                segment_ids=req.segment_ids,
                audio_path=source_path,
                enable_emotion=req.enable_emotion,
                enable_prosody=req.enable_prosody,
                emotion_python=config.ENV_ANALYSIS_PYTHON or None,
                emotion_worker_path=config.ROOT_DIR / "workers" / "emotion_worker.py",
                emotion_model_path=str(config.EMOTION_MODEL_PATH) if config.EMOTION_MODEL_PATH else None,
                emotion_model_id=config.EMOTION_MODEL_ID or None,
                emotion_revision=config.EMOTION_REVISION or None,
                emotion_funasr_version=config.EMOTION_FUNASR_VERSION or None,
                emotion_user_authorized=config.EMOTION_USER_AUTHORIZED,
                emotion_device=config.EMOTION_DEVICE,
                enable_f0=config.ANALYSIS_ENABLE_F0,
                prosody_python=config.ENV_ANALYSIS_PYTHON or None,
                prosody_worker_path=config.ROOT_DIR / "workers" / "prosody_worker.py",
            )
        annotations = apply_analysis_to_evidence(script, report)
        if script.evidence is not None and annotations:
            additions = {annotation.id: annotation for annotation in annotations}
            previous = [annotation for annotation in script.evidence.annotations if annotation.id not in additions]
            evidence = script.evidence.model_copy(update={"annotations": [*previous, *annotations]})
            script = ProjectScript.model_validate({**script.model_dump(), "evidence": evidence.model_dump()})
            save_project_script(script)
        return PerformanceAnalysisResponse(
            project_id=safe_id,
            status=report.status,
            stage_status={"emotion": report.emotion.status, "prosody": report.prosody.status},
            review_reasons=report.review_reasons,
            errors=report.errors,
            script=script,
        )


@app.post("/api/projects/{project_id}/synthesize", response_model=SynthesizeResponse)
def synthesize_project_segments(project_id: str, req: SynthesizeRequest) -> SynthesizeResponse:
    """分组配音并执行双向时长闭环，返回最终剧本和仍偏离时间槽的片段。"""
    return synthesize_project(_validate_project_id(project_id), req.segment_ids)


@app.post("/api/projects/{project_id}/agent/chat", response_model=AgentChatResponse)
def agent_chat(project_id: str, req: AgentChatRequest) -> AgentChatResponse:
    """自然语言批量剪辑对话（5 大确定性原子工具调度 + 时长超标反思改写闭环）。"""
    safe_id = _validate_project_id(project_id)
    return run_agent_chat(
        project_id=safe_id,
        user_message=req.message,
        auto_synthesize=req.auto_synthesize,
    )


@app.post("/api/projects/{project_id}/render", response_model=RenderResponse)
def render_project_video(project_id: str, req: RenderRequest) -> RenderResponse:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装，导出 output.mp4（或虚拟舞台音频）。"""
    safe_id = _validate_project_id(project_id)
    return RenderResponse(
        project_id=safe_id,
        **render_final_video(
            safe_id,
            burn_watermark=req.burn_watermark,
            burn_subtitles=req.burn_subtitles,
            auto_fit_overflow=req.auto_fit_overflow,
        ),
    )
