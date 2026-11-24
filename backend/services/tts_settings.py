"""Persist zero-shot TTS engine selection separately from GPT-SoVITS training recipes."""
from __future__ import annotations

import shutil
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend import config
from backend.services.task_runner import ensure_tts_stopped, gpu_task_guard
from backend.services.training_settings import _atomic_write_text

SETTINGS_PATH = config.BACKEND_DIR / "tts_settings.local.json"
_lock = RLock()

TtsEngineId = Literal["gpt_sovits", "cosyvoice3"]
_ENGINE_LABELS: dict[TtsEngineId, str] = {
    "gpt_sovits": "GPT-SoVITS",
    "cosyvoice3": "CosyVoice 3（零样本）",
}


class TtsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine: TtsEngineId = "gpt_sovits"


class TtsEngineOption(BaseModel):
    id: TtsEngineId
    label: str
    available: bool
    missing_files: list[str]


class TtsSettingsResponse(BaseModel):
    engine: TtsEngineId
    engines: list[TtsEngineOption]


def _python_exists(raw: str) -> bool:
    cmd = (raw or "").strip()
    return bool(cmd and (Path(cmd).is_file() or shutil.which(cmd) is not None))


def _missing_engine_files(engine: TtsEngineId) -> list[str]:
    if engine == "gpt_sovits":
        if not config.TTS_AUTO_MANAGE:
            return [] if config.TTS_API_URL.strip() else ["TTS_API_URL"]
        missing: list[str] = []
        root = config.GPT_SOVITS_ROOT
        if root is None or not (root / "api_v2.py").is_file():
            missing.append("GPT_SOVITS_ROOT/api_v2.py")
        if not _python_exists(config.GPT_SOVITS_PYTHON):
            missing.append("GPT_SOVITS_PYTHON")
        return missing

    missing = []
    if not (config.WORKERS_DIR / "cosyvoice3_server.py").is_file():
        missing.append("workers/cosyvoice3_server.py")
    if not config.TTS_AUTO_MANAGE:
        if not config.COSYVOICE3_API_URL.strip():
            missing.append("COSYVOICE3_API_URL")
        return missing
    cv_root = config.COSYVOICE3_ROOT
    if cv_root is None or not (cv_root / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
        missing.append("COSYVOICE3_ROOT/cosyvoice")
    model_dir = config.COSYVOICE3_MODEL_DIR
    if model_dir is None or not (model_dir / "cosyvoice3.yaml").is_file():
        missing.append("COSYVOICE3_MODEL_DIR/cosyvoice3.yaml")
    if not _python_exists(config.COSYVOICE3_PYTHON):
        missing.append("COSYVOICE3_PYTHON")
    return missing


def read_tts_engine() -> TtsEngineId:
    """Return the persisted active TTS engine, defaulting to gpt_sovits."""
    with _lock:
        if not SETTINGS_PATH.exists():
            return "gpt_sovits"
        try:
            return TtsSettings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8")).engine
        except (OSError, ValueError):
            return "gpt_sovits"


def tts_settings_response() -> TtsSettingsResponse:
    active = read_tts_engine()
    options: list[TtsEngineOption] = []
    for engine_id, label in _ENGINE_LABELS.items():
        missing = _missing_engine_files(engine_id)
        options.append(
            TtsEngineOption(
                id=engine_id,
                label=label,
                available=not missing,
                missing_files=missing,
            )
        )
    return TtsSettingsResponse(engine=active, engines=options)


def save_tts_engine(engine: TtsEngineId) -> TtsSettingsResponse:
    missing = _missing_engine_files(engine)
    if missing:
        raise ValueError(f"配音引擎 {engine} 尚未就绪，缺少：{', '.join(missing)}")
    with gpu_task_guard():
        with _lock:
            previous = read_tts_engine()
            if previous != engine and config.TTS_AUTO_MANAGE:
                ensure_tts_stopped()
            _atomic_write_text(
                SETTINGS_PATH,
                TtsSettings(engine=engine).model_dump_json(indent=2),
                prefix=".tts-settings-",
            )
        return tts_settings_response()
