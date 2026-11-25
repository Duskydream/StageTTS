"""StageTTS 主后端全局配置（纯 CPU 环境 env_backend）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from dotenv import dotenv_values
except ImportError:  # 允许精简环境不依赖 python-dotenv
    dotenv_values = None  # type: ignore[assignment]

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # env_parse 仅安装基础依赖时回退至 BaseModel + dotenv/os.environ
    BaseSettings = None  # type: ignore[assignment,misc]
    SettingsConfigDict = None  # type: ignore[assignment,misc]

# 仓库根目录：StageTTS/
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
BACKEND_DIR: Path = ROOT_DIR / "backend"
WORKERS_DIR: Path = ROOT_DIR / "workers"


def _resolve_dir_path(raw: str, default_path: Path) -> Path:
    cleaned = raw.strip()
    if not cleaned:
        return default_path
    p = Path(cleaned)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _resolve_optional_dir_path(raw: str) -> Optional[Path]:
    cleaned = raw.strip()
    if not cleaned:
        return None
    p = Path(cleaned)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _resolve_optional_path(raw: str) -> Optional[Path]:
    cleaned = raw.strip()
    if not cleaned:
        return None
    p = Path(cleaned).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT_DIR / p).resolve()


class _StageTTSSettingsFields(BaseModel):
    """后端共享强类型配置字段定义。"""

    workspace_dir_raw: str = Field(default="", alias="WORKSPACE_DIR")
    assets_dir_raw: str = Field(default="", alias="ASSETS_DIR")
    gpt_sovits_root_raw: str = Field(default="", alias="GPT_SOVITS_ROOT")
    gpt_sovits_python_raw: str = Field(default="", alias="GPT_SOVITS_PYTHON")
    gpt_sovits_config_raw: str = Field(default="", alias="GPT_SOVITS_CONFIG")
    tts_auto_manage: bool = Field(default=True, alias="TTS_AUTO_MANAGE")
    tts_startup_timeout_sec: float = Field(default=90.0, alias="TTS_STARTUP_TIMEOUT_SEC")

    env_parse_python: str = Field(default="python", alias="ENV_PARSE_PYTHON")
    parse_backend: str = Field(default="moss", alias="PARSE_BACKEND")
    moss_model_path_raw: str = Field(default="", alias="MOSS_MODEL_PATH")
    moss_revision: str = Field(default="", alias="MOSS_REVISION")
    moss_torch_version: str = Field(default="2.8.0+cu128", alias="MOSS_TORCH_VERSION")
    moss_transformers_version: str = Field(default="5.6.0", alias="MOSS_TRANSFORMERS_VERSION")
    moss_custom_code_approved: bool = Field(default=False, alias="MOSS_CUSTOM_CODE_APPROVED")
    moss_max_audio_seconds: float = Field(default=5400, gt=0, le=5400, alias="MOSS_MAX_AUDIO_SECONDS")
    moss_max_new_tokens: int = Field(default=65536, gt=0, le=65536, alias="MOSS_MAX_NEW_TOKENS")
    banquet_root_raw: str = Field(default="", alias="BANQUET_ROOT")
    banquet_checkpoint_raw: str = Field(default="", alias="BANQUET_CHECKPOINT")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    env_analysis_python: str = Field(default="", alias="ENV_ANALYSIS_PYTHON")
    emotion_model_path_raw: str = Field(default="", alias="EMOTION_MODEL_PATH")
    emotion_model_id: str = Field(default="", alias="EMOTION_MODEL_ID")
    emotion_revision: str = Field(default="", alias="EMOTION_REVISION")
    emotion_funasr_version: str = Field(default="", alias="EMOTION_FUNASR_VERSION")
    emotion_user_authorized: bool = Field(default=False, alias="EMOTION_USER_AUTHORIZED")
    emotion_device: str = Field(default="cuda:0", alias="EMOTION_DEVICE")
    analysis_enable_f0: bool = Field(default=True, alias="ANALYSIS_ENABLE_F0")
    tts_api_url: str = Field(default="http://127.0.0.1:9880", alias="TTS_API_URL")
    tts_default_gpt_weights: str = Field(default="", alias="TTS_DEFAULT_GPT_WEIGHTS")
    tts_default_sovits_weights: str = Field(default="", alias="TTS_DEFAULT_SOVITS_WEIGHTS")
    cosyvoice3_root_raw: str = Field(default="", alias="COSYVOICE3_ROOT")
    cosyvoice3_python_raw: str = Field(default="", alias="COSYVOICE3_PYTHON")
    cosyvoice3_model_dir_raw: str = Field(default="", alias="COSYVOICE3_MODEL_DIR")
    cosyvoice3_api_url: str = Field(default="http://127.0.0.1:9881", alias="COSYVOICE3_API_URL")

    hf_home: str = Field(default="", alias="HF_HOME")
    huggingface_hub_cache: str = Field(default="", alias="HUGGINGFACE_HUB_CACHE")
    torch_home: str = Field(default="", alias="TORCH_HOME")

    tts_timeout_sec: float = Field(default=120.0, alias="TTS_TIMEOUT_SEC")
    parse_timeout_sec: float = Field(default=1800.0, alias="PARSE_TIMEOUT_SEC")
    llm_timeout_sec: float = Field(default=60.0, alias="LLM_TIMEOUT_SEC")

    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    director_llm_api_key: str = Field(default="", alias="DIRECTOR_LLM_API_KEY")
    director_llm_base_url: str = Field(default="", alias="DIRECTOR_LLM_BASE_URL")
    director_llm_model: str = Field(default="", alias="DIRECTOR_LLM_MODEL")
    hf_token: str = Field(default="", alias="HF_TOKEN")


def _read_dotenv_fallback(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _load_dotenv_entries(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        return {}
    loaded = dotenv_values(env_file) if dotenv_values is not None else _read_dotenv_fallback(env_file)
    return {key: value for key, value in loaded.items() if value is not None}


def _export_dotenv_to_environ(env_file: Path) -> None:
    """将 backend/.env 中的扩展环境变量（如 NLTK_DATA、HF_HOME、TORCH_HOME）透传给子进程。"""
    for key, value in _load_dotenv_entries(env_file).items():
        os.environ.setdefault(key, value)


def _load_fallback_settings(cls: type[_StageTTSSettingsFields]) -> _StageTTSSettingsFields:
    merged: dict[str, Any] = {}
    for key, value in _load_dotenv_entries(BACKEND_DIR / ".env").items():
        merged[key.upper()] = value
    for key, value in os.environ.items():
        merged[key.upper()] = value
    return cls.model_validate(merged)


_export_dotenv_to_environ(BACKEND_DIR / ".env")


if BaseSettings is not None and SettingsConfigDict is not None:

    class StageTTSSettings(BaseSettings, _StageTTSSettingsFields):
        """基于 pydantic-settings 的强类型后端配置。"""

        model_config = SettingsConfigDict(
            env_file=BACKEND_DIR / ".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    settings: _StageTTSSettingsFields = StageTTSSettings()
else:

    class StageTTSSettings(_StageTTSSettingsFields):  # type: ignore[no-redef]
        """无 pydantic-settings 环境下的回退配置模型（如 env_parse）。"""

        model_config = ConfigDict(extra="ignore")

    settings = _load_fallback_settings(StageTTSSettings)

WORKSPACE_DIR: Path = _resolve_dir_path(settings.workspace_dir_raw, ROOT_DIR / "workspace")
ASSETS_DIR: Path = _resolve_dir_path(settings.assets_dir_raw, ROOT_DIR / "assets")
PARSE_WORKER_SCRIPT: Path = WORKERS_DIR / "parse_worker.py"
GPT_SOVITS_ROOT: Optional[Path] = _resolve_optional_dir_path(settings.gpt_sovits_root_raw)
GPT_SOVITS_PYTHON: str = (
    settings.gpt_sovits_python_raw.strip()
    or (str(GPT_SOVITS_ROOT / "runtime" / "python.exe") if GPT_SOVITS_ROOT else "")
)
GPT_SOVITS_CONFIG: str = (
    settings.gpt_sovits_config_raw.strip()
    or (str(BACKEND_DIR / "tts_v2.yaml") if (BACKEND_DIR / "tts_v2.yaml").is_file() else "")
)
TTS_AUTO_MANAGE: bool = settings.tts_auto_manage
TTS_STARTUP_TIMEOUT_SEC: float = settings.tts_startup_timeout_sec

# 外部算力服务与子进程 Python 路径
ENV_PARSE_PYTHON: str = settings.env_parse_python.strip() or "python"
PARSE_BACKEND: str = settings.parse_backend.strip().lower() or "moss"
if PARSE_BACKEND not in {"moss", "legacy"}:
    raise ValueError("PARSE_BACKEND must be moss or legacy; automatic fallback is disabled")
MOSS_MODEL_PATH: Optional[Path] = _resolve_optional_path(settings.moss_model_path_raw)
MOSS_REVISION: str = settings.moss_revision.strip()
MOSS_TORCH_VERSION: str = settings.moss_torch_version.strip()
MOSS_TRANSFORMERS_VERSION: str = settings.moss_transformers_version.strip()
MOSS_CUSTOM_CODE_APPROVED: bool = settings.moss_custom_code_approved
MOSS_MAX_AUDIO_SECONDS: float = settings.moss_max_audio_seconds
MOSS_MAX_NEW_TOKENS: int = settings.moss_max_new_tokens
BANQUET_ROOT: Optional[Path] = _resolve_optional_path(settings.banquet_root_raw)
BANQUET_CHECKPOINT: Optional[Path] = _resolve_optional_path(settings.banquet_checkpoint_raw)
# Optional isolated analysis runtime. Empty values never trigger downloads or installs.
ENV_ANALYSIS_PYTHON: str = settings.env_analysis_python.strip()
EMOTION_MODEL_PATH: Optional[Path] = _resolve_optional_path(settings.emotion_model_path_raw)
EMOTION_MODEL_ID: str = settings.emotion_model_id.strip()
EMOTION_REVISION: str = settings.emotion_revision.strip()
EMOTION_FUNASR_VERSION: str = settings.emotion_funasr_version.strip()
EMOTION_USER_AUTHORIZED: bool = settings.emotion_user_authorized
EMOTION_DEVICE: str = settings.emotion_device.strip() or "cuda:0"
ANALYSIS_ENABLE_F0: bool = settings.analysis_enable_f0
TRANSLATION_CHUNK_SIZE: int = 25
TTS_API_URL: str = settings.tts_api_url.rstrip("/")
TTS_DEFAULT_GPT_WEIGHTS: str = settings.tts_default_gpt_weights.strip()
TTS_DEFAULT_SOVITS_WEIGHTS: str = settings.tts_default_sovits_weights.strip()
COSYVOICE3_ROOT: Optional[Path] = _resolve_optional_dir_path(settings.cosyvoice3_root_raw)
COSYVOICE3_PYTHON: str = settings.cosyvoice3_python_raw.strip()
COSYVOICE3_MODEL_DIR: Optional[Path] = _resolve_optional_path(settings.cosyvoice3_model_dir_raw)
COSYVOICE3_API_URL: str = settings.cosyvoice3_api_url.strip().rstrip("/") or "http://127.0.0.1:9881"

# 可选模型缓存路径（用于重定向至非 C 盘目录）
HF_HOME: str = settings.hf_home.strip()
HUGGINGFACE_HUB_CACHE: str = settings.huggingface_hub_cache.strip()
TORCH_HOME: str = settings.torch_home.strip()

# 超时控制（秒）
TTS_TIMEOUT_SEC: float = settings.tts_timeout_sec
PARSE_TIMEOUT_SEC: float = settings.parse_timeout_sec
LLM_TIMEOUT_SEC: float = settings.llm_timeout_sec

# 大模型 API（兼容 OpenAI SDK）
LLM_API_KEY: str = settings.llm_api_key.strip()
LLM_BASE_URL: str = settings.llm_base_url.strip()
LLM_MODEL: str = settings.llm_model.strip()

# 导演模型 API（可选，留空时复用主模型）
DIRECTOR_LLM_API_KEY: str = settings.director_llm_api_key.strip()
DIRECTOR_LLM_BASE_URL: str = settings.director_llm_base_url.strip()
DIRECTOR_LLM_MODEL: str = settings.director_llm_model.strip()

# HuggingFace pyannote 鉴权 Token
HF_TOKEN: str = settings.hf_token.strip()

# 时长约束反思闭环核心阈值（见 ARCHITECTURE.md 铁律 #6）
DURATION_OVERFLOW_RATIO: float = 1.15
DUB_TAIL_ALLOWANCE_SECONDS: float = 0.45
DURATION_UNDERFLOW_RATIO: float = 0.74
MIN_UNDERFLOW_GAP_SECONDS: float = 1.15
MIN_DURATION_REFLECTION_SECONDS: float = 1.0
MAX_REFLECTION_ROUNDS: int = 2
MAX_GENTLE_SPEED: float = 1.15
MIN_GENTLE_SPEED: float = 0.88

# CORS 默认放行端口（Vite 开发服务器）
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _resolve_under_root(base_dir: Path, relative_part: str) -> Path:
    base_resolved = base_dir.resolve()
    target = (base_resolved / relative_part.lstrip("/\\")).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"非法路径越界访问: {relative_part}")
    return target


def resolve_media_path(url_or_path: str) -> Path:
    """将 /workspace/*、/assets/*、refs/* 或本地路径安全解析为绝对 Path。"""
    raw = (url_or_path or "").strip()
    if not raw:
        raise ValueError("媒体路径不能为空")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("/workspace/"):
        return _resolve_under_root(WORKSPACE_DIR, normalized[len("/workspace/"):])
    if normalized.startswith("workspace/"):
        return _resolve_under_root(WORKSPACE_DIR, normalized[len("workspace/"):])
    if normalized.startswith("/assets/"):
        return _resolve_under_root(ASSETS_DIR, normalized[len("/assets/"):])
    if normalized.startswith("assets/"):
        return _resolve_under_root(ASSETS_DIR, normalized[len("assets/"):])
    if normalized.startswith("refs/"):
        return _resolve_under_root(ASSETS_DIR / "refs", normalized[len("refs/"):])

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def resolve_weights_path(weights_path: str) -> str:
    """解析 GPT/SoVITS 权重路径：配置了 GPT_SOVITS_ROOT 且相对路径存在时转绝对路径，否则原样透传。"""
    raw = (weights_path or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if GPT_SOVITS_ROOT is not None:
        candidate = (GPT_SOVITS_ROOT / p).resolve()
        if candidate.exists():
            return str(candidate)
    return raw


def is_llm_configured() -> bool:
    """Resolve UI preferences as well as environment defaults."""
    from backend.services.llm_settings import get_llm_snapshot
    return get_llm_snapshot().configured()


def is_hf_token_configured() -> bool:
    """判断 HuggingFace Token 是否已有效配置。"""
    if not HF_TOKEN:
        return False
    return HF_TOKEN != "hf_your_token_here" and not HF_TOKEN.startswith("hf_your_")


def _banquet_env_value(name: str, fallback: Optional[Path]) -> str:
    if name in os.environ:
        return os.environ[name].strip()
    return str(fallback) if fallback is not None else ""


def resolve_banquet_paths() -> tuple[Path, Path]:
    """解析并校验 Banquet 源码目录与权重文件路径。"""
    root_raw = _banquet_env_value("BANQUET_ROOT", BANQUET_ROOT)
    if not root_raw:
        raise RuntimeError("Banquet 四轨需要在 backend/.env 配置 BANQUET_ROOT。")
    ckpt_raw = _banquet_env_value("BANQUET_CHECKPOINT", BANQUET_CHECKPOINT)
    if not ckpt_raw:
        raise RuntimeError("Banquet 四轨需要在 backend/.env 配置 BANQUET_CHECKPOINT。")
    source_root = (ROOT_DIR / Path(root_raw).expanduser()).resolve()
    checkpoint = (ROOT_DIR / Path(ckpt_raw).expanduser()).resolve()
    if (
        not (source_root / "src/models/bandit/bandit.py").is_file()
        or not (source_root / "src/models/bandit/film.py").is_file()
        or not checkpoint.is_file()
    ):
        raise RuntimeError("Banquet 源码或权重缺失，请检查 BANQUET_ROOT 和 BANQUET_CHECKPOINT。")
    return source_root, checkpoint


def is_banquet_configured() -> bool:
    """判断 Banquet 源码与权重文件是否就绪。"""
    try:
        resolve_banquet_paths()
        return True
    except RuntimeError:
        return False
