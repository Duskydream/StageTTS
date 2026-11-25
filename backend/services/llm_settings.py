"""Private, atomic application LLM preferences; never served as static assets."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator
from backend import config

SETTINGS_PATH = config.ROOT_DIR / "backend" / "llm_settings.local.json"
_lock = RLock()
_PRIMARY_KEYS = ("base_url", "model", "api_key")
_DIRECTOR_KEYS = ("director_base_url", "director_model", "director_api_key")


def _validate_http_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("接口地址必须是 HTTP(S) URL，且不能包含用户名、密码、查询参数或片段")
    return value


def _is_valid_key(key: str, base_url: str) -> bool:
    if not key or key.startswith("your_"):
        return False
    if key in {"sk-xxx", "changeme", "local-gateway-no-key"}:
        return urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"}
    return True


def _validate_secret_key(secret: SecretStr | None, *, clear: bool, clear_error: str) -> str:
    key = secret.get_secret_value().strip() if secret is not None else ""
    if clear and key:
        raise ValueError(clear_error)
    if len(key) > 8192 or any(ord(char) < 32 for char in key):
        raise ValueError("API Key 格式无效")
    return key


class LLMSettingsUpdate(BaseModel):
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=256)
    api_key: SecretStr | None = None
    clear_api_key: bool = False
    director_base_url: str | None = Field(default=None, max_length=2048)
    director_model: str | None = Field(default=None, max_length=256)
    director_api_key: SecretStr | None = None
    clear_director_api_key: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_http_url(value)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("请填写有效的模型名称")
        return value

    @field_validator("director_base_url")
    @classmethod
    def validate_director_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return _validate_http_url(cleaned) if cleaned else ""

    @field_validator("director_model")
    @classmethod
    def validate_director_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if any(ord(char) < 32 for char in cleaned):
            raise ValueError("导演模型名称格式无效")
        return cleaned


class LLMSettingsResponse(BaseModel):
    base_url: str
    model: str
    api_key_configured: bool
    configured: bool
    director_base_url: str = ""
    director_model: str = ""
    director_api_key_configured: bool = False
    director_configured: bool = False


@dataclass(frozen=True)
class LLMSnapshot:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    director_base_url: str = ""
    director_model: str = ""
    director_api_key: str = field(default="", repr=False)

    def configured(self) -> bool:
        return _is_valid_key(self.api_key, self.base_url)

    def has_custom_director(self) -> bool:
        return bool(self.director_model.strip())

    def effective_director(self) -> tuple[str, str, str]:
        if not self.has_custom_director():
            return (self.base_url, self.model, self.api_key)
        d_url = self.director_base_url.strip() or self.base_url
        d_model = self.director_model.strip()
        d_key = self.director_api_key.strip() or self.api_key
        return (d_url, d_model, d_key)

    def director_configured(self) -> bool:
        if not self.has_custom_director():
            return False
        d_url, _, d_key = self.effective_director()
        return _is_valid_key(d_key, d_url)

    def public(self) -> LLMSettingsResponse:
        return LLMSettingsResponse(
            base_url=self.base_url,
            model=self.model,
            api_key_configured=bool(self.api_key),
            configured=self.configured(),
            director_base_url=self.director_base_url,
            director_model=self.director_model,
            director_api_key_configured=bool(self.director_api_key),
            director_configured=self.director_configured(),
        )


def get_llm_snapshot() -> LLMSnapshot:
    with _lock:
        defaults = dict(
            base_url=config.LLM_BASE_URL,
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            director_base_url=config.DIRECTOR_LLM_BASE_URL,
            director_model=config.DIRECTOR_LLM_MODEL,
            director_api_key=config.DIRECTOR_LLM_API_KEY,
        )
        # Parse subprocess receives a frozen snapshot from its parent process.
        if os.environ.get("STAGETTS_LLM_SNAPSHOT") != "1" and SETTINGS_PATH.exists():
            try:
                saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if not isinstance(saved, dict) or not all(isinstance(saved.get(key), str) for key in _PRIMARY_KEYS):
                    raise ValueError()
                for key in _PRIMARY_KEYS:
                    defaults[key] = saved[key]
                for key in _DIRECTOR_KEYS:
                    if key in saved:
                        if not isinstance(saved[key], str):
                            raise ValueError()
                        defaults[key] = saved[key]
            except (OSError, ValueError):
                raise ValueError("无法读取本地 LLM 设置，请检查设置文件格式") from None
        return LLMSnapshot(**defaults)


def save_llm_settings(request: LLMSettingsUpdate) -> LLMSettingsResponse:
    with _lock:
        previous = get_llm_snapshot()
        key = _validate_secret_key(
            request.api_key,
            clear=request.clear_api_key,
            clear_error="清除 API Key 时不能同时填写新 Key",
        )
        director_key = _validate_secret_key(
            request.director_api_key,
            clear=request.clear_director_api_key,
            clear_error="清除导演模型 API Key 时不能同时填写新 Key",
        )
        current = LLMSnapshot(
            base_url=request.base_url,
            model=request.model,
            api_key="" if request.clear_api_key else key or previous.api_key,
            director_base_url=previous.director_base_url if request.director_base_url is None else request.director_base_url,
            director_model=previous.director_model if request.director_model is None else request.director_model,
            director_api_key="" if request.clear_director_api_key else director_key or previous.director_api_key,
        )
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
                                             prefix=".llm-settings-", suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump(
                    dict(
                        base_url=current.base_url,
                        model=current.model,
                        api_key=current.api_key,
                        director_base_url=current.director_base_url,
                        director_model=current.director_model,
                        director_api_key=current.director_api_key,
                    ),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, SETTINGS_PATH)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return current.public()
