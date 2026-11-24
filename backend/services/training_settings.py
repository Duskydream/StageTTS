"""Persist a future fine-tuning recipe, independently of active inference weights."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from backend import config
from backend.services.task_runner import ensure_tts_stopped, gpu_task_guard
from backend.services.tts_client import reset_active_weights_cache

SETTINGS_PATH = config.BACKEND_DIR / 'training_settings.local.json'
_lock = RLock()
BASE_MODELS = {
    'v2': ('gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
           'gsv-v2final-pretrained/s2G2333k.pth', 'gsv-v2final-pretrained/s2D2333k.pth'),
    'v2Pro': ('s1v3.ckpt', 'v2Pro/s2Gv2Pro.pth', 'v2Pro/s2Dv2Pro.pth'),
    'v2ProPlus': ('s1v3.ckpt', 'v2Pro/s2Gv2ProPlus.pth', 'v2Pro/s2Dv2ProPlus.pth'),
}


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    base_model: Literal['v2', 'v2Pro', 'v2ProPlus'] = 'v2Pro'
    gpt_epochs: int = Field(default=15, ge=1, le=100, strict=True)
    sovits_epochs: int = Field(default=10, ge=1, le=100, strict=True)


class InferenceModelUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    base_model: Literal['v2', 'v2Pro', 'v2ProPlus']


class TrainingModelOption(BaseModel):
    id: str
    available: bool
    missing_files: list[str]


class TrainingSettingsResponse(BaseModel):
    settings: TrainingSettings
    models: list[TrainingModelOption]
    inference_version: str | None
    training_started: bool = False


def read_training_settings() -> TrainingSettings:
    with _lock:
        if not SETTINGS_PATH.exists():
            return TrainingSettings()
        try:
            return TrainingSettings.model_validate_json(SETTINGS_PATH.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise ValueError('无法读取微调设置，请检查本地配置文件') from None


def _required_model_files(model: str, *, include_discriminator: bool = True) -> list[str]:
    t2s_rel, s2g_rel, s2d_rel = BASE_MODELS[model]
    required = [t2s_rel, s2g_rel]
    if include_discriminator:
        required.append(s2d_rel)
    if model != 'v2':
        required.append('sv/pretrained_eres2netv2w24s4ep4.ckpt')
    return required


def _missing_model_files(model: str, *, include_discriminator: bool = True) -> list[str]:
    root = config.GPT_SOVITS_ROOT
    return [
        file
        for file in _required_model_files(model, include_discriminator=include_discriminator)
        if root is None or not (root / 'GPT_SoVITS' / 'pretrained_models' / file).is_file()
    ]


def _atomic_write_text(path: Path, content: str, *, prefix: str = '.tmp-') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=prefix,
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def training_settings_response() -> TrainingSettingsResponse:
    recipe = read_training_settings()
    options = []
    for model in BASE_MODELS:
        missing = _missing_model_files(model, include_discriminator=True)
        options.append(TrainingModelOption(id=model, available=not missing, missing_files=missing))
    version = None
    if config.GPT_SOVITS_CONFIG:
        try:
            data = yaml.safe_load(Path(config.GPT_SOVITS_CONFIG).read_text(encoding='utf-8')) or {}
            version = (data.get('custom') or {}).get('version')
        except (OSError, ValueError, yaml.YAMLError):
            pass
    return TrainingSettingsResponse(settings=recipe, models=options, inference_version=version)


def save_training_settings(recipe: TrainingSettings) -> TrainingSettingsResponse:
    with _lock:
        _atomic_write_text(SETTINGS_PATH, recipe.model_dump_json(indent=2), prefix='.training-settings-')
        return training_settings_response()


def apply_inference_base_model(base_model: Literal['v2', 'v2Pro', 'v2ProPlus']) -> TrainingSettingsResponse:
    """将指定 GPT-SoVITS 底模设为当前默认零样本推理配置，并停止旧服务以便重载。"""
    if base_model not in BASE_MODELS:
        raise ValueError(f'不支持的推理底模: {base_model}')
    if not config.GPT_SOVITS_ROOT or not config.GPT_SOVITS_CONFIG:
        raise ValueError('未配置 GPT_SOVITS_ROOT 或 GPT_SOVITS_CONFIG，无法切换默认推理底模')
    config_path = Path(config.GPT_SOVITS_CONFIG)
    if not config_path.is_file():
        raise ValueError(f'找不到 GPT-SoVITS 推理配置文件: {config_path}')
    missing = _missing_model_files(base_model, include_discriminator=False)
    if missing:
        raise ValueError(f'底模 {base_model} 缺少推理所需权重文件：{", ".join(missing)}')

    t2s_rel, s2g_rel, _ = BASE_MODELS[base_model]
    with gpu_task_guard():
        with _lock:
            raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            data = dict(raw) if isinstance(raw, dict) else {}
            custom = dict(data.get('custom') or {})
            custom.setdefault('bert_base_path', 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')
            custom.setdefault('cnhuhbert_base_path', 'GPT_SoVITS/pretrained_models/chinese-hubert-base')
            custom.setdefault('device', 'cuda')
            custom.setdefault('is_half', True)
            custom['version'] = base_model
            custom['t2s_weights_path'] = f'GPT_SoVITS/pretrained_models/{t2s_rel}'
            custom['vits_weights_path'] = f'GPT_SoVITS/pretrained_models/{s2g_rel}'
            data['custom'] = custom
            _atomic_write_text(
                config_path,
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                prefix='.tts-config-',
            )
        if config.TTS_AUTO_MANAGE:
            ensure_tts_stopped()
        reset_active_weights_cache()
        return training_settings_response()
