"""Standalone FastAPI inference server wrapping FunAudioLLM/CosyVoice3 for StageTTS."""
from __future__ import annotations

import argparse
import gc
import io
import os
import sys
import threading
import wave
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import uvicorn


class CosyVoiceTTSRequest(BaseModel):
    text: str = Field(min_length=1)
    text_lang: str = "zh"
    ref_audio_path: str = Field(min_length=1)
    prompt_text: str = ""
    prompt_lang: str = "zh"
    speed_factor: float = Field(default=1.0, gt=0.25, le=4.0)
    seed: int | None = None


def _base_lang(lang: str | None) -> str:
    cleaned = (lang or "").strip().lower().removeprefix("all_")
    return cleaned.split("-")[0].split("_")[0]


def format_cosyvoice3_prompt(prompt_text: str) -> str:
    """Ensure CosyVoice 3 prompt carries the required system prefix marker."""
    cleaned = (prompt_text or "").strip()
    if "<|endofprompt|>" in cleaned:
        return cleaned
    return f"You are a helpful assistant.<|endofprompt|>{cleaned}"


def _ensure_cosyvoice_import_paths(cosyvoice_root: Path) -> None:
    root_str = str(cosyvoice_root.resolve())
    matcha_str = str((cosyvoice_root / "third_party" / "Matcha-TTS").resolve())
    for entry in (matcha_str, root_str):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _patch_low_vram_model_loader() -> None:
    """Load fp32 checkpoints on CPU and cast llm/flow to fp16 before moving to CUDA."""
    import torch
    from cosyvoice.cli.model import CosyVoiceModel

    if getattr(CosyVoiceModel.load, "_stagetts_low_vram_patched", False):
        return

    def _low_vram_load(self: Any, llm_model: str, flow_model: str, hift_model: str) -> None:
        use_half = bool(getattr(self, "fp16", False)) and getattr(self.device, "type", "") == "cuda"

        def _load_component(module: Any, ckpt_path: str, half_precision: bool) -> None:
            # Keep mmap state_dict alive while copying tensors straight to CUDA fp16 (avoids Windows mmap unmap & CPU commit).
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
            gpu_state_dict: dict[str, Any] = {}
            for key, tensor in state_dict.items():
                if half_precision and tensor.is_floating_point():
                    gpu_state_dict[key] = tensor.to(device=self.device).half()
                else:
                    gpu_state_dict[key] = tensor.to(device=self.device)
            del state_dict
            module.load_state_dict(gpu_state_dict, strict=True, assign=True)
            del gpu_state_dict
            if half_precision:
                module.half()
            module.to(self.device).eval()
            gc.collect()

        _load_component(self.llm, llm_model, use_half)
        _load_component(self.flow, flow_model, use_half)

        raw_hift = torch.load(hift_model, map_location="cpu", weights_only=True, mmap=True)
        hift_sd = {k.replace("generator.", ""): v.to(device=self.device) for k, v in raw_hift.items()}
        del raw_hift
        self.hift.load_state_dict(hift_sd, strict=True, assign=True)
        del hift_sd
        self.hift.to(self.device).eval()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _low_vram_load._stagetts_low_vram_patched = True  # type: ignore[attr-defined]
    CosyVoiceModel.load = _low_vram_load  # type: ignore[method-assign]


def load_cosyvoice3_model(model_dir: Path, cosyvoice_root: Path) -> Any:
    """Load CosyVoice 3 within 8 GB VRAM & tight Windows commit limits while disabling wetext fetches."""
    _ensure_cosyvoice_import_paths(cosyvoice_root)
    import cosyvoice.cli.cosyvoice as cosyvoice_mod
    import onnxruntime
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_low_vram_model_loader()

    orig_load_hyperpyyaml = cosyvoice_mod.load_hyperpyyaml
    orig_inference_session = onnxruntime.InferenceSession
    use_cuda = torch.cuda.is_available()

    def _low_commit_load_hyperpyyaml(*args: Any, **kwargs: Any) -> Any:
        # Instantiate llm/flow directly in float16 on CUDA hosts so Windows commit never spikes on fp32 init.
        prev_dtype = torch.get_default_dtype()
        if use_cuda:
            torch.set_default_dtype(torch.float16)
        try:
            configs = orig_load_hyperpyyaml(*args, **kwargs)
        finally:
            torch.set_default_dtype(prev_dtype)
        if isinstance(configs, dict) and configs.get("hift") is not None:
            configs["hift"].float()
        return configs

    def _low_commit_inference_session(
        path_or_bytes: Any,
        sess_options: Any = None,
        providers: Any = None,
        **kwargs: Any,
    ) -> Any:
        opts = sess_options if sess_options is not None else onnxruntime.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = 2
        return orig_inference_session(
            path_or_bytes,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
            **kwargs,
        )

    sentinel = object()
    prev_wetext = sys.modules.get("wetext", sentinel)
    # Halt online wetext ModelScope TN fetches so frontend falls back immediately to pure-Python TN.
    sys.modules["wetext"] = None  # type: ignore[assignment]
    cosyvoice_mod.load_hyperpyyaml = _low_commit_load_hyperpyyaml
    onnxruntime.InferenceSession = _low_commit_inference_session  # type: ignore[misc]
    try:
        return AutoModel(model_dir=str(model_dir.resolve()), fp16=torch.cuda.is_available())
    finally:
        cosyvoice_mod.load_hyperpyyaml = orig_load_hyperpyyaml
        onnxruntime.InferenceSession = orig_inference_session
        if prev_wetext is sentinel:
            sys.modules.pop("wetext", None)
        else:
            sys.modules["wetext"] = prev_wetext


def create_app(model_dir: Path, cosyvoice_root: Path) -> FastAPI:
    _ensure_cosyvoice_import_paths(cosyvoice_root)
    import torch
    from cosyvoice.utils.common import set_all_random_seed

    cv_model: Any = load_cosyvoice3_model(model_dir, cosyvoice_root)
    infer_lock = threading.Lock()

    app = FastAPI(title="StageTTS CosyVoice3 Worker")

    @app.get("/")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "cosyvoice3",
            "sample_rate": int(cv_model.sample_rate),
        }

    @app.post("/tts")
    def synthesize(req: CosyVoiceTTSRequest) -> Response:
        ref_path = Path(req.ref_audio_path)
        if not ref_path.is_file():
            raise HTTPException(status_code=400, detail=f"找不到参考音频: {req.ref_audio_path}")

        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="待合成文本不能为空")
        same_language = bool(
            _base_lang(req.prompt_lang)
            and _base_lang(req.text_lang) == _base_lang(req.prompt_lang)
            and req.prompt_text.strip()
        )

        with infer_lock:
            if req.seed is not None:
                set_all_random_seed(int(req.seed))
            if same_language:
                prompt_text = format_cosyvoice3_prompt(req.prompt_text)
                stream_iter = cv_model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=prompt_text,
                    prompt_wav=str(ref_path),
                    stream=False,
                    speed=float(req.speed_factor),
                )
            else:
                # Cross-lingual mode avoids leaking source-language prompt_text (e.g. Japanese Kanji) into target speech.
                stream_iter = cv_model.inference_cross_lingual(
                    tts_text=f"You are a helpful assistant.<|endofprompt|>{text}",
                    prompt_wav=str(ref_path),
                    stream=False,
                    speed=float(req.speed_factor),
                )
            chunks = [out["tts_speech"].detach().cpu() for out in stream_iter]

        if not chunks:
            raise HTTPException(status_code=500, detail="CosyVoice3 未返回任何音频数据")

        audio = torch.cat(chunks, dim=1).squeeze(0).float().clamp(-1.0, 1.0)
        pcm16 = (audio * 32767.0).round().to(torch.int16).numpy()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(cv_model.sample_rate))
            wf.writeframes(pcm16.tobytes())
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StageTTS CosyVoice3 HTTP server")
    parser.add_argument("-a", "--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9881)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cosyvoice-root", required=True)
    args = parser.parse_args()

    app = create_app(Path(args.model_dir), Path(args.cosyvoice_root))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
