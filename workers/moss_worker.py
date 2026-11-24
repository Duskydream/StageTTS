"""Pinned offline MOSS 0.9B short-sample experiment worker, separate from default 3.1.

Input JSON version=1.0, model={local_path,revision,transformers_version,torch_version,
custom_code_approved:true}, source={id,kind:raw|dialogue,audio_path}, device=cuda:0,
max_audio_seconds<=5400, max_new_tokens<=65536. PCM16 mono 16k WAV is explicit.
Output retains raw tagged generation and segment timestamps; no invented word times,
probabilities or stable cross-window speaker identities. Run only in prepared env.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
import wave
from pathlib import Path

# Direct --input/--output launches must resolve sibling worker helpers too.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
REVISION = "10304d926740cf5a5cee05a268086e3265197910"
PROMPT = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
CUSTOM_FILES = ("configuration_moss_transcribe_diarize.py", "modeling_moss_transcribe_diarize.py", "processing_moss_transcribe_diarize.py")
SEGMENT_RE = re.compile(r"\[(-?\d+(?:\.\d+)?)\]\[(S\d{2,})\](.*?)\[(-?\d+(?:\.\d+)?)\]", re.DOTALL)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_generation(text: str, duration: float, *, source_id: str = "unscoped",
                     window_id: str = "whole_request") -> tuple[list[dict], list[str]]:
    """Keep valid simultaneous segments; bad/unknown markup remains explicit review."""
    segments, reasons = [], []
    occurrences = {}
    speaker_scope = f"{source_id}:{window_id}"
    cursor = 0
    for match in SEGMENT_RE.finditer(text):
        if text[cursor:match.start()].strip():
            reasons.append("unparsed_generation_content")
        cursor = match.end()
        start, end = float(match[1]), float(match[4])
        generated_start, generated_end = start, end
        timestamp_reasons = []
        if duration < end <= duration + 0.05:
            end = duration
            timestamp_reasons.append("timestamp_clamped_to_source_end")
        if -0.05 <= start < 0:
            start = 0.0
            timestamp_reasons.append("timestamp_clamped_to_source_start")
        body, speaker = match[3].strip(), match[2]
        if speaker == "S00":
            reasons.append("unknown_generated_speaker")
            speaker = "UNKNOWN"
        if not body or not math.isfinite(start + end) or start < 0 or end <= start or end > duration + 0.001:
            reasons.append("invalid_generated_segment")
            continue
        if re.search(r"\[(?:S[^\]]*|\d[^\]]*)\]", body):
            reasons.append("unexpected_markup_inside_segment")
            continue
        row_reasons = ["joint_generated_timestamps_not_forced_alignment", "anonymous_speaker_within_request_only", *timestamp_reasons]
        if re.search(r"\[[^\]]*\]", body):
            row_reasons.append("unclassified_bracketed_tag_in_text")
            reasons.append("unclassified_bracketed_tag_in_text")
        key = (start, end, speaker, body)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        if occurrence:
            row_reasons.append("duplicate_generated_segment")
            reasons.append("duplicate_generated_segment")
        if segments and start < segments[-1]["start"]:
            row_reasons.append("nonmonotonic_generation_order")
        if any(start < previous["end"] and end > previous["start"] and speaker != previous["speaker"] for previous in segments):
            row_reasons.append("overlapping_generated_segments_require_review")
        segment_id = "moss_" + hashlib.sha256(json.dumps(
            [source_id, window_id, start, end, speaker, body, occurrence], ensure_ascii=False,
        ).encode()).hexdigest()[:24]
        segments.append({"id": segment_id, "start": start, "end": end,
                         "generated_start": generated_start, "generated_end": generated_end,
                         "timestamp_policy": "clamp_source_bounds_tolerance_0.05s",
                         "speaker": speaker,
                         "speaker_scope": speaker_scope, "text": body,
                         "words": [], "score": None, "score_kind": None, "needs_review": True,
                         "review_reasons": row_reasons})
    if text[cursor:].strip():
        reasons.append("unparsed_generation_content")
    if not segments:
        reasons.append("no_valid_generated_segments")
    return segments, list(dict.fromkeys(reasons))


def validate_request(request: dict, *, model_only: bool = False) -> tuple[Path, Path, float, dict]:
    if not isinstance(request, dict) or request.get("version") != "1.0":
        raise ValueError("version must be 1.0")
    model, source = request.get("model", {}), request.get("source", {})
    if not isinstance(model, dict) or not isinstance(source, dict):
        raise ValueError("model/source must be JSON objects")
    if model.get("revision") != REVISION or model.get("id", MODEL_ID) != MODEL_ID:
        raise ValueError("only the fixed reviewed official model revision is supported")
    if model.get("custom_code_approved") is not True:
        raise ValueError("explicit pinned local custom-code approval required")
    root = Path(model.get("local_path", ""))
    audio = Path(source.get("audio_path", ""))
    if not root.is_absolute() or not root.is_dir() or (not model_only and (not audio.is_absolute() or not audio.is_file())):
        raise ValueError("model/audio require existing absolute local paths")
    if source.get("kind") not in {"raw", "dialogue"} or not source.get("id"):
        raise ValueError("source id and kind raw|dialogue required")
    manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("local manifest model/revision mismatch")
    review = json.loads((root / "custom_code_review.json").read_text(encoding="utf-8"))
    if review.get("revision") != REVISION or review.get("approved") is not True:
        raise ValueError("local pinned custom code has not been reviewed")
    listed = {row["path"]: row for row in manifest.get("files", [])}
    for name, row in listed.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or root.resolve() not in path.parents:
            raise ValueError("manifest file must remain inside model root")
        if not path.is_file() or not row.get("sha256") or sha256(path) != row["sha256"]:
            raise ValueError(f"local manifest/hash mismatch: {name}")
    required = [*CUSTOM_FILES, "config.json", "processor_config.json", "preprocessor_config.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json", "model-00000-of-00001.safetensors"]
    for name in required:
        path = root / name
        expected = listed.get(name, {}).get("sha256")
        if not path.is_file() or not expected or sha256(path) != expected:
            raise ValueError(f"local manifest/hash mismatch: {name}")
        if name in CUSTOM_FILES and review.get("files", {}).get(name) != expected:
            raise ValueError(f"reviewed custom code hash mismatch: {name}")
    for package, field in (("transformers", "transformers_version"), ("torch", "torch_version")):
        expected = model.get(field)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"explicit {field} required")
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package} version differs from request")
    if request.get("device", "cuda:0") not in {"cuda:0", "cpu"}:
        raise ValueError("device must be cuda:0 or explicitly cpu")
    seconds_limit = float(request.get("max_audio_seconds", 5400))
    if not 0 < seconds_limit <= 5400:
        raise ValueError("max_audio_seconds must be in (0, 5400] (90 minutes)")
    tokens = request.get("max_new_tokens", 1024)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 65536:
        raise ValueError("max_new_tokens must be 1..65536")
    if model_only:
        return root, audio, 0.0, manifest
    with wave.open(str(audio), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
            raise ValueError("explicit PCM16 mono 16k WAV required; preprocess separately preserving timebase")
        duration = stream.getnframes() / stream.getframerate()
    if not 0 < duration <= seconds_limit + 0.05:
        raise ValueError("audio duration exceeds explicit budget plus 0.05s media-tail tolerance")
    return root, audio, duration, manifest


def _load_model_and_processor(root: Path, device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(str(root), trust_remote_code=True, local_files_only=True,
                                                dtype=dtype, attn_implementation="sdpa").to(device).eval()
    processor = AutoProcessor.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    # The official processor emits 30-second encoder chunks. Encode these
    # sequentially while retaining one full-recording decoder/speaker scope.
    from transformers.modeling_outputs import BaseModelOutput
    encoder = model.model.whisper_encoder
    original_forward = encoder.forward
    def sequential_encoder(input_features, *args, **kwargs):
        if input_features.shape[0] <= 1:
            return original_forward(input_features, *args, **kwargs)
        outputs = [original_forward(chunk, *args, **kwargs).last_hidden_state
                   for chunk in input_features.split(1, dim=0)]
        return BaseModelOutput(last_hidden_state=torch.cat(outputs, dim=0))
    encoder.forward = sequential_encoder
    return model, processor, dtype


def _infer_loaded(model, processor, device: str, dtype, audio: Path, max_new_tokens: int) -> dict:
    import numpy as np
    import torch
    from workers.bounded_attention import bounded_sdpa
    with wave.open(str(audio), "rb") as stream:
        waveform = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    messages = [{"role": "user", "content": [{"type": "audio", "audio": str(audio)}, {"type": "text", "text": PROMPT}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=template, audio=[waveform], return_tensors="pt", max_length=131072).to(device)
    prompt_length = int(inputs["attention_mask"][0].sum().item())
    context_limit = int(model.config.text_config.max_position_embeddings)
    output_budget = min(max_new_tokens, context_limit - prompt_length)
    if output_budget < 1:
        raise ValueError("MOSS prompt/audio fills model context; shorten audio")
    with torch.inference_mode(), bounded_sdpa():
        generated = model.generate(**inputs, max_new_tokens=output_budget, do_sample=False,
                                   logits_to_keep=1)
    ids = generated[0][prompt_length:]
    text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    eos_ids = model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids] if eos_ids is not None else []
    ended_with_eos = bool(ids.numel() and int(ids[-1].item()) in eos_ids)
    return {"raw_text": text, "prompt_tokens": prompt_length, "generated_tokens": int(ids.numel()),
            "generation_limit_reached": int(ids.numel()) >= output_budget,
            "ended_with_eos": ended_with_eos,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None,
            "device": device, "dtype": str(dtype), "attention_backend": "sdpa"}


def _infer(root: Path, audio: Path, request: dict) -> dict:
    device = request.get("device", "cuda:0")
    model, processor, dtype = _load_model_and_processor(root, device)
    return _infer_loaded(model, processor, device, dtype, audio, request.get("max_new_tokens", 65536))


def peak_rss_bytes() -> int | None:
    """OS high-water working set for the isolated Windows worker, not CUDA memory."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if query(process, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except (OSError, AttributeError):
        pass
    return None


def run_request(request: dict) -> dict:
    if not isinstance(request, dict):
        return {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "not_run", "error": "request must be an object"}
    result = {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "not_run",
              "source": request.get("source", {}), "model": request.get("model", {}),
              "segments": [], "raw_text": None, "review_reasons": [], "error": None,
              "word_alignment_available": False, "cross_window_speaker_identity_available": False}
    try:
        root, audio, duration, _ = validate_request(request)
    except (OSError, ValueError, TypeError, KeyError, importlib.metadata.PackageNotFoundError, wave.Error) as exc:
        result["error"] = str(exc)
        return result
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            generation = _infer(root, audio, request)
        result.update(generation)
        result["segments"], result["review_reasons"] = parse_generation(
            generation["raw_text"], duration, source_id=request["source"]["id"],
            window_id=str(request["source"].get("window_id") or sha256(audio)),
        )
        if generation.get("generation_limit_reached"):
            result["review_reasons"].append("generation_token_limit_reached_possible_truncation")
        elif generation.get("ended_with_eos") is False:
            result["review_reasons"].append("generation_did_not_end_with_eos")
        result["status"] = "partial" if result["review_reasons"] else "ok"
        result["duration_seconds"] = duration
        result["audio_budget_seconds"] = float(request.get("max_audio_seconds", 5400))
        result["media_tail_tolerance_seconds"] = 0.05
        if duration > result["audio_budget_seconds"]:
            result["review_reasons"].append("source_wave_tail_within_explicit_0.05s_budget_tolerance")
            result["status"] = "partial"
        result["rtf"] = (time.perf_counter() - started) / duration
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    result["peak_rss_bytes"] = peak_rss_bytes()
    return result


def run_batch_request(request: dict) -> dict:
    """Load MOSS once and transcribe multiple short clips in one worker process."""
    sources = request.get("sources")
    if not isinstance(sources, list) or not sources:
        return {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "not_run", "error": "sources must be a non-empty list", "results": []}
    try:
        root, _, _, _ = validate_request({**request, "source": sources[0]}, model_only=True)
    except (OSError, ValueError, TypeError, KeyError, importlib.metadata.PackageNotFoundError, wave.Error) as exc:
        return {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "not_run", "error": str(exc), "results": []}

    device = request.get("device", "cuda:0")
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model, processor, dtype = _load_model_and_processor(root, device)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "error", "error": f"{type(exc).__name__}: {exc}", "results": []}

    results: list[dict] = []
    max_seconds = float(request.get("max_audio_seconds", 10))
    max_tokens = int(request.get("max_new_tokens", 1024))
    for src in sources:
        item = {"source": src, "status": "not_run", "segments": [], "review_reasons": [], "error": None}
        try:
            audio = Path(str(src.get("audio_path") or ""))
            if not audio.is_absolute() or not audio.is_file():
                raise ValueError("audio_path must be an existing absolute path")
            with wave.open(str(audio), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
                    raise ValueError("explicit PCM16 mono 16k WAV required")
                duration = stream.getnframes() / stream.getframerate()
            if not 0 < duration <= max_seconds + 0.05:
                raise ValueError("audio duration exceeds explicit budget")
            with contextlib.redirect_stdout(sys.stderr):
                generation = _infer_loaded(model, processor, device, dtype, audio, max_tokens)
            item.update(generation)
            item["segments"], item["review_reasons"] = parse_generation(
                generation["raw_text"], duration, source_id=str(src.get("id") or "batch"),
                window_id=str(src.get("window_id") or sha256(audio)),
            )
            if generation.get("generation_limit_reached"):
                item["review_reasons"].append("generation_token_limit_reached_possible_truncation")
            elif generation.get("ended_with_eos") is False:
                item["review_reasons"].append("generation_did_not_end_with_eos")
            item["status"] = "partial" if item["review_reasons"] else "ok"
            item["duration_seconds"] = duration
        except Exception as exc:
            item["status"] = "error"
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    return {
        "version": "1.0",
        "stage": "moss_transcribe_diarize",
        "status": "ok",
        "device": device,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read())
        if isinstance(request, dict) and "sources" in request:
            result = run_batch_request(request)
        else:
            result = run_request(request)
    except (OSError, ValueError) as exc:
        result = {"version": "1.0", "stage": "moss_transcribe_diarize", "status": "not_run", "error": str(exc)}
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
