"""Batch offline acoustic emotion hint worker for ENV_ANALYSIS_PYTHON."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import wave
from typing import Any

_CANONICAL_LABELS = {
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
}


def normalize_emotion_label(raw_label: str) -> str:
    """Normalize bilingual emotion2vec+ labels such as '开心/happy' -> 'happy'."""
    cleaned = str(raw_label or "").strip().lower()
    if "/" in cleaned:
        suffix = cleaned.split("/", 1)[1].strip()
        if suffix in _CANONICAL_LABELS:
            return suffix
    if cleaned in _CANONICAL_LABELS:
        return cleaned
    return cleaned or "unknown"


def _load_span_samples(wav_stream: wave.Wave_read, nframes: int, start: float, end: float):
    import numpy as np

    first = max(0, round(start * 16000))
    last = min(nframes, round(end * 16000))
    if last <= first:
        return None
    duration = (last - first) / 16000.0
    if not 0.4 <= duration <= 30.0:
        return None
    wav_stream.setpos(first)
    samples = np.frombuffer(wav_stream.readframes(last - first), dtype="<i2")
    if len(samples) != last - first or not samples.any():
        return None
    return samples.astype("float32") / 32768.0


def _create_model(model_path: str, device: str) -> tuple[Any, str]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from funasr import AutoModel

    actual_device = device or "cpu"
    try:
        model = AutoModel(
            model=model_path,
            hub="ms",
            device=actual_device,
            disable_update=True,
            check_latest=False,
            trust_remote_code=False,
            batch_size=1,
            ncpu=1,
        )
        return model, actual_device
    except Exception:
        if actual_device == "cpu":
            raise
        model = AutoModel(
            model=model_path,
            hub="ms",
            device="cpu",
            disable_update=True,
            check_latest=False,
            trust_remote_code=False,
            batch_size=1,
            ncpu=1,
        )
        return model, "cpu"


def _parse_output(raw: Any) -> tuple[str, float, dict[str, float]]:
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise ValueError("invalid funasr output structure")
    labels = raw[0].get("labels") or []
    scores = raw[0].get("scores") or []
    if len(labels) != len(scores) or not labels:
        raise ValueError("empty or mismatched emotion labels/scores")

    mapped: dict[str, float] = {}
    for label, score in zip(labels, scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            continue
        key = normalize_emotion_label(str(label))
        val = float(score)
        mapped[key] = max(mapped.get(key, 0.0), val)
    if not mapped:
        raise ValueError("no finite emotion scores")
    top_label = max(mapped, key=mapped.__getitem__)
    return top_label, mapped[top_label], mapped


def run_batch(request: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(request.get("audio_16k_wav") or ""))
    model_path = Path(str(request.get("model_path") or ""))
    requested_device = str(request.get("device") or "cpu")
    spans = request.get("spans") or []

    if not audio_path.is_file():
        raise FileNotFoundError(f"audio_16k_wav not found: {audio_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")

    with redirect_stdout(sys.stderr):
        model, actual_device = _create_model(str(model_path), requested_device)
        results: list[dict[str, Any]] = []
        with wave.open(str(audio_path), "rb") as wav_stream:
            if (
                wav_stream.getframerate(),
                wav_stream.getnchannels(),
                wav_stream.getsampwidth(),
                wav_stream.getcomptype(),
            ) != (16000, 1, 2, "NONE"):
                raise ValueError("audio_16k_wav must be 16000Hz mono PCM16 WAV")
            nframes = wav_stream.getnframes()

            for span in spans:
                if not isinstance(span, dict) or "id" not in span:
                    continue
                try:
                    start = float(span["start"])
                    end = float(span["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    continue
                samples = _load_span_samples(wav_stream, nframes, start, end)
                if samples is None:
                    continue
                try:
                    raw = model.generate(
                        input=samples,
                        fs=16000,
                        granularity="utterance",
                        extract_embedding=False,
                        batch_size=1,
                        disable_pbar=True,
                    )
                except Exception:
                    if actual_device == "cpu":
                        continue
                    model, actual_device = _create_model(str(model_path), "cpu")
                    raw = model.generate(
                        input=samples,
                        fs=16000,
                        granularity="utterance",
                        extract_embedding=False,
                        batch_size=1,
                        disable_pbar=True,
                    )
                try:
                    top_label, top_score, scores_map = _parse_output(raw)
                except ValueError:
                    continue
                results.append({
                    "id": span["id"],
                    "top_label": top_label,
                    "top_score": round(top_score, 4),
                    "scores": {k: round(v, 4) for k, v in scores_map.items()},
                })

    return {"status": "ok", "device_used": actual_device, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        payload = run_batch(request)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc), "results": []}
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
