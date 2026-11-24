# OpenMOSS main parser: pinned runtime and installation record

> Current update: application input limit is now 5400 seconds (90 minutes), output token budget 65536, processor context 131072. Earlier 60-second restrictions below are historical validation scope, not the current cap. Long-audio memory, speed and quality are unverified on 8GB hardware.

This document supersedes the earlier draft below it titled “MOSS selected as the new main parser; installation/inference pending.” MOSS is the production parser. WhisperX/pyannote remain isolated legacy/experimental code and are not automatic fallbacks.

## Pinned model and runtime

- Model: [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), revision `10304d926740cf5a5cee05a268086e3265197910`.
- Model checkpoint SHA-256: `9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4`.
- Reviewed custom-code hashes are recorded in the local `custom_code_review.json`; only this immutable local revision is approved. Never set approval true for a floating or unreviewed remote revision.
- Python 3.12.10, PyTorch / torchaudio `2.8.0+cu128`, Transformers `5.6.0`, BF16 + SDPA.
- Developer runtime: `F:/Download/StageTTS/env_moss/python.exe`; model snapshot: `F:/Download/StageTTS/diarization_models/moss-transcribe-diarize`.
- Full measured environment plus retained wheels: 12,248,359,805 bytes (11.41 GiB); model snapshot is separate at 1,833,165,385 bytes. Keep ML environments and model weights outside Git.

## Install a separate environment

The exact top-level tested pins are listed in [`workers/requirements-moss.txt`](../workers/requirements-moss.txt). Install a supported CUDA 12.8 PyTorch/torchaudio pair first using the [official PyTorch selector](https://pytorch.org/get-started/locally/); then install the repository pins and verify actual GPU support.

```powershell
python -m venv F:\Download\StageTTS\env_moss
F:\Download\StageTTS\env_moss\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
F:\Download\StageTTS\env_moss\Scripts\python.exe -m pip install -r workers\requirements-moss.txt
F:\Download\StageTTS\env_moss\Scripts\python.exe -m pip check
F:\Download\StageTTS\env_moss\Scripts\python.exe -c "import torch, torchaudio, transformers; print(torch.__version__, torchaudio.__version__, transformers.__version__, torch.cuda.is_available())"
```

A Windows embedded Python distribution may lack `venv`; use a regular CPython installation for the isolated environment instead of modifying the embedded distribution or the backend interpreter. Do not install the legacy WhisperX/pyannote stack into this environment. Banquet must run here too (same torch ABI); its old torch 2.0 / CUDA 11.8 lock file must not be installed wholesale. The developer runtime passed an actual short Banquet CUDA compatibility smoke test, but other machines still need to verify Banquet's runtime imports and checkpoint.

Acquire the model through the provider's documented route to the exact revision, then verify its files against the manifest and pinned model hash. Review the fixed official custom code and generate the approval record locally. This repository does not contain model weights or custom-code approval records.

## Backend configuration

Set `backend/.env` to the paths on the actual machine:

```dotenv
ENV_PARSE_PYTHON=F:/Download/StageTTS/env_moss/python.exe
PARSE_BACKEND=moss
MOSS_MODEL_PATH=F:/Download/StageTTS/diarization_models/moss-transcribe-diarize
MOSS_REVISION=10304d926740cf5a5cee05a268086e3265197910
MOSS_TORCH_VERSION=2.8.0+cu128
MOSS_TRANSFORMERS_VERSION=5.6.0
MOSS_CUSTOM_CODE_APPROVED=1
MOSS_MAX_AUDIO_SECONDS=5400
MOSS_MAX_NEW_TOKENS=65536
BANQUET_ROOT=F:/Download/StageTTS/banquet/bandit-v2
BANQUET_CHECKPOINT=F:/Download/StageTTS/banquet/banquet-split.ckpt
```

`MOSS_CUSTOM_CODE_APPROVED=1` is only valid for the exact reviewed, hash-verified local snapshot above. The official model code uses Transformers `trust_remote_code`; inference is offline/local-only after acquisition. No HF token or gated diarization model is required for MOSS. The token is not for WhisperX/MOSS setup.

After changing parser configuration, restart the backend; it loads environment configuration at startup. The code fails clearly if MOSS is unavailable or fails. It never silently falls back to WhisperX/pyannote.

## Duration: model capability versus application budget

The upstream model card claims single-pass inputs up to 90 minutes. **StageTTS now permits a 5400-second input budget.** Earlier versions rejected inputs above 60 seconds (apart from a separately logged ≤0.05-second extracted-wave tail tolerance). Real GPU measurements cover Japanese/English at 10, 30 and 60 seconds on an RTX 5060 Laptop 8GB; at 60 seconds peak allocated CUDA memory was 1.942 GiB and reserved memory 1.965 GiB. These are small representative runs, not evidence that a 90-minute run fits, completes in reasonable time, or yields a complete transcript.

`MOSS_MAX_AUDIO_SECONDS` is an application fail-fast budget, not the model's intrinsic limit. Long-audio deployment still requires staged, measured validation, increasing output-token budget if required, memory/time measurements and transcript-completeness checks. Long-video segmentation and cross-window speaker-identity reconciliation are not implemented. Do not split windows and assume `S01` denotes the same speaker across windows.

## Output and known model boundaries

MOSS jointly generates transcript, anonymous in-window speaker labels, and segment timestamps. It does not provide word-level forced alignment, an independent overlap detector, calibrated confidence, or verified cross-window/global speaker identities. Preserve the raw output and provenance; do not interpolate character/word timestamps or silently treat anonymous model IDs as confirmed people. The current Banquet → MOSS → LLM short parse/evidence route and the 10/30/60-second MOSS runs were tested; those do not establish word-level timing or general transcription/diarization accuracy. No gold-labelled accuracy claim is made.

The app's `[media] source → Banquet → MOSS → LLM → script.json` implementation is in `backend/config.py`, `backend/services/task_runner.py`, `workers/parse_worker.py`, and `workers/moss_worker.py`. Keep GPU frameworks isolated from `env_backend`.

---

## Archived: MOSS migration engineering notes

The remainder of this document retains the original MOSS migration engineering diary, including measured download/runtime environment history and experimental notes. Its earlier “pending/not installed” and “production switch not completed” wording is historical and is superseded by the pinned current-state record above and by the live code/configuration.
