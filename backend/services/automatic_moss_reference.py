"""Build clip-specific reference transcripts for MOSS segment-only projects."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import urlparse

from backend import config
from backend.schemas import ProjectScript, ReferencePreview, Segment
from backend.services.text_normalization import normalize_tts_lang, prepare_tts_text


def _select_moss_reference_device() -> str:
    """Use CUDA when 8GB VRAM is free (stopping managed TTS if active); fall back to CPU otherwise."""
    if config.TTS_AUTO_MANAGE:
        try:
            from backend.services.task_runner import ensure_tts_stopped

            if ensure_tts_stopped():
                time.sleep(0.5)
        except Exception:
            pass
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
        try:
            with socket.create_connection((probe_host, port), timeout=0.35):
                return "cpu"
        except OSError:
            pass
    return "cuda:0"


def _moss_model_payload() -> dict:
    return {
        'id': 'OpenMOSS-Team/MOSS-Transcribe-Diarize',
        'local_path': str(config.MOSS_MODEL_PATH),
        'revision': config.MOSS_REVISION,
        'custom_code_approved': config.MOSS_CUSTOM_CODE_APPROVED,
        'torch_version': config.MOSS_TORCH_VERSION,
        'transformers_version': config.MOSS_TRANSFORMERS_VERSION,
    }


def _valid_single_speaker_report(report: dict) -> bool:
    rows = report.get('segments', [])
    return not (
        report.get('status') != 'ok'
        or report.get('review_reasons')
        or not rows
        or report.get('generation_limit_reached')
        or not report.get('ended_with_eos')
        or len({row.get('speaker') for row in rows}) != 1
        or any(row.get('speaker') in {None, 'UNKNOWN'} for row in rows)
    )


def transcribe_reference(audio: Path, directory: Path) -> dict:
    if not config.MOSS_MODEL_PATH or not config.MOSS_CUSTOM_CODE_APPROVED:
        raise ValueError("自动参考音需要已配置并批准的本地 MOSS 模型")
    mono = directory / 'reference_16k.wav'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(mono)],
                   check=True, capture_output=True, timeout=30)
    preferred = _select_moss_reference_device()
    devices = [preferred] if preferred == 'cpu' else [preferred, 'cpu']
    report: dict | None = None
    for idx, device in enumerate(devices):
        request = dict(
            version='1.0',
            model=_moss_model_payload(),
            source={
                'id': 'sha256:' + hashlib.sha256(mono.read_bytes()).hexdigest(),
                'kind': 'dialogue',
                'audio_path': str(mono.resolve()),
                'window_id': 'reference_clip',
            },
            device=device,
            max_audio_seconds=10,
            max_new_tokens=1024,
        )
        incoming, outgoing = directory / 'request.json', directory / 'result.json'
        incoming.write_text(json.dumps(request), encoding='utf-8')
        env = {**os.environ, 'PYTHONUTF8': '1'}
        result = subprocess.run(
            [config.ENV_PARSE_PYTHON, str(config.ROOT_DIR / 'workers' / 'moss_worker.py'),
             '--input', str(incoming), '--output', str(outgoing)],
            env=env, cwd=config.ROOT_DIR, capture_output=True, timeout=config.PARSE_TIMEOUT_SEC,
        )
        if result.returncode or not outgoing.is_file():
            if idx + 1 < len(devices):
                continue
            raise ValueError('自动参考音 MOSS 识别进程失败')
        report = json.loads(outgoing.read_text(encoding='utf-8'))
        if report.get('status') in {'error', 'not_run'}:
            if idx + 1 < len(devices):
                continue
            raise ValueError('自动参考音 MOSS 运行失败：' + str(report.get('error') or report.get('review_reasons')))
        break
    assert report is not None
    if not _valid_single_speaker_report(report):
        raise ValueError('自动参考音识别不完整或含多个说话人，请换一段参考音')
    return report


def extract_moss_segment_reference(script: ProjectScript, seg: Segment) -> ReferencePreview:
    """Slice 3–8s from one MOSS segment, retranscribe the isolated clip, and cache it."""
    lang = normalize_tts_lang(script.source_lang, default='')
    if not lang or seg.speaker == 'UNKNOWN' or not script.media.vocals_wav:
        raise ValueError('缺少源语种或人声音轨，无法生成 MOSS 参考音')
    source = config.resolve_media_path(script.media.vocals_wav)
    project = (config.WORKSPACE_DIR / script.project_id).resolve()
    if not source.is_file() or project not in source.parents:
        raise ValueError('找不到有效的项目人声文件')
    # Preserve the turn onset instead of cutting into its first word.
    start = seg.start
    duration = min(8.0, seg.end - start)
    key = (str(source), source.stat().st_mtime_ns, source.stat().st_size,
           seg.speaker, seg.id, seg.orig_text, start, duration, lang, config.MOSS_REVISION, 'moss_ref_v1')
    stem = 'moss_ref_' + hashlib.sha256(repr(key).encode()).hexdigest()[:24]
    folder = project / 'references'
    audio, metadata = folder / (stem + '.wav'), folder / (stem + '.json')
    if audio.is_file() and metadata.is_file():
        cached = json.loads(metadata.read_text(encoding='utf-8'))
        return ReferencePreview.model_validate(cached['reference'])
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        directory = Path(temporary)
        draft = directory / 'reference.wav'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(start),
                        '-i', str(source), '-t', str(duration), '-ac', '1', '-ar', '32000',
                        '-c:a', 'pcm_s16le', str(draft)], check=True, capture_output=True, timeout=30)
        with wave.open(str(draft), 'rb') as stream:
            actual_duration = stream.getnframes() / stream.getframerate()
        if not 3 <= actual_duration <= 10:
            raise ValueError('裁剪后的参考音长度不在 3–10 秒内')
        report = transcribe_reference(draft, directory)
        text = prepare_tts_text(' '.join(row['text'] for row in report['segments']), lang)
        if not text.strip():
            raise ValueError('参考音识别结果为空')
        reference = ReferencePreview(wav=f'/workspace/{script.project_id}/references/{audio.name}',
                                     text=text, lang=lang, segment_ids=[seg.id])
        record = {'reference': reference.model_dump(), 'source_start': start, 'source_end': start + duration,
                  'speaker_assignment': 'inherited_from_moss_source_turn_not_human_verified',
                  'transcript_source': 'moss_clip_retranscription', 'model_revision': config.MOSS_REVISION,
                  'report': report}
        draft.replace(audio)
        metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return reference


def prewarm_moss_segment_references(
    script: ProjectScript,
    segments: list[Segment],
) -> dict[int, ReferencePreview]:
    """Batch-transcribe uncached MOSS reference clips in a single GPU worker load."""
    lang = normalize_tts_lang(script.source_lang, default='')
    if not lang or not script.media.vocals_wav or not config.MOSS_MODEL_PATH or not config.MOSS_CUSTOM_CODE_APPROVED:
        return {}
    source = config.resolve_media_path(script.media.vocals_wav)
    project = (config.WORKSPACE_DIR / script.project_id).resolve()
    if not source.is_file() or project not in source.parents:
        return {}

    folder = project / 'references'
    folder.mkdir(parents=True, exist_ok=True)
    cached_refs: dict[int, ReferencePreview] = {}
    pending: list[tuple[Segment, float, float, Path, Path]] = []
    seen_ids: set[int] = set()

    for seg in segments:
        if seg.id in seen_ids or seg.speaker == 'UNKNOWN':
            continue
        seen_ids.add(seg.id)
        start = seg.start
        duration = min(8.0, seg.end - start)
        if not 3.0 <= duration <= 10.0:
            continue
        key = (str(source), source.stat().st_mtime_ns, source.stat().st_size,
               seg.speaker, seg.id, seg.orig_text, start, duration, lang, config.MOSS_REVISION, 'moss_ref_v1')
        stem = 'moss_ref_' + hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        audio, metadata = folder / (stem + '.wav'), folder / (stem + '.json')
        if audio.is_file() and metadata.is_file():
            try:
                cached = json.loads(metadata.read_text(encoding='utf-8'))
                cached_refs[seg.id] = ReferencePreview.model_validate(cached['reference'])
                continue
            except (OSError, ValueError):
                pass
        pending.append((seg, start, duration, audio, metadata))

    if not pending:
        return cached_refs

    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        tmp_dir = Path(temporary)
        staged: dict[str, tuple[Segment, float, float, Path, Path, Path]] = {}
        batch_sources: list[dict] = []
        for seg, start, duration, audio_out, meta_out in pending:
            draft_32k = tmp_dir / f'clip_{seg.id}.wav'
            mono_16k = tmp_dir / f'clip_{seg.id}_16k.wav'
            try:
                subprocess.run(
                    ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(start),
                     '-i', str(source), '-t', str(duration), '-ac', '1', '-ar', '32000',
                     '-c:a', 'pcm_s16le', str(draft_32k)],
                    check=True, capture_output=True, timeout=30,
                )
                with wave.open(str(draft_32k), 'rb') as stream:
                    actual_duration = stream.getnframes() / stream.getframerate()
                if not 3 <= actual_duration <= 10:
                    continue
                subprocess.run(
                    ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(draft_32k),
                     '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(mono_16k)],
                    check=True, capture_output=True, timeout=30,
                )
            except (OSError, subprocess.SubprocessError, wave.Error):
                continue
            window_id = f'seg_{seg.id}'
            staged[window_id] = (seg, start, duration, draft_32k, audio_out, meta_out)
            batch_sources.append({
                'id': 'sha256:' + hashlib.sha256(mono_16k.read_bytes()).hexdigest(),
                'kind': 'dialogue',
                'audio_path': str(mono_16k.resolve()),
                'window_id': window_id,
            })

        if not batch_sources:
            return cached_refs

        preferred = _select_moss_reference_device()
        devices = [preferred] if preferred == 'cpu' else [preferred, 'cpu']
        batch_report: dict | None = None
        for idx, device in enumerate(devices):
            incoming, outgoing = tmp_dir / 'batch_request.json', tmp_dir / 'batch_result.json'
            incoming.write_text(
                json.dumps({
                    'version': '1.0',
                    'model': _moss_model_payload(),
                    'sources': batch_sources,
                    'device': device,
                    'max_audio_seconds': 10,
                    'max_new_tokens': 1024,
                }),
                encoding='utf-8',
            )
            env = {**os.environ, 'PYTHONUTF8': '1'}
            proc = subprocess.run(
                [config.ENV_PARSE_PYTHON, str(config.ROOT_DIR / 'workers' / 'moss_worker.py'),
                 '--input', str(incoming), '--output', str(outgoing)],
                env=env, cwd=config.ROOT_DIR, capture_output=True, timeout=config.PARSE_TIMEOUT_SEC,
            )
            if proc.returncode == 0 and outgoing.is_file():
                parsed = json.loads(outgoing.read_text(encoding='utf-8'))
                if parsed.get('status') == 'ok':
                    batch_report = parsed
                    break
            if idx + 1 >= len(devices):
                return cached_refs

        if batch_report:
            for item in batch_report.get('results') or []:
                window_id = (item.get('source') or {}).get('window_id')
                if window_id not in staged or not _valid_single_speaker_report(item):
                    continue
                seg, start, duration, draft_32k, audio_out, meta_out = staged[window_id]
                text = prepare_tts_text(' '.join(row['text'] for row in item['segments']), lang)
                if not text.strip():
                    continue
                reference = ReferencePreview(
                    wav=f'/workspace/{script.project_id}/references/{audio_out.name}',
                    text=text,
                    lang=lang,
                    segment_ids=[seg.id],
                )
                record = {
                    'reference': reference.model_dump(),
                    'source_start': start,
                    'source_end': start + duration,
                    'speaker_assignment': 'inherited_from_moss_source_turn_not_human_verified',
                    'transcript_source': 'moss_clip_retranscription',
                    'model_revision': config.MOSS_REVISION,
                    'report': item,
                }
                draft_32k.replace(audio_out)
                meta_out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                cached_refs[seg.id] = reference

    return cached_refs


def automatic_moss_reference(
    script: ProjectScript,
    speaker: str,
    emotion: str | None = None,
) -> ReferencePreview | None:
    lang = normalize_tts_lang(script.source_lang, default='')
    if not lang or speaker == 'UNKNOWN' or not script.media.vocals_wav:
        return None
    source = config.resolve_media_path(script.media.vocals_wav)
    project = (config.WORKSPACE_DIR / script.project_id).resolve()
    if not source.is_file() or project not in source.parents:
        return None
    candidates = [seg for seg in script.segments if seg.speaker == speaker
                  and seg.timing_source == 'moss_joint_generated' and seg.alignment_granularity == 'segment'
                  and seg.end - seg.start >= 3.2 and seg.orig_text.strip()
                  and not any(other.id != seg.id and other.start < seg.end and other.end > seg.start
                              for other in script.segments)]
    candidates.sort(key=lambda seg: (seg.end - seg.start > 10, abs(seg.end - seg.start - 6)))
    if emotion:
        from backend.services.emotion_routing import select_emotion_reference_candidates

        prioritized = [
            seg
            for seg in select_emotion_reference_candidates(script, speaker, emotion)
            if seg.timing_source == 'moss_joint_generated' and seg.alignment_granularity == 'segment'
        ]
        seen_ids = {seg.id for seg in prioritized}
        candidates = [*prioritized, *(seg for seg in candidates if seg.id not in seen_ids)]
    failures = []
    for seg in candidates:
        try:
            return extract_moss_segment_reference(script, seg)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            failures.append(str(exc))
    if failures:
        raise ValueError('自动参考音生成失败：' + failures[-1])
    return None
