"""音视频后期合成服务：原声保留、响度对齐、人声空间融合、四轨独立混音、自动闪避与 FFmpeg 封装。"""

from __future__ import annotations

import functools
import math
import os
import re
import statistics
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from backend import config
from backend.schemas import ProjectScript, StemVolumes, SubtitleBurnMode, SubtitleLayout
from backend.services.subtitles import write_ass_file
from backend.services.task_runner import probe_media_duration
from workers.prosody_worker import read_wav

_FADE_IN_OUT_MS = 6
_KEEP_ORIG_FADE_MS = 10
_WINDOWS_RESERVED_STEM_RE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)


def check_ffmpeg_available() -> bool:
    """检查系统 PATH 中是否可用 ffmpeg。"""
    return shutil.which("ffmpeg") is not None


def apply_gentle_tempo(wav_path: Path, speed_factor: float, tmp_dir: Path) -> Path:
    """调用 FFmpeg atempo 滤镜对单句做不变调温和变速（限 0.5 ~ 2.0）。"""
    factor = max(0.5, min(2.0, float(speed_factor)))
    if round(factor, 3) == 1.0:
        return wav_path

    out_path = tmp_dir / f"{wav_path.stem}_spd.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-filter:a",
        f"atempo={factor:.3f}",
        "-vn",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg atempo 处理失败: {proc.stderr[-400:]}")
    return out_path


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """转义单引号包裹的 FFmpeg filtergraph 文件路径参数（如 ass=filename='...'）。"""
    text = path.resolve().as_posix()
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\\\\\''")


def _build_ass_subtitle_filter(ass_path: Path) -> str:
    """Build FFmpeg ass= filter string with Windows drive colons and quotes escaped."""
    escaped = (
        str(ass_path.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )
    return f"ass='{escaped}'"


def _find_watermark_font() -> Optional[str]:
    """定位系统中文字体文件并转义为 FFmpeg drawtext fontfile 参数格式。"""
    if os.name == "nt":
        win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [
            win_fonts / "msyh.ttc",
            win_fonts / "simhei.ttf",
            win_fonts / "msyhbd.ttc",
            win_fonts / "arial.ttf",
        ]
    elif sys.platform == "darwin":
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
            Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    for font_path in candidates:
        if font_path.is_file():
            return _escape_ffmpeg_filter_path(font_path)
    return None


def _video_encoder_candidates() -> Tuple[List[str], ...]:
    """按平台返回优先硬件加速与 libx264 兜底视频编码参数。"""
    fallback = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if sys.platform == "darwin":
        return (["-c:v", "h264_videotoolbox", "-q:v", "65"], fallback)
    return (["-c:v", "h264_nvenc", "-preset", "p4"], fallback)


def _measure_source_dialogue_profile(script: ProjectScript) -> tuple[float, float]:
    """Estimate (target_dbfs, room_wetness) from original dialogue stem when available."""
    default_dbfs, default_wet = -18.0, 0.10
    if not script.media.vocals_wav:
        return default_dbfs, default_wet
    try:
        vocals_path = config.resolve_media_path(script.media.vocals_wav)
    except ValueError:
        return default_dbfs, default_wet
    if not vocals_path.is_file():
        return default_dbfs, default_wet

    try:
        with wave.open(str(vocals_path), "rb") as wf:
            wav_dur = wf.getnframes() / float(max(1, wf.getframerate()))
    except (OSError, wave.Error, EOFError):
        return default_dbfs, default_wet

    active_dbs: list[float] = []
    decay_ratios: list[float] = []
    sample_segs = [s for s in script.segments if not s.keep_original and (s.end - s.start) >= 0.6][:15]
    for seg in sample_segs:
        end_t = min(seg.end, wav_dur)
        if end_t - seg.start < 0.5:
            continue
        try:
            samples, rate, _ = read_wav(vocals_path, start=seg.start, end=end_t)
        except (OSError, ValueError, wave.Error, EOFError):
            continue
        hop = max(1, round(rate * 0.02))
        frame_rms = [
            math.sqrt(sum(v * v for v in samples[i:i + hop]) / max(1, len(samples[i:i + hop])))
            for i in range(0, len(samples), hop)
        ]
        if not frame_rms:
            continue
        peak_rms = max(frame_rms)
        if peak_rms <= 1e-6:
            continue
        gate = max(10 ** (-45.0 / 20.0), peak_rms * 0.12)
        active = [r for r in frame_rms if r >= gate]
        silent = [r for r in frame_rms if r < gate]
        if not active:
            continue
        act_rms = math.sqrt(sum(r * r for r in active) / len(active))
        if act_rms > 0:
            active_dbs.append(20.0 * math.log10(act_rms))
            if silent:
                amb_rms = math.sqrt(sum(r * r for r in silent) / len(silent))
                decay_ratios.append(min(1.0, amb_rms / act_rms))

    if not active_dbs:
        return default_dbfs, default_wet
    target_dbfs = round(max(-21.0, min(-15.0, statistics.median(active_dbs))), 2)
    room_wetness = default_wet
    if decay_ratios:
        med_decay = statistics.median(decay_ratios)
        room_wetness = round(max(0.06, min(0.16, 0.06 + med_decay * 0.7)), 3)
    return target_dbfs, room_wetness


def _normalize_clip_loudness(
    clip: AudioSegment,
    target_dbfs: float = -18.0,
    max_gain_db: float = 10.0,
    min_gain_db: float = -10.0,
    peak_ceiling_dbfs: float = -1.0,
) -> AudioSegment:
    """Align loudness on active speech chunks without distorting mostly-silent sentinel audio."""
    if len(clip) == 0 or clip.rms == 0:
        return clip
    threshold = clip.max_possible_amplitude * (10.0 ** (-45.0 / 20.0))
    chunk_ms = 20
    active_sq: list[float] = []
    total_chunks = 0
    for pos in range(0, len(clip), chunk_ms):
        chunk = clip[pos:pos + chunk_ms]
        total_chunks += 1
        if chunk.rms >= threshold:
            active_sq.append(float(chunk.rms) ** 2)
    if total_chunks == 0 or len(active_sq) / total_chunks < 0.15:
        return clip
    active_rms = math.sqrt(sum(active_sq) / len(active_sq))
    if active_rms <= 0:
        return clip
    current_dbfs = 20.0 * math.log10(active_rms / clip.max_possible_amplitude)
    if current_dbfs < -55.0:
        return clip
    desired_gain = max(min_gain_db, min(max_gain_db, target_dbfs - current_dbfs))
    headroom = peak_ceiling_dbfs - clip.max_dBFS
    applied_gain = min(desired_gain, headroom)
    out = clip.apply_gain(applied_gain) if abs(applied_gain) >= 0.25 else clip
    if len(out) >= 80:
        out = out.fade_in(_FADE_IN_OUT_MS).fade_out(_FADE_IN_OUT_MS)
    return out


def _probe_wav_sample_rate(wav_path: Path, default: int = 44100) -> int:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return rate
    except (OSError, wave.Error, EOFError):
        pass
    return default


def _build_vocal_track(
    script: ProjectScript,
    project_dir: Path,
    *,
    timeline_duration: float | None = None,
    auto_fit_overflow: bool = False,
) -> Path:
    """Mix complete clips with active loudness normalization and collision guards."""
    timeline_duration = script.media.duration if timeline_duration is None else timeline_duration
    if timeline_duration <= 0:
        raise RuntimeError("导出待审核：项目时长无效，无法安全安排配音。")
    total_ms = round(timeline_duration * 1000)
    for seg in script.segments:
        if seg.end > timeline_duration + 0.000001:
            raise RuntimeError(f"片段 #{seg.id} 导出待审核：原始时间范围超过媒体结尾。")

    target_dbfs, _ = _measure_source_dialogue_profile(script)

    vocals_audio: Optional[AudioSegment] = None
    if script.media.vocals_wav:
        try:
            vocals_path = config.resolve_media_path(script.media.vocals_wav)
            if vocals_path.is_file():
                with vocals_path.open("rb") as source:
                    vocals_audio = AudioSegment.from_file(source)
        except ValueError:
            vocals_audio = None

    frame_rate = vocals_audio.frame_rate if vocals_audio is not None else 44100
    vocal_track = AudioSegment.silent(duration=total_ms, frame_rate=frame_rate).set_channels(2)

    with tempfile.TemporaryDirectory(prefix="stagetts_mix_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for seg in sorted(script.segments, key=lambda item: (item.start, item.id)):
            pos_ms = max(0, round(seg.start * 1000))
            if seg.keep_original:
                if vocals_audio is None:
                    continue
                end_ms = min(len(vocals_audio), max(pos_ms, int(seg.end * 1000)))
                if end_ms <= pos_ms:
                    continue
                clip = vocals_audio[pos_ms:end_ms]
                fade_ms = min(_KEEP_ORIG_FADE_MS, len(clip) // 4)
                if fade_ms >= 2:
                    clip = clip.fade_in(fade_ms).fade_out(fade_ms)
                clip = _normalize_clip_loudness(clip, target_dbfs=target_dbfs)
                vocal_track = vocal_track.overlay(clip, position=pos_ms)
                continue

            if not seg.tts_wav or seg.status != "ready":
                continue
            try:
                tts_path = config.resolve_media_path(seg.tts_wav)
            except ValueError:
                continue
            if not tts_path.is_file():
                continue

            with tts_path.open("rb") as source:
                raw_clip = AudioSegment.from_wav(source)
            clip = raw_clip
            if seg.applied_speed_factor is None:
                # 旧项目的 WAV 仍沿用导出时变速。
                speed = seg.speed_factor
                raw_sec = len(raw_clip) / 1000.0
                slot_sec = max(0.05, seg.end - seg.start)
                if abs(speed - 1.0) < 0.01 and slot_sec * 1.02 < raw_sec <= slot_sec * config.MAX_GENTLE_SPEED:
                    speed = min(config.MAX_GENTLE_SPEED, round(raw_sec / slot_sec, 3))
                if abs(speed - 1.0) >= 0.02:
                    adjusted_path = apply_gentle_tempo(tts_path, speed, tmp_dir)
                    with adjusted_path.open("rb") as source:
                        clip = AudioSegment.from_wav(source)

            clip_dur = clip.frame_count() / clip.frame_rate
            clip_end = seg.start + clip_dur
            if auto_fit_overflow:
                next_starts = [
                    other.start
                    for other in script.segments
                    if other.id != seg.id and other.start > seg.start
                ]
                next_start = min(next_starts) if next_starts else None
                safe_end = timeline_duration
                if next_start is not None:
                    guard_end = next_start - 0.03 if next_start - 0.03 > seg.start + 0.1 else next_start
                    safe_end = min(safe_end, guard_end)
                collides_next = next_start is not None and clip_end > seg.end and clip_end > next_start
                needs_fit = clip_end > timeline_duration + 0.000001 or collides_next
                if needs_fit and clip_end > safe_end and safe_end > seg.start + 0.1:
                    avail_sec = safe_end - seg.start
                    needed_speed = (clip_dur / avail_sec) * 1.008
                    if needed_speed <= 1.35:
                        fit_src = tmp_dir / f"fit_{seg.id}.wav"
                        with fit_src.open("wb") as fp:
                            clip.export(fp, format="wav")
                        fitted_path = apply_gentle_tempo(fit_src, needed_speed, tmp_dir)
                        with fitted_path.open("rb") as source:
                            clip = AudioSegment.from_wav(source)
                        clip_dur = clip.frame_count() / clip.frame_rate
                        clip_end = seg.start + clip_dur
                        if safe_end < clip_end <= safe_end + 0.05:
                            limit_ms = max(50, int(avail_sec * 1000))
                            clip = clip[:limit_ms].fade_out(min(15, limit_ms))
                            clip_end = seg.start + clip.frame_count() / clip.frame_rate

            if clip_end > timeline_duration + 0.000001:
                raise RuntimeError(
                    f"片段 #{seg.id} 导出待审核：完整配音超过媒体结尾 "
                    f"{clip_end - timeline_duration:.3f} 秒，不能截尾。"
                )
            if clip_end > seg.end:
                collision = next((
                    other for other in script.segments
                    if other.id != seg.id and other.start < clip_end and other.end > seg.end
                ), None)
                if collision is not None:
                    raise RuntimeError(
                        f"片段 #{seg.id} 导出待审核：完整配音侵入片段 #{collision.id}，"
                        "请调整配音或审核排程后导出。"
                    )
            clip = _normalize_clip_loudness(clip, target_dbfs=target_dbfs)
            placed_frames = round(pos_ms * vocal_track.frame_rate / 1000)
            clip_frames = clip.frame_count() * vocal_track.frame_rate / clip.frame_rate
            if placed_frames + clip_frames > vocal_track.frame_count():
                raise RuntimeError(f"片段 #{seg.id} 导出待审核：配音超出混音画布，不能截尾。")
            vocal_track = vocal_track.overlay(clip, position=pos_ms)

    dubbed_path = project_dir / "dubbed_vocals.wav"
    with dubbed_path.open("wb") as output:
        vocal_track.export(output, format="wav")
    return dubbed_path


@functools.lru_cache(maxsize=16)
def _ffmpeg_filter_help(filter_name: str) -> str:
    """查询本机 FFmpeg 滤镜帮助文本，若滤镜不存在则返回空字符串。"""
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"filter={filter_name}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if probe.returncode != 0:
        return ""
    text = probe.stdout + probe.stderr
    if "Unknown filter" in text or f"Filter {filter_name}" not in text:
        return ""
    return text


def _ffmpeg_has_filter(filter_name: str) -> bool:
    return bool(_ffmpeg_filter_help(filter_name))


def _ffmpeg_filter_has_option(help_text: str, option: str) -> bool:
    for line in help_text.splitlines():
        name = line.strip().split("<", 1)[0].strip()
        if name == option:
            return True
    return False


def _amix_normalize_option(inputs: int = 2) -> str:
    """Return the amix no-renormalization option supported by the local FFmpeg."""
    text = _ffmpeg_filter_help("amix")
    if not text or "normalize" in text:
        return f"amix=inputs={inputs}:duration=first:normalize=0"
    if _ffmpeg_filter_has_option(text, "weights"):
        weights = " ".join(["1"] * inputs)
        return f"amix=inputs={inputs}:duration=first:weights={weights},volume={inputs}"
    return f"amix=inputs={inputs}:duration=first,volume={inputs}"


def _alimiter_filter() -> str:
    help_text = _ffmpeg_filter_help("alimiter")
    if not help_text:
        return "volume=0.98"
    options = ["limit=0.98"]
    if _ffmpeg_filter_has_option(help_text, "level"):
        options.append("level=0")
    if _ffmpeg_filter_has_option(help_text, "latency"):
        options.append("latency=1")
    return "alimiter=" + ":".join(options)


def _build_vocal_space_filter(room_wetness: float = 0.10) -> str:
    """Apply subtle EQ warmth, dynamic leveling, and early reflections so TTS sits inside the scene."""
    filters: list[str] = []
    if _ffmpeg_has_filter("highpass"):
        filters.append("highpass=f=75")
    if _ffmpeg_has_filter("equalizer"):
        filters.append("equalizer=f=3400:t=q:w=1.2:g=-1.4")
    if _ffmpeg_has_filter("lowpass"):
        filters.append("lowpass=f=15000")
    if _ffmpeg_has_filter("acompressor"):
        filters.append("acompressor=threshold=-20dB:ratio=2.0:attack=12:release=140:makeup=1.2dB")
    if _ffmpeg_has_filter("aecho"):
        wet = max(0.05, min(0.16, float(room_wetness)))
        tap1 = round(wet, 3)
        tap2 = round(wet * 0.55, 3)
        filters.append(f"aecho=0.90:0.88:23|41:{tap1:.3f}|{tap2:.3f}")
    return ",".join(filters)


def _resolve_background_tracks(
    script: ProjectScript,
) -> Tuple[Optional[Tuple[Path, Path, Path]], Optional[Path]]:
    """优先解析独立三背景轨 (effects, music, singing)，否则回退单 bgm.wav。"""
    if script.media.cinematic_stems is not None:
        stem_urls = (
            script.media.cinematic_stems.effects,
            script.media.cinematic_stems.music,
            script.media.cinematic_stems.singing,
        )
        try:
            stem_paths = tuple(config.resolve_media_path(url) for url in stem_urls)
        except ValueError:
            stem_paths = ()
        if len(stem_paths) == 3 and all(path.is_file() for path in stem_paths):
            return (stem_paths[0], stem_paths[1], stem_paths[2]), None

    bgm_path: Optional[Path] = None
    if script.media.bgm_wav:
        try:
            candidate_bgm = config.resolve_media_path(script.media.bgm_wav)
            if candidate_bgm.is_file():
                bgm_path = candidate_bgm
        except ValueError:
            bgm_path = None
    if script.media.cinematic_stems is not None and bgm_path is None:
        raise FileNotFoundError("影视四轨的混合背景轨缺失，请重新分离后导出。")
    return None, bgm_path


def resolve_effective_auto_ducking(
    script: ProjectScript,
    auto_ducking: Optional[bool] = None,
) -> bool:
    """计算本次导出实际生效的自动闪避开关。"""
    requested = script.media.auto_ducking if auto_ducking is None else bool(auto_ducking)
    return requested and _ffmpeg_has_filter("sidechaincompress")


def _build_audio_mix_graph(
    vocal_idx: int,
    bg_inputs: Tuple[int, ...],
    sample_rate: int,
    script: ProjectScript,
    auto_ducking: Optional[bool] = None,
    *,
    room_wetness: float = 0.10,
) -> Tuple[List[str], str, bool]:
    """构建支持人声空间融合、四轨独立增益与 sidechain 自动闪避的 FFmpeg filter_complex。"""
    master = max(0.0, min(2.0, float(script.media.bgm_volume)))
    sv = script.media.stem_volumes or StemVolumes()
    eff_gain = min(2.0, max(0.0, float(sv.effects)) * master)
    mus_gain = min(2.0, max(0.0, float(sv.music)) * master)
    sng_gain = min(2.0, max(0.0, float(sv.singing)) * master)

    effective_duck = resolve_effective_auto_ducking(script, auto_ducking)
    amix2 = _amix_normalize_option(2)
    amix3 = _amix_normalize_option(3)
    limiter = _alimiter_filter()
    fmt = f"aformat=sample_rates={sample_rate}:channel_layouts=stereo"
    space = _build_vocal_space_filter(room_wetness)
    vocal_fmt = f"{space},{fmt}" if space else fmt

    chains: List[str] = []
    if len(bg_inputs) == 3:
        eff_idx, mus_idx, sng_idx = bg_inputs
        if effective_duck:
            chains.extend([
                f"[{mus_idx}:a]volume={mus_gain:.3f},{fmt}[mus_f]",
                f"[{sng_idx}:a]volume={sng_gain:.3f},{fmt}[sng_f]",
                f"[mus_f][sng_f]{amix2}[ms_bed]",
                f"[{vocal_idx}:a]asplit=2[v_mix][v_sc]",
                f"[v_sc]{fmt}[v_sc_f]",
                "[ms_bed][v_sc_f]sidechaincompress=threshold=0.03:ratio=4:attack=25:release=300:makeup=1[ms_duck]",
                f"[{eff_idx}:a]volume={eff_gain:.3f},{fmt}[eff_f]",
                f"[ms_duck][eff_f]{amix2},{limiter}[bgm_ready]",
                f"[v_mix]{vocal_fmt}[v_mix_f]",
                f"[v_mix_f][bgm_ready]{amix2},{limiter}[aout]",
            ])
        else:
            chains.extend([
                f"[{eff_idx}:a]volume={eff_gain:.3f},{fmt}[eff_f]",
                f"[{mus_idx}:a]volume={mus_gain:.3f},{fmt}[mus_f]",
                f"[{sng_idx}:a]volume={sng_gain:.3f},{fmt}[sng_f]",
                f"[eff_f][mus_f][sng_f]{amix3},{limiter}[bgm_ready]",
                f"[{vocal_idx}:a]{vocal_fmt}[v_mix_f]",
                f"[v_mix_f][bgm_ready]{amix2},{limiter}[aout]",
            ])
        return chains, "[aout]", effective_duck

    (bgm_idx,) = bg_inputs
    if effective_duck:
        chains.extend([
            f"[{bgm_idx}:a]volume={master:.3f},{fmt}[bgm_f]",
            f"[{vocal_idx}:a]asplit=2[v_mix][v_sc]",
            f"[v_sc]{fmt}[v_sc_f]",
            "[bgm_f][v_sc_f]sidechaincompress=threshold=0.03:ratio=4:attack=25:release=300:makeup=1[bgm_duck]",
            f"[v_mix]{vocal_fmt}[v_mix_f]",
            f"[v_mix_f][bgm_duck]{amix2},{limiter}[aout]",
        ])
    else:
        chains.extend([
            f"[{bgm_idx}:a]volume={master:.3f},{fmt}[bgm_f]",
            f"[{vocal_idx}:a]{vocal_fmt}[v_mix_f]",
            f"[v_mix_f][bgm_f]{amix2},{limiter}[aout]",
        ])
    return chains, "[aout]", effective_duck


def _run_ffmpeg_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _build_drawtext_filter() -> str:
    font_spec = _find_watermark_font()
    if font_spec:
        text_expr = "AI 合成语音 · StageTTS"
        font_arg = f"fontfile='{font_spec}':"
    else:
        text_expr = "AI Synthesized Voice - StageTTS"
        font_arg = ""

    return (
        f"drawtext={font_arg}text='{text_expr}':"
        "x=w-tw-24:y=h-th-24:fontsize=22:fontcolor=white@0.88:"
        "box=1:boxcolor=black@0.48:boxborderw=8"
    )


def _sanitize_output_filename(raw_name: Optional[str], expected_ext: str) -> str:
    """校验并规范化导出文件名，防止路径穿越、非法字符或 Windows 保留设备名。"""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return f"output{expected_ext}"
    if (
        "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned)
        or any(ch in '<>:"|?*' for ch in cleaned)
        or cleaned in {".", ".."}
    ):
        raise ValueError("输出文件名不能包含路径分隔符或非法字符")
    suffix = Path(cleaned).suffix.lower()
    if not suffix:
        cleaned = f"{cleaned}{expected_ext}"
    elif suffix != expected_ext:
        raise ValueError(f"当前模式导出文件后缀必须为 {expected_ext}")
    stem = Path(cleaned).stem
    first_part = cleaned.split(".", 1)[0].rstrip(" ")
    if (
        not stem
        or stem.endswith((" ", "."))
        or _WINDOWS_RESERVED_STEM_RE.match(stem)
        or _WINDOWS_RESERVED_STEM_RE.match(first_part)
    ):
        raise ValueError("输出文件名不能使用系统保留名称或以空格/点结尾")
    return cleaned


def _resolve_export_dir(raw_dir: Optional[str]) -> Optional[Path]:
    """校验并创建外部导出目录（必须为绝对路径或以 ~ 开头）。"""
    cleaned = (raw_dir or "").strip()
    if not cleaned:
        return None
    candidate = Path(cleaned).expanduser()
    if not candidate.is_absolute():
        raise ValueError("保存目录请填写绝对路径（支持 ~ 开头）")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"保存路径不是目录：{candidate}")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"无法创建保存目录 {candidate}：{exc}") from exc
    return candidate.resolve()


def _copy_file_atomic(src: Path, dest_dir: Path, filename: str) -> Path:
    """在目标目录内原子拷贝产物文件，兼容跨盘/跨分区移动。"""
    target = (dest_dir / filename).resolve()
    if target == src.resolve():
        return target
    with tempfile.TemporaryDirectory(dir=dest_dir, prefix=".stagetts_export_") as tmp_dir:
        tmp_file = Path(tmp_dir) / filename
        shutil.copyfile(src, tmp_file)
        tmp_file.replace(target)
    return target


def mix_and_render(
    script: ProjectScript,
    burn_watermark: bool = False,
    burn_subtitles: SubtitleBurnMode = "none",
    auto_ducking: Optional[bool] = None,
    output_filename: Optional[str] = None,
    *,
    auto_fit_overflow: bool = False,
) -> Tuple[str, float]:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装导出（含空间融合、四轨混音与单次硬字幕烧录）。"""
    if not check_ffmpeg_available():
        raise RuntimeError("未在系统 PATH 中检测到 ffmpeg，请确认已安装 FFmpeg 6.x+。")

    source_video_path: Optional[Path] = None
    timeline_duration = script.media.duration
    if script.media.source_video:
        candidate_video = config.resolve_media_path(script.media.source_video)
        if candidate_video.is_file():
            source_video_path = candidate_video
            source_duration = probe_media_duration(candidate_video)
            if source_duration <= 0:
                raise RuntimeError("导出待审核：无法确认源视频时长，不能安全安排配音。")
            timeline_duration = min(timeline_duration, source_duration)

    is_audio_export = script.mode == "virtual_stage" and source_video_path is None
    if not is_audio_export and source_video_path is None:
        raise FileNotFoundError(
            f"未找到源视频文件 ({script.media.source_video})，无法封装 output.mp4"
        )
    target_filename = _sanitize_output_filename(output_filename, ".wav" if is_audio_export else ".mp4")
    if not is_audio_export and burn_subtitles != "none" and not _ffmpeg_has_filter("ass"):
        raise RuntimeError("当前 FFmpeg 未编译 libass (ass) 滤镜，无法烧录 ASS 硬字幕。")

    project_dir: Path = config.WORKSPACE_DIR / script.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    stems_paths, bgm_path = _resolve_background_tracks(script)
    output_path = (project_dir / target_filename).resolve()
    protected_paths = {
        (project_dir / "vocals.wav").resolve(),
        (project_dir / "bgm.wav").resolve(),
        (project_dir / "dubbed_vocals.wav").resolve(),
    }
    if source_video_path is not None:
        protected_paths.add(source_video_path.resolve())
    if bgm_path is not None:
        protected_paths.add(bgm_path.resolve())
    if stems_paths is not None:
        protected_paths.update(p.resolve() for p in stems_paths)
    if Path(target_filename).stem.lower() == "input" or output_path in protected_paths:
        raise ValueError(f"输出文件名 {target_filename} 与项目源文件或分轨文件冲突，请更换文件名")

    dubbed_vocals_path = _build_vocal_track(
        script,
        project_dir,
        timeline_duration=timeline_duration,
        auto_fit_overflow=auto_fit_overflow,
    )
    vocal_sr = _probe_wav_sample_rate(dubbed_vocals_path, default=44100)
    _, room_wetness = _measure_source_dialogue_profile(script)

    # 模式 B 虚拟舞台若无源视频文件，导出完整混音音频
    if is_audio_export:
        output_wav = output_path
        with tempfile.TemporaryDirectory(dir=project_dir, prefix=".stagetts_render_") as render_tmp:
            tmp_wav = Path(render_tmp) / target_filename
            if stems_paths is not None or bgm_path is not None:
                cmd = ["ffmpeg", "-y", "-i", str(dubbed_vocals_path)]
                if stems_paths is not None:
                    for p in stems_paths:
                        cmd.extend(["-i", str(p)])
                    bg_inputs: Tuple[int, ...] = (1, 2, 3)
                else:
                    assert bgm_path is not None
                    cmd.extend(["-i", str(bgm_path)])
                    bg_inputs = (1,)
                chains, out_label, _ = _build_audio_mix_graph(
                    vocal_idx=0,
                    bg_inputs=bg_inputs,
                    sample_rate=vocal_sr,
                    script=script,
                    auto_ducking=auto_ducking,
                    room_wetness=room_wetness,
                )
                cmd.extend(["-filter_complex", ";".join(chains), "-map", out_label, str(tmp_wav)])
                proc = _run_ffmpeg_cmd(cmd)
                if proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg 音频混音失败: {proc.stderr[-600:]}")
            else:
                shutil.copyfile(dubbed_vocals_path, tmp_wav)
            if tmp_wav.is_file():
                tmp_wav.replace(output_wav)

        out_duration = probe_media_duration(output_wav) or script.media.duration
        return f"/workspace/{script.project_id}/{target_filename}", out_duration

    assert source_video_path is not None
    output_mp4 = output_path
    base_inputs: List[str] = ["ffmpeg", "-y", "-i", str(source_video_path), "-i", str(dubbed_vocals_path)]
    if stems_paths is not None:
        for p in stems_paths:
            base_inputs.extend(["-i", str(p)])
        chains, out_label, _ = _build_audio_mix_graph(
            vocal_idx=1,
            bg_inputs=(2, 3, 4),
            sample_rate=vocal_sr,
            script=script,
            auto_ducking=auto_ducking,
            room_wetness=room_wetness,
        )
        audio_filter: Optional[str] = ";".join(chains)
        audio_map = ["-map", "0:v:0", "-map", out_label]
    elif bgm_path is not None:
        base_inputs.extend(["-i", str(bgm_path)])
        chains, out_label, _ = _build_audio_mix_graph(
            vocal_idx=1,
            bg_inputs=(2,),
            sample_rate=vocal_sr,
            script=script,
            auto_ducking=auto_ducking,
            room_wetness=room_wetness,
        )
        audio_filter = ";".join(chains)
        audio_map = ["-map", "0:v:0", "-map", out_label]
    else:
        audio_filter = None
        audio_map = ["-map", "0:v:0", "-map", "1:a:0"]

    with tempfile.TemporaryDirectory(dir=project_dir, prefix=".stagetts_render_") as render_tmp:
        render_dir = Path(render_tmp)
        tmp_mp4 = render_dir / target_filename
        video_filters: List[str] = []
        if burn_subtitles != "none":
            ass_path = write_ass_file(script, burn_subtitles, project_dir / "subtitles.ass")
            video_filters.append(_build_ass_subtitle_filter(ass_path))
        if burn_watermark:
            video_filters.append(_build_drawtext_filter())

        if not video_filters:
            cmd = list(base_inputs)
            if audio_filter:
                cmd.extend(["-filter_complex", audio_filter])
            cmd.extend(audio_map)
            cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(tmp_mp4)])
            proc = _run_ffmpeg_cmd(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg 无损封装失败: {proc.stderr[-600:]}")
        else:
            vf_expr = ",".join(video_filters)
            for video_codec_args in _video_encoder_candidates():
                cmd = list(base_inputs)
                if audio_filter:
                    cmd.extend(["-filter_complex", audio_filter])
                cmd.extend(audio_map)
                cmd.extend(["-vf", vf_expr])
                cmd.extend(video_codec_args)
                cmd.extend(["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(tmp_mp4)])
                proc = _run_ffmpeg_cmd(cmd)
                if proc.returncode == 0:
                    break
            else:
                raise RuntimeError(f"FFmpeg 视频重编码导出失败: {proc.stderr[-600:]}")
        if tmp_mp4.is_file():
            tmp_mp4.replace(output_mp4)

    out_duration = probe_media_duration(output_mp4) or script.media.duration
    return f"/workspace/{script.project_id}/{target_filename}", out_duration


def export_project_deliverables(
    script: ProjectScript,
    *,
    burn_watermark: bool = False,
    burn_subtitles: SubtitleBurnMode = "none",
    export_ass_sidecar: Optional[SubtitleLayout] = None,
    auto_ducking: Optional[bool] = None,
    auto_fit_overflow: bool = False,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """导出成片与可选外挂 ASS 字幕，并按需原子落盘至用户指定本机目录。"""
    target_dir = _resolve_export_dir(output_dir)
    has_source_video = bool(
        script.media.source_video and config.resolve_media_path(script.media.source_video).is_file()
    )
    is_audio_export = script.mode == "virtual_stage" and not has_source_video
    effective_watermark = False if is_audio_export else burn_watermark
    effective_burn_subtitles: SubtitleBurnMode = "none" if is_audio_export else burn_subtitles

    output_url, duration = mix_and_render(
        script,
        burn_watermark=effective_watermark,
        burn_subtitles=effective_burn_subtitles,
        auto_ducking=auto_ducking,
        output_filename=output_filename,
        auto_fit_overflow=auto_fit_overflow,
    )
    workspace_output = config.resolve_media_path(output_url)
    saved_path = (
        _copy_file_atomic(workspace_output, target_dir, workspace_output.name)
        if target_dir is not None
        else workspace_output
    )

    subtitle_url: Optional[str] = None
    subtitle_saved_path: Optional[str] = None
    if export_ass_sidecar is not None:
        ass_name = f"{workspace_output.stem}.ass"
        workspace_ass = workspace_output.parent / ass_name
        write_ass_file(script, export_ass_sidecar, workspace_ass)
        subtitle_url = f"/workspace/{script.project_id}/{ass_name}"
        copied_ass = (
            _copy_file_atomic(workspace_ass, target_dir, ass_name)
            if target_dir is not None
            else workspace_ass
        )
        subtitle_saved_path = str(copied_ass)

    return {
        "output_video": output_url,
        "duration": duration,
        "burn_watermark": effective_watermark,
        "burn_subtitles": effective_burn_subtitles,
        "auto_ducking": resolve_effective_auto_ducking(script, auto_ducking),
        "saved_path": str(saved_path),
        "subtitle_url": subtitle_url,
        "subtitle_saved_path": subtitle_saved_path,
    }
