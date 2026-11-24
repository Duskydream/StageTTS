"""ASS 字幕生成与写盘工具（支持单语译文与双语对照布局）。"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from backend.schemas import ProjectScript, SubtitleLayout

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_SPACE_RE = re.compile(r"[ \t]+")


def _format_ass_time(seconds: float) -> str:
    """将秒数格式化为 ASS 时间戳 H:MM:SS.cc（百分之一秒四舍五入，防止 .100 溢出）。"""
    total_cs = max(0, round(float(seconds) * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """转义 ASS 控制字符并规范化换行为 \\N。"""
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("{", "｛").replace("}", "｝")
    cleaned = re.sub(r"\\([Nnh])", r"＼\1", cleaned)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in _NEWLINE_RE.split(cleaned)]
    return r"\N".join(line for line in lines if line)


@functools.lru_cache(maxsize=1)
def _pick_ass_font_name() -> str:
    """按跨平台优先顺序探测清晰无衬线 CJK 字体。"""
    if os.name == "nt":
        return "Microsoft YaHei"
    if sys.platform == "darwin":
        return "PingFang SC"

    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK HK",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
    )
    fc_list = shutil.which("fc-list")
    if fc_list:
        try:
            proc = subprocess.run(
                [fc_list, ":", "family"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if proc.returncode == 0:
                installed = {
                    part.strip()
                    for line in proc.stdout.splitlines()
                    for part in line.split(",")
                    if part.strip()
                }
                for candidate in preferred:
                    if candidate in installed:
                        return candidate
        except (OSError, subprocess.SubprocessError):
            pass
    return "Sans"


def generate_ass_subtitles(
    script: ProjectScript,
    layout: SubtitleLayout = "target",
) -> str:
    """根据 ProjectScript 生成标准 ASS v4+ 字幕文本。"""
    font_name = _pick_ass_font_name()
    title = (script.project_id or "StageTTS").replace("\n", " ").strip()

    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Primary,{font_name},58,&H00FFFFFF,&H000000FF,&H00141414,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.6,1.2,2,48,48,44,1",
        f"Style: Secondary,{font_name},38,&H00D0D0D0,&H000000FF,&H00141414,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.0,1.0,2,48,48,44,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    for seg in sorted(script.segments, key=lambda item: (item.start, item.id)):
        if layout == "source":
            primary_raw = seg.orig_text.strip() or seg.target_text.strip()
        else:
            primary_raw = seg.target_text.strip() or seg.orig_text.strip()
        if not primary_raw:
            continue
        escaped_primary = _escape_ass_text(primary_raw)
        if not escaped_primary:
            continue

        start_sec = max(0.0, float(seg.start))
        end_sec = max(start_sec + 0.05, float(seg.end))
        start_ts = _format_ass_time(start_sec)
        end_ts = _format_ass_time(end_sec)

        char = script.characters.get(seg.speaker)
        speaker_name = re.sub(
            r"[\r\n,]+",
            " ",
            (char.display_name if char else seg.speaker),
        ).strip() or seg.speaker

        orig_raw = seg.orig_text.strip()
        if layout == "bilingual" and orig_raw and orig_raw != primary_raw:
            escaped_orig = _escape_ass_text(orig_raw)
            text_body = rf"{escaped_primary}\N{{\rSecondary}}{escaped_orig}" if escaped_orig else escaped_primary
        else:
            text_body = escaped_primary

        events.append(f"Dialogue: 0,{start_ts},{end_ts},Primary,{speaker_name},0,0,0,,{text_body}")

    return "\n".join([*header, *events, ""])


def write_ass_file(
    script: ProjectScript,
    layout: SubtitleLayout,
    dest_path: Path,
) -> Path:
    """将生成的 ASS 字幕以 UTF-8 BOM 编码写入目标路径。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(generate_ass_subtitles(script, layout=layout), encoding="utf-8-sig")
    return dest_path
