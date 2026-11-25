"""Automatic cache cleanup for GPT-SoVITS intermediate training logs, temp dirs, and stale workspace artifacts."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from backend import config
from backend.schemas import ProjectScript


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _remove_path_tree(path: Path) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    size = _dir_size_bytes(path)
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)
        return True, size
    except OSError:
        return False, 0


def cleanup_finetune_artifacts(
    dataset_dir: Path,
    opt_dir: Path,
    *,
    exp_name: str,
    keep_gpt: Path | None = None,
    keep_sovits: Path | None = None,
    base_model: str | None = None,
    root: Path | None = None,
) -> int:
    """Clean temporary training slices, GPT-SoVITS logs/st_*, and intermediate epoch weights."""
    freed = 0
    for target in (opt_dir, dataset_dir):
        _, sz = _remove_path_tree(target)
        freed += sz

    if root is not None and base_model:
        keep_resolved = {
            p.resolve()
            for p in (keep_gpt, keep_sovits)
            if p is not None and p.exists()
        }
        for folder, ext in (
            (root / f"SoVITS_weights_{base_model}", ".pth"),
            (root / f"GPT_weights_{base_model}", ".ckpt"),
        ):
            if not folder.is_dir():
                continue
            for cand in folder.glob(f"{exp_name}*{ext}"):
                if cand.resolve() not in keep_resolved:
                    _, sz = _remove_path_tree(cand)
                    freed += sz
    return freed


def _referenced_paths_for_project(script: ProjectScript) -> set[Path]:
    refs: set[Path] = set()
    urls: list[str | None] = [
        script.media.source_video,
        script.media.vocals_wav,
        script.media.bgm_wav,
        script.media.output_video,
    ]
    if script.media.cinematic_stems is not None:
        urls.extend([
            script.media.cinematic_stems.dialogue,
            script.media.cinematic_stems.singing,
            script.media.cinematic_stems.music,
            script.media.cinematic_stems.effects,
        ])
    for ch in script.characters.values():
        for emo_ref in ch.emotion_refs.values():
            if emo_ref and emo_ref.wav:
                urls.append(emo_ref.wav)
    for url in urls:
        if not url:
            continue
        try:
            refs.add(config.resolve_media_path(url).resolve())
        except ValueError:
            continue
    return refs


def cleanup_stage_caches(
    *,
    ref_max_age_days: float = 7.0,
    log_max_bytes: int = 5 * 1024 * 1024,
    active_finetune: bool | None = None,
) -> dict[str, Any]:
    """Safely prune stale temp directories, intermediate training logs, and unreferenced caches."""
    if active_finetune is None:
        try:
            from backend.services.gsv_finetune import has_active_finetune_job

            active_finetune = has_active_finetune_job()
        except Exception:
            active_finetune = False

    removed_dirs = 0
    removed_files = 0
    freed_bytes = 0
    now = time.time()
    ref_cutoff = now - max(0.0, ref_max_age_days) * 86400.0

    # 1. System temp stagetts_* directories older than 2 hours
    tmp_root = Path(tempfile.gettempdir())
    if tmp_root.is_dir():
        for cand in tmp_root.glob("stagetts_*"):
            try:
                if now - cand.stat().st_mtime < 7200:
                    continue
            except OSError:
                continue
            ok, sz = _remove_path_tree(cand)
            if ok:
                removed_dirs += 1 if cand.is_dir() or sz >= 0 else 0
                freed_bytes += sz

    # 2. GPT-SoVITS intermediate training logs (logs/st_*)
    if not active_finetune and config.GPT_SOVITS_ROOT is not None:
        gsv_logs = config.GPT_SOVITS_ROOT / "logs"
        if gsv_logs.is_dir():
            for cand in gsv_logs.glob("st_*"):
                if cand.is_dir():
                    ok, sz = _remove_path_tree(cand)
                    if ok:
                        removed_dirs += 1
                        freed_bytes += sz

    # 3. Workspace intermediate finetune slices, unreferenced cinematic-* dirs, and stale auto refs
    ws_root = config.WORKSPACE_DIR
    if ws_root.is_dir():
        tts_log = ws_root / "_tts_server.log"
        if tts_log.is_file():
            try:
                sz = tts_log.stat().st_size
                if sz > log_max_bytes:
                    keep_tail = 512 * 1024
                    with tts_log.open("rb") as fp:
                        fp.seek(max(0, sz - keep_tail))
                        tail_data = fp.read()
                    tts_log.write_bytes(tail_data)
                    freed_bytes += max(0, sz - len(tail_data))
            except OSError:
                pass

        for proj_dir in ws_root.iterdir():
            if not proj_dir.is_dir():
                continue
            if not active_finetune:
                ft_dir = proj_dir / "finetune"
                if ft_dir.is_dir():
                    for cand in ft_dir.glob("st_*"):
                        ok, sz = _remove_path_tree(cand)
                        if ok:
                            removed_dirs += 1
                            freed_bytes += sz

            script_path = proj_dir / "script.json"
            referenced: set[Path] = set()
            if script_path.is_file():
                try:
                    script = ProjectScript.model_validate_json(script_path.read_text(encoding="utf-8"))
                    referenced = _referenced_paths_for_project(script)
                except (OSError, ValueError):
                    referenced = set()

            for cin_dir in proj_dir.glob("cinematic-*"):
                if not cin_dir.is_dir():
                    continue
                resolved_cin = cin_dir.resolve()
                in_use = any(resolved_cin in p.parents or p == resolved_cin for p in referenced)
                complete_marker = cin_dir / "separation_complete.json"
                # Delete incomplete separations or stale superseded cinematic-* folders not referenced by script.json
                if not in_use and (not complete_marker.is_file() or bool(referenced)):
                    try:
                        if now - cin_dir.stat().st_mtime >= 600:
                            ok, sz = _remove_path_tree(cin_dir)
                            if ok:
                                removed_dirs += 1
                                freed_bytes += sz
                    except OSError:
                        pass

            refs_dir = proj_dir / "references"
            if refs_dir.is_dir():
                for prefix in ("auto_*.wav", "aux_*.wav", "moss_ref_*"):
                    for cand in refs_dir.glob(prefix):
                        if not cand.is_file() or cand.resolve() in referenced:
                            continue
                        try:
                            if cand.stat().st_mtime < ref_cutoff:
                                ok, sz = _remove_path_tree(cand)
                                if ok:
                                    removed_files += 1
                                    freed_bytes += sz
                        except OSError:
                            pass

    return {
        "removed_dirs": removed_dirs,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
    }
