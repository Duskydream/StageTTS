"""Run GPT-SoVITS training scripts on Windows with DataLoader workers disabled."""
from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Callable


def patch_windows_dataloader(data_loader_cls: type[Any] | None = None) -> None:
    """Force PyTorch DataLoader single-process mode to prevent Windows spawn MemoryError."""
    if data_loader_cls is None:
        from torch.utils.data import DataLoader as data_loader_cls

    orig_init = data_loader_cls.__init__
    if getattr(orig_init, "_stagetts_patched", False):
        return

    @functools.wraps(orig_init)
    def _single_process_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if len(args) > 5:
            patched_args = list(args)
            patched_args[5] = 0
            args = tuple(patched_args)
        else:
            kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = None
        orig_init(self, *args, **kwargs)

    _single_process_init._stagetts_patched = True  # type: ignore[attr-defined]
    data_loader_cls.__init__ = _single_process_init


def patch_lightning_atomic_save(
    cloud_io_mod: Any | None = None,
    torch_io_mod: Any | None = None,
    save_fn: Callable[[Any, Any], Any] | None = None,
) -> None:
    """Stream Lightning checkpoints straight to disk instead of buffering ~1 GB in BytesIO."""
    if cloud_io_mod is None:
        try:
            from lightning_fabric.plugins.io import torch_io as default_torch_io
            from lightning_fabric.utilities import cloud_io as cloud_io_mod

            if torch_io_mod is None:
                torch_io_mod = default_torch_io
        except ImportError:
            return
    if save_fn is None:
        import torch

        save_fn = torch.save

    orig_save = getattr(cloud_io_mod, "_atomic_save", None)
    if orig_save is None or getattr(orig_save, "_stagetts_patched", False):
        return

    @functools.wraps(orig_save)
    def _streaming_atomic_save(checkpoint: dict[str, Any], filepath: Any) -> None:
        raw_path = str(filepath)
        if "://" in raw_path:
            orig_save(checkpoint, filepath)
            return
        target = Path(raw_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            save_fn(checkpoint, str(tmp))
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    _streaming_atomic_save._stagetts_patched = True  # type: ignore[attr-defined]
    cloud_io_mod._atomic_save = _streaming_atomic_save
    if torch_io_mod is not None and hasattr(torch_io_mod, "_atomic_save"):
        torch_io_mod._atomic_save = _streaming_atomic_save


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gsv_train_launcher.py <script_path> [script_args...]")

    target = Path(sys.argv[1]).resolve()
    cwd_str = str(Path.cwd().resolve())
    target_dir = str(target.parent)

    sys.argv = [sys.argv[1], *sys.argv[2:]]
    for entry in (cwd_str, target_dir):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    patch_windows_dataloader()
    patch_lightning_atomic_save()
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
