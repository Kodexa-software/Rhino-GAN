"""Back up downloaded pretrained models to a local fail-safe folder.

Run this once after a successful first-time download (when every model is
present locally). If an upstream host later 404s, the copies in
``backend/fail-safe-models/`` can be restored to their original locations.

To add more files, append to ``MODEL_PATHS`` below. Each entry may be:
  * an absolute path  -> ``/root/.cache/torch/hub/checkpoints/foo.pth``
  * a path relative to ``backend/``  -> ``pretrained_models/seg.pth``
  * a glob pattern  -> ``cache/*`` or ``pretrained_models/*.pth``

Missing files and unmatched globs are skipped with a warning, so the script
is safe to run repeatedly.
"""
from __future__ import annotations

import shutil
from glob import glob
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
FAILSAFE_DIR = BACKEND_DIR / "fail-safe-models"

MODEL_PATHS: list[str] = [
    # ---- PyTorch hub checkpoints (downloaded by face-alignment / torchvision) ----
    "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth",
    "/root/.cache/torch/hub/checkpoints/3DFAN4-4a694010b9.zip",
    "/root/.cache/torch/hub/checkpoints/depth-6c4283c0e0.zip",
    "/root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth",
    "/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth",

    "pretrained_models/seg.pth",
    "pretrained_models/ffhq.pt",

    "cache/*",
    "pretrained_models/*",

]


def _resolve(entry: str) -> list[Path]:
    """Expand globs and resolve relative paths against backend/."""
    p = Path(entry)
    if not p.is_absolute():
        p = BACKEND_DIR / p

    if any(ch in str(p) for ch in "*?["):
        return [Path(m) for m in glob(str(p))]
    return [p]


def backup_models(paths: list[str] = MODEL_PATHS, dest: Path = FAILSAFE_DIR) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    for entry in paths:
        resolved = _resolve(entry)
        if not resolved:
            missing.append(entry)
            print(f"[MISS] no match for pattern: {entry}")
            continue

        for src in resolved:
            if not src.is_file():
                missing.append(str(src))
                print(f"[MISS] file not found: {src}")
                continue

            dst = dest / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped.append(str(dst))
                print(f"[SKIP] already backed up: {src.name}")
                continue

            shutil.copy2(src, dst)
            copied.append(str(dst))
            print(f"[OK]   {src}  ->  {dst}")

    print()
    print(f"Summary: {len(copied)} copied, {len(skipped)} skipped, {len(missing)} missing")
    print(f"Backup folder: {dest}")
    return {"copied": copied, "skipped": skipped, "missing": missing}


if __name__ == "__main__":
    backup_models()
