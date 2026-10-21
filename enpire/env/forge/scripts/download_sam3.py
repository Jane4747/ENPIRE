#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and validate the facebook/sam3 HuggingFace model.

Usage:
    # Prefer ModelScope when HF_TOKEN is unavailable (no gated access needed):
    uv run python scripts/download_sam3.py --source modelscope

    # Or HuggingFace (gated — requires HF_TOKEN):
    HF_TOKEN=hf_xxx uv run python scripts/download_sam3.py --source huggingface

    # Auto: use HF if HF_TOKEN is set, otherwise ModelScope
    uv run python scripts/download_sam3.py

    # Custom cache root
    HF_HOME=/mnt/shared/<user>/pretrain/huggingface \
        uv run python scripts/download_sam3.py --source modelscope

Environment variables:
    HF_TOKEN          — HuggingFace token (required for --source huggingface)
    HF_HOME           — HuggingFace cache root (default: ~/.cache/huggingface)
    HUGGINGFACE_HUB_CACHE — Direct hub cache path (overrides HF_HOME/hub)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


_REPO_ID = "facebook/sam3"
_USABLE_PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json")


def _hub_cache() -> Path:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")))


def _repo_dir(hub_cache: Path) -> Path:
    return hub_cache / f"models--{_REPO_ID.replace('/', '--')}"


def _usable_snapshot(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any((path / name).is_file() for name in _USABLE_PROCESSOR_FILES)


def _find_cached_snapshot(hub_cache: Path) -> Path | None:
    repo_dir = _repo_dir(hub_cache)
    if not repo_dir.exists():
        return None
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot = repo_dir / "snapshots" / ref_path.read_text().strip()
        if snapshot.exists() and _usable_snapshot(snapshot):
            return snapshot
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir() and _usable_snapshot(p)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _download_huggingface(hub_cache: Path) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[sam3] ERROR: HF_TOKEN not set. facebook/sam3 is a gated model on HuggingFace.")
        print("[sam3] 1. Accept license at https://huggingface.co/facebook/sam3")
        print("[sam3] 2. Set HF_TOKEN=hf_xxx and rerun")
        print("[sam3] Or use: uv run python scripts/download_sam3.py --source modelscope")
        sys.exit(1)

    print(f"[sam3] Downloading {_REPO_ID} from HuggingFace (gated model)...")
    from huggingface_hub import snapshot_download

    path = snapshot_download(_REPO_ID, token=token, cache_dir=str(hub_cache))
    print(f"[sam3] Downloaded to {path}")
    return path


def _download_modelscope(hub_cache: Path) -> str:
    """Fetch ModelScope facebook/sam3 and materialize an HF-hub layout."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except ImportError:
        print("[sam3] ERROR: modelscope is not installed.")
        print("[sam3] Install with: uv pip install modelscope")
        print("[sam3] Or use HuggingFace: HF_TOKEN=hf_xxx uv run python scripts/download_sam3.py --source huggingface")
        sys.exit(1)

    staging = hub_cache / "_modelscope_staging" / "facebook-sam3"
    staging.mkdir(parents=True, exist_ok=True)
    print(f"[sam3] Downloading {_REPO_ID} from ModelScope → {staging}")
    ms_path = Path(
        ms_download(
            _REPO_ID,
            local_dir=str(staging),
            # Official sam3.pt is for the sam3 pip package; transformers uses safetensors.
            ignore_file_pattern=[r"sam3\.pt$"],
        )
    )

    snap_hash = hashlib.sha256(b"facebook/sam3@modelscope").hexdigest()[:40]
    repo_dir = _repo_dir(hub_cache)
    snap = repo_dir / "snapshots" / snap_hash
    snap.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)

    for src in ms_path.iterdir():
        if not src.is_file() or src.name == "sam3.pt":
            continue
        dest = snap / src.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        # Copy into the hub layout so staging can be deleted later.
        shutil.copy2(src, dest)

    (repo_dir / "refs" / "main").write_text(snap_hash)
    print(f"[sam3] HF snapshot ready: {snap}")
    return str(snap)


def _validate(snapshot_path: str) -> None:
    """Validate that the model loads correctly."""
    print("[sam3] Validating model load...")
    try:
        from transformers import Sam3Model, Sam3Processor

        proc = Sam3Processor.from_pretrained(snapshot_path, local_files_only=True)
        print(f"[sam3] Processor: OK ({type(proc).__name__})")

        import torch

        if torch.cuda.is_available():
            model = Sam3Model.from_pretrained(
                snapshot_path, torch_dtype=torch.float16, local_files_only=True
            )
            print(
                f"[sam3] Model: OK ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"
            )
            del model
            torch.cuda.empty_cache()
        else:
            print("[sam3] Model: skipped (no GPU, processor-only validation)")

        print("[sam3] Validation PASSED")
    except Exception as e:
        print(f"[sam3] Validation FAILED: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "huggingface", "modelscope"),
        default="auto",
        help="Weight source. auto uses HuggingFace when HF_TOKEN is set, else ModelScope.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip processor/model load check after download.",
    )
    args = parser.parse_args()

    hub_cache = _hub_cache()
    print(f"[sam3] HF cache: {hub_cache}")

    cached = _find_cached_snapshot(hub_cache)
    if cached is not None:
        print(f"[sam3] Already cached at {cached}")
        files = sorted(p.name for p in cached.iterdir())
        print(f"[sam3] Files: {len(files)} ({', '.join(files[:5])}...)")
        if not args.skip_validate:
            _validate(str(cached))
        return

    source = args.source
    if source == "auto":
        source = "huggingface" if os.environ.get("HF_TOKEN") else "modelscope"
        print(f"[sam3] auto source → {source}")

    if source == "huggingface":
        path = _download_huggingface(hub_cache)
    else:
        path = _download_modelscope(hub_cache)

    if not args.skip_validate:
        _validate(path)


if __name__ == "__main__":
    main()
