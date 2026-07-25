"""Shared filesystem, logging, reproducibility, and serialization helpers."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "pillow",
    "pandas",
    "numpy",
    "scikit-learn",
    "rapidfuzz",
    "matplotlib",
    "pyyaml",
)


def configure_project_environment() -> None:
    """Keep model caches in-project and reduce tokenizer process contention."""
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def project_path(value: str | Path) -> Path:
    """Resolve a repository-relative path while preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def repository_relative(value: str | Path) -> str:
    """Render project-contained paths as portable POSIX-style relative paths."""
    path = project_path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def ensure_artifact_directories() -> None:
    """Create the standard artifact hierarchy."""
    for relative in (
        "artifacts/checkpoints",
        "artifacts/logs",
        "artifacts/predictions",
        "artifacts/figures",
        "artifacts/reports",
    ):
        project_path(relative).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str | Path | None = None, verbose: bool = True) -> logging.Logger:
    """Configure console and optional UTF-8 file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = project_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("receipt_kie")


def write_json(path: str | Path, payload: Any) -> None:
    """Write stable, readable UTF-8 JSON."""
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON Lines with UTF-8 and deterministic separators."""
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def package_versions() -> dict[str, str]:
    """Return exact versions for the core reproducibility stack."""
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def clear_cuda() -> None:
    """Release unused CUDA cache without assuming a GPU exists."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
