"""Collect a reproducible hardware and Python environment report."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))


def _nvidia_query() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
        name, driver, total, free = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "driver_version": driver,
            "memory_total_mib": int(total),
            "memory_free_mib": int(free),
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return {"available": False, "error": str(exc)}


def collect_report() -> dict[str, Any]:
    """Return operating-system, Python, CUDA, memory, and disk details."""
    disk = shutil.disk_usage(PROJECT_ROOT)
    cuda_available = torch.cuda.is_available()
    gpu = _nvidia_query()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
            "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
            "flash_attention_2_installed": _module_available("flash_attn"),
        },
        "gpu": gpu,
        "system_memory": {
            "total_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "available_gib": round(psutil.virtual_memory().available / 2**30, 2),
        },
        "project_drive": {
            "total_gib": round(disk.total / 2**30, 2),
            "used_gib": round(disk.used / 2**30, 2),
            "free_gib": round(disk.free / 2**30, 2),
        },
        "attention_decision": "sdpa",
        "precision_decision": "bf16"
        if cuda_available and torch.cuda.is_bf16_supported()
        else "fp16"
        if cuda_available
        else "fp32",
        "hf_home": os.environ["HF_HOME"],
    }


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main() -> None:
    output_path = PROJECT_ROOT / "artifacts" / "environment_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = collect_report()
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved environment report to {output_path}")


if __name__ == "__main__":
    main()
