"""Record hardware and local-server readiness without downloading weights or claiming a fit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.bolt import MODEL_REPO, MODEL_REVISION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/experiments/receiptops/bolt_preflight.json"
    )
    args = parser.parse_args()
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu = result.stdout.strip() if result.returncode == 0 else "query_failed"
    ram = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        ram = {
            "total_gib": round(memory.total / 2**30, 2),
            "available_gib": round(memory.available / 2**30, 2),
        }
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=2) as response:
            status = response.status
    except (OSError, urllib.error.URLError):
        status = None
    report = {
        "model_repo": MODEL_REPO,
        "verified_repo_revision": MODEL_REVISION,
        "verified_input_output": "text/text",
        "verified_context_limit": 65536,
        "official_model_card": "https://huggingface.co/" + MODEL_REPO,
        "gpu": gpu,
        "ram": ram,
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
        "runtimes_on_path": {
            name: shutil.which(name) for name in ("llama-server", "unsloth", "docker")
        },
        "endpoint": args.base_url,
        "health_http_status": status,
        "bolt_inference_run": False,
        "weights_downloaded_by_this_script": False,
        "notes": [
            "Official Q4_K_M weights are approximately 6.18 GB; 4 GB VRAM is not total memory",
            "Vision-projector files in another distribution do not validate "
            "the official text checkpoint as a receipt VLM",
            "No automatic fit decision: context, offloading, free RAM, and runtime buffers matter",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
