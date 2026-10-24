"""Build an allowlisted, private Kaggle script bundle; never upload local secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
)
SOURCES = (
    "requirements-cloud.txt",
    "scripts/train_qwen_cloud.py",
    "src/receipt_kie/__init__.py",
    "src/receipt_kie/prompts.py",
    "src/receipt_kie/metrics.py",
    "src/receipt_kie/structured.py",
    "src/receipt_kie/cloud_data.py",
    "src/receipt_kie/qwen_cloud.py",
)


def build(username: str, smoke: bool, destination: Path) -> dict:
    if not username or not all(character.isalnum() or character in "_-" for character in username):
        raise ValueError("A valid Kaggle username is required")
    audit = json.loads(
        (ROOT / "artifacts/experiments/receiptops/data_quality.json").read_text("utf-8")
    )
    retained = audit["historical_holdout_audit"]["retained_ids"]
    if len(retained) != 239:
        raise ValueError("Expected the audited 239-record retrospective comparison")
    payload = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCES}
    protocol = {
        "seed": 42,
        "evaluation_ids": retained,
        "epochs": 3,
        "learning_rate": 1e-4,
        "max_pixels": 1048576,
        "max_sequence_length": 2048,
        "selection_metric": "merchant-disjoint official-train validation loss",
        "dataset_source": "urbikn/sroie-datasetv2",
        "reference_smoke_container": DOCKER_IMAGE,
        "evaluation_image_sha256": json.loads(
            (ROOT / "artifacts/experiments/receiptops/evaluation_image_sha256.json").read_text(
                encoding="utf-8"
            )
        ),
        "source_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in payload.items()
        },
    }
    payload["cloud_protocol.json"] = json.dumps(protocol, indent=2)
    mode = "smoke" if smoke else "train"
    script = f'''"""Private ReceiptOps Qwen3-VL {mode} job, from allowlisted sources."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp/receiptops-huggingface"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
deadline = time.monotonic() + {1100 if smoke else 13900}

def run(command, **kwargs):
    remaining = max(1, int(deadline - time.monotonic()))
    subprocess.run(command, check=True, timeout=remaining, **kwargs)

root = Path("/kaggle/working/receiptops-source")
sources = {payload!r}
for relative, text in sources.items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
print("Source SHA256:", {protocol["source_sha256"]!r}, flush=True)
roots = [path for path in Path("/kaggle/input").rglob("SROIE2019")
         if (path / "train/entities").is_dir() and (path / "test/entities").is_dir()]
if len(roots) != 1:
    raise RuntimeError("Attach exactly one urbikn/sroie-datasetv2 dataset: " + str(roots))
print("Runtime:", sys.version, flush=True)
run(["nvidia-smi"])
run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
     "--target", "/tmp/receiptops-bootstrap", "virtualenv==20.39.1"])
bootstrap_environment = dict(os.environ, PYTHONPATH="/tmp/receiptops-bootstrap")
run([sys.executable, "-m", "virtualenv", "--system-site-packages",
     "/tmp/receiptops-env"], env=bootstrap_environment)
python = "/tmp/receiptops-env/bin/python"
run([python, "-m", "pip", "install", "--no-cache-dir", "-r",
     str(root / "requirements-cloud.txt")])
command = [python, str(root / "scripts/train_qwen_cloud.py"),
           "--dataset-root", str(roots[0]), "--protocol", str(root / "cloud_protocol.json"),
           "--output-dir", "/kaggle/working/receiptops-{mode}"]
if {smoke!r}:
    command.append("--smoke")
run(command, cwd=root)
print("ReceiptOps {mode} job complete", flush=True)
'''
    compile(script, "receiptops_job.py", "exec")
    metadata = {
        "id": f"{username}/receiptops-qwen3-vl-{mode}",
        "title": f"ReceiptOps Qwen3 VL {mode.title()}",
        "code_file": "receiptops_job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["urbikn/sroie-datasetv2"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "receiptops_job.py").write_text(script, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/kaggle-training")
    args = parser.parse_args()
    print(json.dumps(build(args.username, args.smoke, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
