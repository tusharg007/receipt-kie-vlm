"""Compare unconstrained and schema-constrained V2 on original validation receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.dataset import load_records  # noqa: E402
from receipt_kie.inference import ReceiptKIEPredictor  # noqa: E402
from receipt_kie.metrics import evaluate_predictions  # noqa: E402
from receipt_kie.utils import (  # noqa: E402
    package_versions,
    project_path,
    repository_relative,
    seed_everything,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/evaluate.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/receipt-kie-lora/dataset_split_manifest.json",
    )
    parser.add_argument("--limit", type=int, help="First N sorted validation IDs; default all 63.")
    parser.add_argument("--image-longest-edge", type=int, help="Explicit memory-budget override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/structured_decoding",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.image_longest_edge is not None:
        if args.image_longest_edge <= 0:
            parser.error("--image-longest-edge must be positive")
        config["model"]["image_longest_edge"] = args.image_longest_edge
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_ids = sorted(manifest["validation_ids"])
    if (
        len(validation_ids) != 63
        or len(set(validation_ids)) != 63
        or set(validation_ids) & set(manifest["train_ids"])
    ):
        raise ValueError("Expected the original disjoint 63-receipt validation split")
    if args.limit is not None:
        if not 1 <= args.limit <= len(validation_ids):
            parser.error("--limit must be between 1 and 63")
        validation_ids = validation_ids[: args.limit]
    source = load_records(config["paths"]["dataset_root"], "train")
    by_id = {record.sample_id: record for record in source}
    records = [by_id[sample_id] for sample_id in validation_ids]
    seed_everything(int(config["evaluation"]["seed"]))
    adapter = project_path(config["paths"]["adapter_path"])
    report = {
        "experiment": "structured_decoding_validation",
        "status": "started",
        "source_split": "official_train_validation",
        "sample_ids": validation_ids,
        "config": config,
        "adapter_sha256": hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "package_versions": package_versions(),
        "constraint_version": importlib.metadata.version("lm-format-enforcer"),
        "variants": {},
    }
    write_json(args.output_dir / "comparison.json", report)
    for name, structured in (("unconstrained", False), ("constrained", True)):
        predictor = ReceiptKIEPredictor(
            config["model"],
            config["paths"]["hf_cache"],
            adapter_path=adapter,
            structured_output=structured,
        )
        rows = []
        try:
            for record in records:
                prediction = predictor.predict(
                    record.image_path,
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    do_sample=bool(config["generation"]["do_sample"]),
                    repetition_penalty=config["generation"].get("repetition_penalty"),
                )
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "image_path": repository_relative(record.image_path),
                        "ground_truth": record.target,
                        **prediction,
                    }
                )
                print(f"{name}: {len(rows)}/{len(records)}", flush=True)
                write_jsonl(args.output_dir / f"{name}.jsonl", rows)
        except (Exception, KeyboardInterrupt) as exc:
            report.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                failed_variant=name,
                error=str(exc) or type(exc).__name__,
            )
            write_json(args.output_dir / "comparison.json", report)
            raise
        finally:
            predictor.close()
        metrics = evaluate_predictions(rows)
        for field in ("schema_valid", "review_required", "generation_limit_hit"):
            metrics[f"{field}_rate"] = sum(bool(row[field]) for row in rows) / len(rows)
        report["variants"][name] = metrics
        write_json(args.output_dir / "comparison.json", report)
    before = report["variants"]["unconstrained"]
    after = report["variants"]["constrained"]
    report["absolute_deltas"] = {
        key: after[key] - before[key]
        for key in (
            "schema_valid_rate",
            "macro_normalized_exact_match",
            "complete_record_normalized_exact_match",
            "total_accuracy",
        )
    }
    report["status"] = "complete"
    write_json(args.output_dir / "comparison.json", report)
    print(json.dumps(report["absolute_deltas"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
