"""Run a tiny LoRA update, reload the adapter, and perform one inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.dataset import load_records  # noqa: E402
from receipt_kie.inference import ReceiptKIEPredictor  # noqa: E402
from receipt_kie.train import run_training  # noqa: E402
from receipt_kie.utils import project_path, write_json  # noqa: E402


def main() -> None:
    config_path = "configs/smoke_test.yaml"
    config = load_config(config_path)
    summary = run_training(config_path)
    output_dir = project_path(config["paths"]["output_dir"])
    adapter_file = output_dir / "adapter_model.safetensors"
    adapter_config = output_dir / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config.is_file():
        raise RuntimeError("Smoke test did not save a reloadable PEFT adapter")
    test_record = load_records(
        config["paths"]["dataset_root"],
        "test",
        limit=1,
        seed=int(config["project"]["seed"]),
    )[0]
    predictor = ReceiptKIEPredictor(
        config["model"],
        config["paths"]["hf_cache"],
        adapter_path=output_dir,
    )
    prediction = predictor.predict(
        test_record.image_path,
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        do_sample=False,
    )
    predictor.close()
    report = {
        "training": summary,
        "adapter_exists": True,
        "adapter_reload_succeeded": True,
        "inference_succeeded": bool(prediction["raw_output"]),
        "parser_succeeded_without_crash": True,
        "prediction": prediction,
        "ground_truth": test_record.target,
        "passed": bool(
            summary["finite_loss_count"]
            and summary["lora_parameter_changed"]
            and prediction["raw_output"]
        ),
    }
    write_json("artifacts/reports/smoke_test.json", report)
    print(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise RuntimeError("Smoke test assertions failed")


if __name__ == "__main__":
    main()
