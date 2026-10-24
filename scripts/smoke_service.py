"""Exercise the real local V2 API once; saves measured results, never a throughput claim."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

from fastapi.testclient import TestClient  # noqa: E402

from receipt_kie.service import ServiceSettings, create_app  # noqa: E402


def main() -> int:
    started = time.perf_counter()
    settings = ServiceSettings(
        state_dir=ROOT / ".cache/receipt-service-smoke", api_key="smoke-only"
    )
    with TestClient(create_app(settings)) as client:
        ready = client.get("/health/ready")
        response = client.post(
            "/v1/extract",
            content=(ROOT / "assets/demo/synthetic_receipt.png").read_bytes(),
            headers={"X-API-Key": "smoke-only", "Content-Type": "image/png"},
        )
        monitoring = client.get("/v1/monitoring", headers={"X-API-Key": "smoke-only"})
        report = {
            "scope": (
                "Real V2 model, one synthetic receipt, in-process ASGI request; "
                "not a network/load benchmark"
            ),
            "readiness_status": ready.status_code,
            "http_status": response.status_code,
            "result": response.json(),
            "service_source_sha256": hashlib.sha256(
                (ROOT / "src/receipt_kie/service.py").read_bytes()
            ).hexdigest(),
            "quality_source_sha256": hashlib.sha256(
                (ROOT / "src/receipt_kie/quality.py").read_bytes()
            ).hexdigest(),
            "quality_artifact_sha256": hashlib.sha256(
                settings.quality_path.read_bytes()
            ).hexdigest(),
            "monitoring": monitoring.json(),
            "image_longest_edge": int(os.environ.get("RECEIPT_IMAGE_EDGE", "1024")),
            "total_with_startup_seconds": time.perf_counter() - started,
        }
    output = ROOT / "artifacts/experiments/receiptops/api_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
