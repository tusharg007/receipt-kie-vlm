"""Single-replica receipt inference API with backpressure, quality routing and telemetry."""

from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, StrictBool
from starlette.concurrency import run_in_threadpool

from receipt_kie.monitoring import AuditStore
from receipt_kie.quality import QualityGate, quality_features
from receipt_kie.structured import strict_receipt

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServiceSettings:
    state_dir: Path = ROOT / ".cache/receipt-service"
    quality_path: Path = ROOT / "artifacts/experiments/quality_gate/quality_gate.json"
    api_key: str | None = None
    max_image_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    max_inflight: int = 1


class FieldLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: StrictBool
    address: StrictBool
    date: StrictBool
    total: StrictBool


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: uuid.UUID
    correct_fields: FieldLabels


def default_predictor() -> Any:
    # Lazy imports keep the API contract tests and operations endpoints GPU-independent.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from receipt_kie.config import load_config
    from receipt_kie.inference import ReceiptKIEPredictor

    config = load_config(ROOT / "configs/evaluate.yaml")
    edge = int(os.environ.get("RECEIPT_IMAGE_EDGE", "1024"))
    if edge <= 0:
        raise ValueError("RECEIPT_IMAGE_EDGE must be positive")
    config["model"]["image_longest_edge"] = edge
    predictor = ReceiptKIEPredictor(
        config["model"],
        os.environ["HF_HOME"],
        adapter_path=ROOT / config["paths"]["adapter_path"],
        structured_output=True,
    )

    class PersistentPredictor:
        def predict(self, path: Path) -> dict:
            return predictor.predict(
                path,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                do_sample=False,
                repetition_penalty=config["generation"]["repetition_penalty"],
            )

        def close(self) -> None:
            predictor.close()

    return PersistentPredictor()


def create_app(
    settings: ServiceSettings | None = None,
    *,
    predictor_factory: Callable[[], Any] = default_predictor,
    quality_gate: QualityGate | None = None,
    reviewer: Any = None,
) -> FastAPI:
    settings = settings or ServiceSettings(
        state_dir=Path(os.environ.get("RECEIPT_STATE_DIR", str(ROOT / ".cache/receipt-service"))),
        api_key=os.environ.get("RECEIPT_API_KEY"),
    )
    if settings.max_inflight < 1 or settings.max_image_bytes < 1 or settings.max_image_pixels < 1:
        raise ValueError("Request limits must be positive")
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    store = AuditStore(settings.state_dir / "audit.sqlite3")
    gate = quality_gate or QualityGate.load(settings.quality_path)
    slots = threading.BoundedSemaphore(settings.max_inflight)
    active_work: set[asyncio.Task] = set()
    registry = CollectorRegistry()
    requests = Counter(
        "receipt_inference_requests", "Receipt inference outcomes", ["outcome"], registry=registry
    )
    latency = Histogram(
        "receipt_inference_seconds",
        "End-to-end extraction latency",
        buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120),
        registry=registry,
    )
    feedback_counter = Counter("receipt_feedback", "New human feedback labels", registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.predictor = await run_in_threadpool(predictor_factory)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            if active_work:
                await asyncio.gather(*list(active_work), return_exceptions=True)
            close = getattr(app.state.predictor, "close", None)
            if close is not None:
                await run_in_threadpool(close)
            if reviewer is not None and getattr(reviewer, "close", None):
                reviewer.close()

    app = FastAPI(title="ReceiptOps", version="0.2.0", lifespan=lifespan)
    app.state.ready = False
    app.state.audit_store = store

    def authenticate(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and (
            not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key)
        ):
            raise HTTPException(401, "Valid X-API-Key required")

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready() -> dict:
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ready", "quality_model": gate.artifact["model_version"]}

    def infer(data: bytes, request_id: str) -> dict:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="receipt-", dir=settings.state_dir) as temporary:
            path = Path(temporary) / "input.image"
            path.write_bytes(data)
            prediction = app.state.predictor.predict(path)
        value, errors = strict_receipt(prediction.get("raw_output", ""))
        assessment = gate.assess(prediction)
        bolt_review = None
        if reviewer is not None and assessment["action"] == "review":
            # The reviewer can add reasons for review, never grant acceptance or change fields.
            bolt_review = reviewer.review(prediction)
        elapsed = time.perf_counter() - started
        store.record(request_id, assessment, quality_features(prediction), elapsed)
        requests.labels(assessment["action"]).inc()
        latency.observe(elapsed)
        return {
            "request_id": request_id,
            "prediction": value,
            "schema_errors": errors,
            **assessment,
            "latency_seconds": elapsed,
            "bolt_review": bolt_review,
        }

    @app.post("/v1/extract", dependencies=[Depends(authenticate)])
    async def extract(request: Request) -> dict:
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready")
        if not slots.acquire(blocking=False):
            requests.labels("overloaded").inc()
            raise HTTPException(429, "Inference capacity reached", headers={"Retry-After": "1"})
        deferred_release = False
        try:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > settings.max_image_bytes:
                    raise HTTPException(413, "Image exceeds request byte limit")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in ("PNG", "JPEG"):
                        raise HTTPException(415, "Only PNG and JPEG images are accepted")
                    if image.width * image.height > settings.max_image_pixels:
                        raise HTTPException(413, "Image exceeds pixel limit")
                    image.verify()
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(400, "Invalid or unsafe image") from exc
            # Keep the admission slot until the worker ends, even if the client disconnects.
            # Detached cancellation must not release GPU capacity while inference is still running.
            work = asyncio.create_task(run_in_threadpool(infer, bytes(data), str(uuid.uuid4())))
            active_work.add(work)
            work.add_done_callback(active_work.discard)
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                deferred_release = True

                def finish_background(task: asyncio.Task) -> None:
                    try:
                        task.result()
                    except BaseException:
                        requests.labels("error").inc()
                    finally:
                        slots.release()

                work.add_done_callback(finish_background)
                raise
        except HTTPException:
            raise
        except Exception as exc:
            requests.labels("error").inc()
            LOGGER.error("Receipt inference failed: %s", type(exc).__name__)
            raise HTTPException(503, "Inference unavailable; retry or request review") from exc
        finally:
            if not deferred_release:
                slots.release()

    @app.post("/v1/feedback", dependencies=[Depends(authenticate)])
    def feedback(body: Feedback) -> dict:
        try:
            status = store.feedback(str(body.request_id), body.correct_fields.model_dump())
        except KeyError as exc:
            raise HTTPException(404, "Unknown request ID") from exc
        except ValueError as exc:
            raise HTTPException(409, "Conflicting feedback already exists") from exc
        if status == "recorded":
            feedback_counter.inc()
        return {"status": status}

    @app.get("/v1/monitoring", dependencies=[Depends(authenticate)])
    def monitoring() -> dict:
        return store.summary(
            gate.artifact.get("drift_reference", {}), model_version=gate.artifact["model_version"]
        )

    @app.get("/metrics", dependencies=[Depends(authenticate)])
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("RECEIPT_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("RECEIPT_API_KEY"):
        raise RuntimeError("Set RECEIPT_API_KEY before binding a non-loopback interface")
    reviewer = None
    if os.environ.get("RECEIPT_BOLT_URL"):
        from receipt_kie.bolt import BoltReviewAgent

        reviewer = BoltReviewAgent(os.environ["RECEIPT_BOLT_URL"])
    uvicorn.run(
        create_app(reviewer=reviewer),
        host=host,
        port=int(os.environ.get("RECEIPT_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
