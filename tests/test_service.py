from __future__ import annotations

import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from receipt_kie.service import ServiceSettings, create_app


class Predictor:
    def __init__(self, *, fail=False, started=None, release=None):
        self.calls = 0
        self.closed = False
        self.fail = fail
        self.started, self.release = started, release

    def predict(self, path):
        assert path.is_file()
        self.calls += 1
        if self.started:
            self.started.set()
            assert self.release.wait(5)
        if self.fail:
            raise RuntimeError("must not expose upstream internals")
        return {
            "raw_output": (
                '{"company":"ACME","address":"1 Road","date":"01/02/2024","total":"12.50"}'
            )
        }

    def close(self):
        self.closed = True


def _image():
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    return image.getvalue()


def test_service_lifecycle_feedback_metrics_and_audit_privacy(tmp_path) -> None:
    predictor = Predictor()
    app = create_app(
        ServiceSettings(state_dir=tmp_path, api_key="test-secret"),
        predictor_factory=lambda: predictor,
    )
    headers = {"X-API-Key": "test-secret"}
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.post("/v1/extract", content=_image()).status_code == 401
        result = client.post("/v1/extract", content=_image(), headers=headers)
        assert result.status_code == 200
        payload = result.json()
        assert payload["prediction"]["total"] == "12.50"
        assert payload["action"] == "review"
        feedback = {
            "request_id": payload["request_id"],
            "correct_fields": {
                "company": True,
                "address": True,
                "date": True,
                "total": False,
            },
        }
        assert (
            client.post("/v1/feedback", json=feedback, headers=headers).json()["status"]
            == "recorded"
        )
        assert (
            client.post("/v1/feedback", json=feedback, headers=headers).json()["status"]
            == "already_recorded"
        )
        feedback["correct_fields"]["total"] = True
        assert client.post("/v1/feedback", json=feedback, headers=headers).status_code == 409
        summary = client.get("/v1/monitoring", headers=headers).json()
        assert summary["feedback_coverage"] == 1
        assert summary["labeled_complete_accuracy"] == 0
        metrics = client.get("/metrics", headers=headers).text
        assert 'receipt_inference_requests_total{outcome="review"} 1.0' in metrics
        assert "receipt_inference_seconds_count 1.0" in metrics
        with app.state.audit_store.connect() as database:
            audit = str(database.execute("SELECT * FROM predictions").fetchall())
        assert "ACME" not in audit and "1 Road" not in audit
        assert not list(tmp_path.glob("receipt-*"))
    assert predictor.calls == 1 and predictor.closed


def test_invalid_images_limits_and_unknown_feedback(tmp_path) -> None:
    predictor = Predictor()
    app = create_app(
        ServiceSettings(state_dir=tmp_path, max_image_bytes=100),
        predictor_factory=lambda: predictor,
    )
    with TestClient(app) as client:
        assert client.post("/v1/extract", content=b"bad image").status_code == 400
        assert client.post("/v1/extract", content=b"x" * 101).status_code == 413
        invalid = {
            "request_id": "00000000-0000-0000-0000-000000000000",
            "correct_fields": {
                "company": True,
                "address": True,
                "date": True,
                "total": True,
            },
        }
        assert client.post("/v1/feedback", json=invalid).status_code == 404
        invalid["correct_fields"]["total"] = "true"
        assert client.post("/v1/feedback", json=invalid).status_code == 422
    assert predictor.calls == 0


def test_backpressure_rejects_concurrent_work_without_starting_another_model(tmp_path) -> None:
    started, release = threading.Event(), threading.Event()
    predictor = Predictor(started=started, release=release)
    app = create_app(ServiceSettings(state_dir=tmp_path), predictor_factory=lambda: predictor)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(client.post, "/v1/extract", content=_image())
        try:
            assert started.wait(3)
            response = client.post("/v1/extract", content=_image())
            assert response.status_code == 429
            assert response.headers["Retry-After"] == "1"
        finally:
            release.set()
        assert work.result().status_code == 200
    assert predictor.calls == 1


def test_inference_failure_is_closed_and_capacity_is_released(tmp_path) -> None:
    predictor = Predictor(fail=True)
    app = create_app(ServiceSettings(state_dir=tmp_path), predictor_factory=lambda: predictor)
    with TestClient(app) as client:
        failed = client.post("/v1/extract", content=_image())
        assert failed.status_code == 503
        assert "upstream internals" not in failed.text
        predictor.fail = False
        assert client.post("/v1/extract", content=_image()).status_code == 200


def test_cancelled_client_does_not_release_a_running_inference_slot(tmp_path) -> None:
    async def scenario():
        started, release = threading.Event(), threading.Event()
        predictor = Predictor(started=started, release=release)
        app = create_app(ServiceSettings(state_dir=tmp_path), predictor_factory=lambda: predictor)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            task = asyncio.create_task(client.post("/v1/extract", content=_image()))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                assert (await client.post("/v1/extract", content=_image())).status_code == 429
            finally:
                release.set()
            # Lifespan shutdown must await the detached work before closing its model.
        assert predictor.closed and predictor.calls == 1

    asyncio.run(scenario())
