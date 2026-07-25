from __future__ import annotations

from pathlib import Path

from receipt_kie.config import deep_merge, get_required, load_config


def test_deep_merge_and_required_lookup() -> None:
    merged = deep_merge({"model": {"id": "base", "size": 1}}, {"model": {"size": 2}})
    assert merged == {"model": {"id": "base", "size": 2}}
    assert get_required(merged, "model.id") == "base"


def test_config_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "model:\n  id: base\ntraining:\n  steps: 10\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\ntraining:\n  steps: 2\n", encoding="utf-8"
    )
    config = load_config(tmp_path / "child.yaml")
    assert config["model"]["id"] == "base"
    assert config["training"]["steps"] == 2
