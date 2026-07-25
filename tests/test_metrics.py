from receipt_kie.metrics import (
    evaluate_predictions,
    normalize_date,
    normalize_text,
    normalize_total,
)


def test_field_normalization() -> None:
    assert normalize_text("  ACME, SDN. BHD  ") == "acme sdn bhd"
    assert normalize_date("01/02/2024") == "2024-02-01"
    assert normalize_total("RM 1,234.5") == "1234.50"


def test_metric_calculation() -> None:
    rows = [
        {
            "ground_truth": {
                "company": "ACME",
                "address": "1 Road",
                "date": "01/02/2024",
                "total": "RM 12.30",
            },
            "valid_json": True,
            "parsed_prediction": {
                "company": "acme",
                "address": "1 road",
                "date": "2024-02-01",
                "total": "12.30",
            },
            "latency_seconds": 0.5,
            "peak_gpu_memory_mib": 100.0,
        }
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["sample_count"] == 1
    assert metrics["valid_json_rate"] == 1.0
    assert metrics["complete_record_normalized_exact_match"] == 1.0
    assert metrics["company_accuracy"] == 1.0
    assert metrics["peak_gpu_memory_mib"] == 100.0
