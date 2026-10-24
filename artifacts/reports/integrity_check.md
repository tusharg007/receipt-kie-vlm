# Result Integrity Check

Generated: 2026-10-24T10:41:03.446472+00:00

| Check | Status | Evidence |
|---|---|---|
| Artifact exists: `models/receipt-kie-lora/adapter_model.safetensors` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/adapter_config.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/training_metadata.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/README.md` | PASS |  |
| Artifact exists: `models/receipt-kie-lora-v2-highres/adapter_model.safetensors` | PASS |  |
| Artifact exists: `models/receipt-kie-lora-v2-highres/adapter_config.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora-v2-highres/training_metadata.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora-v2-highres/README.md` | PASS |  |
| Artifact exists: `models/receipt-kie-qwen3-vl-qlora/adapter_model.safetensors` | PASS |  |
| Artifact exists: `models/receipt-kie-qwen3-vl-qlora/adapter_config.json` | PASS |  |
| Artifact exists: `models/receipt-kie-qwen3-vl-qlora/training_metadata.json` | PASS |  |
| Artifact exists: `models/receipt-kie-qwen3-vl-qlora/README.md` | PASS |  |
| Artifact exists: `models/receipt-kie-qwen3-vl-qlora/LICENSE` | PASS |  |
| Artifact exists: `QWEN3_VL_RESULTS.md` | PASS |  |
| Artifact exists: `HISTORICAL_V1_V2.md` | PASS |  |
| Artifact exists: `MODEL_COMPARISON.md` | PASS |  |
| Artifact exists: `assets/demo/synthetic_receipt.png` | PASS |  |
| Artifact exists: `assets/demo/expected_output.json` | PASS |  |
| Artifact exists: `artifacts/reports/base_metrics.json` | PASS |  |
| Artifact exists: `artifacts/reports/lora_metrics.json` | PASS |  |
| Artifact exists: `artifacts/predictions/base_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/predictions/lora_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/figures/training_loss.png` | PASS |  |
| Artifact exists: `artifacts/figures/base_vs_lora.png` | PASS |  |
| Artifact exists: `artifacts/figures/field_accuracy.png` | PASS |  |
| Artifact exists: `artifacts/figures/robustness_results.png` | PASS |  |
| Artifact exists: `artifacts/figures/qualitative_lora_improvement.png` | PASS |  |
| Artifact exists: `artifacts/figures/qualitative_failure_analysis.png` | PASS |  |
| Artifact exists: `artifacts/reports/qualitative_results_manifest.json` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/test_usage_manifest.json` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/final_holdout_ids.json` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/base_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/v1_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/v2_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/experiments/highres_training_v2/results.json` | PASS |  |
| Artifact exists: `artifacts/figures/highres_training_v2_comparison.png` | PASS |  |
| Artifact exists: `artifacts/figures/highres_training_v2_qualitative/v2_complete_record_success.png` | PASS |  |
| Artifact exists: `artifacts/figures/highres_training_v2_qualitative/v2_improvement_over_v1.png` | PASS |  |
| Artifact exists: `artifacts/figures/highres_training_v2_qualitative/v2_failure_or_regression.png` | PASS |  |
| Artifact exists: `README.md` | PASS |  |
| V1 adapter checksum matches metadata and release | PASS | 94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28 |
| V1 adapter size matches metadata and release | PASS | 10956944 bytes |
| V1 adapter is a real safetensors binary | PASS |  |
| V1 adapter is tracked by normal Git | PASS |  |
| V2 adapter checksum matches metadata and release | PASS | 3e0e5a88c36f0d6a0db6baf2a3b521e40be4ef84b212ed2eafecab431604bf79 |
| V2 adapter size matches metadata and release | PASS | 10956944 bytes |
| V2 adapter is a real safetensors binary | PASS |  |
| V2 adapter is tracked by normal Git | PASS |  |
| Qwen QLoRA adapter checksum matches metadata and release | PASS | 90583f801f2668c931732c9e3a8a25f0ee76c9c34d3c6f41441d65e66fe433a5 |
| Qwen QLoRA adapter size matches metadata and release | PASS | 69788264 bytes |
| Qwen QLoRA adapter is a real safetensors binary | PASS |  |
| Qwen QLoRA adapter is tracked by normal Git | PASS |  |
| Local training adapter matches committed adapter | PASS |  |
| A newly initialized LoRA parameter changed | PASS | steps=140 |
| Trainer state contains real optimization steps | PASS |  |
| Loss history is non-empty and finite | PASS | rows=28 |
| Validation loss history is non-empty and finite | PASS | rows=4 |
| Training and validation both derive from official train split | PASS | train=train, validation=train |
| Training and validation IDs are disjoint | PASS | train=563, validation=63 |
| Leakage-free split has expected 563/63 counts | PASS | train=563, validation=63 |
| Base and LoRA use the same non-zero sample IDs | PASS | count=100 |
| Training-time validation IDs do not overlap final test evaluation | PASS | overlap=0 |
| Base metrics recompute from prediction JSONL | PASS |  |
| Lora metrics recompute from prediction JSONL | PASS |  |
| V2 holdout has 246 unique aligned IDs | PASS | count=246 |
| V2 holdout is disjoint from every prior evaluated ID | PASS | prior=101, holdout=246 |
| V2-release Base metrics recompute from prediction JSONL | PASS |  |
| V2-release V1 metrics recompute from prediction JSONL | PASS |  |
| V2-release V2 metrics recompute from prediction JSONL | PASS |  |
| V2 paired bootstrap values match the seeded release result | PASS | 2,000 resamples; release gate PASS |
| Qualitative manifest values match prediction JSONL | PASS | improvement=X51005301666, failure=X51005433556 |
| Qualitative figure and receipt hashes match manifest | PASS |  |
| Historical archive retains V1/V2 reported metric values | PASS | 19%, 85%, 26%, 18%, 25%, 51.92%, 99.2%, 69.1%, 90.7%, 85.8%, 64.6%, 18.3% |
| README latest comparison matches verified V2/Qwen results | PASS | same 239 records; latest Qwen adapter included |
| No raw dataset is tracked | PASS |  |
| No intermediate checkpoints, caches, environments, or logs tracked | PASS |  |
| No tracked file exceeds 100 MiB | PASS |  |
| All local Markdown links and images exist | PASS | checked=122 |
| No credential-like strings in tracked text | PASS |  |
| No local absolute paths in tracked text | PASS |  |

**Overall: PASS**

This check validates the committed adapter and current evidence. It does not assert production readiness.
