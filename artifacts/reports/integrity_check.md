# Result Integrity Check

Generated: 2026-07-26T01:51:20.056755+00:00

| Check | Status | Evidence |
|---|---|---|
| Artifact exists: `models/receipt-kie-lora/adapter_model.safetensors` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/adapter_config.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/training_metadata.json` | PASS |  |
| Artifact exists: `models/receipt-kie-lora/README.md` | PASS |  |
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
| Artifact exists: `README.md` | PASS |  |
| Committed adapter checksum matches training metadata | PASS | 94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28 |
| Committed adapter size matches training metadata | PASS | 10956944 bytes |
| Committed adapter is a real safetensors binary | PASS |  |
| Final adapter is tracked by normal Git | PASS |  |
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
| Qualitative manifest values match prediction JSONL | PASS | improvement=X51005301666, failure=X51005433556 |
| Qualitative figure and receipt hashes match manifest | PASS |  |
| README contains current metric values | PASS | 19%, 85%, 26%, 18%, 25%, 51.92% |
| No raw dataset is tracked | PASS |  |
| No credential-like strings in tracked text | PASS |  |
| No local absolute paths in tracked text | PASS |  |

**Overall: PASS**

This check validates the committed adapter and current evidence. It does not assert production readiness.
