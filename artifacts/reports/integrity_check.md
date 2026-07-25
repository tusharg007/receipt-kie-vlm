# Result Integrity Check

Generated: 2026-07-25T23:24:13.693540+00:00

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
| Artifact exists: `README.md` | PASS |  |
| Committed adapter checksum matches training metadata | PASS | 7fc369231dd37e064bb9a18777f66ba418d1dd7ae9720db71ba7d2183e6e4ba4 |
| Committed adapter size matches training metadata | PASS | 10956944 bytes |
| Committed adapter is a real safetensors binary | PASS |  |
| Final adapter is tracked by normal Git | PASS |  |
| Local training adapter matches committed adapter | PASS |  |
| A newly initialized LoRA parameter changed | PASS | steps=156 |
| Trainer state contains real optimization steps | PASS |  |
| Loss history is non-empty and finite | PASS | rows=31 |
| Base and LoRA use the same non-zero sample IDs | PASS | count=100 |
| Base metrics recompute from prediction JSONL | PASS |  |
| Lora metrics recompute from prediction JSONL | PASS |  |
| README contains current metric values | PASS | 19%, 85%, 23%, 26%, 29%, 51.74% |
| No raw dataset is tracked | PASS |  |
| No credential-like strings in tracked text | PASS |  |
| No local absolute paths in tracked text | PASS |  |

**Overall: PASS**

This check validates the committed adapter and current evidence. It does not assert production readiness.
