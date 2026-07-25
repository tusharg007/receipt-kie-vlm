# Result Integrity Check

Generated: 2026-07-25T22:36:07.586347+00:00

| Check | Status | Evidence |
|---|---|---|
| Artifact exists: `artifacts/checkpoints/receipt-kie-lora/adapter_model.safetensors` | PASS |  |
| Artifact exists: `artifacts/checkpoints/receipt-kie-lora/adapter_config.json` | PASS |  |
| Artifact exists: `artifacts/checkpoints/receipt-kie-lora/loss_history.json` | PASS |  |
| Artifact exists: `artifacts/checkpoints/receipt-kie-lora/trainer_state.json` | PASS |  |
| Artifact exists: `artifacts/logs/training.log` | PASS |  |
| Artifact exists: `artifacts/reports/base_metrics.json` | PASS |  |
| Artifact exists: `artifacts/reports/lora_metrics.json` | PASS |  |
| Artifact exists: `artifacts/predictions/base_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/predictions/lora_predictions.jsonl` | PASS |  |
| Artifact exists: `artifacts/figures/training_loss.png` | PASS |  |
| Artifact exists: `artifacts/figures/base_vs_lora.png` | PASS |  |
| Artifact exists: `artifacts/figures/field_accuracy.png` | PASS |  |
| Artifact exists: `README.md` | PASS |  |
| Adapter timestamp corresponds to the current run | PASS | adapter=2026-07-25T21:37:57.426339+00:00 |
| A newly initialized LoRA parameter changed | PASS | steps=156 |
| Trainer state contains real optimization steps | PASS |  |
| Loss history is non-empty and finite | PASS | rows=31 |
| Base and LoRA use the same non-zero sample IDs | PASS | count=100 |
| Base metrics recompute from prediction JSONL | PASS |  |
| Lora metrics recompute from prediction JSONL | PASS |  |
| README contains current metric values | PASS | 19%, 85%, 23%, 26%, 29%, 51.74% |
| No raw dataset is tracked | PASS |  |
| No credential-like strings in tracked text | PASS |  |

**Overall: PASS**

This check validates the current local run. It does not assert production readiness.
