# ReceiptKIE-VLM V2 High-Resolution Adapter

This directory contains the recommended ReceiptKIE-VLM adapter. V2 continued
training from the preserved V1 LoRA adapter; it is not a replacement for the
base SmolVLM weights.

## Inference configuration

| Setting | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Longest image edge | 1536 px |
| Maximum patch edge | 512 px |
| Image splitting | enabled |
| Maximum new tokens | 256 |
| Sampling | disabled |
| Repetition penalty | 1.08 |

The adapter uses rank-16 LoRA modules on the `q_proj`, `k_proj`, `v_proj`, and
`o_proj` projections. It was continued for three epochs and 210 optimizer steps
on the same leakage-free 563/63 train/validation split as V1. Checkpoint 210 and
the decoding policy above were selected using only the 63 validation receipts.

## Unseen-holdout result

High-resolution V2 continued adaptation achieved 99.2% valid JSON, 69.1%
company exact match, 90.7% address similarity, 85.8% date exact match, 64.6%
total exact match and 18.3% complete-record exact match on 246 previously unseen
SROIE test receipts.

Address is normalized similarity. Company, date, total, and complete-record
results are normalized exact match; complete-record exact requires all four
fields to be correct.

V2 is a research prototype, not a production financial extractor.

## Integrity

- Adapter size: 10,956,944 bytes
- SHA-256:
  `3e0e5a88c36f0d6a0db6baf2a3b521e40be4ef84b212ed2eafecab431604bf79`
- Full experiment report:
  [`HIGH_RESOLUTION_TRAINING_V2.md`](../../HIGH_RESOLUTION_TRAINING_V2.md)
- Machine-readable results:
  [`results.json`](../../artifacts/experiments/highres_training_v2/results.json)

Run the default demo with:

```bash
python scripts/demo_inference.py --model-version v2
```
