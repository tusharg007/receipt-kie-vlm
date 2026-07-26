# ReceiptKIE-VLM Model Versions

Both adapters are committed and independently reproducible. V2 is recommended
for inference; V1 remains available as the historical baseline.

| Property | V1 historical baseline | V2 recommended |
|---|---|---|
| Adapter path | `models/receipt-kie-lora` | `models/receipt-kie-lora-v2-highres` |
| Training origin | Fresh LoRA initialization | Continued from V1 |
| Longest image edge | 512 px | 1536 px |
| Maximum patch edge | 512 px | 512 px |
| Image splitting | enabled | enabled |
| Additional epochs | 2 original epochs | 3 continued epochs |
| Optimizer steps | 140 | 210 continued steps |
| Maximum new tokens | 128 | 256 |
| Sampling | disabled | disabled |
| Repetition penalty | none | 1.08 |
| Status | retained for reproducibility | recommended research model |

## V1

V1 established the leakage-free, low-resolution LoRA baseline. It used 512 px
training and deterministic 128-token generation. Its original 100-receipt
evaluation, predictions, plots, metadata, and limitations remain unchanged.

## V2

V2 continued from V1 using 1536 px tiled images, a 512 px maximum patch edge,
image splitting, and the same canonical four-field JSON target. Resolution,
checkpoint, and decoding policy were chosen using only the 63-receipt validation
split. The final comparison used 246 official-test receipts absent from every
prior evaluation and ablation artifact.

Use:

```bash
python scripts/demo_inference.py --model-version v2
python scripts/demo_inference.py --model-version v1
```

See [`HIGH_RESOLUTION_TRAINING_V2.md`](HIGH_RESOLUTION_TRAINING_V2.md) for the
full training, uncertainty, latency, memory, and failure analysis.
