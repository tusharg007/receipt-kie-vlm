# ReceiptOps model versions

The **latest trained model is Qwen3-VL-2B QLoRA**. It has the best measured
extraction accuracy and a verified standalone inference CLI. V2 is the legacy
SmolVLM model currently used by the FastAPI service. Version names describe
separate saved adapters; the base model architectures are different.

## Current comparison

These are normalized metrics on the same 239 content-disjoint retrospective
receipts, using the same tolerant field parser. Strict JSON validity is counted
separately. This evaluation is not fresh test data.

| Metric | Legacy SmolVLM V2 | Qwen3-VL NF4 base | Latest Qwen3-VL QLoRA |
|---|---:|---:|---:|
| Macro field exact | 63.1% | 83.6% | **92.2%** |
| Complete-record exact | 18.4% | 52.3% | **73.6%** |
| Strict JSON validity | 98.7% | 0.0% | **100.0%** |

All Qwen base responses were recoverable after removing JSON code fences.
Qwen base and QLoRA use identical image processing, prompts, quantization and
greedy decoding. The V2 comparison uses the same receipts and reference labels
with V2's saved preprocessing and prompt. See
[the full verified report](QWEN3_VL_RESULTS.md) for per-field metrics and paired
uncertainty, and [the latest comparison images](artifacts/experiments/qwen3_vl/full_run/v2_image_comparisons/README.md).

## Saved adapters

| Property | Historical V1 | Legacy V2 | Latest Qwen QLoRA |
|---|---|---|---|
| Base | SmolVLM-256M | SmolVLM-256M | Qwen3-VL-2B-Instruct |
| Adapter path | `models/receipt-kie-lora` | `models/receipt-kie-lora-v2-highres` | `models/receipt-kie-qwen3-vl-qlora` |
| Training method | Fresh rank-16 LoRA | Continued V1 LoRA | Fresh rank-16 NF4 QLoRA |
| Training steps | 140 | 210 additional | 168 |
| Selected checkpoint | 140 | 210 | 112, validation-loss minimum |
| Preprocessing | 512 px longest edge, tiles | 1536 px longest edge, tiles | Qwen processor, at most 1,048,576 pixels |
| Training hardware | 4 GiB RTX 3050 | 4 GiB RTX 3050 | Tesla T4 |
| Inference entry | `demo_inference.py --model-version v1` | `demo_inference.py --model-version v2` | `qwen_inference.py --adapter models/receipt-kie-qwen3-vl-qlora` |
| Serving status | Historical baseline | Existing FastAPI backend | Standalone CLI; API integration pending |

All three adapters are committed. Qwen uses a separate environment with the
cloud dependency pins. Its base and processor are pinned to revision
`89644892e4d85e24eaac8bacfd4f463576704203`; the adapter package includes its
checksum, metadata and the base-model license.

## Run the latest adapter

In the separate Qwen CUDA environment described in [the README](README.md):

```bash
python scripts/qwen_inference.py --image path/to/bill.jpg --adapter models/receipt-kie-qwen3-vl-qlora --output .cache/qwen-result.json
```

For the controlled base comparison:

```bash
python scripts/qwen_inference.py --image path/to/bill.jpg --base --output .cache/qwen-base.json
```

Every Qwen result requires review. Replacing the legacy API backend also requires
integration and quality-policy calibration on Qwen outputs. The V2 quality ranker
has not been validated on Qwen. None of the versions has demonstrated production
hosting or distributed throughput.

## Earlier experiments

The historical V1 100-image and V2 246-ID results predate the exact-content audit.
Seven overlapping/repeated IDs were removed from the current comparison. The
corrected V1/V2 retrospective macro scores are 43.8%/63.1%, and complete-record
exact scores are 4.2%/18.4%.

The original plots and tables remain in [the historical archive](HISTORICAL_V1_V2.md),
[the V2 training report](HIGH_RESOLUTION_TRAINING_V2.md) and
[the resolution ablation](HIGH_RESOLUTION_ABLATION.md). They are preserved as
historical experiments and do not represent the latest trained model.
