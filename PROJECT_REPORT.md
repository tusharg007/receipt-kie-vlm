# ReceiptKIE-VLM Experiment Report

## Executive summary

This experiment transformed a notebook-oriented full-receipt OCR repository into
a reproducible structured key-information extraction project. A fresh LoRA
adapter was trained on 563 SROIE receipt/entity pairs for the fields `company`,
`address`, `date`, and `total`, with 63 disjoint official-train receipts reserved
for validation. On 100 untouched official-test receipts, the adapter improved
valid JSON from 19% to 85%, company accuracy from 1% to 26%, date accuracy from
1% to 18%, total accuracy from 0% to 25%, and address similarity from 2.75% to
51.92%. Complete-record exact match remained 0%.

## Research question

Can parameter-efficient supervised fine-tuning turn a small general
vision-language instruction model into a structured receipt KIE model on a
4 GiB consumer GPU?

The experiment tests format following and field extraction, not unrestricted
transcription. Base and adapted models are compared on the same held-out images
and deterministic generation settings.

## Repository and data provenance

The original MIT licence is preserved. Its notebooks were inspected, sanitized
of an embedded legacy credential, and moved intact to `research_notebooks/` after
useful model-processing ideas were reimplemented and tested.

SROIE is an external dataset. The public mirror contained 626 train and 347 test
receipts. Automated discovery validated 973 image/entity pairs and found no
missing or invalid pairs. One address and one total were empty; these were
retained as empty strings rather than silently discarded.

## Task formulation

Input:

1. A receipt image.
2. A concise extraction instruction.

Target:

```json
{"company":"value","address":"value","date":"value","total":"value"}
```

Targets use stable ordering and compact JSON. The collator invokes the native
Idefics3 processor twice—prompt-only and complete conversation—to locate the
assistant boundary. Prompt, padding, and image placeholder labels are set to
`-100`, leaving only the assistant JSON under loss.

## Model and adaptation

- Base: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Processor: Idefics3, longest image edge 512 px
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Rank 16, alpha 32, dropout 0.05
- 2,727,936 trainable parameters (1.052%)
- 259,212,864 total parameters with adapters

Target modules were validated against loaded linear-module suffixes. The full
model was not trained.

## Runtime decisions

The environment used Python 3.12.13, PyTorch 2.5.1+cu124, CUDA 12.4, and an
NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM. BF16 was supported and used.
FlashAttention 2 was absent. PyTorch SDPA was attempted, but the installed
Transformers Idefics3 vision implementation rejected it; the loader logged the
reason and fell back to eager attention.

The smoke test completed three optimizer steps, produced finite loss, changed a
LoRA tensor checksum, saved and reloaded the adapter, and generated inference
without parser failure. A ten-step calibration measured 14.82 seconds per
optimization step and 827.29 MiB peak allocated VRAM.

## Training

The final run used:

- 563 training receipts from the official train split
- 63 disjoint official-train receipts for validation loss
- two epochs, 140 optimizer steps
- batch size 1, gradient accumulation 8
- AdamW fused, learning rate 2e-4, weight decay 0.01
- 5% warmup, gradient checkpointing, seed 42
- BF16 and 512 px longest-edge image processing

Training took 1,084.19 seconds (18.07 minutes) and recorded 28 finite loss
entries. Mean training loss was 1.3045. Validation loss progressed:

| Step | Epoch | Validation loss |
|---:|---:|---:|
| 35 | 0.50 | 1.3650 |
| 70 | 0.99 | 1.0618 |
| 105 | 1.49 | 0.9373 |
| 140 | 1.99 | 0.9034 |

Peak allocated GPU memory was 827.32 MiB. A saved before/after checksum verified
that the new adapter changed during this run.

## Evaluation design

The base and LoRA variants used the same 100 samples selected deterministically
from the 347-receipt test split. Generation used identical prompts, 512 px image
processing, greedy decoding, and a 128-new-token maximum.

The parser attempts direct JSON, code-fence removal, first balanced-object
extraction, trailing-comma repair, and safe Python-literal dictionaries. It never
substitutes ground truth. Unparseable outputs receive empty predictions.

Company/address normalization applies Unicode normalization, case folding,
punctuation removal, and whitespace collapse. Dates use a conservative list of
known formats. Totals remove currency/grouping symbols and quantize parseable
decimals to two places.

## Results

| Metric | Base | LoRA | Change |
|---|---:|---:|---:|
| Valid JSON | 19% | 85% | +66 pp |
| Company raw exact | 0% | 24% | +24 pp |
| Company normalized exact | 1% | 26% | +25 pp |
| Company similarity | 7.48% | 68.58% | +61.10 pp |
| Address raw exact | 0% | 1% | +1 pp |
| Address normalized exact | 0% | 3% | +3 pp |
| Address similarity | 2.75% | 51.92% | +49.17 pp |
| Date raw exact | 1% | 17% | +16 pp |
| Date normalized exact | 1% | 18% | +17 pp |
| Date similarity | 4.64% | 66.96% | +62.32 pp |
| Total raw exact | 0% | 25% | +25 pp |
| Total normalized exact | 0% | 25% | +25 pp |
| Total similarity | 1.77% | 59.10% | +57.32 pp |
| Complete-record raw exact | 0% | 0% | 0 pp |
| Complete-record normalized exact | 0% | 0% | 0 pp |

LoRA average/median latency was 7.640/7.440 seconds versus 5.895/5.458 seconds
for base. Inference peak allocated memory was 653.17 MiB for LoRA and
642.70 MiB for base.

## Failure analysis

The main result is improved structure and partial field extraction, not solved
receipt understanding. No sample had all four normalized fields exactly correct.
Long addresses amplify small character errors, making exact match severe. Fifteen
LoRA outputs were invalid; manual inspection shows repeated address fragments
that reach the generation cap. The conservative parser correctly marks these
invalid rather than inventing a closing object.

The model is also limited by its 256M parameter scale and by resizing receipts
whose original dimensions reach 4,961×7,016 pixels to a 512 px longest edge.
This configuration was necessary for a reproducible consumer-GPU experiment but
discards fine-print detail.

## Robustness pilot

Twenty fixed receipts were evaluated clean and under four mild corruptions.
Gaussian blur reduced valid JSON from 85% to 80%, company accuracy from 25% to
15%, and address similarity from 57.51% to 52.60%. Reduced brightness lowered
company/date accuracy to 10%/20%, while rotation lowered date accuracy to 15%.
JPEG quality 45 happened to score higher on several metrics in this small fixed
sample, underscoring why these values are descriptive rather than definitive.

These values are a pilot, not confidence-bounded robustness estimates.

## Conclusions and next experiments

LoRA substantially improved schema following and field-level extraction with
only 1.052% trainable parameters on a 4 GiB GPU. The zero complete-record score
shows that the project is not production ready. Highest-value follow-ups are:

1. Increase supported image resolution or use tiling/crops for fine print.
2. Add constrained JSON decoding and repetition controls.
3. Train longer or test a larger VLM with the same leakage-free split.
4. Use field-aware loss/evaluation and targeted address augmentation.
5. Evaluate all 347 test receipts with confidence intervals.
6. Test multilingual and private-receipt distributions under explicit data
   governance.

All claims in this report derive from local JSONL, JSON, CSV, and trainer-state
artifacts. `artifacts/reports/integrity_check.md` records the final verification.
