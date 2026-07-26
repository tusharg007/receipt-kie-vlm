# ReceiptKIE-VLM Interview Guide

## The project in one sentence

I fine-tuned an open-weight SmolVLM with multimodal supervised fine-tuning and
LoRA to extract `company`, `address`, `date`, and `total` as canonical JSON from
receipt images, then compared the new adapter against the unchanged base model
on the same held-out receipts.

## Problem and architecture

The original repository focused on full-receipt transcription. I changed the
learning objective to structured key-information extraction using SROIE entity
annotations. The pipeline is:

1. Discover and validate receipt image/entity pairs.
2. Canonicalize four string fields into deterministic JSON.
3. Build an Idefics3 multimodal chat with image, instruction, and assistant
   target.
4. Mask all labels except assistant answer tokens.
5. Attach LoRA to validated attention projections in SmolVLM-256M.
6. Train a fresh adapter with BF16 and gradient checkpointing.
7. Generate deterministically with base and LoRA models on identical samples.
8. Parse conservatively and calculate raw, normalized, similarity, latency, and
   memory metrics.

## Data pipeline

The audit found 973 valid SROIE image/entity pairs: 626 train and 347 test. No
pair was excluded. One address and one total were empty and remained `""`.
Targets use the fixed key order `company`, `address`, `date`, `total`, compact
JSON separators, UTF-8, and string values.

The 626-receipt official train split is deterministically partitioned into 563
training and 63 validation receipts with zero ID overlap. The official test split
is never used during training or model selection; a fixed 100-receipt test subset
is used only for the final paired generation evaluation.

## Multimodal SFT and the collator

The processor receives a system message, a user message containing one image
and an extraction instruction, and an assistant message containing ground-truth
JSON. The custom collator processes the prompt-only and full conversations,
finds their common token prefix, and masks that prefix. It also masks padding
and image placeholder IDs.

This matters because causal-language-model loss otherwise teaches the model to
reproduce the user prompt or special tokens. The desired objective is:

```text
loss = cross_entropy(model tokens, ground-truth JSON tokens only)
```

The smoke test inspected real token counts and confirmed that assistant tokens
remain unmasked.

## LoRA and training loop

LoRA represents a weight update as a product of low-rank matrices while the
original matrix remains frozen. I used rank 16, alpha 32, dropout 0.05 on
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. The loaded architecture was inspected
first to verify these module suffixes.

Only 2,727,936 parameters were trainable—1.052% of 259,212,864 parameters with
adapters. Training used batch size 1, eight-step gradient accumulation, BF16,
gradient checkpointing, fused AdamW, 2e-4 learning rate, and seed 42.

The final two-epoch run completed 140 optimizer steps in 18.07 minutes on an
RTX 3050 Laptop GPU. Validation loss improved from 1.3650 at step 35 to 0.9034
at step 140.

## Evaluation and main results

Base and LoRA models used the same 100 receipts, prompt, preprocessing, greedy
decoding, and 128-token limit.

| Metric | Base | LoRA |
|---|---:|---:|
| Valid JSON | 19% | 85% |
| Company normalized exact match | 1% | 26% |
| Address similarity | 2.75% | 51.92% |
| Date normalized exact match | 1% | 18% |
| Total normalized exact match | 0% | 25% |
| Complete-record normalized exact match | 0% | 0% |

The correct interpretation is that LoRA learned schema following and useful
partial extraction, but the model did not solve full-record extraction.

## Failure cases and design decisions

- Fifteen LoRA outputs are invalid JSON because generation repeats long
  address-like spans until the token cap.
- Address normalized exact match is only 3%; long strings make one character error enough
  to fail exact match, so similarity is also reported.
- Downscaling to 512 px makes training easy on 4 GiB VRAM but loses fine print.
- LoRA latency is higher because outputs are longer: 7.640 s average versus
  5.895 s for base.
- Eager attention was used only after the installed Idefics3 vision
  implementation rejected SDPA.
- I did not use bitsandbytes because native BF16 LoRA fit comfortably and a
  Windows dependency was unnecessary.
- I retained invalid predictions as failures and never repaired field values
  from ground truth.

## Fifteen likely interview questions and answers

### 1. Why use a VLM instead of OCR followed by rules?

A VLM can jointly use layout, visual context, and language to select semantic
fields and serialize them. OCR-plus-rules is often faster and easier to debug,
and it may be the better production baseline. This experiment asks whether a
small open-weight VLM can learn the end-to-end structured mapping. A production
study should compare both approaches on accuracy, latency, and maintenance cost.

### 2. What exactly did you train?

I started from SmolVLM-256M-Instruct and trained a newly initialized PEFT LoRA
adapter on four SROIE entity fields. I did not train SmolVLM from scratch and did
not reuse an upstream adapter for the reported results.

### 3. Why LoRA?

LoRA reduces memory and checkpoint size by freezing base weights and learning
low-rank updates. It let me train only 1.052% of the parameters on a 4 GiB GPU
while preserving a standard base checkpoint.

### 4. How did you choose LoRA target modules?

I inspected the upstream working approach, enumerated actual linear-module
suffixes in the loaded model, and validated the requested q/k/v/o projections
before attachment. The code fails clearly if a configured target does not
exist.

### 5. How did you prevent target leakage?

Training records come only from the SROIE train split. Generation evaluation
uses the test split. Base and LoRA use identical sample IDs and settings.
Ground truth is passed only to metric calculation, never to generation or parser
repair.

### 6. Explain assistant-only label masking.

The full chat input contains system, user, image, and assistant tokens. I
process both prompt-only and full forms, find the common prefix, and set prefix,
padding, and image-token labels to `-100`. Cross-entropy then applies only to
the assistant JSON.

### 7. How did you know image tokens were not truncated?

The processor call explicitly disables truncation in the collator. Real smoke
batches had roughly 237–244 total tokens, with a 512-token configured ceiling,
and the image placeholder was present and masked. The processor itself performs
supported image resizing to a 512 px longest edge.

### 8. Why use both exact match and similarity?

Exact match measures operational correctness, but it is harsh for long
addresses. Similarity shows partial reading progress. I report raw exact,
normalized exact, and normalized similarity so improvements cannot hide behind
one forgiving metric.

### 9. What normalization is allowed?

Company/address normalization handles Unicode, case, punctuation, and
whitespace. Dates use a limited set of explicit formats. Totals remove currency
and grouping symbols and quantize safely parsed decimals. No fuzzy value is
replaced with ground truth.

### 10. What is the strongest result?

Valid JSON rose from 19% to 85%, showing that SFT strongly changed schema
following. Field metrics also rose materially: total accuracy reached 25% and
address similarity 51.92%.

### 11. What is the biggest limitation?

Complete-record normalized exact match is still 0%. The model produces useful
partial extraction but is not reliable enough for automated financial use.

### 12. Why did inference get slower after LoRA?

The adapter itself adds little compute, but the generated sequences are longer.
The base often terminates early with irrelevant output, while LoRA more often
attempts a full JSON object and sometimes repeats until the 128-token cap.

### 13. How did you validate that training really happened?

The trainer state contains 140 optimization steps and finite loss history. A
fresh LoRA tensor checksum changed. Adapter and loss artifacts have matching run
timestamps. Metrics recompute from saved JSONL predictions, and an automated
integrity report checks these conditions.

### 14. What did the robustness pilot show?

On 20 paired receipts, Gaussian blur reduced valid JSON from 85% to 80% and
company accuracy from 25% to 15%. Reduced brightness and rotation also reduced
field accuracy. The subset is too small for broad
claims, so I label it a pilot.

### 15. What would you do next?

I would test higher-resolution tiling, repetition penalties or constrained JSON
decoding, a larger open VLM, field-balanced augmentation, and the full test set
with confidence intervals. I would also build an OCR-plus-layout baseline to
quantify whether end-to-end VLM complexity is justified.

## Two-minute explanation

ReceiptKIE-VLM converts receipt images directly into JSON with company, address,
date, and total. I began with a repository focused on full OCR, preserved its
MIT licence and research notebooks, and rebuilt the project as a tested Python
package. I audited all 973 SROIE image/entity pairs and used the 626-receipt
training split for structured multimodal SFT.

The model is SmolVLM-256M-Instruct. I attached rank-16 LoRA adapters to validated
attention projections, so only 2.73 million parameters—1.052%—were trainable.
The custom collator masks the entire system/user/image prompt and computes loss
only on the assistant JSON. After a smoke test and timed calibration, I trained
on 563 receipts, validated on 63 disjoint official-train receipts, and completed
two epochs on an RTX 3050 Laptop GPU in 18.07 minutes.

For evaluation, base and LoRA models generated on the exact same 100 held-out
receipts. LoRA increased valid JSON from 19% to 85%, company accuracy from 1%
to 26%, date from 1% to 18%, total from 0% to 25%, and address similarity from
2.75% to 51.92%. The honest limitation is that complete-record exact match
remained 0%, mainly because long addresses and repetition failures are hard for
this small model at 512 px. I retained every raw prediction and added an
integrity check that recomputes metrics, so the result is reproducible and
auditable rather than a demo-only claim.

## Five-minute explanation

The problem I chose is structured receipt understanding. Traditional OCR gives
all text, but downstream systems need named values. I reformulated the original
repository's transcription objective into four-field JSON extraction using
SROIE's entity annotations.

First, I built a discovery and audit layer instead of hardcoding one dataset
path. It pairs image and entity stems, validates JSON and images, logs every
exclusion, and produces field and dimension statistics. The dataset had 973
valid pairs with 626 train and 347 test samples. I kept missing values as empty
strings and canonicalized targets with a fixed schema.

Second, I designed the multimodal SFT representation using the installed
Idefics3 chat template. The user content contains the receipt image and an
instruction to return exactly four JSON keys. The assistant target is canonical
ground truth. The subtle part is label masking: I process prompt-only and full
versions, calculate their token prefix boundary, and mask prompt, padding, and
image placeholders. That makes the loss focus only on target JSON.

Third, I loaded SmolVLM-256M-Instruct and inspected its real module names before
attaching LoRA to q, k, v, and output attention projections. Rank 16 produced
2.73 million trainable parameters, only 1.052% of the model with adapters.
Hardware preflight detected BF16 support on a 4 GiB RTX 3050. SDPA was attempted
but the Idefics3 vision tower in this Transformers version rejected it, so the
loader safely fell back to eager attention. Native BF16 fit under one GiB of
allocated training memory, so I avoided fragile Windows quantization
dependencies.

Before full training, a three-step smoke test proved forward/backward, finite
loss, adapter modification, save/reload, and inference. A ten-step benchmark
estimated runtime. The final two-epoch run used batch size one, accumulation
eight, gradient checkpointing, and 2e-4 learning rate. It trained on 563
receipts, validated on 63 disjoint official-train receipts, and completed 140
optimizer steps in 18.07 minutes. Validation loss fell from 1.3650 at step 35
to 0.9034 at step 140.

The evaluation is deliberately paired. Both variants see the same fixed 100
test receipts with greedy decoding and a 128-token cap. The parser handles
direct JSON, fences, balanced objects, trailing commas, and safe literal
dictionaries, but never replaces values from ground truth. I report raw exact,
normalized exact, and similarity.

LoRA improved valid JSON by 66 percentage points to 85%. Company accuracy rose
to 26%, date to 18%, total to 25%, and address similarity to 51.92%. Those are
real improvements, but complete-record exact match stayed at zero. Fifteen LoRA
outputs repeat address fragments until truncation, and long addresses remain
hard at 512 px. LoRA average latency is 7.640 seconds versus 5.895 seconds for
base because sequences are longer.

Finally, I ran a paired 20-image robustness pilot. Brightness reduction hurt the
most, and small rotation also reduced field accuracy. Every raw prediction,
metric file, loss record, plot, package version, and sample manifest is stored
locally. An integrity script verifies adapter timestamps and updates, recomputes
metrics from JSONL, checks README values, and scans tracked text for likely
secrets. My conclusion is that parameter-efficient SFT clearly teaches
structured behavior on consumer hardware, but accuracy is not yet sufficient
for production financial automation.
