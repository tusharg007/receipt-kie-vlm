# ReceiptKIE-VLM Interview Guide

## The project in one sentence

I built a leakage-controlled receipt KIE system with SmolVLM and versioned LoRA
adapters, then showed that validation-selected 1536 px continued adaptation
materially improved V1 on 246 previously unseen receipts.

## Research progression

### V1 baseline

V1 was trained from a fresh rank-16 LoRA initialization at 512 px for two epochs
and 140 optimizer steps. It established the assistant-only structured-JSON
pipeline and remains committed as a historical baseline.

### Resolution development

A deterministic 30-receipt development ablation showed strong gains at 2048 px.
I did not present it as untouched-test evidence. A 2048 px training smoke test
then exceeded the predefined 3.7 GiB allocated-memory threshold, while 1536 px
retained a 657 MiB margin, so full continued training used 1536 px.

### V2 continued adaptation

V2 continued from V1 for three epochs and 210 steps using 1536 px images,
512 px tiles, image splitting, and the same 563/63 training/validation IDs.
Checkpoint 210 and repetition penalty 1.08 were selected on validation only.
The final comparison used 246 test IDs absent from every prior experiment.

## Main result

| Metric | Base | V1 | V2 |
|---|---:|---:|---:|
| Valid JSON | 41.5% | 98.4% | 99.2% |
| Company exact | 8.1% | 52.4% | 69.1% |
| Address similarity | 12.2% | 82.2% | 90.7% |
| Date exact | 15.4% | 63.4% | 85.8% |
| Total exact | 0.4% | 43.9% | 64.6% |
| Complete-record exact | 0.0% | 4.1% | 18.3% |
| Macro exact | 6.2% | 43.9% | 63.3% |

Address is normalized similarity. Other reported fields are normalized exact
match. V2 improved macro exact by 19.41 percentage points, with a paired
bootstrap 95% interval of +16.26 to +22.76 points.

## Fifteen likely interview questions

### 1. Why a VLM rather than OCR plus rules?

A VLM can jointly use visual layout and language to select semantic fields.
OCR plus rules remains a strong production baseline and should be compared on
accuracy, latency, explainability, and maintenance. This project tests the
end-to-end VLM approach rather than claiming it is always preferable.

### 2. What did you train?

The base is `HuggingFaceTB/SmolVLM-256M-Instruct`. V1 trained a new PEFT LoRA;
V2 continued that adapter. The base weights remained frozen.

### 3. How did you prevent leakage?

Training and validation use only the official train split. A manifest unions
every historical test artifact. Selection used 63 validation receipts, and the
final 246 IDs were disjoint from all 101 previously evaluated test IDs.

### 4. What is assistant-only masking?

The collator processes prompt-only and full conversations, finds the assistant
boundary, and sets system, user, image, and padding labels to `-100`. Loss
therefore applies only to target JSON tokens.

### 5. Why LoRA?

Rank-16 LoRA updates 2.73 million parameters—about 1.05% of the model with
adapters—making training and versioned 10.96 MB checkpoints practical on a
4 GiB GPU.

### 6. Why 1536 px rather than 2048 px?

2048 px improved inference but allocated 3,943 MiB during the training smoke
test, above the 3.7 GiB safety gate. The 1536 px run allocated 3,132 MiB and was
about 3.5 times faster per optimizer step.

### 7. How was the V2 checkpoint selected?

Checkpoints 70, 140, and 210 were evaluated on all 63 validation receipts under
no penalty, penalty 1.08, and adaptive retry. A frozen score combined macro
exact/similarity, complete exact, valid JSON, limit hits, and repetition.
Checkpoint 210 with always-on penalty 1.08 won.

### 8. Why exact match and address similarity?

Exact match measures operational correctness. Long addresses are especially
sensitive to minor punctuation or character differences, so normalized
similarity exposes partial reading progress without relabeling it as exact.

### 9. What normalization is allowed?

Company/address normalization handles Unicode, case, punctuation, and
whitespace. Dates use explicit formats. Totals remove currency/grouping symbols
and quantize parsed decimals. Ground truth is never used to repair predictions.

### 10. What is the strongest result?

On the 246-receipt unseen holdout, V2 reached 63.3% macro exact versus 43.9% for
V1 and 18.3% complete-record exact versus 4.1%. The paired confidence intervals
for both gains exclude zero.

### 11. What is the biggest limitation?

Complete-record exact is still only 18.3%, so most receipts contain at least one
wrong field. Results are SROIE-specific and do not establish reliability under
private, multilingual, or shifted data.

### 12. What are the latency and memory costs?

At identical high-resolution inference, V1 and V2 both peaked near 1,877 MiB.
Median latency was 10.27 s for V1 and 10.74 s for V2. V2 training took 84.37
minutes and peaked at 3,174 MiB allocated.

### 13. How do you know training and evaluation are real?

Trainer history records 210 optimizer steps and six improving validation
losses. LoRA tensor hashes changed, adapters reload as active PEFT models,
metrics recompute exactly from committed JSONL, and seeded bootstrap values
reproduce exactly.

### 14. Did you hide failures?

No. Invalid outputs remain failures. The qualitative release set uses the first
holdout example satisfying each documented category and includes a V2 date
regression alongside success and improvement panels.

### 15. What would you do next?

I would compare an OCR-plus-layout KIE baseline, evaluate dataset shift and
multilingual receipts, improve constrained generation, and investigate
field-balanced or OCR-plus-KIE multi-task curricula. I would not promote the
current model to financial automation without substantially stronger
complete-record performance and external validation.

## Two-minute explanation

ReceiptKIE-VLM extracts company, address, date, and total into canonical JSON. I
started with a leakage-free 512 px LoRA baseline and retained all of its results.
A controlled resolution ablation showed that visual resolution was a primary
bottleneck, but 2048 px training exceeded a predefined memory threshold on the
4 GiB GPU. I therefore continued V1 at 1536 px with 512 px tiles.

The custom collator masks everything except assistant JSON tokens. V2 used the
same 563 training and 63 validation IDs, trained for three additional epochs,
and selected checkpoint 210 plus deterministic repetition penalty 1.08 using
validation only. A test-usage manifest identified 246 official-test receipts
absent from every earlier evaluation.

On those receipts, V2 reached 99.2% valid JSON, 69.1% company exact, 90.7%
address similarity, 85.8% date exact, 64.6% total exact, and 18.3%
complete-record exact. Macro exact improved 19.4 points over V1, with a paired
95% interval of +16.3 to +22.8. The model is still a research prototype because
complete-record exact remains low, but the experiment provides an auditable
example of validation-controlled iteration on consumer hardware.
