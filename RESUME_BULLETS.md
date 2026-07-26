# Resume and Application Materials

## Recommended future résumé bullet

- Continued high-resolution LoRA adaptation of SmolVLM at 1536 px and evaluated
  on 246 previously unseen receipts; achieved 99.2% valid JSON, 69.1% company,
  85.8% date and 64.6% total exact match, 90.7% address similarity and 18.3%
  complete-record accuracy, improving macro exact match by 19.4 percentage
  points over V1.

## Supporting bullets

- Built a leakage-controlled multimodal KIE pipeline with assistant-only label
  masking, deterministic JSON generation, test-usage accounting, validation-only
  checkpoint selection, paired bootstrap intervals, and auditable predictions.
- Continued a rank-16 q/k/v/o LoRA adapter for three epochs and 210 optimizer
  steps at 1536 px on a 4 GiB RTX 3050 GPU, completing in 84.37 minutes with
  3.17 GiB peak allocated training memory.
- Preserved the original 512 px V1 baseline while improving V2 macro exact match
  from 43.9% to 63.3% and complete-record exact match from 4.1% to 18.3% under
  identical high-resolution inference controls.

## Application-form description

ReceiptKIE-VLM is a reproducible vision-language research project for extracting
company, address, date, and total fields from receipt images. I implemented
dataset auditing, multimodal assistant-only SFT, PEFT LoRA training, configurable
tiled preprocessing, validation-controlled selection, conservative parsing,
field-aware metrics, seeded confidence intervals, failure analysis, and
fresh-clone verification. V2 remains a research prototype rather than a
production financial extractor.

## GitHub repository description

Structured receipt KIE with SmolVLM-256M and versioned LoRA adapters:
leakage-controlled training, high-resolution tiling, auditable Base/V1/V2
evaluation, confidence intervals, tests, and reproducible demos.

## Technology keywords

Python, PyTorch, Transformers, SmolVLM, vision-language models, multimodal
learning, PEFT, LoRA, supervised fine-tuning, Idefics3, SROIE, document AI,
key-information extraction, JSON generation, CUDA, BF16, gradient checkpointing,
bootstrap evaluation, pytest, Ruff.
