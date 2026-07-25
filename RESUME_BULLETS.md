# Resume and Application Materials

## Truthful résumé bullets

- Fine-tuned open-weight SmolVLM-256M with supervised multimodal LoRA on 626
  SROIE receipts, updating 2.73M parameters (1.052%) in 26.39 minutes on a 4 GiB
  RTX 3050 Laptop GPU.
- Built a leakage-free structured receipt KIE pipeline with assistant-only label
  masking, deterministic JSON generation, conservative field normalization, and
  auditable base-versus-LoRA evaluation across 100 held-out receipts.
- Improved valid JSON generation from 19% to 85%, company accuracy from 1% to
  23%, date accuracy from 1% to 26%, total accuracy from 0% to 29%, and address
  similarity from 2.75% to 51.74%; documented the remaining 0% complete-record
  exact-match limitation.

## Application-form description

ReceiptKIE-VLM is a reproducible vision-language research project for extracting
merchant, address, date, and total fields from receipt images. I implemented the
SROIE pairing/audit pipeline, multimodal chat formulation, assistant-only loss
masking, PEFT LoRA training, robust JSON parsing, field-aware metrics, paired
base-model evaluation, and a small corruption benchmark. I trained a new adapter
locally on an RTX 3050 and retained raw predictions, trainer state, plots, and
integrity checks so every reported metric is auditable.

## GitHub repository description

Structured receipt KIE with SmolVLM-256M + LoRA: reproducible multimodal SFT,
base-vs-adapter metrics, raw predictions, robustness pilot, tests, and reports.

## Technology keywords

Python, PyTorch, Transformers, SmolVLM, vision-language models, multimodal
learning, PEFT, LoRA, TRL SFTTrainer, supervised fine-tuning, Idefics3, SROIE,
document AI, key-information extraction, JSON generation, CUDA, BF16, gradient
checkpointing, experiment tracking, evaluation, pytest, Ruff.
