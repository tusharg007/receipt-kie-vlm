# ReceiptKIE-VLM Execution Status

Last updated: 2026-07-26 (Asia/Calcutta)

## Current phase

Phase 5 — GPU smoke test and training calibration.

## Completed tasks

- Inspected the complete legacy notebook workflow and preserved it under
  `research_notebooks/`.
- Sanitized a legacy Kaggle credential before creating baseline commit `a39fc0f`.
- Verified GitHub, Hugging Face, and Kaggle authentication.
- Rebuilt `.venv` with Python 3.12 and CUDA-enabled PyTorch 2.5.1+cu124.
- Verified CUDA, BF16, SDPA, and the NVIDIA RTX 3050 Laptop GPU with 4 GiB VRAM.
- Downloaded SROIE2019 to the ignored `data/raw/sroie/` directory.
- Validated all 973 image/entity pairs: 626 train and 347 test, with no exclusions.
- Generated `artifacts/environment_report.json` and dataset audit reports.
- Added reproducible YAML configurations, package metadata, requirements, and scripts.
- Implemented structured prompts, canonical JSON targets, SROIE loading, assistant-only
  masking, LoRA model setup, deterministic inference, metrics, and evaluation reporting.
- Added parser, normalization, dataset, prompt, and configuration tests.
- Passed 12 unit tests and Ruff checks for all production Python paths.
- Passed a three-step real-model smoke test with adapter save/reload and inference.
- Timed ten full gradient-accumulated optimization steps at 14.82 seconds per step.

## Important decisions

- Preserve the MIT licence and explicitly acknowledge adapted upstream ideas.
- Train a genuinely new adapter; do not use an upstream adapter for reported results.
- Use `HuggingFaceTB/SmolVLM-256M-Instruct` and four SROIE fields:
  `company`, `address`, `date`, and `total`.
- Apply training loss only to assistant answer tokens.
- Use native BF16 LoRA and PyTorch SDPA; avoid mandatory bitsandbytes/FlashAttention.
- Resize images to a 512-pixel longest edge to fit the 4 GiB GPU.
- Train for two epochs (158 expected optimizer steps), projected at roughly 39 minutes from
  calibration.
- Keep model caches inside `.cache/huggingface/` and never commit raw data or caches.
- Do not push to GitHub until the user gives explicit final approval.

## Errors encountered and fixes

- Rebuilt a broken virtual environment that referenced a removed Python 3.11 install.
- Replaced a CPU-only PyTorch wheel with the CUDA 12.4 build.
- Redirected temporary package storage to the project drive after the system drive filled.
- Pinned NumPy below 2 and compatible SciPy/scikit-learn ranges to resolve ABI drift.
- Redirected pytest temporary files into the project after the system temp directory denied
  access.
- Excluded preserved legacy notebooks from production lint checks.

## Current training state

Smoke and calibration completed. Full two-epoch training is the next gate.

## Remaining tasks

- Train and checksum a new LoRA adapter.
- Evaluate the same held-out examples with base and LoRA models.
- Run optional robustness evaluation if the core run remains inside the time budget.
- Generate final plots, examples, documentation, integrity checks, and dependency lock.
- Create intentional local phase commits; request approval before any GitHub push.
