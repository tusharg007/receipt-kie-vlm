# ReceiptKIE-VLM Execution Status

Last updated: 2026-07-26 (Asia/Calcutta)

## Current phase

Leakage-free retraining, evaluation refresh, and recruiter-facing evidence
upgrade complete.

## Completed tasks

- Inspected every legacy notebook and preserved it under `research_notebooks/`.
- Sanitized a legacy Kaggle credential before baseline commit `a39fc0f`.
- Verified GitHub, Hugging Face, and Kaggle authentication.
- Rebuilt `.venv` with Python 3.12 and CUDA-enabled PyTorch 2.5.1+cu124.
- Verified CUDA, BF16, SDPA availability, and the RTX 3050 Laptop GPU with 4 GiB VRAM.
- Downloaded SROIE2019 into the ignored `data/raw/sroie/` directory.
- Validated all 973 image/entity pairs: 626 train and 347 test, no exclusions.
- Generated the environment report and complete dataset audit.
- Implemented the package, YAML configs, CLIs, tests, logging, and deterministic seeds.
- Passed a three-step real-model smoke test, including adapter save/reload and inference.
- Timed ten full gradient-accumulated calibration steps at 14.82 seconds/step.
- Created a deterministic 563/63 train/validation partition entirely within the
  official train split; the official test split is untouched until final evaluation.
- Trained a genuinely new rank-16 LoRA adapter for two epochs and 140 optimizer steps.
- Verified 28 finite training-loss records, four validation points, and a changed
  LoRA parameter checksum.
- Evaluated base and LoRA variants on the same 100 held-out test receipts.
- Generated raw predictions, JSON/CSV metrics, figures, and curated examples.
- Completed a five-condition paired robustness pilot on 20 fixed receipts.
- Rewrote the README and created the project report, interview guide, and resume bullets.
- Generated `requirements-lock.txt` from the successful environment.
- Passed 16 unit tests, Ruff, `pip check`, metric recomputation, and all integrity checks.
- Committed the minimal 10.45 MiB LoRA adapter and recruiter-ready inference demo.
- Passed a `git clone --no-local` test with a new Python 3.11 virtual environment,
  isolated model cache, CPU model loading, adapter loading, and parsed-JSON inference.

## Commands executed

- `git init`, local baseline and phase commits
- `.venv\Scripts\python.exe scripts\preflight.py`
- `.venv\Scripts\python.exe scripts\prepare_dataset.py`
- `.venv\Scripts\python.exe -m pytest -q`
- `.venv\Scripts\ruff.exe check src\receipt_kie scripts tests`
- `.venv\Scripts\python.exe scripts\run_smoke_test.py`
- `.venv\Scripts\python.exe scripts\run_training.py --config configs\train_lora.yaml`
- `.venv\Scripts\python.exe scripts\run_evaluation.py --config configs\evaluate.yaml`
- `.venv\Scripts\python.exe scripts\run_robustness.py --config configs\evaluate.yaml`
- `.venv\Scripts\python.exe scripts\build_report.py`

## Important decisions

- Preserve the MIT licence and explicitly acknowledge adapted upstream ideas.
- Train a new adapter; do not use an upstream checkpoint for reported results.
- Use SmolVLM-256M-Instruct and SROIE `company`, `address`, `date`, and `total`.
- Apply loss only to assistant JSON tokens.
- Use native BF16 LoRA; avoid mandatory bitsandbytes and FlashAttention.
- Attempt SDPA, then use logged eager fallback because this Idefics3 vision tower rejects it.
- Resize images through the supported 512-pixel longest-edge processor option.
- Train for two epochs because calibration projected completion well under 100 minutes.
- Keep model caches, raw data, corruptions, and trainer checkpoints out of Git;
  track only the minimal final adapter under `models/receipt-kie-lora/`.
- Never push to GitHub until the user gives explicit final approval.

## Errors encountered and fixes

- Broken Python 3.11 virtual environment: rebuilt with Python 3.12.13.
- An overly strict Python 3.12 package declaration blocked the first clean-clone
  verifier: confirmed the supported stack on Python 3.11 and declared Python 3.11+.
- CPU-only Windows PyTorch wheel: replaced with CUDA 12.4 build.
- System temporary drive exhausted during wheel install: redirected temporary storage.
- NumPy 2 ABI drift: pinned NumPy below 2 and compatible SciPy/scikit-learn ranges.
- System pytest temp access denied: moved pytest temp files into the project.
- Matplotlib attempted a Tk GUI: selected the headless Agg backend.
- Idefics3 vision SDPA unsupported: automatically and transparently fell back to eager.
- Initial smoke generation produced invalid JSON: accepted as a three-step quality limitation;
  full training was required before quality evaluation.
- Fifteen final LoRA predictions repeat until truncation: retained as invalid failures.

## Current training and evaluation state

- Training complete: 563 training samples, 63 validation samples, two epochs,
  140 steps, 1,084.19 seconds.
- Validation loss: 1.3650 (step 35), 1.0618 (step 70), 0.9373 (step 105),
  0.9034 (step 140).
- Peak allocated training VRAM: 827.32 MiB.
- Evaluation complete: 100 held-out receipts per variant.
- Base/LoRA valid JSON: 19% / 85%.
- Base/LoRA company accuracy: 1% / 26%.
- Base/LoRA address similarity: 2.75% / 51.92%.
- Base/LoRA date accuracy: 1% / 18%.
- Base/LoRA total accuracy: 0% / 25%.
- Complete-record normalized exact match: 0% for both.

## Remaining tasks

- No required implementation work remains for this evidence refresh.
- Future research should evaluate the full test split, add confidence intervals,
  and improve complete-record exact match beyond the current 0%.
