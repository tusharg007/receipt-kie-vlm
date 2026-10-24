# Qwen3-VL training on Kaggle

This is a separate vision-model challenger, not a replacement for the shipped V2
adapter until accuracy and operational measurements support promotion. Training
uses the official `Qwen/Qwen3-VL-2B-Instruct` vision-language checkpoint, revision
`89644892e4d85e24eaac8bacfd4f463576704203`, with NF4 QLoRA on one CUDA device.

## Protocol

The public `urbikn/sroie-datasetv2` attachment supplies images and four-field
annotations. Exact-content copies are deduplicated before training. Conflicting
duplicate annotations and incomplete annotations are quarantined. Normalized
merchant names define groups; a seeded 15% group holdout supplies validation.
On the current data this yields **445 fit / 174 validation / 239 evaluation**
records. Validation is approximately 15% of groups, not necessarily 15% of images.

No official-test image enters training or checkpoint selection. The 239-record
evaluation uses the content-audited historical subset and remains retrospective.
It is not a fresh test set. Near duplicates, spelling variants of merchant names,
and possible pretraining exposure remain limitations.

Declared first-run settings: rank 16 / alpha 32 / dropout 0.05, language-model
attention and MLP projection adapters, frozen base and visual weights, NF4 double
quantization, three epochs, learning rate 1e-4, batch one, accumulation eight,
gradient checkpointing, at most 1,048,576 image pixels and 2,048 sequence tokens.
FP16 is used on T4; BF16 is used only when supported by the actual device.

Assistant-only loss masks the complete image/user prompt and visual tokens.
Token overflow fails explicitly; it does not silently truncate a target or image.
Checkpoints are selected by minimum merchant-disjoint validation loss. The chosen
adapter and the same quantized base model are evaluated with identical prompts,
pixels, greedy decoding, a 256-token budget, and strict four-string parsing.
Every failed or invalid prediction remains in the metric denominator.

Evaluation image bytes must match the published SHA-256 manifest. Before any
evaluation predictions, the adapter/base selection is frozen using validation
macro exact with no complete-record exact regression; both evaluation results
are still reported. The default service is not promoted by this selection.

Two T4 GPUs do not form one shared VRAM pool. This protocol exposes only device
zero; it is a single-GPU experiment, not distributed training.

## Build and run

Use Kaggle's authenticated CLI. Generated bundles are private by default and
contain only the source allowlist in `scripts/build_kaggle_training.py`, a public
dataset reference, and the evaluation ID manifest. Local credentials, private
documents, images and model downloads are not bundled.

```powershell
python scripts/build_kaggle_training.py --username YOUR_USERNAME --smoke --output-dir .cache/kaggle-training/smoke
python -m kaggle kernels push -p .cache/kaggle-training/smoke --accelerator NvidiaTeslaT4 -t 1200
python -m kaggle kernels status YOUR_USERNAME/receiptops-qwen3-vl-smoke
```

The smoke test must produce finite losses and a nonzero LoRA parameter change.
It records actual GPU, software versions, peak memory, source hashes and split
manifest. A submitted or running kernel is not evidence of successful training.
Inspect `training_summary.json` before proceeding.

```powershell
python scripts/build_kaggle_training.py --username YOUR_USERNAME --output-dir .cache/kaggle-training/train
python -m kaggle kernels push -p .cache/kaggle-training/train --accelerator NvidiaTeslaT4 -t 10800
python -m kaggle kernels status YOUR_USERNAME/receiptops-qwen3-vl-train
python -m kaggle kernels output YOUR_USERNAME/receiptops-qwen3-vl-train -p .cache/kaggle-training/download
```

Check the account's actual quota first using `python -m kaggle quota`. Availability
and weekly limits depend on the account and platform allocation. The timeout is
a resource bound, not a prediction of how long training takes.

## Environment and artifacts

Kaggle's current image uses Python 3.13. Its `ensurepip` bootstrap failed in the
first job, so the launcher uses `virtualenv` with system packages and installs
the cloud dependency pins in that isolated environment. Kaggle's matched CUDA
Torch/torchvision pair is retained and recorded. Transformers 4.57.6 and the cloud
PEFT versions are not installed into the local V2 environment.

The first unsuccessful bootstrap is preserved in the private kernel's version
history. A successful smoke does not establish trained extraction accuracy.
Full-run outputs include the selected adapter, training summary, partition
manifest, validation metrics, paired base/adapted predictions and comparison
metrics. Actual results belong under `artifacts/experiments/qwen3_vl/` only after
retrieval and verification. Keep optimizer states and weight caches out of Git.

The original FastAPI service continues to use V2. Any Qwen promotion also needs
an inference integration, memory/latency checks, a new quality-model calibration,
and a documented evaluation decision. This package does not silently swap models.

Primary references: [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
[Qwen fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune),
[Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks),
[Kaggle CLI kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
