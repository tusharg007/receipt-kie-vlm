# ReceiptOps Qwen3-VL receipt adapter

This is the selected PEFT LoRA adapter for the experimental Qwen3-VL receipt
extractor. Load it with the pinned base model and the repository's
`scripts/qwen_inference.py` command. The adapter does not contain the base model
or image processor.

- Base: `Qwen/Qwen3-VL-2B-Instruct`, revision
  `89644892e4d85e24eaac8bacfd4f463576704203` (Apache-2.0).
- Selected checkpoint: step 112, selected by lowest merchant-disjoint
  validation loss during a 168-step, three-epoch run.
- Training: QLoRA rank 16, alpha 32, dropout 0.05; NF4 double quantization;
  frozen base and vision weights; one Tesla T4.
- Retrospective evaluation: 239 content-disjoint records. Normalized macro
  field exact match 92.2%; normalized complete-record exact match 73.6%; strict
  four-field JSON 239/239. The same NF4 base scored 83.6%, 52.3%, and 0/239
  strict JSON respectively; its fenced JSON was recoverable by the tolerant
  parser. This is retrospective evidence, not a fresh test.
- Training peak allocated GPU memory: 5,523 MiB. This is not a 4 GB training
  claim. Measured adapter inference averaged 10.8 seconds per image on the T4
  under the evaluated settings; this is not a scale benchmark.

The adapter is experimental. The service still defaults to the V2 model, every
result needs human review, and this evaluation does not approve deployment.
See [the experiment report](../../QWEN3_VL_RESULTS.md) and
[Kaggle protocol](../../KAGGLE_TRAINING.md). The base model's Apache-2.0 license
is included here; the repository's adapter weights are released under the
repository license.
