# ReceiptKIE-VLM Execution Status

Last updated: 2026-07-26 (Asia/Calcutta)

## Current phase

V2 release promotion and public-clone verification.

## Version status

- **V1 historical baseline:** preserved at `models/receipt-kie-lora`.
- **V2 recommended model:** preserved at
  `models/receipt-kie-lora-v2-highres`.
- **Main README:** retains V1 results and adds the V2 research progression.
- **Raw dataset:** ignored and not required for either demo.

## Completed research

- Leakage-free 563/63 official-train split with zero overlap.
- V1: 512 px, two epochs, 140 optimizer steps.
- Fixed 30-receipt resolution-development ablation.
- 2048/1536 training-memory smoke tests with a predefined safety gate.
- V2: continued from V1 at 1536 px for three epochs and 210 steps.
- Validation-only checkpoint and decoding-policy selection.
- One-time Base/V1/V2 comparison on 246 previously unevaluated test receipts.
- Seeded 2,000-resample confidence intervals and three honest qualitative panels.
- Exact metric recomputation from all committed prediction JSONL files.

## Recommended V2 result

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Valid JSON | 98.4% | 99.2% | +0.8 pp |
| Company exact | 52.4% | 69.1% | +16.7 pp |
| Address similarity | 82.2% | 90.7% | +8.5 pp |
| Date exact | 63.4% | 85.8% | +22.4 pp |
| Total exact | 43.9% | 64.6% | +20.7 pp |
| Complete-record exact | 4.1% | 18.3% | +14.2 pp |
| Macro exact | 43.9% | 63.3% | +19.4 pp |

These are paired results on 246 previously unseen SROIE test receipts at the
same frozen high-resolution inference setting. Address is normalized similarity;
other reported fields are normalized exact match.

## V2 runtime

- Training resolution: 1536 px with 512 px patches and image splitting.
- Duration: 84.37 minutes.
- Peak allocated training VRAM: 3,174.26 MiB.
- Selected checkpoint: 210.
- Generation: deterministic, 256 new tokens, repetition penalty 1.08.
- V2 average/median holdout latency: 12.46/10.74 seconds.
- V2 peak inference memory: 1,876.55 MiB.

## Verification commands

```text
python -m pytest -q
python -m ruff check src scripts tests
python -m pip check
python scripts/integrity_check.py
python scripts/verify_installation.py
python scripts/demo_inference.py --model-version v1
python scripts/demo_inference.py --model-version v2
git diff --check
```

## Local verification result

- Pytest: 25 passed.
- Ruff: passed.
- `pip check`: no broken requirements.
- Integrity verification: PASS.
- Installation verification: PASS for both exact adapter checksums.
- V1 synthetic demo: parsed JSON, 9.721 s, 651.92 MiB peak, one visual tile.
- V2 synthetic demo: parsed JSON, 9.512 s, 1,877.30 MiB peak, ten visual tiles.
- Adapter loading: both versions confirmed as active PEFT models; no base-only
  fallback.

The final public-clone result and exact environment details are recorded in
`CLONE_VERIFICATION.md`.

## Limitations

- V2 is a research prototype, not a production financial extractor.
- Complete-record exact match is 18.3%, so most receipts still contain at least
  one incorrect field.
- Bootstrap intervals measure sampling uncertainty, not dataset shift.
- SROIE does not establish performance on private or multilingual receipts.
