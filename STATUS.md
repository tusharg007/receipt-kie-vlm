# ReceiptKIE-VLM Execution Status

## Current update — 2026-10-24 (Asia/Calcutta)

ReceiptOps adds a class-weighted quality classifier, bounded text-only Bolt
review adapter, persistent FastAPI inference, Prometheus metrics, SQLite
feedback/export, feature PSI, input/auth/concurrency checks, CI and Docker/ECS
blueprints. Full local suite: 98 passing tests at this update. A real V2 API
request returned HTTP 200 in about 8.3 seconds at 1024 px and was routed to review.

The content audit found seven cross-official-split duplicate image groups. Seven
historical evaluation IDs are excluded for train/prior-test/repeated content,
leaving 239 retrospective, content-disjoint records. V2 macro exact is 63.1%
versus V1 43.8%. After quarantining one incomplete validation annotation, quality
AP is 0.275 versus prevalence 0.184. The candidate's 80% calibration precision
fell to 37.0% on retrospective evaluation, so promotion is disabled and every
service output requires review. See [ReceiptOps runbook](RECEIPTOPS.md).

The official Bolt Q4_K_M weights were downloaded, hash-verified and tested using
llama.cpp b11459 on CPU. The corrected schema-constrained review cycle completed
with abstention. A fixed three-bill OCR + Bolt development pilot scored 33.3%
macro exact versus saved V2's 25.0%; neither got a complete record correct and
one Bolt request exhausted its output budget. Bolt remains experimental. See
[Bolt pilot](BOLT_PILOT.md). No Bolt GPU-fit, fine-tuning or cloud claim is made.

Kaggle training update: Qwen3-VL-2B NF4 QLoRA completed three epochs (168 steps)
on a Tesla T4. Checkpoint 112 was selected by merchant-disjoint validation loss.
On 239 content-disjoint retrospective records, the adapted model scored 92.2%
macro field exact and 73.6% complete-record exact, up from 83.6%/52.3% for the
same pinned Qwen base and 63.1%/18.4% for V2. The base emitted code-fenced JSON;
strict schema validity is 0% for that base and 100% for QLoRA, while field
accuracy uses a common tolerant parser. Training peak allocated memory was
5.4 GiB. This retrospective evaluation is not fresh test data, and the API still
uses V2. See [full results](QWEN3_VL_RESULTS.md) and the
[Kaggle protocol](KAGGLE_TRAINING.md). A fresh T4 process exactly reproduced the
base and adapter raw outputs for one validation image; this is a reproducibility
check, not an accuracy estimate. See [the reload record](artifacts/experiments/qwen3_vl/full_run/export_reload_verification.json).
The earlier CI dependency failure was fixed with PyArrow 19.0.1; remote run
40070331478 passed 94 tests on commit 432d59e. Current local validation passes
98 tests, including raw-output consistency checks and portfolio outcome-count checks.

The README now includes the system design, experiment progression, implementation
map and measured tradeoffs. Its expanded gallery has 14 distinct bills covering
field corrections, unchanged outputs and regressions. CI audits the gallery's
selection, displayed values, source outputs and figure checksums without raw images.

The following July release status is archived. Its 246-ID evaluation novelty
claims are superseded by the content audit; original artifacts remain unchanged.

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
