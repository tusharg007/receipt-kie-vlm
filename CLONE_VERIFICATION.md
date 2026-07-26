# Public Fresh-Clone Verification — V2 Release

**Result: PASS**

Verified from a normal network clone of:

`https://github.com/tusharg007/receipt-kie-vlm.git`

The successful verification used a separate temporary directory, a new Python
3.11 virtual environment inside the clone, and a clone-local Hugging Face
cache. The source workspace, its virtual environment and model cache, raw
SROIE data, Kaggle credentials, and local untracked files were not used.

## Repository identity

- Clone command:
  `git clone https://github.com/tusharg007/receipt-kie-vlm.git`
- Clone timestamp: `2026-07-26T17:12:00.4739906Z`
- Checked-out branch: `main`
- Public HEAD tested:
  `550ceaa42db3d6d4e31b66fb088d6cfe51d0ed0c`
- Public `origin/main` at the end of verification:
  `550ceaa42db3d6d4e31b66fb088d6cfe51d0ed0c`
- Experiment commit:
  `7b1aa11e1e664c504d213b1577c08452d3d043e3`
- V2 training and evaluation commit:
  `82c3a6709b619da3543978282ad7b1b0953e463c`
- V2 promotion commit:
  `e38066a7abbb586ac727ea0d2705cd3de2cbe89d`
- CPU demo reporting fix:
  `550ceaa42db3d6d4e31b66fb088d6cfe51d0ed0c`
- Initial tracked working tree: clean
- Clone source: public GitHub network remote, not local Git objects
- Git submodules required: no
- Git LFS used: no
- Raw dataset files tracked: no

## Adapter verification

| Version | Adapter | Size | SHA-256 |
|---|---|---:|---|
| V1 historical baseline | `models/receipt-kie-lora/adapter_model.safetensors` | 10,956,944 bytes | `94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28` |
| V2 recommended | `models/receipt-kie-lora-v2-highres/adapter_model.safetensors` | 10,956,944 bytes | `3e0e5a88c36f0d6a0db6baf2a3b521e40be4ef84b212ed2eafecab431604bf79` |

For both adapters:

- The expected checksum and size matched.
- The file is a real safetensors binary, not a Git LFS pointer.
- The file is tracked by normal Git.
- PEFT configuration identifies LoRA for
  `HuggingFaceTB/SmolVLM-256M-Instruct`.
- The corresponding demo loaded the adapter as the active PEFT model.
- Adapter failure is fail-closed; there is no silent base-model fallback.

## Isolated installation

- Python: 3.11.9
- PyTorch: 2.5.1+cpu
- CUDA: unavailable in the clean environment; CPU fallback used
- Virtual environment: newly created at `<PUBLIC_CLONE>/.venv`
- Hugging Face cache: `<PUBLIC_CLONE>/.cache/huggingface`
- Model entries in the cache before the first demo: zero
- Installation source: only the public tracked `requirements-dev.txt` and its
  included `requirements.txt`
- Installation result: PASS
- `python scripts/verify_installation.py`: PASS

The installation verifier passed every check for Python compatibility,
dependencies, CPU fallback, both adapter binaries, exact checksums and sizes,
PEFT configuration, the synthetic demo image, and the absence of any SROIE or
Kaggle requirement for demo inference.

## First-run base-model download

- Public base-model download: PASS
- Authentication requested: no
- Download location: clone-local Hugging Face cache
- Cache after the first demo: 517,886,727 bytes (493.895 MiB)
- First-run end-to-end wall time: 196.400 seconds

The wall time includes the public model and processor download, base-model and
V2-adapter loading, and CPU inference. Download time was not measured
separately.

## V2 default demo

Command:

```powershell
python scripts/demo_inference.py
```

Result: **PASS**

- Selected version: `v2`
- Adapter: `models/receipt-kie-lora-v2-highres`
- Processor longest edge: 1536
- Patch longest edge: 512
- Image splitting: enabled
- Visual tiles: 10
- Decoding: deterministic, 256 new tokens, repetition penalty 1.08
- Device: CPU
- Reported inference latency: 16.229 seconds
- Peak GPU memory: not available on CPU
- Synthetic image loaded: yes
- Raw model output produced: yes
- Parsed four-field JSON produced: yes
- Silent base-model fallback: no

Raw and parsed output:

```json
{
  "company": "DEMO MART",
  "address": "12 SAMPLE ROAD FICTIONAL CITY 10000",
  "date": "12/07/2026",
  "total": "42.50"
}
```

## V1 optional demo

Command:

```powershell
python scripts/demo_inference.py --model-version v1
```

Result: **PASS**

- Selected version: `v1`
- Adapter: `models/receipt-kie-lora`
- Processor longest edge: 512
- Patch longest edge: 512
- Image splitting: enabled
- Visual tiles: 1
- Decoding: deterministic, 128 new tokens, no repetition penalty
- Device: CPU
- Reported inference latency: 6.908 seconds
- Peak GPU memory: not available on CPU
- Raw model output produced: yes
- Parsed four-field JSON produced: yes
- Silent base-model fallback: no

Raw and parsed output:

```json
{
  "company": "FICTIONAL CITY 10000",
  "address": "12 SAMPLE ROAD, FICTIONAL CITY 10000",
  "date": "12/07/2026",
  "total": "42.50"
}
```

The V1 output is retained as a reproducibility demonstration, not as a claim
that every field in the synthetic example is correct.

## Tests and integrity checks

| Command | Result |
|---|---|
| `python -m pytest -q` | PASS — 26 passed in 42.17 seconds |
| `python -m ruff check src scripts tests` | PASS — all checks passed |
| `python -m pip check` | PASS — no broken requirements |
| `python scripts/verify_installation.py` | PASS |
| `python scripts/integrity_check.py` | PASS |

The integrity verifier also confirmed:

- All release metrics recompute from their tracked prediction JSONL files.
- The 246-receipt V2 holdout is disjoint from every prior evaluated ID.
- The seeded 2,000-resample paired-bootstrap values match the release result.
- All 37 local Markdown links and referenced images exist.
- No raw dataset, cache, environment, checkpoint, execution log, credential
  value, or local absolute path is tracked.

Running the integrity script in a public clone regenerates its tracked report
timestamp and substitutes clone-appropriate adapter-provenance checks for
checks that can only use private trainer state. The disposable clone therefore
showed that report as modified after the run; its overall result remained PASS.

## Repository size

At the tested public commit:

- Tracked files: 179
- Total tracked size: 38,315,558 bytes (36.540 MiB)
- Git metadata size in the fresh clone: 39,959,597 bytes (38.108 MiB)
- Tracked files plus Git metadata: 78,275,155 bytes (74.649 MiB)
- Files above 25 MiB: none

Largest tracked files:

| File | Size |
|---|---:|
| `models/receipt-kie-lora/adapter_model.safetensors` | 10,956,944 bytes |
| `models/receipt-kie-lora-v2-highres/adapter_model.safetensors` | 10,956,944 bytes |
| `artifacts/predictions/examples/both_fail__X51005361908.jpg` | 1,919,659 bytes |
| `artifacts/predictions/examples/lora_failure__X51005361908.jpg` | 1,919,659 bytes |

## Isolation and security evidence

- Original source-workspace files copied or read by the clone: none
- Existing virtual environment used: no
- Existing Hugging Face cache used: no
- Raw SROIE dataset used: no
- Kaggle account or credentials used: no
- Hugging Face authentication used: no
- Clone-local synthetic image used: yes
- Actionable local-path or credential scan matches: none

The text scan found only generic programming variables named `key`; no
credential-like value was present.

## Non-fatal warnings

1. `image_seq_len` is unused by the installed processor version.
2. Optional `hf_xet` is absent, so Hugging Face used regular HTTPS.
3. The Idefics3 vision tower rejected SDPA; the explicit loader retried with
   eager attention successfully.
4. Peak GPU memory is unavailable during CPU-only verification and is reported
   explicitly as such.

An earlier attempt to build a clean environment on the nearly full system
drive failed for insufficient space. That temporary clone was removed, and the
results above come only from a new normal network clone on a drive with
sufficient space.

## Conclusion

**PASS.** An external recruiter can clone the current public repository,
install it in a fresh Python 3.11 environment, download the public base model,
and run either committed LoRA adapter on CPU. V2 is selected by default, both
adapters remain active without silent fallback, and parsed JSON is produced
without the source workspace, raw SROIE data, Kaggle credentials, Hugging Face
authentication, or private files.
