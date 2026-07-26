# Public Fresh-Clone Verification

**Result: PASS**

Verified on 2026-07-26 from a normal network clone of:

`https://github.com/tusharg007/receipt-kie-vlm.git`

The successful verification used a newly created temporary directory, a fresh
Python 3.11 virtual environment inside the clone, and a clone-local Hugging Face
cache. It did not use the source workspace, its virtual environment, its model
cache, raw SROIE data, Kaggle credentials, or untracked files.

## Repository identity

- Clone command:
  `git clone https://github.com/tusharg007/receipt-kie-vlm.git`
- Clone timestamp: `2026-07-26T02:02:09.7480430Z`
- Checked-out branch: `main`
- HEAD tested: `dbf6a3b51fec5a854777028c46b7302a9541504d`
- Required leakage-free-results commit present: yes
- Tracked working tree before and after verification: clean
- Clone source: public GitHub network remote, not local Git objects

## Adapter verification

- File: `models/receipt-kie-lora/adapter_model.safetensors`
- Size: 10,956,944 bytes
- SHA-256:
  `94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28`
- Expected size matched: yes
- Expected checksum matched: yes
- Real safetensors binary: yes
- Git LFS pointer: no
- Tracked by normal Git: yes
- PEFT configuration: LoRA for
  `HuggingFaceTB/SmolVLM-256M-Instruct`

The committed `training_metadata.json` contains:

| Field | Verified value |
|---|---:|
| `training_sample_count` | 563 |
| `validation_sample_count` | 63 |
| `official_test_used_during_training` | `false` |
| `optimizer_step_count` | 140 |
| `final_validation_loss` | 0.9034055471420288 |

## Isolated installation

- Python: 3.11.9
- PyTorch: 2.5.1+cpu
- CUDA: unavailable in the clean environment; CPU fallback used
- Virtual environment: newly created at `<ISOLATED_CLONE>/.venv`
- Hugging Face cache: `<ISOLATED_CLONE>/.cache/huggingface`
- Cache before the demo: empty
- Runtime installation from tracked `requirements.txt`: PASS
- Runtime installation time: 572.504 seconds
- `scripts/verify_installation.py`: PASS

The runtime requirements intentionally omit developer tools. The exact initial
`python -m pytest -q` command therefore reported that `pytest` was unavailable.
The publicly tracked `requirements-dev.txt`, which includes `requirements.txt`,
was then installed; this is the repository's documented setup for tests and
Ruff. No local or private dependency file was used.

The installation verifier confirmed Python 3.11 compatibility, CPU fallback,
the current adapter files and checksum, a real safetensors binary, the expected
PEFT configuration, the synthetic demo image, and that raw training data is not
required for inference.

## First-run base-model download

- Public base-model download: PASS
- Authentication requested: no
- Download location: clone-local Hugging Face cache
- Cache after demo: 517,886,727 bytes (493.895 MiB)
- First-run end-to-end wall time: 177.979 seconds

The wall time includes public model and processor download, loading the base
model and adapter, and CPU inference. The download portion was not timed
separately.

## Demo inference

Command:

```powershell
python scripts/demo_inference.py
```

Result: **PASS**

- Device: CPU
- Adapter path: `models/receipt-kie-lora`
- Adapter loaded through `PeftModel.from_pretrained`: yes
- Silent base-model fallback: no
- Synthetic image loaded: yes
- Raw model output produced: yes
- Parsed JSON produced: yes
- Reported inference latency: 8.195 seconds
- SROIE dataset required: no
- Kaggle account or credentials required: no

Raw output:

```json
{"company": "FICTIONAL CITY 10000", "address": "12 SAMPLE ROAD, FICTIONAL CITY 10000", "date": "12/07/2026", "total": "42.50"}
```

Parsed output:

```json
{
  "company": "FICTIONAL CITY 10000",
  "address": "12 SAMPLE ROAD, FICTIONAL CITY 10000",
  "date": "12/07/2026",
  "total": "42.50"
}
```

The demo always supplies the committed adapter path. Missing adapter files
return a non-zero exit code, `PeftModel.from_pretrained` is called directly,
and any adapter-loading exception returns a non-zero exit code. There is no
exception path that silently continues with the base model.

## Tests and dependency checks

After installing the tracked development requirements:

| Command | Result |
|---|---|
| `python -m pytest -q` | PASS — 16 passed in 4.43 seconds |
| `python -m ruff check src scripts tests` | PASS — all checks passed |
| `python -m pip check` | PASS — no broken requirements |

## Isolation evidence

- Original source-workspace files copied or read by the clone: none
- Existing virtual environment used: no
- Existing Hugging Face cache used: no
- Raw SROIE dataset used: no
- Kaggle account or credentials used: no
- Clone-local synthetic image used: yes
- Clone tracked files modified by verification: no

## Warnings

The successful demo emitted three non-fatal warnings:

1. `image_seq_len` is unused by the installed processor version.
2. Optional `hf_xet` is absent, so Hugging Face used regular HTTPS.
3. The Idefics3 vision tower rejected SDPA; the explicit loader retried with
   eager attention successfully.

An initial temporary clone on the nearly full system drive could not finish
installing the CPU environment. That exact failed temporary directory was
validated, removed, and replaced by a completely new normal network clone on a
drive with sufficient space. The successful verification results above come
only from the second clean clone.

## Conclusion

**PASS.** A recruiter can normally clone the current public repository, verify
the leakage-free adapter and metadata, download the public base model into an
empty local cache, and produce parsed JSON with the committed LoRA adapter on
CPU without the source workspace, SROIE, Kaggle credentials, or private files.
