# Fresh-Clone Verification

**Result: PASS**

Verified on 2026-07-26 (Asia/Calcutta) from commit
`b9cd2bd33bf74c8abef88b709c4ff7e9b6aec499`.

## Procedure

The source repository was copied as independent Git objects:

```powershell
git clone --no-local . ..\receipt-kie-vlm-fresh-clone-test
cd ..\receipt-kie-vlm-fresh-clone-test
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:HF_HOME = Join-Path (Get-Location) ".cache\huggingface"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_installation.py
.\.venv\Scripts\python.exe scripts\demo_inference.py
```

No file was manually copied from the source workspace. The clone did not use the
source virtual environment, Hugging Face cache, raw SROIE data, or Kaggle credentials.

## Environment and installation

- Python: 3.11.9
- PyTorch: 2.5.1+cpu
- CUDA: unavailable in this isolated environment; CPU fallback used
- Dependency installation: PASS in 160.336 seconds
- `pip check`: PASS, no broken requirements
- `scripts/verify_installation.py`: PASS

## Adapter verification

- Path: `models/receipt-kie-lora/adapter_model.safetensors`
- SHA-256: `7fc369231dd37e064bb9a18777f66ba418d1dd7ae9720db71ba7d2183e6e4ba4`
- Size: 10,956,944 bytes
- Storage: real safetensors binary tracked by normal Git
- PEFT configuration: loadable LoRA for `HuggingFaceTB/SmolVLM-256M-Instruct`

## Model loading and inference

- Public base-model download required: yes, on the first run
- Isolated Hugging Face cache before demo: absent
- Clone-local Hugging Face cache after demo: 493.895 MiB
- Base model loading: PASS
- Committed LoRA adapter loading: PASS
- Device: CPU
- End-to-end first-run wall time, including download and loading: 179.358 seconds
- Reported inference latency: 11.260 seconds
- Parsed JSON: PASS

The model returned:

```json
{
  "company": "12 SAMPLE ROAD FICTIONAL CITY 10000",
  "address": "12/07/2026, FICTIONAL CITY 10000",
  "date": "12/07/2026",
  "total": "15.90"
}
```

This output is valid JSON and proves end-to-end adapter inference. It does not
match every field in `assets/demo/expected_output.json`; that file describes the
fictional receipt content and is not presented as a guaranteed prediction.

## Conclusion

The repository can be cloned normally, installed in a new Python 3.11 environment,
verified without the training dataset, and used for inference with the committed
adapter. The clean-clone inference requirement passes.
