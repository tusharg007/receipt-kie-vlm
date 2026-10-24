# Type-safe receipt extraction

This upgrade applies TypeLLM's typed-generation idea in two ways: optional JSON
Schema token filtering for the existing SmolVLM/LoRA adapters, and a separate native
TypeLLM client for a compatible vision-language model served by SGLang. Neither
path changes the committed adapter weights.

Schema compliance and OCR accuracy are different measurements. A model can return
a perfectly typed, incorrect merchant or total. Review signals below identify some
errors; a receipt with no review signals is **not** verified correct.

## Run your existing model with constrained decoding

In your existing project environment:

```powershell
python -m pip install -r requirements-structured.txt
python scripts/demo_inference.py --structured --image path/to/receipt.jpg
```

The model retains its four-field training contract:

```json
{"company":"...","address":"...","date":"...","total":"..."}
```

All four keys are required, all values are strings, and extra keys are forbidden.
An empty string represents an unreadable/missing value. Total remains a string to
preserve the existing training labels, decimal text, and evaluation interface.
The schema deliberately leaves merchant names, addresses, date notation, and
amount notation open rather than restricting them to an invented vocabulary.

`ReceiptKIEPredictor(..., structured_output=True)` uses LM Format Enforcer's
Transformers integration to filter allowed tokens during generation. Tokenizer
indexing is reused; each prediction receives fresh parsing state. This is a local
implementation of the typed-generation methodology, **not** the TypeLLM runtime
or its single-token categorical algorithm.

After decoding, strict validation rejects malformed/incomplete objects, duplicate
keys, missing/extra keys, and non-string values. There is no repair or unconstrained
fallback in structured mode. The raw output remains available for diagnosis. A
token budget can still end generation before JSON completes: such output is
rejected, not presented as schema-guaranteed success.

Every local prediction now also reports:

- `schema_valid` and `schema_errors`, based on the actual raw JSON.
- `review_required` and `review_issues` for missing fields, invalid/unrecognized
  dates, suspicious monetary notation, or a reached output-token budget.
- `structured_output`, indicating whether decoding was constrained.

Date checks accept both day-first and month-first formats without rewriting the
printed value. Checks flag unfamiliar formats for review instead of changing
them. They do not verify image evidence or detect every plausible wrong value.
Without `--structured`, the previous resilient parser and generation behavior
remain available for reproducibility; `valid_json` retains its historical meaning
(safely parseable/repaired output), while `schema_valid` is strict.

### Limited GPU memory

The installed 4 GB GPU ran out of memory on some 1536 px inputs, including the
square synthetic demo. An explicit lower-resolution run is available:

```powershell
python scripts/demo_inference.py --structured --image-longest-edge 1024
```

The override is printed in the processor settings. It trades image detail for
memory use and does not change the default V2 configuration. A 1024 px experiment
must not be compared directly with the published 1536 px V2 scores as evidence of
an accuracy improvement.

## Native TypeLLM and SGLang

The checked TypeLLM 0.7.2 package requires Transformers >=5.0. This project pins
Transformers 4.46.3 for its reproducible SmolVLM training stack. Install the native
client in a **separate environment**:

```powershell
python -m venv .venv-typellm
.\.venv-typellm\Scripts\python.exe -m pip install -r requirements-typellm.txt
.\.venv-typellm\Scripts\python.exe scripts/typellm_inference.py --image assets/demo/synthetic_receipt.png --base-url http://127.0.0.1:30000
```

This requires an already running, compatible SGLang **vision-language** model.
Use `--model <served-checkpoint-id>` when discovery is unsuitable. The adapter
explicitly selects the supplied SGLang endpoint, even if `TYPELLM_API_KEY` exists;
there is no automatic hosted-service selection. Images are sent to that endpoint.
The client does not launch a server, download model weights, load the SmolVLM
LoRA adapter, or assume the large model in the reference demo fits this GPU.

The native workflow extracts company, address, date and total independently with
nullable types. Total uses numeric decoding; missing/unreadable values can abstain
with null instead of being forced to zero. A dependent `total_source` enum then
rechecks whether the extracted amount is the printed final total, a subtotal,
tendered cash/change, or unclear. That explicit enum uses balanced permutation
averaging and returns probabilities. A non-final/unclear source triggers review.
These probabilities describe the model's decision and are not calibrated OCR
confidence or proof of image grounding.

Optional bounded thinking is available for total extraction and its source check:

```powershell
.\.venv-typellm\Scripts\python.exe scripts/typellm_inference.py --image path/to/receipt.jpg --thinking-budget 256
```

The output includes the original `typed_result`, token/cache `usage`, and a
`parsed_prediction` compatible with the existing four-string evaluation schema.
Numeric totals gain at least two fractional digits without rounding away further
digits; null becomes an empty string. Original numeric spelling is therefore not
preserved by this projection. `raw_output` on this backend is the serialized
four-field projection, not a free-form model transcript. String transcriptions
and numeric decoding consume multiple tokens; the single-token benefit concerns
categorical decisions such as `total_source`. Prefix/KV reuse is provided by the
SGLang server configuration, not by the local Transformers constraint.

The native adapter has contract tests with a simulated client. Its request schemas
were also compiled by the actual pinned TypeLLM 0.7.2 schema compiler, with and
without thinking; see `artifacts/experiments/structured_decoding/typellm_contract.json`.
An end-to-end TypeLLM accuracy benchmark requires the separate SGLang VLM server
and has not been run in this workspace.

## Measure accuracy before adopting a new default

Compare the same V2 adapter, images, prompt, token budget and repetition penalty,
changing only token constraints:

```powershell
python scripts/compare_structured.py
```

This uses the original 63 **official-train validation** receipts, not official-test
data. It writes separate unconstrained/constrained prediction JSONL files and a
`comparison.json` under `artifacts/experiments/structured_decoding/`. The report
includes sample IDs, adapter checksum, settings, versions, strict schema rate,
field/record exact match, review rate, token-limit rate, latency, and memory.
It does not overwrite published V1/V2 benchmark files. Predictions, including
those requiring review, remain in the metric denominator.

For a smaller, explicitly lower-resolution pilot:

```powershell
python scripts/compare_structured.py --limit 10 --image-longest-edge 1024 --output-dir artifacts/experiments/structured_decoding/pilot10_1024
```

Select settings on validation data. Any later accuracy claim needs a frozen
comparison on newly held-out receipts; the old published test set has already
been used. Do not equate a higher schema-valid rate with better transcription.

### Recorded local pilot

The paired run on the first 10 sorted original validation IDs completed at 1024 px
using the V2 adapter, deterministic 256-token decoding, and repetition penalty
1.08. Both variants scored:

| Metric | Unconstrained | Constrained |
|---|---:|---:|
| Strict schema-valid output | 100% | 100% |
| Macro normalized field exact match | 42.5% | 42.5% |
| Total normalized exact match | 60% | 60% |
| Complete-record normalized exact match | 10% | 10% |
| Review-required rate | 10% | 10% |

This run demonstrated **no OCR accuracy gain**. The baseline already produced
schema-valid JSON throughout this small sample. Constraints enforce the contract
on future outputs; they cannot recover merchant names or digits the model misread.
The opt-in mode therefore remains an experiment rather than replacing the
recommended V2 default. Raw paired predictions and exact settings are in
`artifacts/experiments/structured_decoding/pilot10_1024/`.

## References

- [TypeLLM usage, image input, dependencies, and supported models](https://github.com/TypeLLM/TypeLLM)
- [TypeLLM dependency declaration](https://github.com/TypeLLM/TypeLLM/blob/main/pyproject.toml)
- [LM Format Enforcer Transformers integration and limitations](https://github.com/noamgat/lm-format-enforcer)
