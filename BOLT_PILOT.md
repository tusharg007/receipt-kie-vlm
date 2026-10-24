# Bolt receipt challenger: measured development pilot

The official NVIDIA Bolt text checkpoint ran locally, but this pilot does not
justify replacing the existing V2 extractor. It evaluates a separate pipeline:
actual receipt pixels → RapidOCR 1.4.4 / ONNX Runtime 1.20.1 on CPU → Bolt text
extraction. No dataset transcripts or reference labels enter the model context.

## Fixed selection and results

The first three sorted IDs from the existing ten-image original-validation pilot
were selected before any challenger output. V2 predictions were already saved
at 1024 px. Challenger preprocessing differs. These are development examples,
not the 239-record retrospective evaluation or an untouched test set.

| Receipt ID | V2 exact fields / 4 | OCR + Bolt exact fields / 4 | Challenger outcome |
|---|---:|---:|---|
| X00016469672 | 2 | 0 | Exhausted 256-token budget; counted as invalid |
| X51005361900 | 1 | 3 | Merchant, date and total match; address spacing differs |
| X51005361946 | 0 | 1 | Total matches after currency normalization; merchant/date wrong |

Normalized macro field exact: **25.0% → 33.3%**. Complete-record exact:
**0/3 → 0/3**. Valid output: **3/3 → 2/3**. Macro field similarity fell from
72.1% to 56.8%. The small gain in exact fields therefore does not establish a
reliable overall improvement. No threshold, prompt or sample selection was
tuned to obtain a positive result.

All challenger outputs require review. The V2 quality classifier is not applied
to this backend because its feature distribution and errors differ.

## Runtime and provenance

- Repository: `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`.
- Revision: `1db54d5b12533ff1bcaa8e42cf4c667988a1f1be`.
- File: `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`, 6,180,446,208 bytes.
- Verified SHA-256: `c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612`.
- llama.cpp b11459 / `f498f864f`, Windows x64 CPU, **zero GPU layers**,
  context 2048, one slot, six threads, batch 128 / microbatch 64.
- Temperature 0, max 256 output tokens, explicit `reasoning_effort: none`,
  nested JSON Schema response format, unchanged four-string receipt schema.
- Measured server peak working set at the recorded snapshot: about 5.20 GiB.
  This excludes other processes and is not total system memory or GPU-fit proof.

Pilot request times were about 35.3, 38.4 and 15.1 seconds. Prefix caching and
an interleaved review smoke request affect these timings. Historical V2 timings
used GPU inference. These are diagnostic observations, not a controlled speed
comparison or a throughput benchmark. The local server was stopped after tests.

## Integration failure preserved

The first run followed the server README's direct `schema` example with type
`json_schema`. Inspection of b11459's parser showed that type requires the nested
`json_schema.schema` wrapper; otherwise only generic JSON was enforced. A template
keyword alone also left reasoning enabled. Both protocol settings were corrected
and the same three IDs rerun. Initial results remain in `initial_protocol/`.
Only the corrected run above is reported as the constrained pilot.

The separate bounded review agent now completes its one allowed diagnostic tool
call and a schema-valid abstention on the fictional API receipt. This tests
protocol behavior, not the quality of its judgment or verification of pixels.
Initial failed agent runs remain recorded as well.

## Inspect and reproduce

- [Final metrics](artifacts/experiments/ocr_bolt_pilot/comparison.json)
- [Every output and OCR line](artifacts/experiments/ocr_bolt_pilot/ocr_bolt_predictions.jsonl)
- [Weight verification](artifacts/experiments/receiptops/bolt_download.json)
- [Runtime snapshot](artifacts/experiments/receiptops/bolt_runtime.json)
- [Corrected live review smoke](artifacts/experiments/receiptops/bolt_live_verified_schema.json)

Install `requirements-ocr.txt`, start the verified text model in llama-server,
then run `python scripts/compare_ocr_bolt.py --base-url http://127.0.0.1:8081 --limit 3`.
Raw SROIE images are required; they are not redistributed here. A repeat run may
overwrite final outputs, so choose `--output-dir` to preserve this measurement.

NVIDIA declares text/text and a 64K model limit. This experiment used 2048 tokens
and CPU execution. Direct vision, 4GB GPU fit, 64K runtime memory, Bolt fine-tuning,
cloud deployment and fresh-data accuracy remain untested.

Primary references: [official NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB),
[Unsloth deployment guide](https://unsloth.ai/docs/models/nemotron-3.5-bolt),
[b11459 response-format parser](https://github.com/ggml-org/llama.cpp/blob/b11459/tools/server/server-common.cpp#L1210),
[RapidOCR](https://github.com/RapidAI/RapidOCR).
