# `v12/deploy/` — exported, production-ready students

Torch-free export targets. One subfolder per exported checkpoint, written by
`../scripts/export_student.py` (Phase 6), which quantizes to INT8, verifies the
quantized outputs against the PyTorch model and fits the calibration
temperatures stored alongside.

| Folder | Model |
|---|---|
| `student_kd_s42/` | the distilled student, seed 42 — **what the Streamlit app serves by default** |

This is the only part of `v12/` that runs in production. `../../app.py` loads it
through `polyglot12.runtime.StudentRuntime`, which needs numpy and onnxruntime
and nothing else — no torch, no transformers, no training code.

## Why some weights are committed and some are not

`.gitignore` keeps `encoder.onnx` (the fp32 export, ~12 MB) out of the repo but
**tracks** `embeddings.npz` and `encoder.int8.onnx`. Streamlit Community Cloud
deploys by cloning the repo, so the files the app actually loads have to be in
it; the fp32 encoder is never loaded at runtime and would only be repo weight.

Adding a new export? Commit its `meta.json`, `embeddings.npz` and
`encoder.int8.onnx`, and leave its `encoder.onnx` ignored.
