# `student_kd_s42/` — the deployed v1.2 student

The distilled student (seed 42) that `app.py` serves by default. 11,878,419
parameters, exported from `runs/03_student/runs/student_kd_s42/student.pt` by
`../../scripts/export_student.py`.

Test accuracy: **intent 0.921, sentiment 0.916** (190 rows) — against 0.905 /
0.847 for the v1.1 `full.npz`.

| File | Bytes | Loaded at runtime | Role |
|---|---:|:---:|---|
| `meta.json` | 2 KB | yes | architecture, label order, calibration temperatures, review thresholds, parameter counts |
| `embeddings.npz` | 34 MB | yes | piece and position embedding tables, plus the v1.0 linear branch |
| `encoder.int8.onnx` | 3.4 MB | yes | the INT8-quantized transformer encoder |
| `encoder.onnx` | 12 MB | no | fp32 encoder, kept locally for requantizing — git-ignored |

Split this way on purpose: the embedding tables are a plain lookup that NumPy
does faster than an ONNX gather, so only the transformer stack is exported to
ONNX. Hashing, the embedding mean, the linear branch and calibration all run in
NumPy in `polyglot12/runtime.py`.

```python
import sys; sys.path.insert(0, 'v12')
from polyglot12.runtime import StudentRuntime
rt = StudentRuntime('v12/deploy/student_kd_s42', precision='int8', threads=2)
rt.predict('refund kab milega bhai')
```

`meta.json` must stay consistent with the two weight files — it carries the
label order the logits are indexed by and the temperatures the probabilities are
divided by. Replace all three together or not at all.
