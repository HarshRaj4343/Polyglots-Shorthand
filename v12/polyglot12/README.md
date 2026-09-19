# `v12/polyglot12/` — the v1.2 Python package

Shared library code for v1.2. Every script in `../scripts/` and every generated
notebook in `../notebooks/` imports from here, which is why the package is kept
free of notebook- and path-specific logic.

| Module | Responsibility |
|---|---|
| `common.py` | labels, JSONL I/O, seeding, platform detection, checkpoint helpers |
| `hashing.py` | v1.1 hashing **copied verbatim** — see the note below |
| `teacher.py` | pretrained Hinglish encoder + 6-way intent and 3-way sentiment heads |
| `student.py` | hashed pieces → 4 pre-LN transformer blocks → attention pooling, plus the v1.0 linear branch |
| `train_student.py` | student training loop with resume |
| `augment.py` | meaning-preserving character noise for consistency training and KD |
| `neardup.py` | char-3-gram Jaccard near-duplicate filter |
| `evaluate.py` | scores any system from saved logits using the v1.1 protocol |
| `runtime.py` | torch-free inference for the exported student (NumPy + onnxruntime) |

## Two constraints worth knowing before editing

**`hashing.py` is a frozen copy, not an import.** It duplicates the v1.1 hashing
in the repo root's `model.py` so the Colab/Kaggle bundle does not have to carry
the whole repo. The copy must stay bit-identical: `../tests/test_hashing.py`
replays 50 fixtures captured from the original `model.py` and fails on any drift.
Changing the hashing here without regenerating those fixtures silently
invalidates every v1.2 number.

**`runtime.py` must not import torch.** It is what `../../app.py` loads in
production, and the deployed app installs only numpy + onnxruntime. Training
code belongs in `student.py`/`train_student.py`; inference code belongs here.
