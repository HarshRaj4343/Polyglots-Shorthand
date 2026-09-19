# The Polyglot's Shorthand v1.2

Goal: distill a pretrained Hinglish encoder (teacher) into a <=15M-parameter PyTorch student that keeps the
v1.1 hashed word/alias/char-n-gram pieces, and test whether pretrained knowledge and more diverse data fix
the v1.1 failures. v1.1 is untouched and still runs from the repo root (`./run.sh`).

All six phases are complete (`PROGRESS.md`); the exported student in `deploy/student_kd_s42/` is what the
Streamlit app serves by default. Every folder below has its own README.

| Where | What |
|---|---|
| `polyglot12/` | shared code: hashing (v1.1 copy), teacher, student, augmentation, near-dup filter |
| `configs/` | teacher / student configs (paths relative to `v12/`) |
| `data/v11/` | EXACT v1.1 splits (digest-checked) + `test_stress` |
| `data/contrast.jsonl` | 40 hand-written diagnostic pairs, never trained on |
| `data/gen/` | hand-authored generated messages (TRAIN only) |
| `data/kd/` | augmented + teacher-labeled distillation sets (TRAIN only) |
| `notebooks/` | self-contained GPU notebooks (Colab/Kaggle, "Run all", auto-resume) |
| `tests/` | hash-consistency test + 50 fixtures from v1.1 `model.py` |
| `scripts/` | CPU-side pipeline steps: data prep, evaluation, export, bundling |
| `results/` | all reported numbers |
| `deploy/` | exported torch-free students — what `../app.py` loads in production |
| `runs/` | unpacked notebook outputs (git-ignored) |

Local environment: `uv venv --python 3.12 .venv-v12 && uv pip install --python .venv-v12/bin/python -r v12/requirements-v12.txt`

```sh
cd v12
make smoke        # CPU smoke test of every training script -> results/smoke_test.json
make bundle       # notebooks + bundle.zip for Colab/Kaggle
./run_v12.sh      # export splits, hash test, smoke test, notebooks, bundle
make deploy       # re-export deploy/student_kd_s42 from a student checkpoint
make compare      # rebuild the Phase 5 comparison table

python3 predict.py "refund kab milega bhai"
```
