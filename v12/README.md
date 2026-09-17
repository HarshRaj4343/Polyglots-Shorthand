# The Polyglot's Shorthand v1.2 (work in progress)

Goal: distill a pretrained Hinglish encoder (teacher) into a <=15M-parameter PyTorch student that keeps the
v1.1 hashed word/alias/char-n-gram pieces, and test whether pretrained knowledge and more diverse data fix
the v1.1 failures. v1.1 is untouched and still runs from the repo root (`./run.sh`).

| Where | What |
|---|---|
| `polyglot12/` | shared code: hashing (v1.1 copy), teacher, student, augmentation, near-dup filter |
| `configs/` | teacher / student configs (paths relative to `v12/`) |
| `data/v11/` | EXACT v1.1 splits (digest-checked) + `test_stress` |
| `data/contrast.jsonl` | 40 hand-written diagnostic pairs, never trained on |
| `notebooks/` | self-contained GPU notebooks (Colab/Kaggle, "Run all", auto-resume) |
| `tests/` | hash-consistency test + 50 fixtures from v1.1 `model.py` |
| `results/` | all reported numbers |
| `runs/` | unpacked notebook outputs (git-ignored) |

Local environment: `uv venv --python 3.12 .venv-v12 && uv pip install --python .venv-v12/bin/python -r v12/requirements-v12.txt`

```sh
cd v12
make smoke        # CPU smoke test of every training script -> results/smoke_test.json
make bundle       # notebooks + bundle.zip for Colab/Kaggle
./run_v12.sh      # export splits, hash test, smoke test, notebooks, bundle
```
