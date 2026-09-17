#!/usr/bin/env sh
# v1.2 local pipeline (v1.1 stays runnable via ../run.sh). GPU steps run in notebooks/.
set -eu
cd "$(dirname "$0")"
PY="${PY:-../.venv-v12/bin/python}"
"$PY" scripts/export_v11_splits.py      # exact v1.1 splits (digest-checked)
"$PY" tests/test_hashing.py             # hash consistency vs stored fixtures
"$PY" scripts/smoke_test.py             # CPU smoke test of every training script
"$PY" scripts/build_notebooks.py
"$PY" scripts/build_bundle.py           # -> bundle.zip for Colab/Kaggle
