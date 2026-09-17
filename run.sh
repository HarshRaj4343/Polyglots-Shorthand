#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" solution.py reproduce
"$PYTHON" audit.py
"$PYTHON" -m unittest -v test_solution.py
