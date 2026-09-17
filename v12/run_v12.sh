#!/usr/bin/env sh
# v1.2 local pipeline (v1.1 stays runnable via ../run.sh). GPU steps run in notebooks/ (01, 02, 03);
# each later step runs only if the notebook outputs it needs have been unpacked into runs/.
set -eu
cd "$(dirname "$0")"
PY="${PY:-../.venv-v12/bin/python}"
"$PY" scripts/export_v11_splits.py                 # exact v1.1 splits (digest-checked)
"$PY" tests/test_hashing.py                        # hash consistency vs stored fixtures
"$PY" scripts/smoke_test.py                        # CPU smoke test of every training script
"$PY" scripts/v11_logits.py                        # v1.1 scored by the shared evaluator
[ -f cache/public/sentimix/train_14k_split_conll.txt ] && "$PY" scripts/build_public_data.py
"$PY" scripts/build_generated.py
"$PY" scripts/build_aug.py
[ -f runs/01_teacher/summary.json ] && "$PY" scripts/eval_teacher.py
[ -f runs/02_teacher_kd/summary.json ] && "$PY" scripts/eval_teacher.py runs/02_teacher_kd results/teacher_v2.json && "$PY" scripts/build_kd.py
if [ -f runs/03_student/runs/student_kd_s42/student.pt ]; then
  "$PY" scripts/export_student.py --student runs/03_student/runs/student_kd_s42/student.pt --out deploy/student_kd_s42
  "$PY" scripts/latency.py deploy/student_kd_s42
fi
"$PY" scripts/eval_systems.py
"$PY" scripts/build_notebooks.py
"$PY" scripts/build_bundle.py                      # -> bundle.zip for Colab/Kaggle
