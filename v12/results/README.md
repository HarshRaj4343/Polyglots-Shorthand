# `v12/results/` — every reported v1.2 number

Machine-written JSON. Each file is produced by a named script or notebook and is
the source the whitepaper and `../SECTION12_DRAFT.md` quote from; no number is
transcribed by hand. Nothing here is an input to training.

| File | Written by | Holds |
|---|---|---|
| `phase5_comparison.json` | `scripts/eval_systems.py` | the main table: v1.1, teachers and student on every split |
| `phase5_table.md` | `scripts/eval_systems.py` | the same table rendered for pasting into prose |
| `teacher_v11data.json` | `scripts/eval_teacher.py` | Phase 1 teacher, trained on v1.1 train only |
| `teacher_v2.json` | notebook 02 | teacher v2, trained on the enlarged data |
| `latency_student_kd_s42.json` | `scripts/latency.py` | deployed-student CPU latency, v1.1 benchmark protocol |
| `generated_data_stats.json` | `scripts/build_generated.py` | row/label counts and near-dup rejections for the generated set |
| `public_data_stats.json` | `scripts/build_public_data.py` | the same for the cleaned public data |
| `aug_stats.json` | `scripts/build_aug.py` | augmentation counts |
| `kd_stats.json` | `scripts/build_kd.py` | KD set sizes and teacher-logit coverage |
| `smoke_test.json` | `scripts/smoke_test.py` | last CPU smoke run, per training script |

`phase5_comparison.json` is also read at runtime: `../../app.py` pulls
`full_reports.deploy_int8.test` from it to show the student's test accuracy in
the sidebar. Keep it in the repo even though it is generated.
