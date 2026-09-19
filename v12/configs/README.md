# `v12/configs/` — training run configurations

One JSON file per training run, read by the scripts in `../scripts/` and by the
generated notebooks. **All paths inside are relative to `v12/`**, not to this
folder and not to the repo root, so the notebooks resolve them identically on
Colab, Kaggle and a local CPU.

| File | Run |
|---|---|
| `teacher_v11data.json` | Phase 1 — teacher fine-tuned on the v1.1 train split only, the honest like-for-like comparison against v1.1 |
| `teacher_kd.json` | Phase 2 — teacher v2 trained on v1.1 + generated + public data, used only to produce KD logits |
| `student.json` | Phase 3 — the distilled student: architecture, optimizer, KD temperature and loss weights |

`student.json`'s `model_cfg` block is the architecture of record; the same values
are copied into `../deploy/student_kd_s42/meta.json` at export time, and the two
must agree for the deployed model to load.

To try a variant, copy a file rather than editing one in place — the committed
configs are what the reported numbers in `../results/` were produced with.
