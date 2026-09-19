# `v12/notebooks/` — GPU training notebooks

The three phases that need a GPU. Each notebook is self-contained and designed
for "Run all" on a free Colab or Kaggle instance, with auto-resume so a session
timeout costs nothing: re-running picks up from the last checkpoint.

| Notebook | Phase | Produces |
|---|---|---|
| `01_teacher.ipynb` | 1 | teacher fine-tuned on the v1.1 train split only — the like-for-like comparison against v1.1 |
| `02_teacher_kd.ipynb` | 2 | teacher v2 on the enlarged data, and the KD logits Phase 3 consumes |
| `03_student.ipynb` | 3 | the distilled student checkpoint that Phase 6 exports for deployment |

## These files are generated — do not edit them

`../scripts/build_notebooks.py` writes all three. Edits made in Colab or Jupyter
are overwritten the next time it runs. To change a notebook, change that script
(or the `polyglot12` module it calls into) and regenerate:

```sh
cd v12
make bundle      # regenerates notebooks/ and packs bundle.zip
```

## Workflow

1. `make bundle` → upload `../bundle.zip` to the GPU host.
2. Run the notebook; it writes checkpoints and a results JSON.
3. Download its output zip back into `../runs/` (git-ignored) and continue on CPU
   with the matching script in `../scripts/`.

Before spending GPU time, `python3 scripts/notebook_dryrun.py <notebook>` runs
the cells locally on CPU with the smoke config to catch cell-level bugs.
