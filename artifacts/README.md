# `artifacts/` — trained v1.1 weights and reported metrics

Everything in here is **build output**. `python3 solution.py reproduce`
regenerates all of it from `data/` in one pass; nothing here is edited by hand.

## Models

Four `.npz` archives, each a full trained model (weights, feature config,
calibration temperatures, review thresholds) loadable with `Model.load()`:

| File | Feature set |
|---|---|
| `full.npz` | words, word pairs, char n-grams, aliases, emoji context + attention branch — the v1.1 model of record |
| `char_word.npz` | words, word pairs and char n-grams (no aliases/emoji) |
| `word_only.npz` | words and word pairs only |
| `strip_symbols.npz` | same features as `full`, but emoji and punctuation stripped first |

The last three exist to support the ablation table in the whitepaper. All four
are selectable in the Streamlit app's sidebar; the app's *default* model is the
newer v1.2 student in `v12/deploy/`, not these.

## Reports

| File | Contents |
|---|---|
| `results.json` | per-model metrics on every split, error lists, benchmark timings, parameter counts, environment, and `dataset_sha256` for reproducibility |
| `audit_results.json` | metrics and errors on the separate 24-row `audit.jsonl` |
| `test_run.txt` | captured output of `python3 -m unittest test_solution.py` |

`app.py` reads `results.json` to show per-model test accuracy in the sidebar,
so deleting it degrades the UI (the app still runs).
