# `v12/data/` — v1.2 datasets

**`DATA_CARD.md` in this folder is the authoritative description**: row counts,
provenance, licensing and the near-duplicate audit. This file is only a map of
the folders.

| Path | What | Split it feeds |
|---|---|---|
| `v11/` | the EXACT v1.1 splits, reused not regenerated | train / validation / test |
| `contrast.jsonl` | 40 hand-written minimal pairs (80 rows) | never trained on — diagnostic only |
| `gen/` | hand-authored generated support messages | TRAIN only |
| `kd/` | augmented and teacher-labeled distillation sets | TRAIN only |

## The rule that governs this folder

**Every v1.2 addition goes to TRAIN only.** The v1.1 validation, test, audit and
stress splits are reused byte-for-byte, and validation remains the only split
used for early stopping, selection and calibration. That is what makes the v1.2
numbers comparable to v1.1 rather than merely better-looking.

Two guards enforce it: `v11/MANIFEST.json` digest-checks the reused splits, and
every generated or augmented row is rejected if it is a near-duplicate
(char-3-gram Jaccard > 0.6) of anything in validation, test, audit or
`contrast.jsonl`.

`contrast.jsonl` was written in Phase 0, *before* any generated training data
existed, precisely so it could not be contaminated by it.

If `human_test.jsonl` is added here (fields: id, text, intent, sentiment,
group_id), every evaluation scores and reports it separately — it is the most
trustworthy test set available.
