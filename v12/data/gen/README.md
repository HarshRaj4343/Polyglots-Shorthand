# `v12/data/gen/` — hand-authored generated data

Support messages written by hand to cover what v1.1's seeds underrepresented —
positive sentiment outside `feedback`, mixed-intent phrasing, and respelling
variety. **TRAIN only.**

| Path | What |
|---|---|
| `src/` | the human-editable source, one pipe-delimited line per message |
| `gen_v12.jsonl` | 1,202 generated rows — build output, do not hand-edit |

`../../scripts/build_generated.py` compiles `src/` into `gen_v12.jsonl` and
writes counts and rejections to `../../results/generated_data_stats.json`:

```sh
cd v12 && python3 scripts/build_generated.py
```

The build is not a straight conversion — it assigns group ids so paraphrases stay
together, and drops any row that is a near-duplicate (char-3-gram Jaccard > 0.6)
of a v1.1 validation, test or audit message, or of `../contrast.jsonl`. The
rejection count in the stats file is worth reading after every edit; a sharp rise
means the new lines are rewording data the model is scored on.

Edit `src/`, never `gen_v12.jsonl` — the next build overwrites it.
