# `v12/data/kd/` — distillation training sets

The Phase 3 inputs: training rows carrying **teacher-v2 logits** alongside their
gold labels, so the student learns the teacher's full distribution rather than
just the argmax. TRAIN only.

| File | Rows | Committed | Contents |
|---|---:|:---:|---|
| `aug_train.jsonl` | 2,323 | yes | v1.1 train + generated rows, plus augmented variants |
| `kd_support.jsonl` | 4,061 | yes | the above with gold labels **and** teacher logits |
| `kd_public_labeled.jsonl` | 12,105 | no | public tweets: gold sentiment, intent null, teacher logits |
| `kd_pool.jsonl` | 50,000 | no | unlabeled pool: teacher logits only |

The last two are git-ignored for size and licensing; rebuild them locally with
`scripts/build_public_data.py` then `scripts/build_kd.py`. The student trains
without them, on less data.

## How they are built

`build_aug.py` writes `aug_train.jsonl`: up to two variants per source row, each
either a v1.1 shorthand/respelling map applied per token or `polyglot12.augment`
character noise (vowel drop, keyboard swap, elongation, transposition). Variants
inherit the source labels and `group_id`, and are rejected if they introduce a
v1.1 stress-map output the source lacked, or near-duplicate (Jaccard > 0.6) any
validation/test/audit/contrast message.

`build_kd.py` then attaches teacher-v2 logits — it must run **after** notebook
02, and reads the run selected there by validation NLL.

## The intent mask

A row gets intent distillation only if it is a support row **and** the teacher's
top intent probability (T=1) is ≥ 0.6. Public and pool rows never get intent KD:
their intent labels do not exist, and an unmasked teacher guess there would
teach the student the teacher's errors. Sentiment KD applies throughout. The
resulting coverage is recorded in `../../results/kd_stats.json`.
