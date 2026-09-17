# v1.2 data card

Human-collected test set: if `data/human_test.jsonl` is added (fields id, text, intent, sentiment, group_id), every
evaluation also scores it and reports it separately; it is the most trustworthy test set.

All v1.2 additions go to TRAIN only. The v1.1 train/validation/test/audit splits and group ids are reused
exactly (see `v11/MANIFEST.json`; source digests match `artifacts/results.json`). Validation is the only split
used for early stopping, selection and calibration.

## v1.1 splits (`data/v11/`)
| File | Rows | Groups | Source | Use |
|---|---:|---:|---|---|
| train.jsonl | 536 | 130 | LLM-authored seeds + rule augmentation (v1.1 `make_data.py`) | training |
| validation.jsonl | 158 | 39 | same | early stopping, selection, calibration |
| test.jsonl | 190 | 44 | same | development test (report only) |
| test_stress.jsonl | 190 | 44 | test with the v1.1 14-entry stress map | robustness (report only) |
| audit.jsonl | 24 | 24 | separately authored v1.1 audit | report only |

Note: 5 of the 536 v1.1 train rows already have char-3-gram Jaccard > 0.6 with a validation/test/audit message
(v1.1 split by paraphrase group, not by surface similarity). They are left unchanged because the splits must
be reused exactly; new v1.2 data is filtered against validation/test/audit/contrast.

## Diagnostic contrast set (`data/contrast.jsonl`)
- 40 hand-written minimal pairs (80 rows), written in Phase 0 BEFORE any generated training data, never used
  for training, tuning, checkpointing or calibration.
- Phenomena (10 pairs each): `negation_scope`, `neutral_nahi`, `negated_negative`, `unseen_affect`.
- Max char-3-gram Jaccard to any v1.1 train/val/test/audit message: 0.37.
- All `unseen_affect` affect words (raahat, tang, sukoon, nirash, chinta, dukhi, chidh, fikar, khinn, pareshan,
  ...) are absent from the v1.1 training split (checked on normalized tokens).
- Caveat: written by the same LLM assistant that wrote most training data.

## Public data (Phase 2a)
Built by `scripts/build_public_data.py` (deterministic, seed 42) from downloads in `v12/cache/` (git-ignored). The output
JSONL files are **git-ignored** (Twitter content, non-commercial / unclear licenses) but go into the private notebook bundle.
Stats: `results/public_data_stats.json`.

### Used
| Source | ID / URL | License | Raw size | Label mapping | Use |
|---|---|---|---:|---|---|
| SemEval-2020 Task 9 SentiMix (Hinglish) | HF `RTT1/SentiMix` (re-upload of the official train_14k / dev_3k / test + test labels, CoNLL token format) | Tagged `openrail` by the uploader; the original task data are research-use Twitter data. **Unclear**: research use only, not redistributed | 20,000 tweets | positive/negative/neutral -> same | labeled sentiment, intent loss masked; KD pool |
| Code-mixed tweets sentiment | HF `Abhishek4896/hindi-english-code-mixed-tweets-sentiment` | MIT | 498 (heavily templated, 385 exact duplicates) | same | labeled sentiment, intent masked |
| PHINC | HF `LingoIITGN/PHINC` (official; `veezbo/phinc` is an identical copy) | CC-BY-4.0 | 13,738 tweets (+ English translations, unused) | none | unlabeled KD pool (sentiment KD only) |
| L3Cube-HingLID | github.com/l3cube-pune/code-mixed-nlp `L3Cube-HingLID/{train,validation,test}.txt` (token/LID format) | CC-BY-NC-SA-4.0 | 44,455 tweets | LID tags dropped | unlabeled KD pool |
| CMU Hinglish DoG | HF `festvox/cmu_hinglish_dog` (`translation.hi_en`) | CC-BY-SA-3.0 / GFDL | 9,962 utterances | none | unlabeled KD pool |

Cleaning: repair cp1252 mojibake (drop if unrepairable), remove @mentions, URLs, `t.co` fragments, `#` signs, `RT`;
keep only Latin-script letters (emoji/punctuation allowed); 3-60 words, <=512 code points; require >=2 distinct Romanized
Hindi function words (English-ambiguous `the/to/me/do/hi/ho/na` excluded); exact-duplicate removal on lowercased word
sequences; near-duplicate removal (char-3-gram Jaccard > 0.6) against v1.1 validation/test/audit and the contrast set.

| Source | Filtering |
|---|---|
| SentiMix | 20,000 raw -> 12,080 kept (not Hinglish 6,987; non-Latin 123; length 126; empty/unrepairable 236; exact dup 448; near-dup of protected 0) |
| Abhishek tweets | 498 raw -> 25 kept (not Hinglish 88; non-Latin 0; length 0; empty/unrepairable 0; exact dup 385; near-dup of protected 0) |
| PHINC | 13,738 raw -> 9,561 kept (not Hinglish 3,886; non-Latin 19; length 79; empty/unrepairable 114; exact dup 79; near-dup of protected 0) |
| HingLID | 44,455 raw -> 42,543 kept (not Hinglish 922; non-Latin 0; length 475; empty/unrepairable 0; exact dup 515; near-dup of protected 0) |
| CMU DoG | 9,962 raw -> 7,146 kept (not Hinglish 1,419; non-Latin 3; length 1,092; empty/unrepairable 0; exact dup 302; near-dup of protected 0) |

**Near-duplicates of protected messages removed: 0** (tweets/chit-chat are far from support messages).

Outputs:
- `data/public/sentiment_labeled.jsonl`: 12,105 rows (positive 2,784, negative 4,777, neutral 4,544); sha256 683b70ca5c7fa754...
- `data/public/unlabeled_pool.jsonl`: 50,000 rows, capped at 50,000 by seeded sampling (cmu_hinglish_dog 6,043, hinglid 35,957, phinc 8,000); sha256 dab576ddd8e515d4...

Observations from 20 samples per source: SentiMix is political/cricket Twitter with frequent abuse and visible label
noise (e.g. "Bht acha kiya ... kr dena chahiye tha" labelled neutral); HingLID is lowercased without punctuation and
contains some Gujarati/Punjabi romanized tweets; CMU DoG is movie chit-chat, partly formal transliteration; PHINC is
casual tweets. **All public data is out of domain** (not customer support) and is used only for sentiment.

### Checked and not used
| Candidate | Reason |
|---|---|
| `md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset` (100k, CC-BY-NC-ND-4.0) | Loads, but it is machine-translated Amazon reviews mixing Devanagari, Bengali script and English - not Romanized Hinglish; ND license |
| L3Cube-HingCorpus (52.93M tweets, CC-BY-NC-SA-4.0) | Single multi-GB Google Drive file; not needed for a 50k pool; HingLID (same L3Cube Twitter source) used instead |
| L3Cube labeled Hinglish sentiment | None published in `l3cube-pune/code-mixed-nlp` or on the l3cube-pune HF organisation |
| `ujs/hinglish` | Speech dataset (audio) |
| `adealvii/codemixed-synthetic-sarc-11k` | Indonesian-English |
| `sudipghosh1728/Code_Mixed_video_Complaint` | YouTube video-level complaint metadata, not message text |
| `findnitai/english-to-hinglish`, `rvv-karma/English-Hinglish` | Machine/LLM translations of English corpora; not needed given real tweets |
| Hinglish customer-support INTENT data | None found on HF (searched "hinglish", "code-mixed", "codemixed", "hindi-english") |

## Generated support data (Phase 2b)
- `data/gen/gen_v12.jsonl` (1,202 rows, 591 paraphrase families = `group_id`, `source="gen_v12"`), built by
  `scripts/build_generated.py` from the hand-authored sources `data/gen/src/<intent>[.b2|.b3].txt`
  (`family|sentiment|persona|tags|text`). Written by the LLM assistant in three batches; TRAIN only.
- **Short of the ~1,500 target: 1,202 rows** (augmentation in Phase 3 multiplies them).
- Exact duplicates removed: 0; **near-duplicates (char-3-gram Jaccard > 0.6) of v1.1 validation/test/audit
  or contrast removed: 1**; max remaining Jaccard 0.582.
- Sentiment: {'negative': 351, 'positive': 308, 'neutral': 543}. Words per message 2-36 (mean 11.3); emoji rate 10.4%.
- Personas: {'ar': 86, 'el': 138, 'gen': 494, 'sk': 107, 'pro': 124, 'st': 153, 'hw': 100} (st student, el elderly, sk shopkeeper, ar angry repeat customer, pro professional, hw homemaker, gen generic).
- Targeted phenomena (tags): {'pr': 82, 'af': 151, 'ns': 97, 'mx': 66, 'ng': 99, 'nn': 119} (ns negation scope, nn neutral with nahi, ng negated negative, af varied affect word,
  pr praise-then-request, mx mixed intent labelled by the main request).
- Also: regional spellings (kaiku/apun, humra, kithe/labh raha, kado/aavega), formal Hindi (nivedan, nirast, avgat),
  English-heavy professional messages, heavy shorthand/typos (cncl, delvrd, pacel).
- Self-audit for label consistency (neutral rows with affect words, polarity-conflicting words) fixed 2 rows.

| Intent | negative | neutral | positive |
|---|---:|---:|---:|
| cancel_order | 65 | 127 | 56 |
| refund | 62 | 98 | 54 |
| track_order | 59 | 97 | 48 |
| not_received | 54 | 84 | 42 |
| damaged_item | 50 | 80 | 42 |
| feedback | 61 | 57 | 66 |

### Overlap risk (affect vocabulary)
| Split | affect words present | also in generated | already in v1.1 train | new via generated only | vocab coverage v1.1 train -> + generated |
|---|---:|---:|---:|---|---|
| validation | 13 | 12 | 9 | frustrating, loot, useless | 0.687 -> 0.949 |
| test | 15 | 13 | 12 | fraud, pareshan | 0.701 -> 0.944 |
| audit | 6 | 6 | 4 | badhiya, bakwas | 0.770 -> 0.952 |
| contrast | 32 | 31 | 13 | aabhaar, aabhari, chidh, chinta, dhokha, dukhi, fraud, irritate, khinn, nirash, pareshaan, pareshan, prabhavit, prasann, raahat, sukoon, tang, tension | 0.567 -> 0.945 |

**Consequence:** the contrast set's `unseen_affect` words are no longer unseen for any model trained on generated data
(18 of its 32 affect words enter training only through `gen_v12`). Results on that phenomenon for teacher v2 / student
measure learning from the generated data, not generalisation to new words. Generated data also raises coverage of the
validation/test vocabulary from ~0.69 to ~0.94, because the same kind of generator wrote all of them.
