# v1.2 progress

## Phase 0 - local setup (done)
- Exact v1.1 splits exported (digests match results.json) + test_stress; hashing copied into `polyglot12/` and verified byte-identical to v1.1 on 50 fixtures and all 1,148 rows under Python 3.10/3.12/3.14 (Unicode 13/15/16).
- Teacher and student trainers written with per-epoch checkpoints and auto-resume; CPU smoke test (32 rows, simulated disconnect + resume, all 4 teacher tokenizers, 3 student variants, augmentation-safety check) passes -> `results/smoke_test.json`; notebook 01 dry-run passes.
- Finding: hing-bert and MuRIL tokenize every emoji as [UNK] (261/908 rows affected); only the hing-roberta models can see emoji. Contrast set (40 pairs) written early so notebook 01 scores it.

## Phase 1 - teacher diagnostic (done; notebook 01 on Colab T4, 38 min, no disconnects)
- 12/12 runs finished; selected by validation NLL: hing-roberta-mixed lr 5e-5 (278M). Colab hash test identical (Python 3.13/Unicode 15.1). Local evaluator reproduces every v1.1 whitepaper number exactly.
- Sentiment fixed on all sets: test F1 1.000 vs 0.845 (paired +0.159 [+0.056, +0.282]), emoji pairs 12/12, contrast sentiment 0.96 avg vs 0.70; audit errors 1 vs 6. Test sentiment 1.000 is a ceiling (validation 0.943).
- Intent NOT improved: +0.032 [-0.061, +0.142]; stress consistency 92.6% vs 97.4%; negation-scope pairs 4/10; the "cancel mat karna" audit error remains. -> results/teacher_v11data.json

## Phase 2 - data (done)
- 2a public: SemEval-2020 SentiMix (HF RTT1/SentiMix) + small MIT tweet set -> 12,105 labeled sentiment rows (intent masked); PHINC + L3Cube-HingLID + CMU DoG -> 50,000-row unlabeled pool; 0 near-duplicates of protected sets; md-nishat-008 rejected (not Romanized). Files git-ignored (licenses), rebuilt by `scripts/build_public_data.py`.
- 2b generated: 1,202 support messages / 591 families (short of ~1,500), all 6 intents x 3 sentiments, 7 personas, 499 tagged failure-type rows; 1 near-duplicate removed. Overlap risk: 18 contrast affect words now appear only via generated data.
- 2c contrast set written in Phase 0 (40 pairs). 2d `human_test.jsonl` hook in the evaluator (none provided yet).
