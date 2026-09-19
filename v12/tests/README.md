# `v12/tests/` — the hash-consistency guard

One test, guarding one thing: that v1.2 hashes text exactly as v1.1 did.

| File | Role |
|---|---|
| `test_hashing.py` | replays every fixture and asserts the indices match |
| `hash_fixtures.json` | 50 cases captured from the ORIGINAL v1.1 `model.py` — the ground truth |

`../polyglot12/hashing.py` is a verbatim copy of the v1.1 hashing, duplicated so
the GPU bundle does not need the repo root. A copy can drift from its original
silently, and if it does, every v1.2 feature index shifts: the student trains on
one vocabulary and the deployed runtime hashes into another, with no error and
badly degraded accuracy. This test is what makes that drift loud.

```sh
cd v12 && make test-hash
```

`make test-hash` runs the file as a script under **both** interpreters —
`.venv-v12` (Python 3.12, unicodedata 15.0.0) and `.venv` (Python 3.14,
unicodedata 16.0.0) — because the risk being guarded against is partly a Unicode
version difference: the same text can normalize differently across releases, and
the hashing must survive that. Running it under one interpreter only tests half
of what matters. The test is not discoverable via `unittest discover`; invoke it
as `python tests/test_hashing.py` if you need to run it directly.

If it fails, fix `polyglot12/hashing.py`. Regenerate the fixtures with
`scripts/make_hash_fixtures.py` **only** when the v1.1 hashing itself genuinely
changed — regenerating to silence a failure destroys the guarantee and
invalidates every trained checkpoint and every number in `../results/`.

The v1.1 implementation tests are separate, at the repo root
(`../../test_solution.py`, 13 tests).
