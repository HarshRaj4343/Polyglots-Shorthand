# `v12/data/gen/src/` — source lines for the generated set

The hand-written originals. This is the only folder in `gen/` meant to be edited
by a human; `../gen_v12.jsonl` is compiled from it by
`../../../scripts/build_generated.py`.

## Line format

```text
family|sentiment|persona|tags|text
```

| Field | Values |
|---|---|
| `family` | paraphrase family — lines sharing one stay in the same split |
| `sentiment` | `p` positive, `n` negative, `u` neutral |
| `persona` | speaker register the line was written in |
| `tags` | free-form markers for later slicing |
| `text` | the message itself |

Lines starting with `#` are comments; the first line of each file is the header
shown above.

## File naming

One group of files per intent, and the build globs `*.txt`, so the suffix is
free-form batching, not semantics:

| Pattern | Meaning |
|---|---|
| `<intent>.txt` | the first authoring pass |
| `<intent>.b2.txt`, `<intent>.b3.txt` | later batches, added as gaps were found |

Six intents × three batches = the 18 files here. A new batch is just a new
`<intent>.b4.txt`; nothing needs registering.

**The `family` field is what keeps the evaluation honest.** Paraphrases of one
message must share a family, or the same sentence can land in both train and a
reported split. Reuse the family value across every rewording of a message.
