"""Phase 3 input: meaning-preserving augmented variants of v1.1 train + generated rows (TRAIN only).

    .venv-v12/bin/python v12/scripts/build_aug.py  -> v12/data/kd/aug_train.jsonl, v12/results/aug_stats.json

Per source row, up to two variants (deduplicated, must differ from the source):
  map   : v1.1 make_data maps (shorthand SHORT, respelling SPELLING), applied at random per token
  noise : polyglot12.augment.char_noise (vowel drop, keyboard swap, elongation, transposition)
Variants keep the source labels and group_id (same paraphrase family). A variant is rejected if it contains a
v1.1 stress-map output that was not already in the source, or is a near-duplicate (char-3-gram Jaccard > 0.6)
of any v1.1 validation/test/audit or contrast message.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V12 = ROOT / 'v12'
sys.path.insert(0, str(V12))
sys.path.insert(0, str(ROOT))
from make_data import SHORT, SPELLING  # noqa: E402  (v1.1 maps, imported not copied)
from polyglot12.augment import PROTECTED, char_noise  # noqa: E402
from polyglot12.common import STRESS_OUTPUTS, read_jsonl, write_jsonl  # noqa: E402
from polyglot12.neardup import NearDupIndex  # noqa: E402

SEED = 42
MAPS = {**SHORT, **SPELLING}


def map_variant(text, rng, p=0.6):
    out = []
    for tok in text.split(' '):
        low = tok.lower()
        if low in MAPS and low not in PROTECTED and rng.random() < p:
            out.append(MAPS[low])
        else:
            out.append(tok)
    return ' '.join(out)


def introduces_stress_output(src, var):
    before = {t.lower() for t in src.split()}
    return any(t.lower() in STRESS_OUTPUTS and t.lower() not in before for t in var.split())


def main():
    rng = random.Random(SEED)
    sources = [dict(r, _src='v11_train') for r in read_jsonl(V12 / 'data/v11/train.jsonl')]
    sources += [dict(r, _src='gen_v12') for r in read_jsonl(V12 / 'data/gen/gen_v12.jsonl')]
    protected = [r['text'] for s in ('validation', 'test', 'audit') for r in read_jsonl(V12 / 'data/v11' / f'{s}.jsonl')]
    protected += [r['text'] for r in read_jsonl(V12 / 'data/contrast.jsonl')]
    index = NearDupIndex(protected)
    existing = {r['text'].lower() for r in sources}
    out, st = [], Counter()
    for r in sources:
        cands = [('map', map_variant(r['text'], rng)), ('noise', char_noise(r['text'], rng, p=0.3))]
        seen = {r['text'].lower()}
        for kind, v in cands:
            if v.lower() in seen or v.lower() in existing:
                st[f'{kind}_unchanged_or_duplicate'] += 1
                continue
            if introduces_stress_output(r['text'], v):
                st[f'{kind}_rejected_stress_output'] += 1
                continue
            if index.is_near_dup(v):
                st[f'{kind}_rejected_near_duplicate'] += 1
                continue
            seen.add(v.lower())
            st[f'{kind}_kept'] += 1
            out.append({'id': f'aug-{kind}-{r["id"]}', 'text': v, 'intent': r['intent'], 'sentiment': r['sentiment'],
                        'group_id': r['group_id'], 'source': f'aug_{kind}_{r["_src"]}', 'parent_id': r['id']})
    write_jsonl(V12 / 'data/kd/aug_train.jsonl', out)
    stats = {'source_rows': len(sources), 'variants': len(out), 'counts': dict(st),
             'near_duplicates_removed': st['map_rejected_near_duplicate'] + st['noise_rejected_near_duplicate'],
             'by_source': dict(Counter(r['source'] for r in out)), 'sentiment': dict(Counter(r['sentiment'] for r in out))}
    (V12 / 'results/aug_stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    for r in rng.sample(out, 8):
        print('  ', r['source'], '|', r['text'])


if __name__ == '__main__':
    main()
