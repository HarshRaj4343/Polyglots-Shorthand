"""Phase 3 (local, after notebook 02): attach teacher-v2 logits to the KD training sets.

    .venv-v12/bin/python v12/scripts/build_kd.py [runs/02_teacher_kd]

Reads the SELECTED run's logits (selection by validation NLL inside the notebook) and writes:
  data/kd/kd_support.jsonl         v1.1 train + generated + augmented: gold labels + teacher logits      (committed)
  data/kd/kd_public_labeled.jsonl  public tweets: gold sentiment, intent null, teacher logits           (git-ignored)
  data/kd/kd_pool.jsonl            unlabeled pool: no labels, teacher logits                            (git-ignored)
  results/kd_stats.json

kd_intent_mask = support row AND teacher max intent probability (T=1) >= 0.6. Public/pool rows never get intent KD.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.common import INTENTS, SENTIMENTS, read_jsonl, write_jsonl  # noqa: E402

THRESH = 0.6
FILES = {
    'kd_train': ('data/v11/train.jsonl', 'support'),
    'kd_gen': ('data/gen/gen_v12.jsonl', 'support'),
    'kd_aug': ('data/kd/aug_train.jsonl', 'support'),
    'kd_public_labeled': ('data/public/sentiment_labeled.jsonl', 'public'),
    'kd_pool': ('data/public/unlabeled_pool.jsonl', 'pool'),
}


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    run = V12 / (sys.argv[1] if len(sys.argv) > 1 else 'runs/02_teacher_kd')
    summary = json.loads((run / 'summary.json').read_text())
    logit_dir = run / 'selected_logits'
    out = {'support': [], 'public': [], 'pool': []}
    stats = {'teacher_run': summary['selected_run'], 'intent_mask_threshold': THRESH, 'files': {}}
    for split, (path, kind) in FILES.items():
        rows = read_jsonl(V12 / path)
        z = np.load(logit_dir / f'{split}.npz')
        pos = {i: k for k, i in enumerate(z['ids'].tolist())}
        zi, zs = z['intent'][[pos[r['id']] for r in rows]], z['sentiment'][[pos[r['id']] for r in rows]]
        pi, ps = softmax(zi.astype(np.float64)), softmax(zs.astype(np.float64))
        st = {'rows': len(rows)}
        if kind == 'support':
            yi = np.array([INTENTS.index(r['intent']) for r in rows])
            ys = np.array([SENTIMENTS.index(r['sentiment']) for r in rows])
            st['teacher_agrees_with_gold_intent'] = float((pi.argmax(1) == yi).mean())
            st['teacher_agrees_with_gold_sentiment'] = float((ps.argmax(1) == ys).mean())
            mask = pi.max(1) >= THRESH
            st['intent_kd_kept'] = float(mask.mean())
        else:
            mask = np.zeros(len(rows), bool)
            if kind == 'public':
                ys = np.array([SENTIMENTS.index(r['sentiment']) for r in rows])
                st['teacher_agrees_with_gold_sentiment'] = float((ps.argmax(1) == ys).mean())
            st['teacher_sentiment_distribution'] = dict(Counter(SENTIMENTS[k] for k in ps.argmax(1)))
        for r, a, b, m in zip(rows, zi, zs, mask):
            rec = {'id': r['id'], 'text': r['text'], 'intent': r.get('intent') if kind == 'support' else None,
                   'sentiment': r.get('sentiment') if kind != 'pool' else None, 'group_id': r.get('group_id', r['id']),
                   'source': r.get('source', split), 't_intent_logits': [round(float(x), 4) for x in a],
                   't_sentiment_logits': [round(float(x), 4) for x in b], 'kd_intent_mask': bool(m)}
            out[kind].append(rec)
        stats['files'][split] = st
    write_jsonl(V12 / 'data/kd/kd_support.jsonl', out['support'])
    write_jsonl(V12 / 'data/kd/kd_public_labeled.jsonl', out['public'])
    write_jsonl(V12 / 'data/kd/kd_pool.jsonl', out['pool'])
    (V12 / 'results/kd_stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
