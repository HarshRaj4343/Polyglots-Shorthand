"""Phase 2b: convert hand-authored generated support messages to JSONL (TRAIN only) and report stats.

    .venv-v12/bin/python v12/scripts/build_generated.py

Source: v12/data/gen/src/<intent>*.txt, lines  family|sentiment(p/n/u)|persona|tags|text
Output: v12/data/gen/gen_v12.jsonl  (id, text, intent, sentiment, group_id, source="gen_v12", persona, tags)
        v12/results/generated_data_stats.json
Messages with char-3-gram Jaccard > 0.6 to any v1.1 validation/test/audit or contrast message are removed.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.common import INTENTS, read_jsonl, write_jsonl  # noqa: E402
from polyglot12.hashing import TOKEN, features, normalize  # noqa: E402
from polyglot12.neardup import NearDupIndex  # noqa: E402

SENT = {'p': 'positive', 'n': 'negative', 'u': 'neutral'}
PERSONAS = {'st', 'el', 'sk', 'ar', 'pro', 'hw', 'gen'}
TAGS = {'ns', 'nn', 'ng', 'af', 'pr', 'mx', '-'}
# Affect vocabulary (normalized single tokens) used for the overlap-risk report.
AFFECT = sorted(set('''pareshan pareshaan tang dukhi khush shukriya dhanyavaad aabhaar aabhari raahat sukoon nirash chinta
tension gussa naraz naraaz chidh irritate irritating khinn prasann prabhavit mast badiya badhiya zabardast kamaal lajawab
shandaar bekar bakwaas bakwas ghatiya faltu worst useless bura accha acchi achha acha pasand nafrat pachhtava santusht
disappointed frustrating happy love great awesome superb excellent amazing thanks thank sad angry fraud jhooth jhoothe
dhokha loot sharam badtameezi'''.split()))


def words(text):
    return {t for t in TOKEN.findall(normalize(text)) if t.isalpha()}


def main():
    rows, problems = [], []
    for path in sorted((V12 / 'data/gen/src').glob('*.txt')):
        intent = path.stem.split('.')[0]
        if intent not in INTENTS:
            raise SystemExit(f'unknown intent file {path.name}')
        fam_count = Counter()
        for ln, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split('|', 4)
            if len(parts) != 5:
                problems.append(f'{path.name}:{ln} malformed')
                continue
            fam, s, persona, tags, text = (p.strip() for p in parts)
            if s not in SENT or persona not in PERSONAS or any(t not in TAGS for t in tags.split(',')):
                problems.append(f'{path.name}:{ln} bad field {s} {persona} {tags}')
                continue
            features(text)  # validates length / non-empty
            k = fam_count[fam]
            fam_count[fam] += 1
            rows.append({'id': f'gen-{intent}-{fam}-{k}', 'text': text, 'intent': intent, 'sentiment': SENT[s],
                         'group_id': f'gen-{intent}-{fam}', 'source': 'gen_v12', 'persona': persona,
                         'tags': [] if tags == '-' else tags.split(',')})
    if problems:
        raise SystemExit('\n'.join(problems))
    # Exact duplicates inside generated data (case/whitespace-insensitive).
    seen, dedup, exact_dups = set(), [], 0
    for r in rows:
        key = normalize(r['text'])
        if key in seen:
            exact_dups += 1
            continue
        seen.add(key)
        dedup.append(r)
    # Families must have a single label pair (they are paraphrase families).
    fam_labels = {}
    for r in dedup:
        fam_labels.setdefault(r['group_id'], set()).add(r['sentiment'])
    mixed_fams = sorted(g for g, v in fam_labels.items() if len(v) > 1)

    v11 = {s: read_jsonl(V12 / 'data/v11' / f'{s}.jsonl') for s in ('train', 'validation', 'test', 'audit')}
    contrast = read_jsonl(V12 / 'data/contrast.jsonl')
    protected = v11['validation'] + v11['test'] + v11['audit'] + contrast
    index = NearDupIndex([r['text'] for r in protected])
    kept, removed = [], []
    for r in dedup:
        j, match = index.best_match(r['text'])
        if j > 0.6:
            removed.append({'text': r['text'], 'jaccard': round(j, 3), 'matched': match})
        else:
            kept.append(r)
    write_jsonl(V12 / 'data/gen/gen_v12.jsonl', kept)

    # ---- stats
    nwords = [len(r['text'].split()) for r in kept]
    has_emoji = sum(any(ord(c) > 0x2100 for c in r['text']) for r in kept)
    gen_words = set().union(*(words(r['text']) for r in kept))
    train_words = set().union(*(words(r['text']) for r in v11['train']))
    overlap = {}
    for name, rs in (('validation', v11['validation']), ('test', v11['test']), ('audit', v11['audit']), ('contrast', contrast)):
        split_words = set().union(*(words(r['text']) for r in rs))
        aff = sorted(w for w in AFFECT if w in split_words)
        overlap[name] = {
            'affect_words_in_split': aff,
            'also_in_generated': sorted(w for w in aff if w in gen_words),
            'also_in_v11_train': sorted(w for w in aff if w in train_words),
            'new_via_generated_only': sorted(w for w in aff if w in gen_words and w not in train_words),
            'vocab_coverage_by_v11_train': round(len(split_words & train_words) / len(split_words), 3),
            'vocab_coverage_by_v11_train_plus_generated': round(len(split_words & (train_words | gen_words)) / len(split_words), 3),
        }
    max_j = sorted((index.best_match(r['text'])[0] for r in kept), reverse=True)
    stats = {
        'rows': len(kept), 'families': len({r['group_id'] for r in kept}),
        'authored_lines': len(rows), 'exact_duplicates_removed': exact_dups,
        'near_duplicates_removed_vs_val_test_audit_contrast': len(removed), 'near_duplicate_examples': removed[:20],
        'max_jaccard_to_protected_after_filter': round(max_j[0], 3) if max_j else None,
        'families_with_mixed_sentiment': mixed_fams,
        'intent_x_sentiment': {i: dict(Counter(r['sentiment'] for r in kept if r['intent'] == i)) for i in INTENTS},
        'sentiment': dict(Counter(r['sentiment'] for r in kept)),
        'persona': dict(Counter(r['persona'] for r in kept)),
        'tags': dict(Counter(t for r in kept for t in r['tags'])),
        'words': {'min': min(nwords), 'max': max(nwords), 'mean': round(sum(nwords) / len(nwords), 1)},
        'emoji_rate': round(has_emoji / len(kept), 3),
        'affect_word_overlap_risk': overlap,
    }
    (V12 / 'results/generated_data_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ('affect_word_overlap_risk', 'near_duplicate_examples')}, indent=1, ensure_ascii=False))
    for k, v in overlap.items():
        print(k, {x: (len(y) if isinstance(y, list) else y) for x, y in v.items()}, 'new via gen:', v['new_via_generated_only'])


if __name__ == '__main__':
    main()
