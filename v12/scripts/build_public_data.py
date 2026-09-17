"""Phase 2a: build cleaned public Hinglish data (TRAIN-only use) from downloaded sources.

    .venv-v12/bin/python v12/scripts/build_public_data.py [--samples]

Downloads (not committed; re-created deterministically):
  v12/cache/public/sentimix/*          HF dataset RTT1/SentiMix (SemEval-2020 Task 9 Hinglish files)
  v12/cache/public/hinglid_*.txt       github.com/l3cube-pune/code-mixed-nlp  L3Cube-HingLID
  HF datasets LingoIITGN/PHINC, festvox/cmu_hinglish_dog, Abhishek4896/hindi-english-code-mixed-tweets-sentiment

Outputs (git-ignored because of source licenses / Twitter terms; included in the private bundle):
  v12/data/public/sentiment_labeled.jsonl   labeled sentiment, intent masked (null)
  v12/data/public/unlabeled_pool.jsonl      unlabeled KD pool (<= 50,000 sentences)
  v12/results/public_data_stats.json        counts, filters, near-duplicate removals, label distribution
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.common import read_jsonl, text_sha, write_jsonl  # noqa: E402
from polyglot12.hashing import MAX_CHARS  # noqa: E402
from polyglot12.neardup import NearDupIndex  # noqa: E402

CACHE = V12 / 'cache' / 'public'
HF_CACHE = V12 / 'cache' / 'hf_datasets'
POOL_CAP = 50_000
SEED = 42

# Frequent Romanized Hindi function words; a sentence must contain at least two distinct ones.
HINDI_MARKERS = frozenset('''hai hain hu hun tha thi nahi nhi nahin mat ka ki ke ko se mein mai main bhi
kya kyu kyun kyon kaise kab kahan yeh ye woh wo vo aur toh bhai yaar ji aap tum tu mera meri mere tera teri
apna apni apne kuch sab bahut bohot bht kar karo karna kiya kia raha rahe rahi rha gaya gayi gya diya liya hoga hogi
abhi phir fir par lekin agar jab tab ek koi kisi unka uska iska wala wali wale bas sirf acha accha achha ne hua hui
laga lagta lag chal dekh dekho bolo bola theek thik bilkul kyunki isliye matlab jaldi zyada pe tak hum hame hume
humne unhe usne mujhe tujhe'''.split())
# Deliberately excluded (also common English tokens): the to me do hi ho na
MENTION = re.compile(r'@\s*\w+')
URL = re.compile(r'(https?\s*:\s*/\s*/\S*|www\.\S+|\S+\.(com|in|co|ly|org)\S*)', re.I)


def latin_only(text):
    """Keep only Romanized text: letters must be ASCII/Latin (emoji and punctuation allowed)."""
    for c in text:
        if c.isalpha() and not ('LATIN' in unicodedata.name(c, '')):
            return False
    return True


MOJIBAKE = re.compile('[\u00c2\u00c3\u00e2\u00f0][\u0080-\u00bf\u2018-\u203a\u2122\u20ac\u0152-\u0178]')
TCO = re.compile(r'\bt\s*\.?\s*co\s*/\s*\S+', re.I)
MENTION_STUB = re.compile(r'(^|\s)_\s*\w+')


def fix_mojibake(text):
    """Repair UTF-8 read as cp1252 (e.g. 'ðŸ˜ƒ' -> '😃'); None if it cannot be repaired."""
    if not MOJIBAKE.search(text):
        return text
    try:
        return text.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def clean(text):
    text = fix_mojibake(text)
    if text is None:
        return ''
    t = TCO.sub(' ', text)
    t = MENTION_STUB.sub(' ', t)
    t = MENTION.sub(' ', t)
    t = URL.sub(' ', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'\bRT\b', ' ', t)
    t = t.replace('…', '...')
    return ' '.join(t.split())


def is_hinglish(text):
    words = re.findall(r'[a-z]+', text.lower())
    return len(set(words) & HINDI_MARKERS) >= 2


def keep_reason(text):
    if not text:
        return 'empty'
    if len(text) > MAX_CHARS:
        return 'too_long'
    if not latin_only(text):
        return 'non_latin_script'
    n = len(text.split())
    if n < 3 or n > 60:
        return 'length'
    if not is_hinglish(text):
        return 'not_hinglish'
    return None


def join_conll(tokens):
    """SentiMix/HingLID token lists -> text; re-attach '@'/'#' to the next token, drop URL fragments."""
    out, skip_url = [], False
    for tok in tokens:
        if tok in ('http', 'https') or tok.startswith(('http', 'www')):
            skip_url = True
            continue
        if skip_url and (tok in (':', '/', '//', '.', '_', '-', '?', '=', '&') or re.fullmatch(r'[\w./?=&-]+', tok) and ('/' in tok or '.' in tok)):
            continue
        skip_url = False
        if out and out[-1] in ('@', '#'):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return ' '.join(out)


def read_sentimix():
    rows = []
    labels = {}
    lab_path = CACHE / 'sentimix' / 'test_labels_hinglish.txt'
    for line in lab_path.read_text().splitlines()[1:]:
        if ',' in line:
            uid, lab = line.strip().split(',')
            labels[uid] = lab
    for fname, split in (('train_14k_split_conll.txt', 'train'), ('dev_3k_split_conll.txt', 'dev'),
                         ('Hindi_test_unalbelled_conll_updated.txt', 'test')):
        uid, lab, toks = None, None, []

        def flush():
            if uid is not None:
                label = lab if lab else labels.get(uid)
                rows.append({'raw': join_conll(toks), 'sentiment': label, 'uid': f'{split}-{uid}'})
        for line in (CACHE / 'sentimix' / fname).read_text(encoding='utf-8').splitlines():
            parts = line.split('\t')
            if parts[0] == 'meta':
                flush()
                uid, lab, toks = parts[1], (parts[2] if len(parts) > 2 else None), []
            elif line.strip():
                toks.append(parts[0])
        flush()
    return rows


def read_hinglid():
    rows = []
    for split in ('train', 'validation', 'test'):
        toks = []
        for i, line in enumerate((CACHE / f'hinglid_{split}.txt').read_text(encoding='utf-8').splitlines() + ['']):
            if line.strip():
                toks.append(line.split('\t')[0])
            elif toks:
                rows.append({'raw': join_conll(toks), 'uid': f'{split}-{len(rows)}'})
                toks = []
    return rows


def read_hf():
    from datasets import load_dataset
    phinc = load_dataset('LingoIITGN/PHINC', cache_dir=str(HF_CACHE))['train']
    dog = load_dataset('festvox/cmu_hinglish_dog', cache_dir=str(HF_CACHE))
    abhi = load_dataset('Abhishek4896/hindi-english-code-mixed-tweets-sentiment', cache_dir=str(HF_CACHE))['train']
    out = {
        'phinc': [{'raw': r['Sentence'], 'uid': str(i)} for i, r in enumerate(phinc)],
        'cmu_hinglish_dog': [{'raw': r['translation']['hi_en'], 'uid': f'{s}-{i}'} for s in ('train', 'validation', 'test')
                             for i, r in enumerate(dog[s])],
        'abhishek_tweets': [{'raw': r['tweet'], 'sentiment': r['sentiment'].strip().lower(), 'uid': str(i)} for i, r in enumerate(abhi)],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', action='store_true', help='print 20 raw->clean samples per source and exit')
    args = ap.parse_args()
    sources = {'sentimix': read_sentimix(), 'hinglid': read_hinglid(), **read_hf()}
    rng = random.Random(SEED)
    if args.samples:
        for name, rows in sources.items():
            print(f'===== {name}: {len(rows)} raw rows')
            for r in rng.sample(rows, 20):
                c = clean(r['raw'])
                print(f"  [{r.get('sentiment', '-')}] {keep_reason(c) or 'KEEP':16s} {c[:150]!r}")
        return

    protected_rows = [r for s in ('validation', 'test', 'audit') for r in read_jsonl(V12 / 'data/v11' / f'{s}.jsonl')]
    protected_rows += read_jsonl(V12 / 'data/contrast.jsonl')
    index = NearDupIndex([r['text'] for r in protected_rows])
    stats = {'sources': {}, 'near_dup_threshold': 0.6, 'protected_sets': 'v1.1 validation, test, audit + contrast'}
    seen = set()
    labeled, pool = [], []
    for name, rows in sources.items():
        st = Counter()
        for r in rows:
            text = clean(r['raw'])
            reason = keep_reason(text)
            if reason:
                st[reason] += 1
                continue
            key = ' '.join(re.findall(r'\w+', text.lower()))
            if key in seen:
                st['exact_duplicate'] += 1
                continue
            if index.is_near_dup(text):
                st['near_duplicate_of_protected'] += 1
                continue
            seen.add(key)
            st['kept'] += 1
            row = {'id': f'pub-{name}-{r["uid"]}', 'text': text, 'intent': None, 'group_id': f'pub-{name}-{r["uid"]}', 'source': name}
            if r.get('sentiment') in ('positive', 'negative', 'neutral'):
                labeled.append({**row, 'sentiment': r['sentiment']})
            else:
                if 'sentiment' in r:
                    st['missing_label'] += 1
                pool.append({**row, 'sentiment': None})
        stats['sources'][name] = {'raw_rows': len(rows), **dict(st)}
    # Cap the unlabeled pool at 50k with a seeded sample, proportional to what each source contributes.
    rng.shuffle(pool)
    if len(pool) > POOL_CAP:
        pool = pool[:POOL_CAP]
    pool.sort(key=lambda r: r['id'])
    labeled.sort(key=lambda r: r['id'])
    write_jsonl(V12 / 'data/public/sentiment_labeled.jsonl', labeled)
    write_jsonl(V12 / 'data/public/unlabeled_pool.jsonl', pool)
    stats['sentiment_labeled'] = {'rows': len(labeled), 'by_source': dict(Counter(r['source'] for r in labeled)),
                                  'labels': dict(Counter(r['sentiment'] for r in labeled))}
    stats['unlabeled_pool'] = {'rows': len(pool), 'cap': POOL_CAP, 'by_source': dict(Counter(r['source'] for r in pool))}
    stats['near_duplicates_removed_total'] = sum(v.get('near_duplicate_of_protected', 0) for v in stats['sources'].values())
    (V12 / 'results').mkdir(exist_ok=True)
    (V12 / 'results/public_data_stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
