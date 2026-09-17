"""v1.1 hashing, copied verbatim so the Colab/Kaggle bundle does not need the repo root.

normalize / clean / features must stay byte-identical to model.py. token_pieces is
model.tokens with the bucket count as a parameter (v1.1 used 16384).
tests/test_hashing.py checks this against stored indices on every platform.
"""
import re
import unicodedata
import zlib
from collections import Counter

import numpy as np

DIM = 32768
MAX_CHARS = 512
MAX_TOKENS = 128
V11_ATT_BUCKETS = 16384

ALIASES = {'krdo':'kar do', 'kr':'kar', 'kro':'karo', 'krna':'karna',
           'nhi':'nahi', 'nai':'nahi', 'nh':'nahi', 'rha':'raha', 'rhe':'rahe',
           'plz':'please', 'pls':'please', 'mera':'mera', 'bht':'bahut',
           'bohot':'bahut', 'bohut':'bahut', 'acha':'accha', 'achha':'accha',
           'kaha':'kahan', 'kabtk':'kab tak', 'paisa':'paise',
           'ordr':'order', 'delivry':'delivery', 'servis':'service', 'parcl':'parcel',
           'thnk':'thank', 'chaiye':'chahiye', 'mujhko':'mujhe'}
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def normalize(text):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if len(text) > MAX_CHARS:
        raise ValueError(f'maximum input is {MAX_CHARS} Unicode code points; split long chats upstream')
    text = unicodedata.normalize('NFC', text).lower()
    text = ' '.join(text.split())
    if not text:
        raise ValueError('text must not be empty')
    return text


def clean(text, mode='full'):
    s = normalize(text)
    if mode == 'strip_symbols':
        s = ' '.join(''.join(c for c in s if not unicodedata.category(c).startswith(('S', 'P'))).split())
    return s


def bucket(f, size=DIM):
    return zlib.crc32(f.encode('utf-8')) % size


def features(text, mode='full'):
    """Sparse v1.0 linear-branch features: (sorted bucket ids int32, l2-normalised float32 values)."""
    s = clean(text, mode)
    ts = TOKEN.findall(s)
    fs = ['w:' + t for t in ts]
    fs += ['b:' + a + '|' + b for a, b in zip(ts, ts[1:])]
    if mode != 'word_only':
        padded = '^' + s + '$'
        fs += [f'c{n}:' + padded[i:i+n] for n in (3, 4, 5) for i in range(len(padded)-n+1)]
    if mode in ('full', 'strip_symbols'):
        canon = [u for t in ts for u in ALIASES.get(t, t).split()]
        fs += ['a:' + t for t in canon]
        fs += ['ab:' + a + '|' + b for a, b in zip(canon, canon[1:])]
        symbols = list(dict.fromkeys(t for t in ts if any(unicodedata.category(c).startswith('S') for c in t)))[:8]
        words = list(dict.fromkeys(t for t in ts if any(c.isalnum() for c in t)))[:64]
        fs += ['e:' + e + '|' + t for e in symbols for t in words]
    counts = Counter(bucket(f) for f in fs)
    ix = np.array(sorted(counts), dtype=np.int32)
    vals = np.array([1 + np.log(counts[int(i)]) for i in ix], dtype=np.float32)
    vals /= max(float(np.linalg.norm(vals)), 1e-12)
    return ix, vals


def token_pieces(text, n_buckets=V11_ATT_BUCKETS, mode='full'):
    """Per-token hashed pieces (word, alias, char 3/4-grams): (tokens, flat ids int64, pieces per token int64)."""
    ts = TOKEN.findall(clean(text, mode))[:MAX_TOKENS]
    flat, lens = [], []
    for t in ts:
        keys = ['tw:' + t]
        if mode in ('full', 'strip_symbols'):
            keys.append('ta:' + ALIASES.get(t, t))
        if mode != 'word_only':
            p = '<' + t + '>'
            keys += [f'tc{n}:' + p[i:i+n] for n in (3, 4) for i in range(len(p)-n+1)]
        ids = sorted({bucket(k, n_buckets) for k in keys})
        flat += ids
        lens.append(len(ids))
    return ts, np.array(flat, dtype=np.int64), np.array(lens, dtype=np.int64)
