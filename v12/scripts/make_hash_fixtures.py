"""Create tests/hash_fixtures.json from the ORIGINAL v1.1 model.py (ground truth).

Run once with the v1.1 environment:  .venv/bin/python v12/scripts/make_hash_fixtures.py
The 50 strings mix real v1.1 messages, shorthand, emoji (incl. ZWJ / skin tone),
other scripts, punctuation runs, whitespace/case edge cases and a 512-code-point input.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import model  # noqa: E402

HAND = [
    'bhai order cancel kar do please urgent meeting hai',
    'item delivered bol rha h but mila hi nhi 😒',
    'bohot badhiya service 😒',
    'aapki help se khush hu thank you 😊',
    'cancel mat karna sirf order track karke batao',
    'mujhe booking cancel karni padegi kal ghar pe nahi hu',
    'bahut pareshan hu mera order cancel karo',
    'cancel ho gaya bina kisi hassle ke superb experience',
    'koi problem nahi hui, shukriya!!!',
    'tang aa gaya hu refund ke liye 😤😤',
    'REFUND   KAB   MILEGA???',
    '  ordr   delivry  servis  parcl thnk chaiye mujhko  ',
    'krdo kabtk nhi nai nh rha rhe plz pls bht bohut acha achha kaha paisa',
    'nahii karr hey milaa meraa kyaa pleez orrder delivary refnd serrvice',
    'love it ❤️ five star 👍🏽',
    '👩🏽‍💻 fixed my order',
    '🥲?!',
    'हिंदी accha hai',
    'தமிழ் super service',
    'বাংলা bhalo',
    'order #4521 ka refund ₹499 kab aayega?',
    'delivery...late...again!!!',
    "don't cancel, it's fine :)",
    'e-mail pe invoice bhejo',
    'x',
    'ok',
    '😡',
    'Ⅻ ﬁ ǅ ß İ',
    'café wala order',
    'tab\tand\nnewline   message',
]


def main():
    rows = []
    for split in ('train', 'validation', 'test'):
        rows += [json.loads(l)['text'] for l in (ROOT / 'data' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    rows += [json.loads(l)['text'] for l in (ROOT / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    extra = [t for t in dict.fromkeys(rows) if t not in HAND]
    # Deterministic pick: every k-th unique message.
    step = max(1, len(extra) // 19)
    picked = extra[::step][:19]
    long_text = ('mera order delivered bol raha hai but mila hi nahi 😒 ' * 20)[:512]
    strings = HAND + picked + [long_text]
    assert len(strings) == 50, len(strings)
    out = []
    for s in strings:
        ix, vals = model.features(s, 'full')
        toks, flat, lens = model.tokens(s, 'full')
        ix_s, vals_s = model.features(s, 'strip_symbols')
        toks32 = [sorted({model.bucket(k, 32768) for k in keys}) for keys in _keys(toks)]
        out.append({'text': s, 'normalized': model.normalize(s),
                    'features_full_ix': ix.tolist(), 'features_full_vals': [float(v) for v in vals],
                    'features_strip_ix': ix_s.tolist(),
                    'tokens': toks, 'pieces16384_flat': flat.tolist(), 'pieces16384_lens': lens.tolist(),
                    'pieces32768_flat': [i for ids in toks32 for i in ids],
                    'pieces32768_lens': [len(ids) for ids in toks32]})
    meta = {'generated_with_python': sys.version.split()[0], 'numpy': model.np.__version__,
            'note': 'ground truth from v1.1 model.py'}
    path = ROOT / 'v12' / 'tests' / 'hash_fixtures.json'
    path.write_text(json.dumps({'meta': meta, 'cases': out}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', path, len(out), 'cases')


def _keys(toks):
    for t in toks:
        keys = ['tw:' + t, 'ta:' + model.ALIASES.get(t, t)]
        p = '<' + t + '>'
        keys += [f'tc{n}:' + p[i:i+n] for n in (3, 4) for i in range(len(p)-n+1)]
        yield keys


if __name__ == '__main__':
    main()
