"""Hash-consistency test: polyglot12.hashing must reproduce the stored v1.1 indices exactly.

Runs on the Mac and inside every notebook:  python tests/test_hashing.py   (exit code 1 on mismatch)
Bucket ids must match exactly; float feature values within 1e-6 (numpy builds may differ in last bits).
"""
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from polyglot12.hashing import features, normalize, token_pieces  # noqa: E402


def run():
    data = json.loads((HERE / 'hash_fixtures.json').read_text(encoding='utf-8'))
    failures = []
    for i, c in enumerate(data['cases']):
        s = c['text']
        checks = {}
        checks['normalized'] = normalize(s) == c['normalized']
        ix, vals = features(s, 'full')
        checks['features_full_ix'] = ix.tolist() == c['features_full_ix']
        checks['features_full_vals'] = len(vals) == len(c['features_full_vals']) and all(
            abs(float(a) - b) <= 1e-6 for a, b in zip(vals, c['features_full_vals']))
        checks['features_strip_ix'] = features(s, 'strip_symbols')[0].tolist() == c['features_strip_ix']
        for n in (16384, 32768):
            toks, flat, lens = token_pieces(s, n)
            checks[f'tokens'] = toks == c['tokens']
            checks[f'pieces{n}'] = flat.tolist() == c[f'pieces{n}_flat'] and lens.tolist() == c[f'pieces{n}_lens']
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            failures.append((i, s[:40], bad))
    print(f'hash test: {len(data["cases"]) - len(failures)}/{len(data["cases"])} cases identical '
          f'(python {sys.version.split()[0]}, unicodedata {unicodedata.unidata_version}; '
          f'fixtures from python {data["meta"]["generated_with_python"]})')
    for f in failures:
        print('  MISMATCH', f)
    return not failures


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
