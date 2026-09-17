"""Export the EXACT v1.1 splits (never regenerated) to v12/data/v11/*.jsonl.

Checks the source files against the SHA-256 digests recorded in artifacts/results.json,
keeps v1.1 group ids, and adds a test_stress file (the v1.1 14-entry stress map applied to
test; same labels and groups) so GPU notebooks can score it.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V12 = ROOT / 'v12'
sys.path.insert(0, str(V12))
sys.path.insert(0, str(ROOT))

from polyglot12.common import read_jsonl, stress_text, write_jsonl  # noqa: E402
import solution  # noqa: E402  (v1.1 code, only used to cross-check the stress map)

OUT = V12 / 'data' / 'v11'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    expected = json.loads((ROOT / 'artifacts' / 'results.json').read_text())['dataset_sha256']
    manifest = {'source_sha256': {}, 'export_sha256': {}, 'rows': {}}
    splits = {}
    for split in ('train', 'validation', 'test'):
        src = ROOT / 'data' / f'{split}.jsonl'
        digest = sha256(src)
        if digest != expected[split]:
            raise SystemExit(f'{src} does not match the v1.1 results.json digest; refusing to export')
        manifest['source_sha256'][split] = digest
        splits[split] = [{'id': f'{split}-{i:04d}', 'text': r['text'], 'intent': r['intent'],
                          'sentiment': r['sentiment'], 'group_id': r['group'], 'source': 'v11'}
                         for i, r in enumerate(read_jsonl(src))]
    # audit.py assigns group ids audit-00..audit-23 in file order.
    audit_src = ROOT / 'audit.jsonl'
    manifest['source_sha256']['audit'] = sha256(audit_src)
    splits['audit'] = [{'id': f'audit-{i:04d}', 'text': r['text'], 'intent': r['intent'],
                        'sentiment': r['sentiment'], 'group_id': f'audit-{i:02d}', 'source': 'v11'}
                       for i, r in enumerate(read_jsonl(audit_src))]
    splits['test_stress'] = []
    for r in splits['test']:
        assert stress_text(r['text']) == solution.stress_text(r['text'])
        splits['test_stress'].append(dict(r, id=r['id'].replace('test-', 'test_stress-'), text=stress_text(r['text'])))
    for split, rows in splits.items():
        path = OUT / f'{split}.jsonl'
        write_jsonl(path, rows)
        manifest['export_sha256'][split] = sha256(path)
        manifest['rows'][split] = len(rows)
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['rows']))


if __name__ == '__main__':
    main()
