"""Digest of hashed pieces + feature ids over every exported v1.1 row (compare across machines)."""
import hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polyglot12.common import read_jsonl
from polyglot12.hashing import features, token_pieces
h = hashlib.sha256()
d = Path(__file__).resolve().parents[1] / 'data' / 'v11'
for split in ('train', 'validation', 'test', 'audit', 'test_stress'):
    for r in read_jsonl(d / f'{split}.jsonl'):
        h.update(features(r['text'])[0].tobytes()); h.update(token_pieces(r['text'], 32768)[1].tobytes())
print(h.hexdigest())
