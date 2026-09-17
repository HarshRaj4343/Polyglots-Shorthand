"""Build v12/bundle.zip: only what the GPU notebooks need (code, configs, JSONL data, hash test)."""
import hashlib
import zipfile
from pathlib import Path

V12 = Path(__file__).resolve().parents[1]
INCLUDE = [
    'polyglot12/*.py',
    'configs/*.json',
    'tests/test_hashing.py',
    'tests/hash_fixtures.json',
    'data/v11/*.jsonl',
    'data/v11/MANIFEST.json',
    'data/contrast.jsonl',
    'data/gen/*.jsonl',
    'data/public/*.jsonl',
    'data/kd/*.jsonl',
]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)  # deterministic zip


def main():
    files = sorted({p for pat in INCLUDE for p in V12.glob(pat) if p.is_file()})
    out = V12 / 'bundle.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            info = zipfile.ZipInfo(str(f.relative_to(V12)), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f'{out} {out.stat().st_size / 1024:.0f} KB, {len(files)} files, sha256 {digest}')
    for f in files:
        print('  ', f.relative_to(V12))


if __name__ == '__main__':
    main()
