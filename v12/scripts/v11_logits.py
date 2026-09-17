"""Write logits of the frozen v1.1 full model (artifacts/full.npz) for every v1.2 evaluation split,
so v1.1 is scored by exactly the same evaluator as the teacher and student.

    .venv-v12/bin/python v12/scripts/v11_logits.py   -> v12/runs/v11_full/logits/*.npz
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'v12'))
from model import Model  # noqa: E402
from polyglot12.common import read_jsonl  # noqa: E402

OUT = ROOT / 'v12' / 'runs' / 'v11_full' / 'logits'


def main():
    m = Model.load(ROOT / 'artifacts' / 'full.npz')
    OUT.mkdir(parents=True, exist_ok=True)
    files = {s: ROOT / 'v12' / 'data' / 'v11' / f'{s}.jsonl' for s in ('train', 'validation', 'test', 'test_stress', 'audit')}
    files['contrast'] = ROOT / 'v12' / 'data' / 'contrast.jsonl'
    files['human_test'] = ROOT / 'v12' / 'data' / 'human_test.jsonl'
    for split, path in files.items():
        if not path.exists():
            continue
        rows = read_jsonl(path)
        z = np.array([m.logits(r['text']) for r in rows], dtype=np.float32)
        np.savez_compressed(OUT / f'{split}.npz', ids=np.array([r['id'] for r in rows]), intent=z[:, :6], sentiment=z[:, 6:])
        print(split, z.shape)
    print('stored v1.1 calibration:', m.temperature, m.threshold)


if __name__ == '__main__':
    main()
