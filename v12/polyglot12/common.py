"""Labels, JSONL I/O, seeding, platform detection and checkpoint helpers shared by all v1.2 scripts."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

INTENTS = ['cancel_order', 'refund', 'track_order', 'not_received', 'damaged_item', 'feedback']
SENTIMENTS = ['negative', 'neutral', 'positive']

# The 14-entry v1.1 stress map (solution.stress_text), copied verbatim. Training-time
# noise must never produce any of these outputs, so the stress test stays unseen.
STRESS_MAP = {'nahi':'nahii','nhi':'nahii','kar':'karr','kr':'karr','hai':'hey',
              'h':'hey','mila':'milaa','mera':'meraa','kya':'kyaa','please':'pleez',
              'order':'orrder','delivery':'delivary','refund':'refnd','service':'serrvice'}
STRESS_OUTPUTS = frozenset(STRESS_MAP.values())


def stress_text(text):
    return ' '.join(STRESS_MAP.get(t, t) for t in text.lower().split())


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def text_sha(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def label_index(row, key):
    """Index of the row's label for 'intent'/'sentiment', or -1 when unlabeled."""
    labels = INTENTS if key == 'intent' else SENTIMENTS
    v = row.get(key)
    return labels.index(v) if v in labels else -1


def class_weights(ys, k):
    """v1.1 class balancing N/(K*n_c), over labeled rows only (y >= 0)."""
    ys = np.asarray([y for y in ys if y >= 0])
    counts = np.bincount(ys, minlength=k) if len(ys) else np.zeros(k)
    return (len(ys) / (k * np.maximum(counts, 1))).astype(np.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_platform():
    if 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU'):
        return 'colab'
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or Path('/kaggle/working').exists():
        return 'kaggle'
    return 'local'


def persist_dir(notebook, local_root=None):
    """Directory that survives a disconnect: Drive on Colab, /kaggle/working on Kaggle."""
    plat = detect_platform()
    if plat == 'colab':
        base = Path('/content/drive/MyDrive/polyglot12')
    elif plat == 'kaggle':
        base = Path('/kaggle/working')
    else:
        base = Path(local_root) if local_root else Path(__file__).resolve().parents[1] / 'runs'
    d = base / notebook
    d.mkdir(parents=True, exist_ok=True)
    return d


def rng_state():
    import torch
    st = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    import torch
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'])
    if 'cuda' in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


def atomic_torch_save(obj, path):
    """Write to a temp file then rename, so a disconnect mid-write never corrupts the last checkpoint."""
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)
