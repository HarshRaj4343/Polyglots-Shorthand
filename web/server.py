# ============================================================================
# web/server.py - minimal FastAPI backend for the web frontend.
# ----------------------------------------------------------------------------
# Run:  .venv/bin/uvicorn web.server:app --reload --port 8000
# Serves the single-page frontend at "/" and two JSON endpoints:
#   GET  /api/meta            the selectable models and their facts
#   POST /api/predict         {"texts": [...], "model": "<key>"} -> one result each
# Models: the v1.2 distilled student (ONNX INT8) plus the v1.1 artifacts/*.npz,
# the same set the Streamlit sidebar offers. Each is loaded on first use and cached.
# ============================================================================

import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / 'static'
ART = ROOT / 'artifacts'
V12_DEPLOY = ROOT / 'v12' / 'deploy' / 'student_kd_s42'
V12_KEY = 'v1.2 student + KD (ONNX INT8)'
MAX_CHARS = 512
MAX_BATCH = 200

sys.path.insert(0, str(ROOT))
from model import INTENTS, SENTIMENTS, Model  # noqa: E402

# Short descriptions, in the order the picker should show them.
MODEL_INFO = {
    V12_KEY:             'v1.2: hashed word/alias/char pieces into 4 transformer blocks plus the v1.0 linear '
                         'branch, distilled from hing-roberta-mixed; 11.9M parameters, INT8 ONNX. '
                         'Best available accuracy.',
    'full.npz':          'All features: words, char n-grams, aliases, emoji context + attention',
    'strip_symbols.npz': 'Like full, but emojis and punctuation are removed first',
    'word_only.npz':     'Words and word pairs only (no char n-grams / aliases) + attention',
    'char_word.npz':     'Words, word pairs and character n-grams + attention',
}


class StudentAdapter:
    """Gives the v1.2 ONNX student the same interface as a v1.1 Model."""

    def __init__(self, deploy_dir):
        sys.path.insert(0, str(ROOT / 'v12'))
        from polyglot12.runtime import StudentRuntime
        self.rt = StudentRuntime(deploy_dir, precision='int8', threads=2)
        cal = self.rt.meta['calibration']
        self.mode = 'full (hashed pieces + transformer)'
        self.att = True
        self.temperature = [cal['intent']['temperature'], cal['sentiment']['temperature']]
        self.threshold = [cal['intent']['review_threshold'], cal['sentiment']['review_threshold']]

    def parameter_count(self):
        return self.rt.meta['params']['total']

    def predict(self, text):
        return self.rt.predict(text)

    def probabilities(self, text):
        z = self.rt.logits(text)[0]
        soft = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        return [soft(z[:6] / self.temperature[0]), soft(z[6:] / self.temperature[1])]

    def attention_weights(self, text):
        _, toks, alpha = self.rt.logits(text)
        return list(zip(toks, (float(a) for a in alpha)))


def _accuracy():
    """Test-set accuracy (190 rows) per model, from the evaluation reports."""
    acc = {}
    try:
        r = json.loads((ART / 'results.json').read_text())['models']
        for mode, v in r.items():
            acc[f'{mode}.npz'] = (v['test']['intent']['accuracy'], v['test']['sentiment']['accuracy'])
    except (OSError, KeyError, ValueError):
        pass
    try:
        c = json.loads((ROOT / 'v12' / 'results' / 'phase5_comparison.json').read_text())
        c = c['full_reports']['deploy_int8']['test']
        acc[V12_KEY] = (c['intent']['accuracy'], c['sentiment']['accuracy'])
    except (OSError, KeyError, ValueError):
        pass
    return acc


ACCURACY = _accuracy()


def _student_available():
    if not (V12_DEPLOY / 'meta.json').exists() or not (V12_DEPLOY / 'encoder.int8.onnx').exists():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _available():
    """Model keys that can actually be loaded here, best first."""
    keys = [V12_KEY] if _student_available() else []
    order = list(MODEL_INFO)
    return keys + sorted((p.name for p in ART.glob('*.npz')),
                         key=lambda n: (order.index(n) if n in MODEL_INFO else len(order), n))


AVAILABLE = _available()
_loaded = {}


def get_model(key):
    """Load on first use, then reuse. Raises KeyError for an unknown key."""
    if key not in AVAILABLE:
        raise KeyError(key)
    if key not in _loaded:
        _loaded[key] = StudentAdapter(V12_DEPLOY) if key == V12_KEY else Model.load(ART / key)
    return _loaded[key]


def _facts(key):
    m = get_model(key)
    a = ACCURACY.get(key, (None, None))
    return {
        'key': key,
        'description': MODEL_INFO.get(key, ''),
        'parameters': int(m.parameter_count()),
        'feature_mode': m.mode,
        'attention': m.att is not None,
        'accuracy': {'intent': a[0], 'sentiment': a[1]},
        'temperature': {'intent': float(m.temperature[0]), 'sentiment': float(m.temperature[1])},
        'review_threshold': {'intent': float(m.threshold[0]), 'sentiment': float(m.threshold[1])},
    }


def analyse(key, text):
    """One message -> labels, confidence, all class probabilities and attention."""
    text = (text or '').strip()
    if not text:
        raise ValueError('message is empty')
    if len(text) > MAX_CHARS:
        raise ValueError(f'message is longer than {MAX_CHARS} characters')

    m = get_model(key)
    out = {'text': text, **m.predict(text)}
    for head, labels, p in zip(('intent', 'sentiment'), (INTENTS, SENTIMENTS), m.probabilities(text)):
        out[head]['probabilities'] = {l: float(v) for l, v in zip(labels, p)}
    out['attention'] = [{'token': t, 'weight': float(a)} for t, a in m.attention_weights(text)]
    return out


class PredictIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    model: str | None = None


app = FastAPI(title="The Polyglot's Shorthand", docs_url='/api/docs')

if not AVAILABLE:
    raise RuntimeError('No model found. Run `python solution.py reproduce` for the v1.1 artifacts.')
get_model(AVAILABLE[0])   # warm the default so the first request is fast


@app.get('/api/meta')
def meta():
    return {
        'models': [_facts(k) for k in AVAILABLE],
        'default': AVAILABLE[0],
        'intents': INTENTS,
        'sentiments': SENTIMENTS,
        'max_chars': MAX_CHARS,
        'max_batch': MAX_BATCH,
    }


@app.post('/api/predict')
def predict(body: PredictIn):
    key = body.model or AVAILABLE[0]
    if key not in AVAILABLE:
        raise HTTPException(status_code=400, detail=f'unknown model {key!r}; available: {AVAILABLE}')

    results = []
    for text in body.texts:
        try:
            results.append(analyse(key, text))
        except ValueError as e:
            results.append({'text': text, 'error': str(e)})
    if len(results) == 1 and 'error' in results[0]:
        raise HTTPException(status_code=400, detail=results[0]['error'])
    return {'model': key, 'results': results}


app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')
