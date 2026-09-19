# ============================================================================
# web/server.py - minimal FastAPI backend for the web frontend.
# ----------------------------------------------------------------------------
# Run:  .venv/bin/uvicorn web.server:app --reload --port 8000
# Serves the single-page frontend at "/" and two JSON endpoints:
#   GET  /api/meta            model facts shown in the page
#   POST /api/predict         {"texts": [...]} -> one result per message
# The model is the v1.2 distilled student (ONNX INT8, v12/deploy/student_kd_s42).
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
DEPLOY = ROOT / 'v12' / 'deploy' / 'student_kd_s42'
MAX_CHARS = 512
MAX_BATCH = 200

INTENTS = ['cancel_order', 'refund', 'track_order', 'not_received', 'damaged_item', 'feedback']
SENTIMENTS = ['negative', 'neutral', 'positive']

sys.path.insert(0, str(ROOT / 'v12'))
from polyglot12.runtime import StudentRuntime  # noqa: E402

runtime = StudentRuntime(DEPLOY, precision='int8', threads=2)
CAL = runtime.meta['calibration']


def _accuracy():
    try:
        t = json.loads((ROOT / 'v12' / 'results' / 'phase5_comparison.json').read_text())
        t = t['full_reports']['deploy_int8']['test']
        return t['intent']['accuracy'], t['sentiment']['accuracy']
    except (OSError, KeyError, ValueError):
        return None, None


ACC_INTENT, ACC_SENTIMENT = _accuracy()


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def analyse(text):
    """One message -> labels, confidence, all class probabilities and attention."""
    text = (text or '').strip()
    if not text:
        raise ValueError('message is empty')
    if len(text) > MAX_CHARS:
        raise ValueError(f'message is longer than {MAX_CHARS} characters')

    z, toks, alpha = runtime.logits(text)
    out = {'text': text}
    for key, labels, zz in (('intent', INTENTS, z[:6]), ('sentiment', SENTIMENTS, z[6:])):
        p = _softmax(zz / CAL[key]['temperature'])
        j = int(np.argmax(p))
        out[key] = {
            'label': labels[j],
            'confidence': float(p[j]),
            'review_required': bool(float(p[j]) < CAL[key]['review_threshold']),
            'probabilities': {l: float(v) for l, v in zip(labels, p)},
        }
    out['attention'] = [{'token': t, 'weight': float(a)} for t, a in zip(toks, alpha)]
    return out


class PredictIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


app = FastAPI(title="The Polyglot's Shorthand", docs_url='/api/docs')


@app.get('/api/meta')
def meta():
    return {
        'model': 'v1.2 student + KD (ONNX INT8)',
        'parameters': runtime.meta['params']['total'],
        'accuracy': {'intent': ACC_INTENT, 'sentiment': ACC_SENTIMENT},
        'temperature': {k: CAL[k]['temperature'] for k in ('intent', 'sentiment')},
        'review_threshold': {k: CAL[k]['review_threshold'] for k in ('intent', 'sentiment')},
        'intents': INTENTS,
        'sentiments': SENTIMENTS,
        'max_chars': MAX_CHARS,
        'max_batch': MAX_BATCH,
    }


@app.post('/api/predict')
def predict(body: PredictIn):
    results = []
    for text in body.texts:
        try:
            results.append(analyse(text))
        except ValueError as e:
            results.append({'text': text, 'error': str(e)})
    if len(results) == 1 and 'error' in results[0]:
        raise HTTPException(status_code=400, detail=results[0]['error'])
    return {'results': results}


app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')
