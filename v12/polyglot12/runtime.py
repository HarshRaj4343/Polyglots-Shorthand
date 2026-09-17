"""Deployed student runtime: NumPy (hashing, piece-embedding mean, linear branch) + onnxruntime (encoder).

No torch at inference. A deploy directory contains:
  embeddings.npz     pieces (n_buckets x d), linear_w (DIM x 9), linear_b (9)   [linear_* absent without the branch]
  encoder.onnx       fp32 encoder: inputs X (1,T,d), mask (1,T) -> logits (1,9), alpha (1,T)
  encoder.int8.onnx  dynamic INT8 quantization of encoder.onnx
  meta.json          model config, calibration {intent|sentiment: temperature, review_threshold}, provenance
"""
import json
from pathlib import Path

import numpy as np

from .common import INTENTS, SENTIMENTS
from .hashing import features, token_pieces


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class StudentRuntime:
    def __init__(self, deploy_dir, precision='int8', threads=1):
        import onnxruntime as ort
        d = Path(deploy_dir)
        self.meta = json.loads((d / 'meta.json').read_text())
        emb = np.load(d / 'embeddings.npz')
        self.pieces = emb['pieces']
        self.linear_w = emb['linear_w'] if 'linear_w' in emb else None
        self.linear_b = emb['linear_b'] if 'linear_b' in emb else None
        self.n_buckets = self.meta['model_cfg']['n_buckets']
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        name = 'encoder.int8.onnx' if precision == 'int8' else 'encoder.onnx'
        self.session = ort.InferenceSession(str(d / name), so, providers=['CPUExecutionProvider'])
        self.precision = precision

    def logits(self, text):
        """Raw 9 logits (uncalibrated) and attention-pooling weights, batch size 1."""
        toks, flat, lens = token_pieces(text, self.n_buckets)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        X = (np.add.reduceat(self.pieces[flat], starts, axis=0) / lens[:, None])[None].astype(np.float32)
        mask = np.ones((1, len(lens)), np.float32)
        z, alpha = self.session.run(None, {'X': X, 'mask': mask})
        z = z[0].astype(np.float64)
        if self.linear_w is not None:
            ix, vals = features(text)
            z = z + vals @ self.linear_w[ix] + self.linear_b
        return z, toks, alpha[0]

    def predict(self, text, with_attention=False):
        """Same JSON shape as v1.1 Model.predict."""
        z, toks, alpha = self.logits(text)
        cal = self.meta['calibration']
        out = {}
        for key, labels, zz in (('intent', INTENTS, z[:6]), ('sentiment', SENTIMENTS, z[6:])):
            p = _softmax(zz / cal[key]['temperature'])
            j = int(np.argmax(p))
            out[key] = {'label': labels[j], 'confidence': float(p[j]),
                        'review_required': bool(float(p[j]) < cal[key]['review_threshold'])}
        if with_attention:
            out['attention'] = [[t, round(float(a), 4)] for t, a in zip(toks, alpha)]
        return out
