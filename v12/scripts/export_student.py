"""Phase 6: export a trained student to the torch-free deploy format, quantize, verify, calibrate.

    .venv-v12/bin/python v12/scripts/export_student.py --student runs/03_student/runs/student_kd_s42/student.pt \
        --out deploy/student_kd_s42 [--name student_kd_s42]

Steps
 1. embeddings.npz + encoder.onnx (opset 17, dynamic token axis)
 2. encoder.int8.onnx via onnxruntime dynamic quantization (INT8 weights for MatMul/Gemm)
 3. parity: torch vs ONNX-fp32 logits on every evaluation message (max |diff|)
 4. logits of the fp32 and int8 runtimes for validation/test/test_stress/audit/contrast[/human_test]
    -> runs/deploy_<name>_{fp32,int8}/logits/*.npz (scored later by the shared evaluator)
 5. calibration (v1.1 algorithm) fitted on the deployed INT8 model's VALIDATION logits -> meta.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.common import INTENTS, SENTIMENTS, read_jsonl  # noqa: E402
from polyglot12.evaluate import calibrate  # noqa: E402
from polyglot12.runtime import StudentRuntime  # noqa: E402
from polyglot12.student import Student, collate, hash_row  # noqa: E402

SPLITS = {'validation': 'data/v11/validation.jsonl', 'test': 'data/v11/test.jsonl', 'test_stress': 'data/v11/test_stress.jsonl',
          'audit': 'data/v11/audit.jsonl', 'contrast': 'data/contrast.jsonl', 'human_test': 'data/human_test.jsonl'}


class EncoderWrapper(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.s = student

    def forward(self, X, mask):
        return self.s.encode(X, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    ck = torch.load(V12 / args.student, map_location='cpu')
    model = Student(**ck['model_cfg'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    out = V12 / args.out
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or out.name

    arrays = {'pieces': model.pieces.weight.detach().numpy().astype(np.float32)}
    if ck['model_cfg']['linear_branch']:
        arrays['linear_w'] = model.linear.weight.detach().numpy().astype(np.float32)
        arrays['linear_b'] = model.lin_bias.detach().numpy().astype(np.float32)
    np.savez(out / 'embeddings.npz', **arrays)

    d = ck['model_cfg']['d']
    X = torch.randn(1, 12, d)
    mask = torch.ones(1, 12)
    wrapper = EncoderWrapper(model).eval()
    torch.onnx.export(wrapper, (X, mask), str(out / 'encoder.onnx'), input_names=['X', 'mask'],
                      output_names=['logits', 'alpha'], opset_version=17, dynamo=False,
                      dynamic_axes={'X': {1: 'T'}, 'mask': {1: 'T'}, 'alpha': {1: 'T'}})
    model.eval()  # the exporter restores the wrapper's training flag recursively; make sure dropout is off
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(out / 'encoder.onnx'), str(out / 'encoder.int8.onnx'), weight_type=QuantType.QInt8)

    meta = {'name': name, 'model_cfg': ck['model_cfg'], 'source_checkpoint': args.student,
            'params': model.param_breakdown(), 'intents': INTENTS, 'sentiments': SENTIMENTS,
            'calibration': {'intent': {'temperature': 1.0, 'review_threshold': 0.0},
                            'sentiment': {'temperature': 1.0, 'review_threshold': 0.0}},
            'files_bytes': {}}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))

    rt = {p: StudentRuntime(out, p) for p in ('fp32', 'int8')}
    report = {'parity_torch_vs_onnx_fp32_max_abs': 0.0, 'splits': {}}
    for split, path in SPLITS.items():
        p = V12 / path
        if not p.exists():
            continue
        rows = read_jsonl(p)
        with torch.no_grad():
            zt = torch.cat([model(collate([hash_row(r['text'], ck['model_cfg']['n_buckets'])], 'cpu'))[0] for r in rows]).numpy()
        Z = {k: np.stack([rt[k].logits(r['text'])[0] for r in rows]) for k in rt}
        report['parity_torch_vs_onnx_fp32_max_abs'] = max(report['parity_torch_vs_onnx_fp32_max_abs'], float(np.abs(zt - Z['fp32']).max()))
        agree = {h: float((Z['fp32'][:, sl].argmax(1) == Z['int8'][:, sl].argmax(1)).mean())
                 for h, sl in (('intent', slice(0, 6)), ('sentiment', slice(6, 9)))}
        report['splits'][split] = {'rows': len(rows), 'int8_vs_fp32_argmax_agreement': agree,
                                   'int8_vs_fp32_max_abs_logit_diff': float(np.abs(Z['fp32'] - Z['int8']).max())}
        for k, z in Z.items():
            ldir = V12 / 'runs' / f'deploy_{name}_{k}' / 'logits'
            ldir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ldir / f'{split}.npz', ids=np.array([r['id'] for r in rows]),
                                intent=z[:, :6].astype(np.float32), sentiment=z[:, 6:].astype(np.float32))
            if split == 'validation':
                y = {h: np.array([lab.index(r[h]) for r in rows]) for h, lab in (('intent', INTENTS), ('sentiment', SENTIMENTS))}
                if k == 'int8':
                    for h, sl in (('intent', slice(0, 6)), ('sentiment', slice(6, 9))):
                        T, th = calibrate(z[:, sl], y[h])
                        meta['calibration'][h] = {'temperature': T, 'review_threshold': th,
                                                  'fitted_on': 'validation logits of the INT8 runtime'}
    meta['files_bytes'] = {f.name: f.stat().st_size for f in out.iterdir() if f.is_file()}
    meta['export_report'] = report
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'params': meta['params'], 'files_bytes': meta['files_bytes'], **report}, indent=1))


if __name__ == '__main__':
    main()
