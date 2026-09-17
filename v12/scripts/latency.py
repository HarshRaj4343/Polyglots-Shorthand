"""Phase 6 latency on the Mac CPU, whitepaper protocol (v1.1 solution.benchmark):
80 warm-up + 600 timed calls, batch 1, warm and serial; each call = hashing + embedding mean + ONNX encoder +
linear branch + calibration + JSON serialization. Workloads: real test messages and one sentence repeated to
32/128/512 code points. Excludes process start-up and model loading.

    .venv-v12/bin/python v12/scripts/latency.py deploy/student_kd_s42 [--out results/latency_student_kd_s42.json]
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V12 = ROOT / 'v12'
sys.path.insert(0, str(V12))
from polyglot12.common import read_jsonl  # noqa: E402
from polyglot12.runtime import StudentRuntime  # noqa: E402

SAMPLE = 'mera order delivered bol raha hai but mila hi nahi 😒 '


def measure(predict, texts, repeats=600, warmup=80):
    for i in range(warmup):
        json.dumps(predict(texts[i % len(texts)]), ensure_ascii=False)
    times = []
    begin = time.perf_counter_ns()
    for i in range(repeats):
        s = time.perf_counter_ns()
        json.dumps(predict(texts[i % len(texts)]), ensure_ascii=False)
        times.append((time.perf_counter_ns() - s) / 1e6)
    elapsed = (time.perf_counter_ns() - begin) / 1e9
    return {'requests': repeats, 'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)), 'messages_per_second': repeats / elapsed}


def bench(predict, texts):
    return {'test_messages': measure(predict, texts),
            'fixed_length_synthetic': {str(n): measure(predict, [(SAMPLE * 20)[:n]]) for n in (32, 128, 512)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deploy')
    ap.add_argument('--out')
    args = ap.parse_args()
    texts = [r['text'] for r in read_jsonl(V12 / 'data/v11/test.jsonl')]
    result = {'protocol': 'Warm, serial, batch=1, 80 warm-up + 600 timed calls. Includes hashing, piece-embedding mean, ONNX encoder, '
                          'linear branch, calibration and JSON serialization. Excludes process start-up, model loading, network, queueing.',
              'configs': {}}
    for precision in ('int8', 'fp32'):
        for threads in (1, 2, 4):
            rt = StudentRuntime(V12 / args.deploy, precision, threads)
            key = f'{precision}_threads{threads}'
            result['configs'][key] = bench(rt.predict, texts)
            r = result['configs'][key]
            print(key, 'test p50/p95/p99 %.3f/%.3f/%.3f ms' % (r['test_messages']['p50_ms'], r['test_messages']['p95_ms'], r['test_messages']['p99_ms']),
                  '| 512cp p95 %.3f ms' % r['fixed_length_synthetic']['512']['p95_ms'], flush=True)
    # v1.1 on the same machine, same protocol, for reference
    sys.path.insert(0, str(ROOT))
    from model import Model
    v11 = Model.load(ROOT / 'artifacts/full.npz')
    result['v11_full_same_machine'] = bench(v11.predict, texts)
    print('v1.1 test p95 %.3f ms | 512cp p95 %.3f ms' % (result['v11_full_same_machine']['test_messages']['p95_ms'],
                                                          result['v11_full_same_machine']['fixed_length_synthetic']['512']['p95_ms']))
    import onnxruntime
    cpu = platform.processor()
    try:
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    except Exception:
        pass
    result['environment'] = {'python': sys.version.split()[0], 'numpy': np.__version__, 'onnxruntime': onnxruntime.__version__,
                             'system': platform.system(), 'release': platform.release(), 'machine': platform.machine(), 'cpu': cpu}
    meta = json.loads((V12 / args.deploy / 'meta.json').read_text())
    result['model'] = {'name': meta['name'], 'params': meta['params'], 'files_bytes': meta['files_bytes']}
    out = V12 / (args.out or f'results/latency_{meta["name"]}.json')
    out.write_text(json.dumps(result, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
