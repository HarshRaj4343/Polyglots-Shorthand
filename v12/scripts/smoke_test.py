"""CPU smoke test of every v1.2 training script (a few steps on 32 rows).

    .venv-v12/bin/python v12/scripts/smoke_test.py      (writes v12/results/smoke_test.json)

1. hash-consistency test (50 fixtures)
2. augmentation safety: never emits a stress-map output, never edits negators/emoji/punctuation,
   never changes the token count (checked over every v1.1 train/val/test text x 20 seeds)
3. teacher trainer with all four candidate tokenizers/configs (tiny random weights), 2 epochs:
   simulated disconnect after epoch 1 -> resume -> finish -> rerun is skipped
4. student trainer, full size (d=256, 4 blocks): KD + consistency + pool rows, with the same
   disconnect/resume check; plus 1-epoch no-KD and no-linear-branch variants
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.augment import PROTECTED, noisy_variant  # noqa: E402
from polyglot12.common import STRESS_OUTPUTS, read_jsonl, write_jsonl  # noqa: E402

PY = sys.executable
SMOKE = V12 / 'runs' / 'smoke'


def sh(args, expect_fail=False):
    t = time.time()
    p = subprocess.run([PY, *args], cwd=V12, capture_output=True, text=True)
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    if not ok:
        print(p.stdout[-4000:], p.stderr[-4000:])
        raise SystemExit(f'FAILED: {args}')
    return p.stdout + p.stderr, time.time() - t


def main():
    import shutil
    shutil.rmtree(SMOKE, ignore_errors=True)
    SMOKE.mkdir(parents=True)
    report = {}

    out, secs = sh(['tests/test_hashing.py'])
    report['hash_test'] = {'output': out.strip(), 'seconds': round(secs, 1)}
    print(out.strip())

    texts = [r['text'] for s in ('train', 'validation', 'test') for r in read_jsonl(V12 / 'data/v11' / f'{s}.jsonl')]
    n_changed = n_total = 0
    for seed in range(20):
        for i, t in enumerate(texts):
            v = noisy_variant(t, seed * 100003 + i)
            a, b = t.split(' '), v.split(' ')
            assert len(a) == len(b), (t, v)
            for x, y in zip(a, b):
                if x != y:
                    assert y.lower() not in STRESS_OUTPUTS, (x, y)
                    assert x.lower() not in PROTECTED and x.isalpha() and x.isascii(), (x, y)
            n_changed += v != t
            n_total += 1
    report['augment_safety'] = {'texts_checked': n_total, 'fraction_changed': round(n_changed / n_total, 3),
                                'stress_outputs_emitted': 0, 'protected_or_symbol_tokens_edited': 0}
    print('augment safety ok:', report['augment_safety'])

    train = read_jsonl(V12 / 'data/v11/train.jsonl')[:32]
    val = read_jsonl(V12 / 'data/v11/validation.jsonl')[:16]
    write_jsonl(SMOKE / 'train32.jsonl', train)
    write_jsonl(SMOKE / 'val16.jsonl', val)

    # ---- teacher
    tcfg = json.loads((V12 / 'configs/teacher_v11data.json').read_text())
    tcfg.update({'notebook': 'smoke_teacher', 'learning_rates': [5e-5], 'max_epochs': 2, 'patience': 2, 'batch_size': 8,
                 'train_files': [{'path': str(SMOKE / 'train32.jsonl')}], 'validation_file': str(SMOKE / 'val16.jsonl'),
                 'predict_files': {'validation': str(SMOKE / 'val16.jsonl'), 'train': str(SMOKE / 'train32.jsonl')}})
    (SMOKE / 'teacher.json').write_text(json.dumps(tcfg))
    tout = SMOKE / 'teacher_out'
    base = ['-m', 'polyglot12.teacher', '--config', str(SMOKE / 'teacher.json'), '--smoke', '--out', str(tout)]
    o1, s1 = sh(base + ['--stop-after-epochs', '1'], expect_fail=True)
    assert 'simulated disconnect' in o1
    o2, s2 = sh(base)
    o3, s3 = sh(base)
    summ = json.loads((tout / 'summary.json').read_text())
    runs = summ['runs']
    first = next(iter(runs.values()))
    assert first['resumed_at_epochs'] == [1], first
    for name, r in runs.items():
        assert r['state'] == 'done', (name, r)
        z = np.load(tout / 'runs' / name / 'logits' / 'validation.npz')
        assert z['intent'].shape == (16, 6) and z['sentiment'].shape == (16, 3) and np.isfinite(z['intent']).all()
    assert o3.count('already done, skipping') == len(runs)
    report['teacher'] = {'runs': {k: {'state': v['state'], 'val_nll': round(v['best_val_nll'], 4), 'epochs': v['epochs_run'],
                                      'resumed_at_epochs': v['resumed_at_epochs']} for k, v in runs.items()},
                         'selected': summ['selected_run'], 'seconds_crash_resume_rerun': [round(s1, 1), round(s2, 1), round(s3, 1)],
                         'note': 'tiny random encoders (2 layers, hidden 32) with the real tokenizers/configs; the first candidate was interrupted after epoch 1 and resumed'}
    print('teacher smoke ok:', json.dumps(report['teacher']['runs']))

    # ---- student
    rng = np.random.default_rng(0)
    kd_rows = [dict(r, t_intent_logits=rng.normal(size=6).round(3).tolist(), t_sentiment_logits=rng.normal(size=3).round(3).tolist(),
                    kd_intent_mask=bool(i % 3)) for i, r in enumerate(train)]
    pool_rows = [{'id': f'pool-{i}', 'text': r['text'], 't_intent_logits': None, 't_sentiment_logits': rng.normal(size=3).round(3).tolist()}
                 for i, r in enumerate(read_jsonl(V12 / 'data/v11/test.jsonl')[:16])]  # smoke only; texts irrelevant
    write_jsonl(SMOKE / 'kd32.jsonl', kd_rows)
    write_jsonl(SMOKE / 'pool16.jsonl', pool_rows)
    scfg = json.loads((V12 / 'configs/student.json').read_text())
    scfg.update({'notebook': 'smoke_student', 'max_epochs': 2, 'patience': 2, 'batch_size': 16, 'pool_per_epoch': 8,
                 'train_files': [{'path': str(SMOKE / 'kd32.jsonl'), 'intent': True, 'sentiment': True, 'kd': True},
                                 {'path': str(SMOKE / 'pool16.jsonl'), 'intent': False, 'sentiment': False, 'kd': True}],
                 'validation_file': str(SMOKE / 'val16.jsonl'), 'predict_files': {'validation': str(SMOKE / 'val16.jsonl')}})
    for v in scfg['variants']:
        v['max_epochs'] = 2 if v['name'] == 'student_kd_s42' else 1
    (SMOKE / 'student.json').write_text(json.dumps(scfg))
    sout = SMOKE / 'student_out'
    sbase = ['-m', 'polyglot12.train_student', '--config', str(SMOKE / 'student.json'), '--out', str(sout)]
    o1, s1 = sh(sbase + ['--only', 'student_kd_s42', '--stop-after-epochs', '1'], expect_fail=True)
    assert 'simulated disconnect' in o1
    o2, s2 = sh(sbase + ['--only', 'student_kd_s42,student_nokd_s42,student_kd_nolinear_s42'])
    assert 'resumed after epoch 1' in o2
    sruns = json.loads((sout / 'summary.json').read_text())['runs']
    for name, r in sruns.items():
        assert r['state'] == 'done', (name, r)
        z = np.load(sout / 'runs' / name / 'logits' / 'validation.npz')
        assert z['intent'].shape == (16, 6) and np.isfinite(z['intent']).all()
    kd = sruns['student_kd_s42']
    assert kd['history'][0]['kd'] > 0 and kd['history'][0]['cons'] > 0 and kd['pool_rows'] == 16
    assert sruns['student_nokd_s42']['history'][0]['kd'] == 0 and sruns['student_nokd_s42']['pool_rows'] == 0
    report['student'] = {k: {'val_nll': round(v['best_val_nll'], 4), 'epochs': v['epochs_run'], 'params': v['params'],
                             'train_rows': v['train_rows'], 'pool_rows': v['pool_rows'],
                             'first_epoch_losses': {x: round(v['history'][0][x], 4) for x in ('ce', 'kd', 'cons')},
                             'seconds_per_epoch_cpu': round(v['history'][0]['seconds'], 2)} for k, v in sruns.items()}
    report['student_seconds_crash_resume'] = [round(s1, 1), round(s2, 1)]
    print('student smoke ok:', json.dumps({k: (v['val_nll'], v['params']['total']) for k, v in report['student'].items()}))
    (V12 / 'results').mkdir(exist_ok=True)
    (V12 / 'results' / 'smoke_test.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('SMOKE TEST PASSED -> results/smoke_test.json')


if __name__ == '__main__':
    main()
