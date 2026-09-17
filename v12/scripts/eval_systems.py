"""Phase 5: one table for every system, scored by the shared v1.1-protocol evaluator.

    .venv-v12/bin/python v12/scripts/eval_systems.py  -> results/phase5_comparison.json, results/phase5_table.md

Rows that have not been trained are reported as "not run". Paired bootstrap deltas are vs v1.1 on identical
resamples (5,000 test-group resamples, seed 0). Data ablation: teacher v2 vs Phase 1 teacher (same encoder,
same selection rule, different training data). Test/audit/contrast are never used for any choice here.
"""
import json
import sys
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.evaluate import bootstrap_samples, evaluate_system, load_rows, strip_private  # noqa: E402

V11_PARAMS = 32768 * 9 + 9 + 532_800


def selected(run_root):
    s = V12 / run_root / 'summary.json'
    if not s.exists():
        return None
    sel = json.loads(s.read_text()).get('selected_run')
    return (V12 / run_root / 'runs' / sel) if sel else None


def student_dir(name):
    d = V12 / 'runs/03_student/runs' / name
    return d if (d / 'logits' / 'test.npz').exists() else None


def params_of(system_dir, key):
    if key == 'v11_full':
        return V11_PARAMS
    st = system_dir / 'status.json' if system_dir else None
    if st and st.exists():
        s = json.loads(st.read_text())
        return s.get('n_params') or s.get('params', {}).get('total')
    m = V12 / 'deploy' / 'student_kd_s42' / 'meta.json'
    if key.startswith('deploy') and m.exists():
        return json.loads(m.read_text())['params']['total']
    return None


def latency_of(key):
    lat = V12 / 'results' / 'latency_student_kd_s42.json'
    if not lat.exists():
        return None
    L = json.loads(lat.read_text())
    pick = lambda b: {'p50_ms': b['test_messages']['p50_ms'], 'p95_ms': b['test_messages']['p95_ms'], 'p99_ms': b['test_messages']['p99_ms'],
                      'p95_ms_512cp': b['fixed_length_synthetic']['512']['p95_ms']}
    if key == 'v11_full':
        return pick(L['v11_full_same_machine'])
    if key == 'deploy_int8':
        return {k: pick(v) for k, v in L['configs'].items() if k.startswith('int8')}
    if key == 'student_kd_s42':
        return {k: pick(v) for k, v in L['configs'].items() if k.startswith('fp32')}
    return None


def row(report, key, label, system_dir):
    t = report['test']
    return {
        'system': label, 'key': key,
        'intent_macro_f1': t['intent']['macro_f1'], 'intent_ci95': t['intent']['macro_f1_ci95'],
        'sentiment_macro_f1': t['sentiment']['macro_f1'], 'sentiment_ci95': t['sentiment']['macro_f1_ci95'],
        'paired_vs_v11': report.get('paired_vs_reference') or None,
        'positive_recall': report['positive_recall'],
        'stress': report.get('stress'), 'emoji_pairs': report['emoji_pairs'],
        'audit_errors': report.get('audit_errors', {}).get('n_rows_with_error'),
        'contrast': {k: {'both_acc': v['both_acc'], 'pairs': f"{v['pairs_fully_correct']}/{v['pairs']}",
                         **({'intent_acc': v['intent_acc'], 'sentiment_acc': v['sentiment_acc']} if k != 'all' else {})}
                     for k, v in report.get('contrast', {}).items()},
        'ece': {h: t[h]['ece'] for h in ('intent', 'sentiment')},
        'coverage': {h: t[h]['coverage'] for h in ('intent', 'sentiment')},
        'accepted_accuracy': {h: t[h]['accepted_accuracy'] for h in ('intent', 'sentiment')},
        'calibration': report['calibration'],
        'human_test': ({h: report['human_test'][h]['macro_f1'] for h in ('intent', 'sentiment')} if 'human_test' in report else None),
        'params': params_of(system_dir, key), 'latency': latency_of(key),
    }


def fmt_ci(v, ci):
    return f'{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]'


def main():
    samples = bootstrap_samples(load_rows('test'))
    systems = [
        ('v11_full', 'v1.1 full (NumPy linear + attention)', V12 / 'runs/v11_full'),
        ('teacher_v11data', 'Teacher, v1.1 data (Phase 1)', selected('runs/01_teacher')),
        ('teacher_v2', 'Teacher v2 (Phase 3)', selected('runs/02_teacher_kd')),
        ('student_kd_s42', 'Student + KD (seed 42)', student_dir('student_kd_s42')),
        ('student_nokd_s42', 'Student no-KD (seed 42)', student_dir('student_nokd_s42')),
        ('student_kd_nolinear_s42', 'Student + KD w/o linear branch', student_dir('student_kd_nolinear_s42')),
        ('student_kd_s43', 'Student + KD (seed 43)', student_dir('student_kd_s43')),
        ('student_kd_s44', 'Student + KD (seed 44)', student_dir('student_kd_s44')),
        ('deploy_int8', 'Student + KD, deployed ONNX INT8', (V12 / 'runs/deploy_student_kd_s42_int8') if (V12 / 'runs/deploy_student_kd_s42_int8/logits/test.npz').exists() else None),
    ]
    base = evaluate_system(V12 / 'runs/v11_full', 'v1.1', samples=samples)
    ref = {'name': 'v1.1 full', 'boot': base['_boot']}
    reports, rows = {}, []
    for key, label, d in systems:
        if d is None:
            rows.append({'system': label, 'key': key, 'status': 'not run'})
            continue
        r = base if key == 'v11_full' else evaluate_system(d, label, reference=ref, samples=samples)
        reports[key] = r
        rows.append({'status': 'done', **row(r, key, label, d)})
    out = {'protocol': 'shared v1.1 evaluator; calibration on validation; 5,000 group-bootstrap resamples (seed 0); paired deltas vs v1.1',
           'rows': rows}
    if 'teacher_v11data' in reports and 'teacher_v2' in reports:
        out['data_ablation_teacher_v2_minus_teacher_v11data'] = {
            h: {'mean_delta': float((reports['teacher_v2']['_boot'][h] - reports['teacher_v11data']['_boot'][h]).mean()),
                'ci95': [float(np.percentile(reports['teacher_v2']['_boot'][h] - reports['teacher_v11data']['_boot'][h], q)) for q in (2.5, 97.5)],
                'p_delta_gt_0': float(((reports['teacher_v2']['_boot'][h] - reports['teacher_v11data']['_boot'][h]) > 0).mean())}
            for h in ('intent', 'sentiment')}
    def paired(a, b):
        return {h: {'mean_delta': float((reports[a]['_boot'][h] - reports[b]['_boot'][h]).mean()),
                    'ci95': [float(np.percentile(reports[a]['_boot'][h] - reports[b]['_boot'][h], q)) for q in (2.5, 97.5)],
                    'p_delta_gt_0': float(((reports[a]['_boot'][h] - reports[b]['_boot'][h]) > 0).mean())} for h in ('intent', 'sentiment')}
    out['paired_comparisons'] = {}
    for a, b, name in (('student_kd_s42', 'student_nokd_s42', 'kd_effect (student+KD - no-KD, seed 42)'),
                       ('student_kd_s42', 'student_kd_nolinear_s42', 'linear_branch_effect (with - without, seed 42)'),
                       ('student_kd_s42', 'teacher_v2', 'student+KD - teacher v2'),
                       ('deploy_int8', 'student_kd_s42', 'int8 - fp32/torch (student+KD s42)')):
        if a in reports and b in reports:
            out['paired_comparisons'][name] = paired(a, b)
    seeds = [reports[k] for k in ('student_kd_s42', 'student_kd_s43', 'student_kd_s44') if k in reports]
    if len(seeds) > 1:
        out['student_kd_seed_spread'] = {h: {'n_seeds': len(seeds), 'mean': float(np.mean([s['test'][h]['macro_f1'] for s in seeds])),
                                             'sd': float(np.std([s['test'][h]['macro_f1'] for s in seeds], ddof=1))} for h in ('intent', 'sentiment')}
    out['full_reports'] = {k: strip_private(v) for k, v in reports.items()}
    (V12 / 'results/phase5_comparison.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Markdown table
    L = ['| System | Params | Intent F1 [95% CI] | Δ vs v1.1 | Sentiment F1 [95% CI] | Δ vs v1.1 | Pos. recall | Stress F1 int/sent (consist.) | Emoji pairs | Audit err. | Contrast both-correct NS/NN/NG/UA (pairs) | ECE int/sent | Coverage / acc. accepted (sent) | p50/p95/p99 ms |',
         '|---|---:|---|---|---|---|---:|---|---:|---:|---|---|---|---|']
    for r in rows:
        if r['status'] != 'done':
            L.append(f"| {r['system']} | not run |" + ' |' * 12)
            continue
        dv = lambda h: (f"{r['paired_vs_v11'][h]['mean_delta']:+.3f} [{r['paired_vs_v11'][h]['ci95'][0]:+.2f}, {r['paired_vs_v11'][h]['ci95'][1]:+.2f}]"
                        if r['paired_vs_v11'] else '-')
        c = r['contrast']
        cs = '/'.join(f"{c[k]['both_acc']:.2f}" for k in ('negation_scope', 'neutral_nahi', 'negated_negative', 'unseen_affect')) + f" ({c['all']['pairs']})" if c else '-'
        st = r['stress']
        lat = r['latency']
        if isinstance(lat, dict) and 'p95_ms' not in lat:
            lat = lat.get('int8_threads2') or lat.get('fp32_threads2') or lat.get('int8_threads1') or lat.get('fp32_threads1') or next(iter(lat.values()))
        ls = f"{lat['p50_ms']:.2f}/{lat['p95_ms']:.2f}/{lat['p99_ms']:.2f}" if lat else 'not measured'
        L.append(f"| {r['system']} | {r['params']:,} | {fmt_ci(r['intent_macro_f1'], r['intent_ci95'])} | {dv('intent')} | "
                 f"{fmt_ci(r['sentiment_macro_f1'], r['sentiment_ci95'])} | {dv('sentiment')} | {r['positive_recall']:.2f} | "
                 f"{st['intent']['macro_f1']:.3f}/{st['sentiment']['macro_f1']:.3f} ({st['intent']['prediction_consistency']:.1%}/{st['sentiment']['prediction_consistency']:.1%}) | "
                 f"{r['emoji_pairs']['both_correct']}/{r['emoji_pairs']['n_pairs']} | {r['audit_errors']} | {cs} | "
                 f"{r['ece']['intent']:.3f}/{r['ece']['sentiment']:.3f} | {r['coverage']['sentiment']:.2f} / {r['accepted_accuracy']['sentiment']:.3f} | {ls} |")
    if 'data_ablation_teacher_v2_minus_teacher_v11data' in out:
        a = out['data_ablation_teacher_v2_minus_teacher_v11data']
        L.append('')
        L.append('Data ablation (teacher v2 − Phase 1 teacher, same encoder): ' + '; '.join(
            f"{h} {a[h]['mean_delta']:+.3f} [{a[h]['ci95'][0]:+.3f}, {a[h]['ci95'][1]:+.3f}], P(Δ>0)={a[h]['p_delta_gt_0']:.2f}" for h in a))
    for name, d in out.get('paired_comparisons', {}).items():
        L.append(f'Paired {name}: ' + '; '.join(f"{h} {d[h]['mean_delta']:+.3f} [{d[h]['ci95'][0]:+.3f}, {d[h]['ci95'][1]:+.3f}], P(Δ>0)={d[h]['p_delta_gt_0']:.2f}" for h in d))
    if 'student_kd_seed_spread' in out:
        sp = out['student_kd_seed_spread']
        L.append('Student + KD seed spread (42/43/44): ' + '; '.join(f"{h} mean {sp[h]['mean']:.3f} sd {sp[h]['sd']:.3f}" for h in sp))
    (V12 / 'results/phase5_table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
