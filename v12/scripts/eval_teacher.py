"""Phase 1 local evaluation: teacher trained on v1.1 train only -> results/teacher_v11data.json.

The selected run (lowest validation NLL, chosen inside the notebook) is the reported teacher.
All other runs are listed for transparency, but were NOT used to pick anything.

    .venv-v12/bin/python v12/scripts/eval_teacher.py [runs/01_teacher] [results/teacher_v11data.json]
"""
import json
import sys
from pathlib import Path

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
from polyglot12.evaluate import bootstrap_samples, evaluate_system, load_rows, strip_private  # noqa: E402


def compact(r):
    t = r['test']
    return {'test_intent_macro_f1': t['intent']['macro_f1'], 'test_intent_ci95': t['intent']['macro_f1_ci95'],
            'test_sentiment_macro_f1': t['sentiment']['macro_f1'], 'test_sentiment_ci95': t['sentiment']['macro_f1_ci95'],
            'paired_vs_v11': {h: r['paired_vs_reference'].get(h) for h in ('intent', 'sentiment')},
            'positive_recall': r['positive_recall'],
            'stress': r.get('stress'), 'emoji_pairs': r['emoji_pairs'],
            'audit_errors': r.get('audit_errors', {}).get('n_rows_with_error'),
            'contrast_both_acc': {k: v['both_acc'] for k, v in r.get('contrast', {}).items()},
            'contrast_pairs_fully_correct': {k: f"{v['pairs_fully_correct']}/{v['pairs']}" for k, v in r.get('contrast', {}).items()},
            'ece': {h: t[h]['ece'] for h in ('intent', 'sentiment')},
            'coverage': {h: t[h]['coverage'] for h in ('intent', 'sentiment')},
            'accepted_accuracy': {h: t[h]['accepted_accuracy'] for h in ('intent', 'sentiment')},
            'calibration': r['calibration']}


def main():
    run_root = V12 / (sys.argv[1] if len(sys.argv) > 1 else 'runs/01_teacher')
    out = V12 / (sys.argv[2] if len(sys.argv) > 2 else 'results/teacher_v11data.json')
    summary = json.loads((run_root / 'summary.json').read_text())
    samples = bootstrap_samples(load_rows('test'))
    base = evaluate_system(V12 / 'runs/v11_full', 'v1.1 full', samples=samples)
    ref = {'name': 'v1.1 full', 'boot': base['_boot']}
    runs = {}
    for name, st in summary['runs'].items():
        if st.get('state') != 'done':
            runs[name] = {'state': st.get('state')}
            continue
        r = evaluate_system(run_root / 'runs' / name, name, reference=ref, samples=samples)
        runs[name] = {'state': 'done', 'model_id': st['model_id'], 'lr': st['lr'], 'best_val_nll': st['best_val_nll'],
                      'best_epoch': st['best_epoch'], 'epochs_run': st['epochs_run'], 'n_params': st['n_params'],
                      'report': strip_private(r)}
    sel = summary['selected_run']
    result = {
        'phase': 1, 'description': 'Teacher fine-tuned on the v1.1 TRAIN split only; selection by validation NLL.',
        'protocol': 'v1.1 evaluator (reproduces whitepaper v1.1 numbers exactly); calibration on validation; '
                    'group bootstrap 5,000 resamples seed 0; paired deltas vs v1.1 on identical resamples.',
        'training_platform': {k: summary.get(k) for k in ('platform', 'gpu', 'wall_minutes')},
        'selected_run': sel,
        'selection_rule': summary['selection_rule'],
        'selected': {**{k: runs[sel][k] for k in ('model_id', 'lr', 'best_val_nll', 'best_epoch', 'n_params')},
                     'summary': compact(runs[sel]['report']), 'report': runs[sel]['report']},
        'v11_full': {'summary': compact(base), 'report': strip_private(base)},
        'all_runs_for_transparency_not_used_for_selection': {
            k: ({'model_id': v['model_id'], 'lr': v['lr'], 'best_val_nll': v['best_val_nll'], 'best_epoch': v['best_epoch'],
                 **compact(v['report'])} if v['state'] == 'done' else v) for k, v in runs.items()},
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('wrote', out)
    rows = [('v1.1 full', None, compact(base))] + [(k, v['best_val_nll'], compact(v['report'])) for k, v in runs.items() if v['state'] == 'done']
    print(f"{'system':42s} {'valNLL':>6s} {'int F1 [CI]':>20s} {'sent F1 [CI]':>20s} {'dSent':>6s} {'posR':>5s} {'stressI':>7s} {'emoji':>5s} {'aud':>3s} {'contrast':>8s}")
    for name, v, c in rows:
        d = c['paired_vs_v11']['sentiment']
        print(f"{name[:42]:42s} {('%.3f' % v) if v is not None else '  -   ':>6s} "
              f"{c['test_intent_macro_f1']:.3f} [{c['test_intent_ci95'][0]:.2f},{c['test_intent_ci95'][1]:.2f}] "
              f"{c['test_sentiment_macro_f1']:.3f} [{c['test_sentiment_ci95'][0]:.2f},{c['test_sentiment_ci95'][1]:.2f}] "
              f"{('%+.3f' % d['mean_delta']) if d else '   -  ':>6s} {c['positive_recall']:.2f} {c['stress']['intent']['macro_f1']:.3f}   "
              f"{c['emoji_pairs']['both_correct']}/{c['emoji_pairs']['n_pairs']} {c['audit_errors']:>3d} {c['contrast_both_acc']['all']:.3f}")


if __name__ == '__main__':
    main()
