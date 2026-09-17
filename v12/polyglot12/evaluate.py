"""Local evaluation of any system from saved logits, using the v1.1 protocol.

A "system" is a directory with logits/{validation,test,test_stress,audit,contrast}.npz
(arrays: ids, intent (N,6), sentiment (N,3)). Calibration (temperature grid + review
threshold) is fitted on VALIDATION only, with exactly the v1.1 Model.calibrate algorithm.
Test, audit and contrast are only scored.

Group bootstrap: 5,000 resamples of test groups with replacement, numpy default_rng(0),
groups sorted by id -- identical to the whitepaper, so paired deltas use the same resamples.
"""
from pathlib import Path

import numpy as np

from .common import INTENTS, SENTIMENTS, read_jsonl

V12 = Path(__file__).resolve().parents[1]
HEADS = (('intent', INTENTS), ('sentiment', SENTIMENTS))


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def load_logits(system_dir, split, rows):
    f = np.load(Path(system_dir) / 'logits' / f'{split}.npz')
    pos = {i: k for k, i in enumerate(f['ids'].tolist())}
    order = [pos[r['id']] for r in rows]
    return {'intent': f['intent'][order].astype(np.float64), 'sentiment': f['sentiment'][order].astype(np.float64)}


def calibrate(z, y):
    """v1.1 Model.calibrate for one head: T from 30-point grid by NLL, then max-coverage threshold with >=90% acc."""
    candidates = np.geomspace(0.4, 4.0, 30)
    losses = [np.mean([-np.log(max(float(softmax(zi / t)[yi]), 1e-9)) for zi, yi in zip(z, y)]) for t in candidates]
    T = float(candidates[int(np.argmin(losses))])
    p = softmax(z / T)
    conf, correct = p.max(1), p.argmax(1) == y
    options = []
    for th in sorted(set([0.0] + [float(x) for x in conf])):
        keep = conf >= th
        if keep.sum() >= 5 and correct[keep].mean() >= 0.90:
            options.append((int(keep.sum()), -th))
    return T, (-max(options)[1] if options else 1.01)


def confusion(y, p, k):
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def macro_f1(y, p, k):
    cm = confusion(y, p, k)
    f1 = [2 * cm[i, i] / max(2 * cm[i, i] + cm[:, i].sum() - cm[i, i] + cm[i, :].sum() - cm[i, i], 1) for i in range(k)]
    return float(np.mean(f1))


def summarize(y, p, labels, conf=None, threshold=0.0):
    k = len(labels)
    cm = confusion(y, p, k)
    per = {}
    for i, lab in enumerate(labels):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum()) - tp; fn = int(cm[i, :].sum()) - tp
        per[lab] = {'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                    'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'support': int(cm[i, :].sum())}
    out = {'n': len(y), 'accuracy': float(np.mean(np.asarray(y) == np.asarray(p))), 'macro_f1': macro_f1(y, p, k),
           'per_class': per, 'confusion_matrix': cm.tolist(), 'label_order': labels}
    if conf is not None:
        keep = np.asarray(conf) >= threshold
        out['coverage'] = float(keep.mean())
        out['accepted_accuracy'] = float((np.asarray(y)[keep] == np.asarray(p)[keep]).mean()) if keep.any() else None
    return out


def ece(y, p, conf, bins=10):
    y, p, conf = map(np.asarray, (y, p, conf))
    e = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            e += m.mean() * abs((y[m] == p[m]).mean() - conf[m].mean())
    return float(e)


def bootstrap_samples(rows, n=5000, seed=0):
    groups = sorted({r['group_id'] for r in rows})
    gidx = {g: [i for i, r in enumerate(rows) if r['group_id'] == g] for g in groups}
    rng = np.random.default_rng(seed)
    return [np.concatenate([gidx[groups[g]] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(n)]


def boot_f1(y, p, k, samples):
    y, p = np.asarray(y), np.asarray(p)
    return np.array([macro_f1(y[s], p[s], k) for s in samples])


def emoji_pairs(rows, pred_sent):
    """v1.1 solution.emoji_pairs on test: base (positive) vs base + ' 😒' (negative), per lowercased text."""
    lookup = {}
    for i, r in enumerate(rows):
        lookup[r['text'].lower()] = i
    pairs = []
    for s, i in lookup.items():
        if s.endswith(' 😒') and s[:-2] in lookup:
            j = lookup[s[:-2]]
            if rows[j]['sentiment'] == 'positive' and rows[i]['sentiment'] == 'negative':
                pairs.append(pred_sent[j] == 2 and pred_sent[i] == 0)
    return {'n_pairs': len(pairs), 'both_correct': int(sum(pairs)), 'both_correct_rate': float(np.mean(pairs)) if pairs else None}


def load_rows(split):
    path = V12 / 'data' / ('contrast.jsonl' if split == 'contrast' else f'v11/{split}.jsonl')
    if split == 'human_test':
        path = V12 / 'data' / 'human_test.jsonl'
    return read_jsonl(path) if path.exists() else None


def evaluate_system(system_dir, name, reference=None, samples=None):
    """Full v1.1-protocol report for one system. reference: dict with test predictions of the baseline for paired deltas."""
    rows = {s: load_rows(s) for s in ('validation', 'test', 'test_stress', 'audit', 'contrast', 'human_test')}
    Z = {s: load_logits(system_dir, s, r) for s, r in rows.items()
         if r is not None and (Path(system_dir) / 'logits' / f'{s}.npz').exists()}
    samples = samples if samples is not None else bootstrap_samples(rows['test'])
    y = {s: {h: np.array([labels.index(r[h]) for r in rows[s]]) for h, labels in HEADS} for s in Z}
    report = {'system': name, 'calibration': {}, 'test': {}, 'paired_vs_reference': {}}
    pred = {s: {h: Z[s][h].argmax(1) for h, _ in HEADS} for s in Z}
    for h, labels in HEADS:
        T, th = calibrate(Z['validation'][h], y['validation'][h])
        report['calibration'][h] = {'temperature': T, 'review_threshold': th}
        for s in Z:
            conf = softmax(Z[s][h] / T).max(1)
            summ = summarize(y[s][h], pred[s][h], labels, conf, th)
            summ['ece'] = ece(y[s][h], pred[s][h], conf)
            summ['mean_confidence'] = float(conf.mean())
            report.setdefault(s, {})[h] = summ
        b = boot_f1(y['test'][h], pred['test'][h], len(labels), samples)
        report['test'][h]['macro_f1_ci95'] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        if reference is not None:
            d = b - reference['boot'][h]
            report['paired_vs_reference'][h] = {'reference': reference['name'], 'mean_delta': float(d.mean()),
                                                'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                'p_delta_gt_0': float((d > 0).mean())}
        report.setdefault('_boot', {})[h] = b
    report['positive_recall'] = report['test']['sentiment']['per_class']['positive']['recall']
    if 'test_stress' in Z:
        report['stress'] = {h: {'macro_f1': report['test_stress'][h]['macro_f1'],
                                'prediction_consistency': float(np.mean(pred['test'][h] == pred['test_stress'][h]))} for h, _ in HEADS}
    report['emoji_pairs'] = emoji_pairs(rows['test'], pred['test']['sentiment'])
    if 'audit' in Z:
        wrong = (pred['audit']['intent'] != y['audit']['intent']) | (pred['audit']['sentiment'] != y['audit']['sentiment'])
        report['audit_errors'] = {'n_rows_with_error': int(wrong.sum()), 'n': len(wrong),
                                  'errors': [{'text': rows['audit'][i]['text'],
                                              'expected': [rows['audit'][i]['intent'], rows['audit'][i]['sentiment']],
                                              'predicted': [INTENTS[pred['audit']['intent'][i]], SENTIMENTS[pred['audit']['sentiment'][i]]]}
                                             for i in np.where(wrong)[0]]}
    if 'contrast' in Z:
        report['contrast'] = contrast_report(rows['contrast'], pred['contrast'], y['contrast'])
    return report


def contrast_report(rows, pred, y):
    """Per phenomenon: row accuracy of intent, sentiment, both; and pairs where both members are fully correct."""
    out = {}
    both = (pred['intent'] == y['intent']) & (pred['sentiment'] == y['sentiment'])
    for ph in sorted({r['phenomenon'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['phenomenon'] == ph]
        pairs = {}
        for i in idx:
            pairs.setdefault(rows[i]['pair_id'], []).append(bool(both[i]))
        out[ph] = {'rows': len(idx),
                   'intent_acc': float(np.mean(pred['intent'][idx] == y['intent'][idx])),
                   'sentiment_acc': float(np.mean(pred['sentiment'][idx] == y['sentiment'][idx])),
                   'both_acc': float(np.mean(both[idx])),
                   'pairs_fully_correct': int(sum(all(v) for v in pairs.values())), 'pairs': len(pairs),
                   'errors': [{'text': rows[i]['text'], 'expected': [rows[i]['intent'], rows[i]['sentiment']],
                               'predicted': [INTENTS[pred['intent'][i]], SENTIMENTS[pred['sentiment'][i]]]}
                              for i in idx if not both[i]]}
    out['all'] = {'rows': len(rows), 'both_acc': float(both.mean()),
                  'pairs_fully_correct': sum(v['pairs_fully_correct'] for k, v in out.items() if k != 'all'),
                  'pairs': sum(v['pairs'] for k, v in out.items() if k != 'all')}
    return out


def strip_private(report):
    return {k: v for k, v in report.items() if not k.startswith('_')}
