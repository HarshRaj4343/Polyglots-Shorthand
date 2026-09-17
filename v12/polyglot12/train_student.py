"""Train student variants with resume.

    python -m polyglot12.train_student --config configs/student.json [--smoke] [--only NAME] [--stop-after-epochs N]

Loss = class-balanced CE on labeled rows (v1.1 weights N/(K n_c))
     + lambda_kd * T^2 * KL(teacher_T || student_T) per head, masked per row
     + lambda_cons * KL(stopgrad p(x) || p(noisy x)) on meaning-preserving char noise.
Rows with no label and no teacher logits are ignored. Unlabeled KD-pool rows (no labels) are
sub-sampled to pool_per_epoch per epoch. Early stopping on unweighted validation NLL (v1.1).
Full training state is checkpointed after every epoch in the persistent directory.
"""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import noisy_variant
from .common import (INTENTS, SENTIMENTS, atomic_torch_save, class_weights, detect_platform, label_index,
                     persist_dir, read_jsonl, rng_state, seed_everything, set_rng_state, write_json)
from .student import Student, collate, hash_row

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def load_training_rows(cfg, use_kd):
    rows = []
    for spec in cfg['train_files']:
        path = BUNDLE_ROOT / spec['path']
        if not path.exists():
            if spec.get('optional'):
                print(f'optional train file {spec["path"]} missing, skipped', flush=True)
                continue
            raise FileNotFoundError(path)
        for r in read_jsonl(path):
            yi = label_index(r, 'intent') if spec.get('intent', True) else -1
            ys = label_index(r, 'sentiment') if spec.get('sentiment', True) else -1
            ti = r.get('t_intent_logits') if use_kd and spec.get('kd', True) else None
            ts = r.get('t_sentiment_logits') if use_kd and spec.get('kd', True) else None
            row = {'text': r['text'], 'yi': yi, 'ys': ys,
                   'ti': ti, 'ts': ts, 'kd_i': bool(ti is not None and r.get('kd_intent_mask', True)),
                   'kd_s': ts is not None, 'pool': yi < 0 and ys < 0}
            if row['yi'] < 0 and row['ys'] < 0 and not row['kd_i'] and not row['kd_s']:
                continue
            rows.append(row)
    return rows


def kl_soft(student_logits, teacher_logits, T):
    """T^2 * KL(softmax(t/T) || softmax(s/T)), per row."""
    pt = F.softmax(teacher_logits / T, -1)
    return (T * T) * (pt * (F.log_softmax(teacher_logits / T, -1) - F.log_softmax(student_logits.float() / T, -1))).sum(-1)


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, hashed, device, bs=256):
    model.eval()
    zs = []
    for s in range(0, len(hashed), bs):
        z, _ = model(collate(hashed[s:s + bs], device))
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs) if zs else np.zeros((0, 9), np.float32)


def val_nll(z, yi, ys):
    lpi = torch.log_softmax(torch.from_numpy(z[:, :6]), -1).numpy()[np.arange(len(yi)), yi]
    lps = torch.log_softmax(torch.from_numpy(z[:, 6:]), -1).numpy()[np.arange(len(ys)), ys]
    return float(-np.mean(np.concatenate([np.maximum(lpi, np.log(1e-9)), np.maximum(lps, np.log(1e-9))])))


def run_variant(var, cfg, out, device, args):
    name = var['name']
    rdir = out / 'runs' / name
    rdir.mkdir(parents=True, exist_ok=True)
    status_path = rdir / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('state') == 'done':
        print(f'[{name}] already done, skipping', flush=True)
        return status
    c = {**cfg, **var}
    seed_everything(c['seed'])
    model_cfg = {k: c[k] for k in ('n_buckets', 'd', 'layers', 'heads', 'ffn', 'max_tokens', 'dropout', 'linear_branch')}
    model = Student(**model_cfg).to(device)
    rows = load_training_rows(c, c['kd'])
    t0 = time.time()
    hashed = [hash_row(r['text'], c['n_buckets']) for r in rows]
    val = read_jsonl(BUNDLE_ROOT / c['validation_file'])
    val_hashed = [hash_row(r['text'], c['n_buckets']) for r in val]
    vyi = np.array([label_index(r, 'intent') for r in val])
    vys = np.array([label_index(r, 'sentiment') for r in val])
    print(f'[{name}] {len(rows)} training rows hashed in {time.time() - t0:.1f}s; params {model.param_breakdown()}', flush=True)
    wi = torch.tensor(class_weights([r['yi'] for r in rows], 6), device=device)
    ws = torch.tensor(class_weights([r['ys'] for r in rows], 3), device=device)

    core = [i for i, r in enumerate(rows) if not r['pool']]
    pool = [i for i, r in enumerate(rows) if r['pool']]
    per_epoch = len(core) + min(len(pool), c['pool_per_epoch'])
    bs = c['batch_size']
    steps_per_epoch = math.ceil(per_epoch / bs)
    total = steps_per_epoch * c['max_epochs']
    warm = max(1, int(c['warmup_ratio'] * total))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.startswith(('pieces', 'pos', 'linear')) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': c['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=c['lr'])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start, best, best_epoch, bad, history = 0, float('inf'), 0, 0, []
    last = rdir / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best, best_epoch, bad, history = ck['epoch'], ck['best'], ck['best_epoch'], ck['bad'], ck['history']
        set_rng_state(ck['rng'])
        print(f'[{name}] resumed after epoch {start} (best val NLL {best:.4f} @ {best_epoch})', flush=True)
    write_json(status_path, {'state': 'running', 'epoch': start})

    for epoch in range(start, c['max_epochs']):
        if bad >= c['patience']:
            break
        t0 = time.time()
        model.train()
        er = np.random.default_rng(c['seed'] * 1000 + epoch)
        idx = core + (list(er.choice(pool, size=min(len(pool), c['pool_per_epoch']), replace=False)) if pool else [])
        idx = [int(idx[i]) for i in er.permutation(len(idx))]
        sums = {'ce': 0.0, 'kd': 0.0, 'cons': 0.0}
        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]
            br = [rows[i] for i in bi]
            batch = collate([hashed[i] for i in bi], device, c['token_dropout'], er)
            yi = torch.tensor([r['yi'] for r in br], device=device)
            ys = torch.tensor([r['ys'] for r in br], device=device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                z, _ = model(batch)
            z = z.float()
            loss_ce = 0.0
            for logits, y, w in ((z[:, :6], yi, wi), (z[:, 6:], ys, ws)):
                keep = y >= 0
                if keep.any():
                    loss_ce = loss_ce + (w[y[keep]] * F.cross_entropy(logits[keep], y[keep], reduction='none')).sum() / keep.sum()
            loss_kd = z.sum() * 0.0
            if c['kd']:
                mi = torch.tensor([r['kd_i'] for r in br], device=device, dtype=torch.float32)
                ms = torch.tensor([r['kd_s'] for r in br], device=device, dtype=torch.float32)
                if mi.any():
                    ti = torch.tensor([r['ti'] if r['kd_i'] else [0.0] * 6 for r in br], device=device)
                    loss_kd = loss_kd + masked_mean(kl_soft(z[:, :6], ti, c['kd_T']), mi)
                if ms.any():
                    ts = torch.tensor([r['ts'] if r['kd_s'] else [0.0] * 3 for r in br], device=device)
                    loss_kd = loss_kd + masked_mean(kl_soft(z[:, 6:], ts, c['kd_T']), ms)
            loss_cons = z.sum() * 0.0
            if c['lambda_cons'] > 0:
                noisy = [hash_row(noisy_variant(rows[i]['text'], c['seed'] * 7919 + epoch * 104729 + i), c['n_buckets']) for i in bi]
                nb = collate(noisy, device, 0.0, er)
                with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                    zn, _ = model(nb)
                zn = zn.float()
                p = z.detach()
                loss_cons = (kl_soft(zn[:, :6], p[:, :6], 1.0) + kl_soft(zn[:, 6:], p[:, 6:], 1.0)).mean()
            loss = loss_ce + c['lambda_kd'] * loss_kd + c['lambda_cons'] * loss_cons
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = len(br)
            sums['ce'] += float(loss_ce) * n
            sums['kd'] += float(loss_kd) * n
            sums['cons'] += float(loss_cons) * n
        v = val_nll(predict(model, val_hashed, device), vyi, vys)
        improved = v < best - 1e-6
        if improved:
            best, best_epoch, bad = v, epoch + 1, 0
            atomic_torch_save(model.state_dict(), rdir / 'best.pt')
        else:
            bad += 1
        secs = time.time() - t0
        history.append({'epoch': epoch + 1, **{k: sums[k] / len(idx) for k in sums}, 'val_nll': v, 'seconds': secs})
        atomic_torch_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                           'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best': best, 'best_epoch': best_epoch,
                           'bad': bad, 'history': history, 'rng': rng_state()}, last)
        h = history[-1]
        print(f'[{name}] epoch {epoch + 1}/{c["max_epochs"]} ce {h["ce"]:.3f} kd {h["kd"]:.3f} cons {h["cons"]:.3f} '
              f'val NLL {v:.4f}{" *" if improved else ""} ({secs:.1f}s)', flush=True)
        if epoch == start:
            print(f'>>> TIME ESTIMATE [{name}]: ~{secs:.0f}s/epoch -> <= ~{secs * (c["max_epochs"] - start) / 60:.0f} min', flush=True)
        if args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs:
            raise SystemExit(f'--stop-after-epochs {args.stop_after_epochs}: simulated disconnect')

    model.load_state_dict(torch.load(rdir / 'best.pt', map_location='cpu'))
    ldir = rdir / 'logits'
    ldir.mkdir(exist_ok=True)
    for split, path in c['predict_files'].items():
        p = BUNDLE_ROOT / path
        if not p.exists():
            continue
        prow = read_jsonl(p)
        z = predict(model, [hash_row(r['text'], c['n_buckets']) for r in prow], device)
        np.savez_compressed(ldir / f'{split}.npz', ids=np.array([r['id'] for r in prow]), intent=z[:, :6], sentiment=z[:, 6:])
    torch.save({'state_dict': model.state_dict(), 'model_cfg': model_cfg}, rdir / 'student.pt')
    status = {'state': 'done', 'variant': var, 'best_val_nll': best, 'best_epoch': best_epoch, 'epochs_run': len(history),
              'history': history, 'params': model.param_breakdown(), 'train_rows': len(rows),
              'core_rows': len(core), 'pool_rows': len(pool)}
    write_json(status_path, status)
    (rdir / 'last.pt').unlink(missing_ok=True)
    (rdir / 'best.pt').unlink(missing_ok=True)
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out')
    ap.add_argument('--only', help='comma-separated variant names')
    ap.add_argument('--stop-after-epochs', type=int, default=0)
    args = ap.parse_args(argv)
    p = Path(args.config)
    cfg = json.loads((p if p.is_absolute() else BUNDLE_ROOT / p).read_text())
    out = Path(args.out) if args.out else persist_dir(cfg['notebook'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'platform={detect_platform()} device={device} out={out}', flush=True)
    variants = cfg['variants']
    if args.only:
        keep = set(args.only.split(','))
        variants = [v for v in variants if v['name'] in keep]
    t_start = time.time()
    results = {}
    for var in variants:
        if (time.time() - t_start) / 60 > cfg['time_budget_minutes']:
            results[var['name']] = {'state': 'skipped_time'}
            print(f'[{var["name"]}] SKIPPED: time budget reached', flush=True)
            continue
        results[var['name']] = run_variant(var, cfg, out, device, args)
    write_json(out / 'summary.json', {'config': cfg, 'device': str(device), 'runs': results,
                                      'wall_minutes': (time.time() - t_start) / 60})
    return results


if __name__ == '__main__':
    main()
