"""Teacher: pretrained Hinglish encoder + intent (6) and sentiment (3) heads.

    python -m polyglot12.teacher --config configs/teacher_v11data.json [--smoke] [--stop-after-epochs N]

For every (candidate model, learning rate) run: class-balanced CE (v1.1 formula), AdamW with
linear warm-up/decay, fp16 AMP on CUDA, early stopping on unweighted validation NLL (v1.1
definition), a checkpoint after EVERY epoch in the persistent directory (auto-resume), then raw
logits for every predict file. The best run by validation NLL is recorded in summary.json.
Test/audit/contrast logits are only written, never read, here.

Checkpoints store weights in fp16 (half the I/O on Google Drive) plus scheduler, RNG, history and
early-stopping state. Optimizer moments are not stored by default (save_optimizer=false), so a
resumed run restarts AdamW moments at the resumed epoch; this is recorded in status.json.
"""
import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (INTENTS, SENTIMENTS, atomic_torch_save, class_weights, detect_platform, label_index,
                     persist_dir, read_jsonl, rng_state, seed_everything, set_rng_state, write_json)

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


class Teacher(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.intent = nn.Linear(h, len(INTENTS))
        self.sentiment = nn.Linear(h, len(SENTIMENTS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = self.drop((out * m).sum(1) / m.sum(1).clamp(min=1.0))   # masked mean pooling
        return self.intent(pooled), self.sentiment(pooled)


def load_encoder(model_id, smoke, cache_dir):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if smoke:  # real tokenizer + config, tiny random weights (no 1 GB download)
        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = 32, 2, 2, 64
        enc = AutoModel.from_config(cfg)
    else:
        enc = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    return tok, enc


def clean_text(t):
    return ' '.join(t.split())


def load_rows(cfg):
    import random
    rows = []
    for spec in cfg['train_files']:
        file_rows = read_jsonl(BUNDLE_ROOT / spec['path'])
        if spec.get('sample') and len(file_rows) > spec['sample']:
            # fixed seeded subset (same rows every epoch / every run) so large out-of-domain files do not dominate
            file_rows = random.Random(cfg['seed']).sample(file_rows, spec['sample'])
        for r in file_rows:
            rows.append({'text': clean_text(r['text']),
                         'yi': label_index(r, 'intent') if spec.get('intent', True) else -1,
                         'ys': label_index(r, 'sentiment') if spec.get('sentiment', True) else -1})
    if cfg.get('max_train_rows'):
        rows = rows[:cfg['max_train_rows']]
    return rows


def encode(tok, texts, max_length, device):
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return b['input_ids'].to(device), b['attention_mask'].to(device)


def weighted_ce(logits, y, w):
    """sum_i w[y_i] * CE_i / n_labeled  (rows with y=-1 are ignored)."""
    keep = y >= 0
    if not keep.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[keep].float(), y[keep], reduction='none')
    return (w[y[keep]] * ce).sum() / keep.sum()


@torch.no_grad()
def predict_logits(model, tok, texts, cfg, device, batch_size=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    zi = np.zeros((len(texts), len(INTENTS)), np.float32)
    zs = np.zeros((len(texts), len(SENTIMENTS)), np.float32)
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = encode(tok, [texts[i] for i in idx], cfg['max_length'], device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            a, b = model(ids, mask)
        zi[idx] = a.float().cpu().numpy()
        zs[idx] = b.float().cpu().numpy()
    return zi, zs


def val_nll(model, tok, val_rows, cfg, device):
    zi, zs = predict_logits(model, tok, [r['text'] for r in val_rows], cfg, device)
    yi = np.array([r['yi'] for r in val_rows])
    ys = np.array([r['ys'] for r in val_rows])
    lpi = torch.log_softmax(torch.from_numpy(zi), -1).numpy()[np.arange(len(yi)), yi]
    lps = torch.log_softmax(torch.from_numpy(zs), -1).numpy()[np.arange(len(ys)), ys]
    return float(-np.mean(np.concatenate([np.maximum(lpi, np.log(1e-9)), np.maximum(lps, np.log(1e-9))])))


def slug(model_id, lr):
    return re.sub(r'[^A-Za-z0-9]+', '-', model_id).strip('-') + f'_lr{lr:g}'


def run_one(model_id, lr, cfg, out, device, args, plan):
    name = slug(model_id, lr)
    rdir = out / 'runs' / name
    rdir.mkdir(parents=True, exist_ok=True)
    status_path = rdir / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('state') in ('done', 'load_failed'):
        print(f'[{name}] already {status["state"]}, skipping', flush=True)
        return status
    seed_everything(cfg['seed'])
    cache = str(plan['hf_cache']) if plan.get('hf_cache') else None
    try:
        tok, enc = load_encoder(model_id, args.smoke, cache)
    except Exception as e:  # record and move on to the next candidate
        status = {'state': 'load_failed', 'model_id': model_id, 'lr': lr, 'error': repr(e)[:2000]}
        write_json(status_path, status)
        print(f'[{name}] LOAD FAILED: {e!r}', flush=True)
        return status
    model = Teacher(enc, cfg.get('dropout', 0.1)).to(device)
    train_rows, val_rows = plan['train_rows'], plan['val_rows']
    wi = torch.tensor(class_weights([r['yi'] for r in train_rows], len(INTENTS)), device=device)
    ws = torch.tensor(class_weights([r['ys'] for r in train_rows], len(SENTIMENTS)), device=device)
    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight')
    groups = [{'params': [p for n, p in model.named_parameters() if not n.endswith(no_decay)], 'weight_decay': cfg['weight_decay']},
              {'params': [p for n, p in model.named_parameters() if n.endswith(no_decay)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    bs = cfg['batch_size']
    steps_per_epoch = math.ceil(len(train_rows) / bs)
    total = steps_per_epoch * cfg['max_epochs']
    warm = max(1, int(cfg['warmup_ratio'] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_epoch, best, best_epoch, bad, history, resumed = 0, float('inf'), 0, 0, [], []
    last = rdir / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        sched.load_state_dict(ck['sched'])
        if ck.get('opt'):
            opt.load_state_dict(ck['opt'])
        start_epoch, best, best_epoch, bad, history = ck['epoch'], ck['best'], ck['best_epoch'], ck['bad'], ck['history']
        resumed = ck.get('resumed', []) + [start_epoch]
        set_rng_state(ck['rng'])
        print(f'[{name}] resumed after epoch {start_epoch} (best val NLL {best:.4f} @ {best_epoch})', flush=True)
    write_json(status_path, {'state': 'running', 'model_id': model_id, 'lr': lr, 'epoch': start_epoch})

    for epoch in range(start_epoch, cfg['max_epochs']):
        if bad >= cfg['patience']:
            break
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(cfg['seed'] * 1000 + epoch)
        perm = torch.randperm(len(train_rows), generator=g).tolist()
        tot = 0.0
        for s in range(0, len(perm), bs):
            batch = [train_rows[i] for i in perm[s:s + bs]]
            ids, mask = encode(tok, [r['text'] for r in batch], cfg['max_length'], device)
            yi = torch.tensor([r['yi'] for r in batch], device=device)
            ys = torch.tensor([r['ys'] for r in batch], device=device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                zi, zs = model(ids, mask)
            loss = weighted_ce(zi, yi, wi) + weighted_ce(zs, ys, ws)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(batch)
        v = val_nll(model, tok, val_rows, cfg, device)
        improved = v < best - 1e-6
        if improved:
            best, best_epoch, bad = v, epoch + 1, 0
            atomic_torch_save({k: t.half() if t.is_floating_point() else t for k, t in model.state_dict().items()}, rdir / 'best.pt')
        else:
            bad += 1
        secs = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_loss': tot / len(train_rows), 'val_nll': v, 'seconds': secs})
        atomic_torch_save({'model': {k: t.half() if t.is_floating_point() else t for k, t in model.state_dict().items()},
                           'sched': sched.state_dict(), 'opt': opt.state_dict() if cfg.get('save_optimizer') else None,
                           'epoch': epoch + 1, 'best': best, 'best_epoch': best_epoch, 'bad': bad,
                           'history': history, 'rng': rng_state(), 'resumed': resumed}, last)
        print(f'[{name}] epoch {epoch + 1}/{cfg["max_epochs"]} train {tot / len(train_rows):.4f} '
              f'val NLL {v:.4f}{" *" if improved else ""} ({secs:.1f}s incl. checkpoint)', flush=True)
        if not plan.get('estimated'):
            plan['estimated'] = True
            per_run = secs * cfg['max_epochs'] + 60
            print(f'>>> TIME ESTIMATE: ~{secs:.0f}s/epoch -> <= ~{per_run / 60:.1f} min per run, '
                  f'<= ~{per_run * plan["n_runs"] / 60:.0f} min for all {plan["n_runs"]} runs '
                  f'(budget guard {cfg["time_budget_minutes"]} min)', flush=True)
        plan['epoch_seconds'] = secs
        if args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs:
            raise SystemExit(f'--stop-after-epochs {args.stop_after_epochs}: simulated disconnect')

    # Restore best weights and write logits for every predict file.
    model.load_state_dict(torch.load(rdir / 'best.pt', map_location='cpu'))
    model.float()
    ldir = rdir / 'logits'
    ldir.mkdir(exist_ok=True)
    for split, path in cfg['predict_files'].items():
        p = BUNDLE_ROOT / path
        if not p.exists():
            print(f'[{name}] predict file {path} missing, skipped', flush=True)
            continue
        rows = read_jsonl(p)
        zi, zs = predict_logits(model, tok, [clean_text(r['text']) for r in rows], cfg, device)
        np.savez_compressed(ldir / f'{split}.npz', ids=np.array([r['id'] for r in rows]), intent=zi, sentiment=zs)
    status = {'state': 'done', 'model_id': model_id, 'lr': lr, 'best_val_nll': best, 'best_epoch': best_epoch,
              'epochs_run': len(history), 'history': history, 'resumed_at_epochs': resumed,
              'optimizer_state_restored_on_resume': bool(cfg.get('save_optimizer')),
              'n_params': sum(p.numel() for p in model.parameters())}
    write_json(status_path, status)
    if not cfg.get('keep_weights'):
        for f in ('last.pt', 'best.pt'):
            (rdir / f).unlink(missing_ok=True)
    else:
        (rdir / 'last.pt').unlink(missing_ok=True)
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true', help='tiny random encoders, CPU-friendly')
    ap.add_argument('--out', help='override output dir (default: platform persistent dir)')
    ap.add_argument('--stop-after-epochs', type=int, default=0, help='testing: exit after N epochs to test resume')
    ap.add_argument('--override', action='append', default=[], help='testing: key=json_value, e.g. max_epochs=1')
    args = ap.parse_args(argv)
    cfg = json.loads((BUNDLE_ROOT / args.config).read_text() if not Path(args.config).is_absolute() else Path(args.config).read_text())
    for kv in args.override:
        k, v = kv.split('=', 1)
        cfg[k] = json.loads(v)
    out = Path(args.out) if args.out else persist_dir(cfg['notebook'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plat = detect_platform()
    hf_cache = Path('/content/hf_cache') if plat == 'colab' else (Path('/kaggle/tmp/hf_cache') if plat == 'kaggle' else None)
    train_rows = load_rows(cfg)
    val_rows = [{'text': clean_text(r['text']), 'yi': label_index(r, 'intent'), 'ys': label_index(r, 'sentiment')}
                for r in read_jsonl(BUNDLE_ROOT / cfg['validation_file'])]
    runs = [(m, lr) for m in cfg['candidates'] for lr in cfg['learning_rates']]
    plan = {'train_rows': train_rows, 'val_rows': val_rows, 'n_runs': len(runs), 'hf_cache': hf_cache}
    print(f'platform={plat} device={device} out={out} train_rows={len(train_rows)} val_rows={len(val_rows)} runs={len(runs)}', flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0), flush=True)
    t_start = time.time()
    results = {}
    for m, lr in runs:
        elapsed = (time.time() - t_start) / 60
        est = (plan.get('epoch_seconds', 0) * cfg['max_epochs'] + 60) / 60
        if elapsed + est > cfg['time_budget_minutes']:
            print(f'[{slug(m, lr)}] SKIPPED: time budget ({elapsed:.0f} + ~{est:.0f} min > {cfg["time_budget_minutes"]})', flush=True)
            results[slug(m, lr)] = {'state': 'skipped_time', 'model_id': m, 'lr': lr}
            continue
        results[slug(m, lr)] = run_one(m, lr, cfg, out, device, args, plan)
    done = {k: v for k, v in results.items() if v.get('state') == 'done'}
    selected = min(done, key=lambda k: done[k]['best_val_nll']) if done else None
    summary = {'config': cfg, 'smoke': args.smoke, 'platform': plat, 'device': str(device),
               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               'selection_rule': 'lowest validation NLL (unweighted, both heads, T=1) over all done runs',
               'selected_run': selected, 'runs': results, 'wall_minutes': (time.time() - t_start) / 60}
    write_json(out / 'summary.json', summary)
    if selected:
        sel = out / 'selected_logits'
        shutil.rmtree(sel, ignore_errors=True)
        shutil.copytree(out / 'runs' / selected / 'logits', sel)
    print(json.dumps({'selected_run': selected, 'val_nll': {k: v.get('best_val_nll') for k, v in results.items()}}, indent=1), flush=True)
    return summary


def package_outputs(out, zip_path):
    """outputs.zip = summary + per-run status + logits (no weights)."""
    import zipfile
    out = Path(out)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix in ('.json', '.npz', '.txt', '.log') and not f.name.endswith('.tmp'):
                z.write(f, f.relative_to(out))
    return zip_path


if __name__ == '__main__':
    main()
