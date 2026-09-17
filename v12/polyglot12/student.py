"""Student: hashed token pieces -> 4 pre-LN transformer blocks -> attention pooling, plus the
v1.0 hashed sparse linear branch; the two logit vectors are summed before the heads (as in v1.1).

The model is split into embed() (EmbeddingBag lookups, done in NumPy at deployment) and
encode() (the dense transformer, exported to ONNX) so export never has to trace EmbeddingBag.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hashing import DIM, MAX_TOKENS, features, token_pieces

DEFAULT = dict(n_buckets=32768, d=256, layers=4, heads=4, ffn=1024, max_tokens=MAX_TOKENS, dropout=0.1, linear_branch=True)


class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ffn)
        self.ff2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_bias):
        B, T, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // h) + key_bias      # key_bias: (B,1,1,T), 0 or -1e4
        att = self.drop(torch.softmax(att.float(), dim=-1).to(q.dtype))
        y = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff2(self.drop(F.gelu(self.ff1(self.ln2(x))))))


class Student(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = {**DEFAULT, **cfg}
        self.cfg = c
        d = c['d']
        self.pieces = nn.EmbeddingBag(c['n_buckets'], d, mode='mean')
        self.pos = nn.Embedding(c['max_tokens'], d)
        self.blocks = nn.ModuleList(Block(d, c['heads'], c['ffn'], c['dropout']) for _ in range(c['layers']))
        self.ln_f = nn.LayerNorm(d)
        self.pool = nn.Linear(d, 1)
        self.head = nn.Linear(d, 9)
        self.drop = nn.Dropout(c['dropout'])
        nn.init.normal_(self.pieces.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        if c['linear_branch']:
            self.linear = nn.EmbeddingBag(DIM, 9, mode='sum')
            nn.init.zeros_(self.linear.weight)          # v1.0 starts from W = 0
            self.lin_bias = nn.Parameter(torch.zeros(9))

    # ---- embed: hashed pieces -> (B,T,d) token matrix + mask; sparse features -> (B,9) linear logits
    def embed(self, b):
        tok = self.pieces(b['piece_ids'], b['piece_offsets'])                        # (Ntok, d)
        B, T = b['mask'].shape
        X = tok.new_zeros(B * T, tok.shape[-1]).index_copy(0, b['tok_pos'], tok).view(B, T, -1)
        z_lin = None
        if self.cfg['linear_branch']:
            z_lin = self.linear(b['feat_ix'], b['feat_offsets'], per_sample_weights=b['feat_vals']) + self.lin_bias
        return X, z_lin

    # ---- encode: dense part (exported to ONNX)
    def encode(self, X, mask):
        T = X.shape[1]
        x = self.drop(X + self.pos.weight[:T].unsqueeze(0))
        key_bias = (1.0 - mask.to(x.dtype))[:, None, None, :] * -1e4
        for blk in self.blocks:
            x = blk(x, key_bias)
        h = self.ln_f(x)
        score = self.pool(h).squeeze(-1).float() + (1.0 - mask.float()) * -1e4
        alpha = torch.softmax(score, dim=-1)
        pooled = (alpha.unsqueeze(-1).to(h.dtype) * h).sum(1)
        return self.head(pooled), alpha

    def forward(self, b):
        X, z_lin = self.embed(b)
        z, alpha = self.encode(X, b['mask'])
        if z_lin is not None:
            z = z + z_lin.to(z.dtype)
        return z, alpha

    def param_breakdown(self):
        parts = {'piece_embeddings': self.pieces.weight.numel(), 'positions': self.pos.weight.numel(),
                 'transformer_blocks': sum(p.numel() for p in self.blocks.parameters()),
                 'final_ln_pool_head': sum(p.numel() for m in (self.ln_f, self.pool, self.head) for p in m.parameters())}
        if self.cfg['linear_branch']:
            parts['linear_branch'] = self.linear.weight.numel() + self.lin_bias.numel()
        parts['total'] = sum(parts.values())
        return parts


def hash_row(text, n_buckets):
    """Precompute everything the student needs for one message (CPU, NumPy)."""
    _, flat, lens = token_pieces(text, n_buckets)
    ix, vals = features(text)
    return flat, lens, ix, vals


def collate(hashed, device, token_dropout=0.0, rng=None, max_tokens=MAX_TOKENS):
    """hashed: list of (flat, lens, ix, vals). Token dropout removes whole tokens (keeps >= 1)."""
    piece_ids, piece_offsets, tok_pos, feat_ix, feat_offsets, feat_vals = [], [], [], [], [], []
    n_tok = []
    pos_in_flat = 0
    for flat, lens, ix, vals in hashed:
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        keep = np.arange(len(lens))
        if token_dropout and len(lens) > 1:
            m = rng.random(len(lens)) >= token_dropout
            if m.any():
                keep = keep[m]
        n_tok.append(len(keep))
        for t in keep:
            piece_offsets.append(pos_in_flat)
            seg = flat[starts[t]:starts[t] + lens[t]]
            piece_ids.append(seg)
            pos_in_flat += len(seg)
        feat_offsets.append(sum(len(a) for a in feat_ix))
        feat_ix.append(ix)
        feat_vals.append(vals)
    B, T = len(hashed), max(1, max(n_tok))
    mask = np.zeros((B, T), np.float32)
    for i, n in enumerate(n_tok):
        mask[i, :n] = 1
        tok_pos.extend(i * T + np.arange(n))
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {'piece_ids': t(np.concatenate(piece_ids), torch.long), 'piece_offsets': t(piece_offsets, torch.long),
            'tok_pos': t(np.asarray(tok_pos, np.int64), torch.long), 'mask': t(mask, torch.float32),
            'feat_ix': t(np.concatenate(feat_ix).astype(np.int64), torch.long),
            'feat_offsets': t(feat_offsets, torch.long), 'feat_vals': t(np.concatenate(feat_vals), torch.float32)}
