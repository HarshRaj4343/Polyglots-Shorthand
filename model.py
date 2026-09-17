# This turns a text message into numbers ("features") and trains a small
# model that predicts TWO things at once:
#   1. intent    -> what the customer wants (cancel, refund, track, ...)
#   2. sentiment -> how they feel (negative / neutral / positive)
#
# The model has two branches whose scores are ADDED together:
#   a) a linear model over hashed word / char n-gram features (bag of features:
#      fast and strong on keywords, but blind to word order), and
#   b) a self-attention branch: every token gets an embedding plus a position
#      embedding, one self-attention layer lets each token look at the other
#      tokens (so "accha" can see the "nahi" after it), and attention pooling
#      picks the tokens that matter for the final decision.

import json
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
import numpy as np

# The 6 possible intent labels. Their index (0..5) is the model's output slot.
INTENTS = ['cancel_order', 'refund', 'track_order', 'not_received', 'damaged_item', 'feedback']
# The 3 possible sentiment labels. They use output slots 6..8.
SENTIMENTS = ['negative', 'neutral', 'positive']
# Size of the hashed feature space: every feature string is mapped to one of
# 32,768 "buckets". The weight matrix therefore has DIM rows.
DIM = 32768
# Longest message we accept (in Unicode characters).
MAX_CHARS = 512
# Attention branch sizes: hashed token-embedding table, embedding width and the
# number of positions (longer messages are truncated for the attention branch only).
ATT_BUCKETS = 16384
ATT_DIM = 32
MAX_TOKENS = 128

ALIASES = {'krdo':'kar do', 'kr':'kar', 'kro':'karo', 'krna':'karna',
           'nhi':'nahi', 'nai':'nahi', 'nh':'nahi', 'rha':'raha', 'rhe':'rahe',
           'plz':'please', 'pls':'please', 'mera':'mera', 'bht':'bahut',
           'bohot':'bahut', 'bohut':'bahut', 'acha':'accha', 'achha':'accha',
           'kaha':'kahan', 'kabtk':'kab tak', 'paisa':'paise',
           'ordr':'order', 'delivry':'delivery', 'servis':'service', 'parcl':'parcel',
           'thnk':'thank', 'chaiye':'chahiye', 'mujhko':'mujhe'}
# Tokenizer regex: a token is either a run of word characters (letters/digits,
# any script) OR a single non-space symbol (so emojis and "?" / "!" survive).
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def normalize(text):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if len(text) > MAX_CHARS:
        raise ValueError(f'maximum input is {MAX_CHARS} Unicode code points; split long chats upstream')
    # NFC = combine accented/composed characters into a single standard form,
    # then lowercase everything ("ORDER" == "order").
    text = unicodedata.normalize('NFC', text).lower()
    # Collapse repeated spaces/tabs/newlines into single spaces and trim ends.
    text = ' '.join(text.split())
    if not text:
        raise ValueError('text must not be empty')
    return text

# Normalize, and for the 'strip_symbols' ablation also drop emojis/punctuation.
def clean(text, mode='full'):
    s = normalize(text)
    if mode == 'strip_symbols':
        s = ' '.join(''.join(c for c in s if not unicodedata.category(c).startswith(('S', 'P'))).split())
    return s

def bucket(f, size=DIM):
    return zlib.crc32(f.encode('utf-8')) % size

# ---------------------------------------------------------------------------
# features(): convert a message into a sparse, normalized feature vector.
# Returns (ix, vals): the bucket indices that are "on" and their weights.
# `mode` controls which feature families are used (for ablation experiments):
#   'full'          -> everything
#   'word_only'     -> words + word pairs only
#   'char_word'     -> words + word pairs + character n-grams
#   'strip_symbols' -> like 'full' but emojis/punctuation removed first
# ---------------------------------------------------------------------------
def features(text, mode='full'):
    s = clean(text, mode)
    ts = TOKEN.findall(s)
    fs = ['w:' + t for t in ts]
    fs += ['b:' + a + '|' + b for a,b in zip(ts,ts[1:])]
    if mode != 'word_only':
        padded = '^' + s + '$'
        fs += [f'c{n}:' + padded[i:i+n] for n in (3,4,5) for i in range(len(padded)-n+1)]
    if mode in ('full', 'strip_symbols'):
        canon = [u for t in ts for u in ALIASES.get(t,t).split()]
        fs += ['a:' + t for t in canon]
        fs += ['ab:' + a + '|' + b for a,b in zip(canon,canon[1:])]
        symbols = list(dict.fromkeys(t for t in ts if any(unicodedata.category(c).startswith('S') for c in t)))[:8]
        words = list(dict.fromkeys(t for t in ts if any(c.isalnum() for c in t)))[:64]
        fs += ['e:' + e + '|' + t for e in symbols for t in words]
    # Step 4: "hashing trick" - map each feature string to a bucket number with
    # CRC32 % DIM, and count how often each bucket appears.
    counts = Counter(bucket(f) for f in fs)
    # Sorted list of active bucket indices.
    ix = np.array(sorted(counts), dtype=np.int32)
    # Value per bucket = 1 + log(count): repeated features count, but sub-linearly.
    vals = np.array([1 + np.log(counts[int(i)]) for i in ix], dtype=np.float32)
    # L2-normalise so long and short messages have vectors of the same length.
    vals /= max(float(np.linalg.norm(vals)), 1e-12)
    return ix, vals

# ---------------------------------------------------------------------------
# tokens(): the ordered token sequence for the attention branch.
# Each token is represented by several hashed "pieces" (the word itself, its
# alias and its character 3/4-grams, depending on `mode`); its embedding is the
# mean of those pieces, so misspellings like "ordr" still land near "order".
# Returns (token_strings, flat_bucket_ids, pieces_per_token).
# ---------------------------------------------------------------------------
def tokens(text, mode='full'):
    ts = TOKEN.findall(clean(text, mode))[:MAX_TOKENS]
    flat, lens = [], []
    for t in ts:
        keys = ['tw:' + t]
        if mode in ('full', 'strip_symbols'):
            keys.append('ta:' + ALIASES.get(t, t))
        if mode != 'word_only':
            p = '<' + t + '>'
            keys += [f'tc{n}:' + p[i:i+n] for n in (3,4) for i in range(len(p)-n+1)]
        ids = sorted({bucket(k, ATT_BUCKETS) for k in keys})
        flat += ids; lens.append(len(ids))
    return ts, np.array(flat, dtype=np.int64), np.array(lens, dtype=np.int64)

def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ---------------------------------------------------------------------------
# Attention: token embeddings -> 1 self-attention layer -> attention pooling.
#   X     = mean(E[pieces]) + P[position]                 (T x d)
#   A     = softmax(X Wq (X Wk)^T / sqrt(d))                (T x T)  who looks at whom
#   H     = tanh(X + (A X Wv) Wo)                           (T x d)  context-aware tokens
#   alpha = softmax(H u)                                    (T)      which tokens matter
#   z     = (sum_i alpha_i H_i) Wout                        (9)      scores for both heads
# Gradients are written out by hand (no autograd), see backward().
# ---------------------------------------------------------------------------
class Attention:
    DENSE = ('P','Wq','Wk','Wv','Wo','u','Wout')

    def __init__(self, d=ATT_DIM, seed=0):
        rng = np.random.default_rng(seed)
        r = lambda shape, s: rng.normal(0, s, shape).astype(np.float32)
        self.d = d
        self.params = {'E': r((ATT_BUCKETS,d), 0.1), 'P': np.zeros((MAX_TOKENS,d), np.float32),
                       'Wq': r((d,d), d**-0.5), 'Wk': r((d,d), d**-0.5), 'Wv': r((d,d), d**-0.5),
                       'Wo': r((d,d), 0.1), 'u': r(d, 0.1), 'Wout': r((d,9), 0.01)}

    def forward(self, flat, lens):
        p = self.params; T = len(lens)
        if T == 0:                                    # nothing left (e.g. emoji-only text with symbols stripped)
            return np.zeros(9, np.float32), None
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        X = np.add.reduceat(p['E'][flat], starts, axis=0) / lens[:,None] + p['P'][:T]
        Q, K, V = X @ p['Wq'], X @ p['Wk'], X @ p['Wv']
        A = softmax(Q @ K.T / np.sqrt(self.d), axis=1)
        C = A @ V
        H = np.tanh(X + C @ p['Wo'])
        alpha = softmax(H @ p['u'])
        pooled = alpha @ H
        return pooled @ p['Wout'], (flat, lens, X, Q, K, V, A, C, H, alpha, pooled)

    def backward(self, cache, dz):
        p = self.params
        flat, lens, X, Q, K, V, A, C, H, alpha, pooled = cache
        g = {'Wout': np.outer(pooled, dz)}
        dpooled = p['Wout'] @ dz
        dH = np.outer(alpha, dpooled)
        dalpha = H @ dpooled
        ds = alpha * (dalpha - alpha @ dalpha)
        g['u'] = H.T @ ds
        dH += np.outer(ds, p['u'])
        dpre = dH * (1 - H*H)
        g['Wo'] = C.T @ dpre
        dC = dpre @ p['Wo'].T
        dA = dC @ V.T
        dV = A.T @ dC
        dS = A * (dA - (dA*A).sum(axis=1, keepdims=True)) / np.sqrt(self.d)
        dQ, dK = dS @ K, dS.T @ Q
        g['Wq'], g['Wk'], g['Wv'] = X.T @ dQ, X.T @ dK, X.T @ dV
        dX = dpre + dQ @ p['Wq'].T + dK @ p['Wk'].T + dV @ p['Wv'].T
        g['P'] = np.zeros_like(p['P']); g['P'][:len(lens)] = dX
        # Sparse embedding gradient: each piece gets its token's gradient / #pieces.
        rows, inv = np.unique(flat, return_inverse=True)
        gE = np.zeros((len(rows), self.d), np.float32)
        np.add.at(gE, inv, np.repeat(dX / lens[:,None], lens, axis=0))
        g['E'] = (rows, gE)
        return g

# Optimizer for the attention branch: Adam for the small dense matrices and
# sparse Adagrad for the big embedding table (only touched rows are updated).
class AttentionOptimizer:
    def __init__(self, att, lr=0.003, lr_emb=0.05):
        self.att, self.lr, self.lr_emb, self.t = att, lr, lr_emb, 0
        self.m = {k: np.zeros_like(att.params[k]) for k in Attention.DENSE}
        self.v = {k: np.zeros_like(att.params[k]) for k in Attention.DENSE}
        self.G = np.zeros_like(att.params['E'])

    def step(self, g, scale=1.0):
        self.t += 1; b1, b2 = 0.9, 0.999
        for k in Attention.DENSE:
            self.m[k] = b1*self.m[k] + (1-b1)*g[k]
            self.v[k] = b2*self.v[k] + (1-b2)*g[k]*g[k]
            mh = self.m[k]/(1-b1**self.t); vh = self.v[k]/(1-b2**self.t)
            self.att.params[k] -= scale*self.lr*mh/(np.sqrt(vh)+1e-8)
        rows, gE = g['E']
        self.G[rows] += gE*gE
        self.att.params['E'][rows] -= scale*self.lr_emb*gE/(np.sqrt(self.G[rows])+1e-8)

class Model:
    def __init__(self, mode='full', attention=True):
        self.mode = mode                                   # which feature set to use
        self.w = np.zeros((DIM,9), dtype=np.float32)       # weights: 32768 buckets x 9 outputs
        self.b = np.zeros(9,dtype=np.float32)              # bias per output
        self.att = Attention() if attention else None      # context branch (None = linear only)
        self.temperature = [1.0,1.0]                       # calibration temps [intent, sentiment]
        self.threshold = [0.0,0.0]                         # "needs human review" cutoffs [intent, sentiment]

    # ----- logits(): raw 9 scores = linear branch + attention branch -----
    def logits(self, text):
        ix,v = features(text,self.mode)
        # Linear score: sum of the weight rows of active buckets (scaled by value) + bias.
        z = v @ self.w[ix] + self.b
        if self.att is not None:
            _, flat, lens = tokens(text, self.mode)
            z = z + self.att.forward(flat, lens)[0]
        return z

    def probabilities(self, text):
        z = self.logits(text)
        # Split the 9 scores into the two heads, divide by temperature, softmax each.
        return [softmax(z[:6]/self.temperature[0]), softmax(z[6:]/self.temperature[1])]

    # ----- attention_weights(): which tokens the pooling focused on (for display) -----
    def attention_weights(self, text):
        if self.att is None:
            return []
        ts, flat, lens = tokens(text, self.mode)
        cache = self.att.forward(flat, lens)[1]
        return [] if cache is None else list(zip(ts, (float(a) for a in cache[9])))

    # ----- predict(): text -> friendly JSON-ready dict with label + confidence -----
    def predict(self,text):
        ps = self.probabilities(text)
        out = {}
        # Do the same thing for each head (intent, then sentiment).
        for k,labels,p,th in zip(('intent','sentiment'),(INTENTS,SENTIMENTS),ps,self.threshold):
            j = int(np.argmax(p))                          # pick the most likely label
            out[k] = {'label':labels[j], 'confidence':float(p[j]),
                      # flag for human review if the model isn't confident enough
                      'review_required':bool(float(p[j]) < th)}
        return out

    # ----- fit(): train both branches jointly -----
    def fit(self,train,valid,epochs=55,seed=42,att_lr=0.0001,att_lr_emb=0.01,token_dropout=0.2,att_decay=0.01):
        # Attention hyper-parameters were chosen by validation NLL (not test metrics).
        # Pre-compute features, token pieces and numeric labels for every training row.
        xs=[features(r['text'],self.mode) for r in train]
        toks=[tokens(r['text'],self.mode)[1:] for r in train]
        ys=[(INTENTS.index(r['intent']),SENTIMENTS.index(r['sentiment'])) for r in train]
        # Seeded random generator so training is reproducible.
        rng=np.random.default_rng(seed)
        opt=AttentionOptimizer(self.att,att_lr,att_lr_emb) if self.att is not None else None
        # Class-balancing weights: rare labels get a bigger weight so the model
        # doesn't just learn to always predict the most common label.
        counts_a=np.bincount([a for a,b in ys],minlength=6)
        counts_b=np.bincount([b for a,b in ys],minlength=3)
        balance_a=len(ys)/(6*np.maximum(counts_a,1))
        balance_b=len(ys)/(3*np.maximum(counts_b,1))
        # Track the best model seen on the validation set ("early stopping").
        best=float('inf'); best_params=None
        for epoch in range(epochs):
            # Learning rate slowly decays as epochs go on.
            decay=1/(1+epoch/20); lr=0.6*decay
            # Decoupled weight decay once per epoch, not once per active feature.
            self.w *= 0.999
            if self.att is not None and att_decay:
                for k in Attention.DENSE: self.att.params[k] *= 1-att_decay
            # Visit training rows in a random order each epoch.
            for j in rng.permutation(len(train)):
                ix,v=xs[j]; a,b=ys[j]
                # Forward pass: scores -> probabilities for both heads.
                z=v @ self.w[ix]+self.b
                if opt is not None:
                    flat,lens=toks[j]
                    # Token dropout: hide random tokens so the branch must use context, not one keyword.
                    if token_dropout and len(lens)>1:
                        keep=rng.random(len(lens))>=token_dropout
                        if keep.any():
                            flat=flat[np.repeat(keep,lens)]; lens=lens[keep]
                    z_att,cache=self.att.forward(flat,lens); z=z+z_att
                d=np.concatenate([softmax(z[:6]),softmax(z[6:])])
                # Gradient of cross-entropy loss = predicted prob - 1 at the true label.
                d[a]-=1; d[6+b]-=1
                # Scale the gradient by the class-balance weight of the true label.
                d[:6]*=balance_a[a]; d[6:]*=balance_b[b]
                # Update only the rows of active buckets (sparse update = fast).
                self.w[ix] -= lr*v[:,None]*d[None,:]
                # Bias is updated with a 10x smaller step.
                self.b -= lr*0.1*d
                # Backpropagate the same error signal through the attention branch.
                if opt is not None and cache is not None:
                    opt.step(self.att.backward(cache,d.astype(np.float32)),decay)
            # After each epoch, measure loss on validation data; keep the best copy.
            loss=self.nll(valid)
            if loss < best:
                att_copy={k:x.copy() for k,x in self.att.params.items()} if self.att is not None else None
                best=loss; best_params=(self.w.copy(),self.b.copy(),att_copy,epoch+1)
        # Restore the best weights found during training.
        self.w,self.b,att_copy,self.best_epoch=best_params
        if self.att is not None: self.att.params=att_copy
        return self

    # ----- nll(): average negative log-likelihood (lower = better) on some rows -----
    def nll(self,rows):
        terms=[]
        for r in rows:
            p,q=self.probabilities(r['text'])
            # Loss for intent and for sentiment; clamp to 1e-9 to avoid log(0).
            terms += [-np.log(max(float(p[INTENTS.index(r['intent'])]),1e-9)),
                      -np.log(max(float(q[SENTIMENTS.index(r['sentiment'])]),1e-9))]
        return float(np.mean(terms))

    # ----- calibrate(): tune temperature + review threshold on validation data -----
    def calibrate(self,valid):
        # Compute raw scores (logits) once for all validation rows.
        logits=[self.logits(r['text']) for r in valid]
        # Loop over the two heads: task 0 = intent (slots 0..5), task 1 = sentiment (6..8).
        for task,(start,end,labels,key) in enumerate(((0,6,INTENTS,'intent'),(6,9,SENTIMENTS,'sentiment'))):
            targets=[labels.index(r[key]) for r in valid]
            candidates=np.geomspace(0.4,4.0,30)
            losses=[np.mean([-np.log(max(float(softmax(z[start:end]/t)[y]),1e-9)) for z,y in zip(logits,targets)]) for t in candidates]
            self.temperature[task]=float(candidates[int(np.argmin(losses))])
            p=np.array([softmax(z[start:end]/self.temperature[task]) for z in logits])
            confidence=p.max(axis=1); correct=p.argmax(axis=1)==targets
            options=[]
            for th in sorted(set([0.0]+[float(x) for x in confidence])):
                keep=confidence>=th
                if keep.sum()>=5 and correct[keep].mean()>=0.90:
                    options.append((int(keep.sum()),-th))
            self.threshold[task]=-max(options)[1] if options else 1.01

    def parameter_count(self):
        n = self.w.size + self.b.size
        return n + (sum(x.size for x in self.att.params.values()) if self.att is not None else 0)

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        meta={'mode':self.mode,'temperature':self.temperature,'threshold':self.threshold,
              'best_epoch':getattr(self,'best_epoch',None),'dim':DIM,'max_chars':MAX_CHARS,
              'intents':INTENTS,'sentiments':SENTIMENTS,'attention':self.att is not None,
              'att_buckets':ATT_BUCKETS,'att_dim':ATT_DIM,'max_tokens':MAX_TOKENS}
        extra={'att_'+k:x for k,x in self.att.params.items()} if self.att is not None else {}
        np.savez_compressed(path,w=self.w,b=self.b,meta=json.dumps(meta),**extra)

    # ----- load(): rebuild a Model from a saved .npz file -----
    @classmethod
    def load(cls,path):
        # allow_pickle=False for safety (no arbitrary code execution from the file).
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['meta']))
            # Refuse files built with a different feature size or label list.
            if meta['dim']!=DIM or meta['intents']!=INTENTS or meta['sentiments']!=SENTIMENTS:
                raise ValueError('incompatible model schema')
            # Older files (before the attention branch) load as linear-only models.
            attention=meta.get('attention',False)
            if attention and (meta['att_buckets'],meta['att_dim'],meta['max_tokens'])!=(ATT_BUCKETS,ATT_DIM,MAX_TOKENS):
                raise ValueError('incompatible model schema')
            m=cls(meta['mode'],attention=attention); m.w=f['w'].copy(); m.b=f['b'].copy()
            if attention:
                m.att.params={k:f['att_'+k].copy() for k in m.att.params}
        # Restore calibration settings.
        m.temperature=meta['temperature']; m.threshold=meta['threshold']; m.best_epoch=meta['best_epoch']
        return m
