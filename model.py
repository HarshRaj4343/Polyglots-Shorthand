"""Unicode-preserving hashed sparse softmax classifiers. NumPy only."""
# ============================================================================
# model.py - THE BRAIN OF THE PROJECT
# ----------------------------------------------------------------------------
# This file turns a text message into numbers ("features") and trains a small
# linear model that predicts TWO things at once:
#   1. intent    -> what the customer wants (cancel, refund, track, ...)
#   2. sentiment -> how they feel (negative / neutral / positive)
# Flow: text -> normalize() -> features() -> Model.probabilities() -> predict()
# ============================================================================

# ---------------------------------------------------------------------------
# Imports: only the Python standard library + NumPy (no PyTorch/sklearn).
# ---------------------------------------------------------------------------
import json
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
# The 6 possible intent labels. Their index (0..5) is the model's output slot.
INTENTS = ['cancel_order', 'refund', 'track_order', 'not_received', 'damaged_item', 'feedback']
# The 3 possible sentiment labels. They use output slots 6..8.
SENTIMENTS = ['negative', 'neutral', 'positive']
# Size of the hashed feature space: every feature string is mapped to one of
# 32,768 "buckets". The weight matrix therefore has DIM rows.
DIM = 32768
# Longest message we accept (in Unicode characters).
MAX_CHARS = 512
# Spelling-normalisation table for common Hinglish shorthand, e.g. "nhi" -> "nahi".
# Used to create extra "a:" (alias) features so shorthand and full spellings
# share a feature.
ALIASES = {'krdo':'kar do', 'kr':'kar', 'kro':'karo', 'krna':'karna',
           'nhi':'nahi', 'nai':'nahi', 'nh':'nahi', 'rha':'raha', 'rhe':'rahe',
           'plz':'please', 'pls':'please', 'mera':'mera', 'bht':'bahut',
           'bohot':'bahut', 'bohut':'bahut', 'acha':'accha', 'achha':'accha',
           'kaha':'kahan', 'kabtk':'kab tak', 'paisa':'paise'}
# Tokenizer regex: a token is either a run of word characters (letters/digits,
# any script) OR a single non-space symbol (so emojis and "?" / "!" survive).
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)

# ---------------------------------------------------------------------------
# normalize(): clean the raw input text into a consistent form.
# ---------------------------------------------------------------------------
def normalize(text):
    # Reject anything that isn't a string.
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    # Reject overly long messages.
    if len(text) > MAX_CHARS:
        raise ValueError(f'maximum input is {MAX_CHARS} Unicode code points; split long chats upstream')
    # NFC = combine accented/composed characters into a single standard form,
    # then lowercase everything ("ORDER" == "order").
    text = unicodedata.normalize('NFC', text).lower()
    # Collapse repeated spaces/tabs/newlines into single spaces and trim ends.
    text = ' '.join(text.split())
    # Empty (or whitespace-only) input is not allowed.
    if not text:
        raise ValueError('text must not be empty')
    return text

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
    # Step 1: clean the text.
    s = normalize(text)
    # Step 2 (ablation only): drop all Symbol (S*) and Punctuation (P*) chars.
    if mode == 'strip_symbols':
        s = ' '.join(''.join(c for c in s if not unicodedata.category(c).startswith(('S', 'P'))).split())
    # Step 3: split into tokens (words, emojis, punctuation).
    ts = TOKEN.findall(s)
    # Feature family "w:" = each individual word/token (unigram).
    fs = ['w:' + t for t in ts]
    # Feature family "b:" = each pair of neighbouring tokens (bigram),
    # e.g. "nahi|mila" - helps capture short negations.
    fs += ['b:' + a + '|' + b for a,b in zip(ts,ts[1:])]
    # Feature family "c3/c4/c5:" = character n-grams of length 3,4,5 over the
    # whole string (with ^ and $ marking start/end). These make the model robust
    # to spelling variations like "delivry" vs "delivery".
    if mode != 'word_only':
        padded = '^' + s + '$'
        fs += [f'c{n}:' + padded[i:i+n] for n in (3,4,5) for i in range(len(padded)-n+1)]
    # Extra families only used by 'full' and 'strip_symbols' modes.
    if mode in ('full', 'strip_symbols'):
        # Replace shorthand with canonical spelling via ALIASES ("nhi" -> "nahi").
        canon = [u for t in ts for u in ALIASES.get(t,t).split()]
        # "a:" = canonical unigrams, "ab:" = canonical bigrams.
        fs += ['a:' + t for t in canon]
        fs += ['ab:' + a + '|' + b for a,b in zip(canon,canon[1:])]
        # Learned token-symbol interactions; never hard-code an emoji's sentiment.
        # Collect up to 8 unique symbol tokens (emojis etc.) and 64 unique words...
        symbols = list(dict.fromkeys(t for t in ts if any(unicodedata.category(c).startswith('S') for c in t)))[:8]
        words = list(dict.fromkeys(t for t in ts if any(c.isalnum() for c in t)))[:64]
        # ...and pair every emoji with every word ("e:😒|badhiya"), so the model
        # can learn that "badhiya + 😒" is sarcastic/negative.
        fs += ['e:' + e + '|' + t for e in symbols for t in words]
    # Step 4: "hashing trick" - map each feature string to a bucket number with
    # CRC32 % DIM, and count how often each bucket appears.
    counts = Counter(zlib.crc32(f.encode('utf-8')) % DIM for f in fs)
    # Sorted list of active bucket indices.
    ix = np.array(sorted(counts), dtype=np.int32)
    # Value per bucket = 1 + log(count): repeated features count, but sub-linearly.
    vals = np.array([1 + np.log(counts[int(i)]) for i in ix], dtype=np.float32)
    # L2-normalise so long and short messages have vectors of the same length.
    vals /= max(float(np.linalg.norm(vals)), 1e-12)
    return ix, vals

# ---------------------------------------------------------------------------
# softmax(): turn raw scores (logits) into probabilities that sum to 1.
# Subtracting the max first avoids numeric overflow in exp().
# ---------------------------------------------------------------------------
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

# ---------------------------------------------------------------------------
# Model: two linear softmax classifiers sharing ONE weight matrix.
# Columns 0..5 of the output are intent scores, columns 6..8 sentiment scores.
# ---------------------------------------------------------------------------
class Model:
    # ----- Constructor: create an untrained (all-zero) model -----
    def __init__(self, mode='full'):
        self.mode = mode                                   # which feature set to use
        self.w = np.zeros((DIM,9), dtype=np.float32)       # weights: 32768 buckets x 9 outputs
        self.b = np.zeros(9,dtype=np.float32)              # bias per output
        self.temperature = [1.0,1.0]                       # calibration temps [intent, sentiment]
        self.threshold = [0.0,0.0]                         # "needs human review" cutoffs [intent, sentiment]

    # ----- probabilities(): text -> [intent probs (6), sentiment probs (3)] -----
    def probabilities(self, text):
        ix,v = features(text,self.mode)
        # Linear score: sum of the weight rows of active buckets (scaled by value) + bias.
        z = v @ self.w[ix] + self.b
        # Split the 9 scores into the two heads, divide by temperature, softmax each.
        return [softmax(z[:6]/self.temperature[0]), softmax(z[6:]/self.temperature[1])]

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

    # ----- fit(): train the weights with stochastic gradient descent (SGD) -----
    def fit(self,train,valid,epochs=55,seed=42):
        # Pre-compute features and numeric labels for every training row.
        xs=[features(r['text'],self.mode) for r in train]
        ys=[(INTENTS.index(r['intent']),SENTIMENTS.index(r['sentiment'])) for r in train]
        # Seeded random generator so training is reproducible.
        rng=np.random.default_rng(seed)
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
            lr=0.6/(1+epoch/20)
            # Decoupled weight decay once per epoch, not once per active feature.
            self.w *= 0.999
            # Visit training rows in a random order each epoch.
            for j in rng.permutation(len(train)):
                ix,v=xs[j]; a,b=ys[j]
                # Forward pass: scores -> probabilities for both heads.
                z=v @ self.w[ix]+self.b
                d=np.concatenate([softmax(z[:6]),softmax(z[6:])])
                # Gradient of cross-entropy loss = predicted prob - 1 at the true label.
                d[a]-=1; d[6+b]-=1
                # Scale the gradient by the class-balance weight of the true label.
                d[:6]*=balance_a[a]; d[6:]*=balance_b[b]
                # Update only the rows of active buckets (sparse update = fast).
                self.w[ix] -= lr*v[:,None]*d[None,:]
                # Bias is updated with a 10x smaller step.
                self.b -= lr*0.1*d
            # After each epoch, measure loss on validation data; keep the best copy.
            loss=self.nll(valid)
            if loss < best:
                best=loss; best_params=(self.w.copy(),self.b.copy(),epoch+1)
        # Restore the best weights found during training.
        self.w,self.b,self.best_epoch=best_params
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
        logits=[]
        for r in valid:
            ix,v=features(r['text'],self.mode); logits.append(v @ self.w[ix]+self.b)
        # Loop over the two heads: task 0 = intent (slots 0..5), task 1 = sentiment (6..8).
        for task,(start,end,labels,key) in enumerate(((0,6,INTENTS,'intent'),(6,9,SENTIMENTS,'sentiment'))):
            targets=[labels.index(r[key]) for r in valid]
            # Temperature scaling: try 30 temperatures between 0.4 and 4.0 and keep
            # the one that gives the lowest validation loss. This makes the
            # reported "confidence" numbers more honest.
            candidates=np.geomspace(0.4,4.0,30)
            losses=[np.mean([-np.log(max(float(softmax(z[start:end]/t)[y]),1e-9)) for z,y in zip(logits,targets)]) for t in candidates]
            self.temperature[task]=float(candidates[int(np.argmin(losses))])
            # Calibrated probabilities, confidence of top label, and whether it was right.
            p=np.array([softmax(z[start:end]/self.temperature[task]) for z in logits])
            confidence=p.max(axis=1); correct=p.argmax(axis=1)==targets
            # Pick maximum coverage achieving >=90% observed validation precision.
            # This tiny-sample rule is not a statistical precision guarantee.
            # Try every observed confidence as a cutoff; keep ones where the
            # accepted predictions (>=5 of them) are at least 90% correct.
            options=[]
            for th in sorted(set([0.0]+[float(x) for x in confidence])):
                keep=confidence>=th
                if keep.sum()>=5 and correct[keep].mean()>=0.90:
                    options.append((int(keep.sum()),-th))
            # Choose the cutoff that keeps the most rows (ties -> lowest cutoff).
            # If none qualify, 1.01 means "always send to human review".
            self.threshold[task]=-max(options)[1] if options else 1.01

    # ----- save(): write weights + settings to a compressed .npz file -----
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        # Metadata stored as JSON text alongside the weight arrays.
        meta={'mode':self.mode,'temperature':self.temperature,'threshold':self.threshold,
              'best_epoch':getattr(self,'best_epoch',None),'dim':DIM,'max_chars':MAX_CHARS,
              'intents':INTENTS,'sentiments':SENTIMENTS}
        np.savez_compressed(path,w=self.w,b=self.b,meta=json.dumps(meta))

    # ----- load(): rebuild a Model from a saved .npz file -----
    @classmethod
    def load(cls,path):
        # allow_pickle=False for safety (no arbitrary code execution from the file).
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['meta']))
            # Refuse files built with a different feature size or label list.
            if meta['dim']!=DIM or meta['intents']!=INTENTS or meta['sentiments']!=SENTIMENTS:
                raise ValueError('incompatible model schema')
            m=cls(meta['mode']); m.w=f['w'].copy(); m.b=f['b'].copy()
        # Restore calibration settings.
        m.temperature=meta['temperature']; m.threshold=meta['threshold']; m.best_epoch=meta['best_epoch']
        return m
