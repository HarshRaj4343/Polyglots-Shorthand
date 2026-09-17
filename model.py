"""Unicode-preserving hashed sparse softmax classifiers. NumPy only."""
import json
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
import numpy as np

INTENTS = ['cancel_order', 'refund', 'track_order', 'not_received', 'damaged_item', 'feedback']
SENTIMENTS = ['negative', 'neutral', 'positive']
DIM = 32768
MAX_CHARS = 512
ALIASES = {'krdo':'kar do', 'kr':'kar', 'kro':'karo', 'krna':'karna',
           'nhi':'nahi', 'nai':'nahi', 'nh':'nahi', 'rha':'raha', 'rhe':'rahe',
           'plz':'please', 'pls':'please', 'mera':'mera', 'bht':'bahut',
           'bohot':'bahut', 'bohut':'bahut', 'acha':'accha', 'achha':'accha',
           'kaha':'kahan', 'kabtk':'kab tak', 'paisa':'paise'}
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def normalize(text):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if len(text) > MAX_CHARS:
        raise ValueError(f'maximum input is {MAX_CHARS} Unicode code points; split long chats upstream')
    text = unicodedata.normalize('NFC', text).lower()
    text = ' '.join(text.split())
    if not text:
        raise ValueError('text must not be empty')
    return text

def features(text, mode='full'):
    s = normalize(text)
    if mode == 'strip_symbols':
        s = ' '.join(''.join(c for c in s if not unicodedata.category(c).startswith(('S', 'P'))).split())
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
        # Learned token-symbol interactions; never hard-code an emoji's sentiment.
        symbols = list(dict.fromkeys(t for t in ts if any(unicodedata.category(c).startswith('S') for c in t)))[:8]
        words = list(dict.fromkeys(t for t in ts if any(c.isalnum() for c in t)))[:64]
        fs += ['e:' + e + '|' + t for e in symbols for t in words]
    counts = Counter(zlib.crc32(f.encode('utf-8')) % DIM for f in fs)
    ix = np.array(sorted(counts), dtype=np.int32)
    vals = np.array([1 + np.log(counts[int(i)]) for i in ix], dtype=np.float32)
    vals /= max(float(np.linalg.norm(vals)), 1e-12)
    return ix, vals

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

class Model:
    def __init__(self, mode='full'):
        self.mode = mode
        self.w = np.zeros((DIM,9), dtype=np.float32)
        self.b = np.zeros(9,dtype=np.float32)
        self.temperature = [1.0,1.0]
        self.threshold = [0.0,0.0]

    def probabilities(self, text):
        ix,v = features(text,self.mode)
        z = v @ self.w[ix] + self.b
        return [softmax(z[:6]/self.temperature[0]), softmax(z[6:]/self.temperature[1])]

    def predict(self,text):
        ps = self.probabilities(text)
        out = {}
        for k,labels,p,th in zip(('intent','sentiment'),(INTENTS,SENTIMENTS),ps,self.threshold):
            j = int(np.argmax(p))
            out[k] = {'label':labels[j], 'confidence':float(p[j]),
                      'review_required':bool(float(p[j]) < th)}
        return out

    def fit(self,train,valid,epochs=55,seed=42):
        xs=[features(r['text'],self.mode) for r in train]
        ys=[(INTENTS.index(r['intent']),SENTIMENTS.index(r['sentiment'])) for r in train]
        rng=np.random.default_rng(seed)
        counts_a=np.bincount([a for a,b in ys],minlength=6)
        counts_b=np.bincount([b for a,b in ys],minlength=3)
        balance_a=len(ys)/(6*np.maximum(counts_a,1))
        balance_b=len(ys)/(3*np.maximum(counts_b,1))
        best=float('inf'); best_params=None
        for epoch in range(epochs):
            lr=0.6/(1+epoch/20)
            # Decoupled weight decay once per epoch, not once per active feature.
            self.w *= 0.999
            for j in rng.permutation(len(train)):
                ix,v=xs[j]; a,b=ys[j]
                z=v @ self.w[ix]+self.b
                d=np.concatenate([softmax(z[:6]),softmax(z[6:])])
                d[a]-=1; d[6+b]-=1
                d[:6]*=balance_a[a]; d[6:]*=balance_b[b]
                self.w[ix] -= lr*v[:,None]*d[None,:]
                self.b -= lr*0.1*d
            loss=self.nll(valid)
            if loss < best:
                best=loss; best_params=(self.w.copy(),self.b.copy(),epoch+1)
        self.w,self.b,self.best_epoch=best_params
        return self

    def nll(self,rows):
        terms=[]
        for r in rows:
            p,q=self.probabilities(r['text'])
            terms += [-np.log(max(float(p[INTENTS.index(r['intent'])]),1e-9)),
                      -np.log(max(float(q[SENTIMENTS.index(r['sentiment'])]),1e-9))]
        return float(np.mean(terms))

    def calibrate(self,valid):
        logits=[]
        for r in valid:
            ix,v=features(r['text'],self.mode); logits.append(v @ self.w[ix]+self.b)
        for task,(start,end,labels,key) in enumerate(((0,6,INTENTS,'intent'),(6,9,SENTIMENTS,'sentiment'))):
            targets=[labels.index(r[key]) for r in valid]
            candidates=np.geomspace(0.4,4.0,30)
            losses=[np.mean([-np.log(max(float(softmax(z[start:end]/t)[y]),1e-9)) for z,y in zip(logits,targets)]) for t in candidates]
            self.temperature[task]=float(candidates[int(np.argmin(losses))])
            p=np.array([softmax(z[start:end]/self.temperature[task]) for z in logits])
            confidence=p.max(axis=1); correct=p.argmax(axis=1)==targets
            # Pick maximum coverage achieving >=90% observed validation precision.
            # This tiny-sample rule is not a statistical precision guarantee.
            options=[]
            for th in sorted(set([0.0]+[float(x) for x in confidence])):
                keep=confidence>=th
                if keep.sum()>=5 and correct[keep].mean()>=0.90:
                    options.append((int(keep.sum()),-th))
            self.threshold[task]=-max(options)[1] if options else 1.01

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        meta={'mode':self.mode,'temperature':self.temperature,'threshold':self.threshold,
              'best_epoch':getattr(self,'best_epoch',None),'dim':DIM,'max_chars':MAX_CHARS,
              'intents':INTENTS,'sentiments':SENTIMENTS}
        np.savez_compressed(path,w=self.w,b=self.b,meta=json.dumps(meta))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['meta']))
            if meta['dim']!=DIM or meta['intents']!=INTENTS or meta['sentiments']!=SENTIMENTS:
                raise ValueError('incompatible model schema')
            m=cls(meta['mode']); m.w=f['w'].copy(); m.b=f['b'].copy()
        m.temperature=meta['temperature']; m.threshold=meta['threshold']; m.best_epoch=meta['best_epoch']
        return m
