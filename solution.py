"""Reproduce training, evaluation, ablations, latency and JSON inference."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
from model import Model, INTENTS, SENTIMENTS, DIM, features
from make_data import build

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'

def read_data(split):
    return [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]

def validate_data(parts):
    groups={s:{r['group'] for r in rows} for s,rows in parts.items()}
    texts={s:{r['text'].lower() for r in rows} for s,rows in parts.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b] or texts[a]&texts[b]:
            raise ValueError(f'cross-split overlap: {a}/{b}')
    for split,rows in parts.items():
        for r in rows:
            if r['intent'] not in INTENTS or r['sentiment'] not in SENTIMENTS:
                raise ValueError(f'invalid labels in {split}')
            features(r['text'])
        if {r['intent'] for r in rows} != set(INTENTS):
            raise ValueError(f'missing intent in {split}')

def summarize(y,p,labels,conf=None,threshold=0):
    cm=np.zeros((len(labels),len(labels)),dtype=int)
    for a,b in zip(y,p): cm[a,b]+=1
    per={}
    for i,label in enumerate(labels):
        tp=int(cm[i,i]); fp=int(cm[:,i].sum())-tp; fn=int(cm[i,:].sum())-tp
        precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
        per[label]={'precision':precision,'recall':recall,'f1':2*tp/max(2*tp+fp+fn,1),'support':int(cm[i,:].sum())}
    out={'n':len(y),'accuracy':float(np.mean(np.array(y)==p)),
         'macro_f1':float(np.mean([v['f1'] for v in per.values()])),
         'per_class':per,'confusion_matrix':cm.tolist(),'label_order':labels}
    if conf is not None:
        keep=np.array(conf)>=threshold
        out['coverage']=float(keep.mean())
        out['accepted_accuracy']=float((np.array(y)[keep]==np.array(p)[keep]).mean()) if keep.any() else None
    return out

def evaluate(model,rows):
    preds=[model.predict(r['text']) for r in rows]
    out={}
    for j,(key,labels) in enumerate([('intent',INTENTS),('sentiment',SENTIMENTS)]):
        y=[labels.index(r[key]) for r in rows]
        p=[labels.index(r[key]['label']) for r in preds]
        out[key]=summarize(y,p,labels,[r[key]['confidence'] for r in preds],model.threshold[j])
    out['errors']=[{'text':r['text'],'group':r['group'],
                    'expected':{k:r[k] for k in ('intent','sentiment')},'predicted':p}
                   for r,p in zip(rows,preds) if any(r[k]!=p[k]['label'] for k in ('intent','sentiment'))]
    return out

# New transformations are used only for the frozen test set, never for model selection.
def stress_text(text):
    mapping={'nahi':'nahii','nhi':'nahii','kar':'karr','kr':'karr','hai':'hey',
             'h':'hey','mila':'milaa','mera':'meraa','kya':'kyaa','please':'pleez',
             'order':'orrder','delivery':'delivary','refund':'refnd','service':'serrvice'}
    return ' '.join(mapping.get(t,t) for t in text.lower().split())

def stress_evaluate(model,rows):
    changed=[dict(r,text=stress_text(r['text'])) for r in rows]
    out=evaluate(model,changed)
    original=[model.predict(r['text']) for r in rows]
    altered=[model.predict(r['text']) for r in changed]
    out['changed_fraction']=float(np.mean([a['text']!=b['text'] for a,b in zip(rows,changed)]))
    for key in ('intent','sentiment'):
        out[key]['prediction_consistency']=float(np.mean([a[key]['label']==b[key]['label'] for a,b in zip(original,altered)]))
    return out

def emoji_pairs(model,rows):
    # Exact held-out base +/- one emoji; count one pair per lowercased base.
    lookup={r['text'].lower():r for r in rows}
    pairs=[]
    for s,r in lookup.items():
        if s.endswith(' 😒') and s[:-2] in lookup:
            a=lookup[s[:-2]]
            if a['sentiment']=='positive' and r['sentiment']=='negative':
                pa=model.predict(a['text']); pb=model.predict(r['text'])
                pairs.append({'base':a['text'],'with_emoji':r['text'],
                              'base_prediction':pa['sentiment']['label'],'emoji_prediction':pb['sentiment']['label'],
                              'both_correct':pa['sentiment']['label']=='positive' and pb['sentiment']['label']=='negative'})
    return {'n_pairs':len(pairs),'both_correct_rate':float(np.mean([r['both_correct'] for r in pairs])) if pairs else None,'pairs':pairs}

def benchmark(model,rows,repeats=600):
    def measure(texts):
        for i in range(80): json.dumps(model.predict(texts[i%len(texts)]),ensure_ascii=False)
        times=[]
        begin=time.perf_counter_ns()
        for i in range(repeats):
            start=time.perf_counter_ns()
            json.dumps(model.predict(texts[i%len(texts)]),ensure_ascii=False)
            times.append((time.perf_counter_ns()-start)/1e6)
        elapsed=(time.perf_counter_ns()-begin)/1e9
        return {'requests':repeats,'p50_ms':float(np.percentile(times,50)),
                'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),
                'messages_per_second':repeats/elapsed}
    texts=[r['text'] for r in rows]
    sample='mera order delivered bol raha hai but mila hi nahi 😒 '
    lengths={str(n):measure([(sample*20)[:n]]) for n in (32,128,512)}
    return {'scope':'Warm, serial, batch=1. Includes text normalization, features, both heads and JSON serialization. Excludes process startup, model loading, network and queueing.',
            'test_messages':measure(texts),'fixed_length_synthetic':lengths}

def reproduce():
    ART.mkdir(exist_ok=True)
    build(ROOT/'data')
    parts={s:read_data(s) for s in ('train','validation','test')}
    validate_data(parts)
    summary={'seed':42,'data':{s:{'rows':len(rs),'groups':len({r['group'] for r in rs}),
             'intents':dict(Counter(r['intent'] for r in rs)),
             'sentiments':dict(Counter(r['sentiment'] for r in rs))} for s,rs in parts.items()},'models':{}}
    for mode in ('full','word_only','char_word','strip_symbols'):
        print('Training',mode,flush=True)
        start=time.perf_counter()
        model=Model(mode).fit(parts['train'],parts['validation'])
        model.calibrate(parts['validation'])
        model.save(ART/f'{mode}.npz')
        elapsed=time.perf_counter()-start
        result={'training_seconds':elapsed,'best_epoch':model.best_epoch,
                'temperature':model.temperature,'review_threshold':model.threshold,
                'validation':evaluate(model,parts['validation']),
                'test':evaluate(model,parts['test']),
                'stress_test':stress_evaluate(model,parts['test']),
                'emoji_pairs':emoji_pairs(model,parts['test'])}
        summary['models'][mode]=result
        print(json.dumps({'mode':mode,'test_intent_f1':result['test']['intent']['macro_f1'],
                          'test_sentiment_f1':result['test']['sentiment']['macro_f1'],
                          'seconds':elapsed}),flush=True)
    model=Model.load(ART/'full.npz')
    summary['benchmark']=benchmark(model,parts['test'])
    summary['parameter_count']=int(model.w.size+model.b.size)
    summary['weight_bytes']=int(model.w.nbytes+model.b.nbytes)
    summary['saved_model_bytes']=(ART/'full.npz').stat().st_size
    cpu=platform.processor()
    if sys.platform=='darwin':
        try: cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,stderr=subprocess.DEVNULL).strip()
        except Exception: pass
    summary['environment']={'python':sys.version.split()[0],'numpy':np.__version__,
                            'system':platform.system(),'release':platform.release(),
                            'machine':platform.machine(),'cpu':cpu}
    summary['dataset_sha256']={s:hashlib.sha256((ROOT/'data'/f'{s}.jsonl').read_bytes()).hexdigest() for s in parts}
    (ART/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved artifacts/results.json',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('reproduce')
    pr=sub.add_parser('predict'); pr.add_argument('text',nargs='?'); pr.add_argument('--stdin',action='store_true')
    tr=sub.add_parser('train'); tr.add_argument('--mode',choices=['full','word_only','char_word','strip_symbols'],default='full')
    sub.add_parser('evaluate'); sub.add_parser('benchmark')
    args=p.parse_args()
    if args.command=='reproduce': return reproduce()
    if args.command=='train':
        parts={s:read_data(s) for s in ('train','validation','test')}; validate_data(parts)
        m=Model(args.mode).fit(parts['train'],parts['validation']);m.calibrate(parts['validation'])
        m.save(ART/f'{args.mode}.npz');print('Model saved');return
    m=Model.load(ART/'full.npz')
    if args.command=='predict':
        if args.stdin:
            for line in sys.stdin:
                try: print(json.dumps(m.predict(line.rstrip('\n')),ensure_ascii=False),flush=True)
                except (ValueError,TypeError) as e: print(json.dumps({'error':str(e)}),flush=True)
        elif args.text is not None:
            try: print(json.dumps(m.predict(args.text),ensure_ascii=False,indent=2))
            except (ValueError,TypeError) as e: p.error(str(e))
        else: p.error('supply text or --stdin')
    elif args.command=='evaluate': print(json.dumps(evaluate(m,read_data('test')),ensure_ascii=False,indent=2))
    else: print(json.dumps(benchmark(m,read_data('test')),indent=2))

if __name__=='__main__':main()
