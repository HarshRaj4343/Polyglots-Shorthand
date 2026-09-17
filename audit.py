"""Frozen post-development audit. Never consumed by training or calibration."""
import json
from pathlib import Path
from model import Model
from solution import evaluate
root=Path(__file__).parent
rows=[json.loads(s) for s in (root/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
for i,r in enumerate(rows): r['group']=f'audit-{i:02d}'
m=Model.load(root/'artifacts/full.npz')
result=evaluate(m,rows)
result['protocol']='24 additional authored examples, created after the final model configuration. Not used for training, calibration or further tuning. Still synthetic; some paraphrase the original problem examples.'
(root/'artifacts/audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:{'accuracy':result[k]['accuracy'],'macro_f1':result[k]['macro_f1']} for k in ('intent','sentiment')},indent=2))
