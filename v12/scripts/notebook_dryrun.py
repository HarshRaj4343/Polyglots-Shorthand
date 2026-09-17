"""Execute a generated notebook's cells locally as if on Kaggle (CPU, smoke config) to catch cell-level bugs.

    ../.venv-v12/bin/python scripts/notebook_dryrun.py 01_teacher
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

V12 = Path(__file__).resolve().parents[1]
name = sys.argv[1]
nb = json.loads((V12 / 'notebooks' / f'{name}.ipynb').read_text())
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
tmp = V12 / 'runs' / 'dryrun' / name
shutil.rmtree(tmp, ignore_errors=True)
(tmp / 'input').mkdir(parents=True)
shutil.copy(V12 / 'bundle.zip', tmp / 'input' / 'bundle.zip')
smoke_cfg = json.loads((V12 / 'runs' / 'smoke' / ('teacher.json' if 'teacher' in name else 'student.json')).read_text())
subs = [("os.path.exists('/kaggle/working')", 'True'), ("'google.colab' in sys.modules", 'False'),
        ('/kaggle/working', str(tmp / 'working')), ('/kaggle/tmp', str(tmp / 'kaggle_tmp')), ('/kaggle/input', str(tmp / 'input'))]
g = {}
# cell 1 (setup)
src = cells[0]
for a, b in subs:
    src = src.replace(a, b)
exec(src, g)
# cell 2 (GPU) skipped on CPU; provide torch
import torch  # noqa: E402
g['torch'] = torch
# cell 3 (unpack + hash test)
src = cells[2]
for a, b in subs:
    src = src.replace(a, b)
exec(src, g)
assert (Path(g['PERSIST']) / 'hash_test.txt').exists()
# cell 4 (train) with the smoke config written into the unpacked bundle
Path(g['WORK'], 'configs', 'dryrun.json').write_text(json.dumps(smoke_cfg))
import re
src = re.sub(r"'--config', 'configs/teacher_[a-z0-9_]+\.json'", "'--config', 'configs/dryrun.json', '--smoke'", cells[3])
src = src.replace("'--config', 'configs/student.json'", "'--config', 'configs/dryrun.json'")
exec(src, g)
# cell 5 (package); md.version works locally, cuda name needs a stub
torch.cuda.get_device_name = lambda i=0: 'cpu-dryrun'
exec(cells[4], g)
with zipfile.ZipFile(g['OUT_ZIP']) as z:
    names = z.namelist()
print('outputs.zip entries:', len(names))
for n in names[:12]:
    print('  ', n)
assert any(n.endswith('summary.json') for n in names) and any(n.endswith('.npz') for n in names) and 'hash_test.txt' in names
print('NOTEBOOK DRY RUN PASSED')
