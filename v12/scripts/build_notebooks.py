"""Generate the self-contained GPU notebooks in v12/notebooks/ ("Run all" on Colab or Kaggle)."""
import sys
from pathlib import Path

import nbformat as nbf

V12 = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    '01_teacher': {
        'title': '01 - Teacher diagnostic (v1.1 train split only)',
        'what': 'Fine-tunes hing-roberta, hing-roberta-mixed, hing-bert and MuRIL (3 learning rates each) on the '
                'v1.1 TRAIN split, early-stops on validation NLL and saves logits for train/val/test/test_stress/audit/contrast.',
        'gpu': '~35-60 min on a T4 (12 short runs; most time is model download and checkpoint writes). Hard guard: 105 min.',
        'module': 'polyglot12.teacher',
        'config': 'configs/teacher_v11data.json',
    },
    '02_teacher_kd': {
        'title': '02 - Teacher v2 (v1.1 train + generated + augmented + public sentiment) and KD pseudo-labels',
        'what': 'Fine-tunes hing-roberta-mixed at 2 learning rates on v1.1 train + 1,202 generated support messages + 2,323 '
                'augmented variants + 4,000 public sentiment tweets (intent loss masked), early-stops on v1.1 validation NLL, '
                'and writes logits for validation/test/test_stress/audit/contrast plus the KD sets (train, generated, '
                'augmented, 12,105 public labeled, 50,000 unlabeled pool).',
        'gpu': '~30-50 min on a T4 (2 runs x <=6 epochs of ~8k rows, then ~75k rows of inference per run). Hard guard: 100 min.',
        'module': 'polyglot12.teacher',
        'config': 'configs/teacher_kd.json',
    },
    '03_student': {
        'title': '03 - Student distillation (hashed pieces + 4-layer transformer + linear branch)',
        'what': 'Trains the 11.9M-parameter student in order: student + KD (seed 42), student without KD (same labeled '
                'data), student + KD without the linear branch, then seeds 43 and 44 while the time budget allows. '
                'Early stopping on v1.1 validation NLL; writes logits for validation/test/test_stress/audit/contrast '
                'and the weights of student_kd_s42 for ONNX export.',
        'gpu': '~40-60 min on a T4 (5 variants x ~5-10 min; the budget guard stops starting new variants after 100 min).',
        'module': 'polyglot12.train_student',
        'config': 'configs/student.json',
        'keep_pt': ['runs/student_kd_s42/student.pt'],
    },
}

SETUP = r'''# ---- 1. Platform, persistent storage, bundle location --------------------------------
import glob, json, os, shutil, subprocess, sys, time, zipfile
NOTEBOOK = '{name}'
IS_COLAB = 'google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ
IS_KAGGLE = os.path.exists('/kaggle/working')
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')                      # checkpoints survive disconnects here
    PERSIST = f'/content/drive/MyDrive/polyglot12/{{NOTEBOOK}}'
    WORK = '/content/v12bundle'
    OUT_ZIP = f'/content/drive/MyDrive/polyglot12/{{NOTEBOOK}}_outputs.zip'
elif IS_KAGGLE:
    PERSIST = f'/kaggle/working/{{NOTEBOOK}}'
    WORK = '/kaggle/tmp/v12bundle'
    OUT_ZIP = '/kaggle/working/outputs.zip'
else:
    raise SystemExit('Run this notebook on Google Colab or Kaggle (GPU).')
os.makedirs(PERSIST, exist_ok=True)
print('platform:', 'colab' if IS_COLAB else 'kaggle', '| persistent dir:', PERSIST)
'''

GPU = r'''# ---- 2. GPU check + dependencies (torch is preinstalled; do not reinstall it) -----------
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.'
cap = torch.cuda.get_device_capability(0)
print('GPU:', torch.cuda.get_device_name(0), 'compute capability', cap)
if cap < (7, 0):
    print('WARNING: P100-class GPU. Recent PyTorch wheels may not support it; on Kaggle pick "GPU T4 x2".')
if IS_KAGGLE:
    import urllib.request
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=10)
    except Exception as e:
        raise SystemExit('No internet: Kaggle > Settings > Internet ON (needed to download models).') from e
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.46,<6', 'sentencepiece', 'protobuf'], check=True)
'''

UNPACK = r'''# ---- 3. Unpack bundle.zip and run the hash-consistency test ----------------------------
def find_bundle():
    cands = (['/content/drive/MyDrive/polyglot12/bundle.zip', '/content/bundle.zip'] if IS_COLAB
             else glob.glob('/kaggle/input/**/bundle.zip', recursive=True) + ['/kaggle/working/bundle.zip'])
    for c in cands:
        if os.path.exists(c):
            return ('zip', c)
    if IS_KAGGLE:  # Kaggle datasets auto-extract zips: look for the unpacked tree instead
        hits = glob.glob('/kaggle/input/**/polyglot12/common.py', recursive=True)
        if hits:
            return ('dir', str(os.path.dirname(os.path.dirname(hits[0]))))
    if IS_COLAB:
        from google.colab import files
        print('bundle.zip not found on Drive (MyDrive/polyglot12/bundle.zip) - upload it now:')
        up = files.upload()
        return ('zip', '/content/' + next(iter(up)))
    raise SystemExit('bundle.zip not found. Kaggle: Add Data > upload bundle.zip as a dataset.')

kind, src = find_bundle()
shutil.rmtree(WORK, ignore_errors=True)
if kind == 'zip':
    with zipfile.ZipFile(src) as z:
        z.extractall(WORK)
else:
    shutil.copytree(src, WORK)
print('bundle from', src)
sys.path.insert(0, WORK)
r = subprocess.run([sys.executable, 'tests/test_hashing.py'], cwd=WORK, capture_output=True, text=True)
print(r.stdout, r.stderr)
open(f'{{PERSIST}}/hash_test.txt', 'w').write(r.stdout + r.stderr)
assert r.returncode == 0, 'Hashing differs from the Mac fixtures - stop and report this.'
'''

TRAIN = r'''# ---- 4. Train (auto-resumes: just "Run all" again after a disconnect) -------------------
cmd = [sys.executable, '-u', '-m', '{module}', '--config', '{config}', '--out', PERSIST]
print(' '.join(cmd))
t0 = time.time()
p = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     env={{**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'}})
with open(f'{{PERSIST}}/train_log.txt', 'a') as log:
    for line in p.stdout:
        print(line, end='')
        log.write(line)
rc = p.wait()
print(f'finished with exit code {{rc}} after {{(time.time() - t0) / 60:.1f}} min')
assert rc == 0, 'Training failed - see the log above (re-running resumes from the last checkpoint).'
'''

PACKAGE = r'''# ---- 5. Package outputs.zip (logits, status, summary, logs; no weights) ------------------
import importlib.metadata as md
env = {{'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
       'transformers': md.version('transformers'), 'python': sys.version.split()[0],
       'platform': 'colab' if IS_COLAB else 'kaggle'}}
json.dump(env, open(f'{{PERSIST}}/env.json', 'w'), indent=1)
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(PERSIST):
        for f in fs:
            full = os.path.join(root, f)
            if f.endswith(('.json', '.npz', '.txt')) or os.path.relpath(full, PERSIST) in {keep_pt!r}:
                z.write(full, os.path.relpath(full, PERSIST))
print('wrote', OUT_ZIP, f'{{os.path.getsize(OUT_ZIP) / 1e6:.1f}} MB')
if IS_COLAB:
    from google.colab import files
    files.download(OUT_ZIP)
else:
    print('Kaggle: download outputs.zip from the Output panel (right side, /kaggle/working).')
'''


def build(name, spec):
    nb = nbf.v4.new_notebook()
    nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                   'language_info': {'name': 'python'}, 'accelerator': 'GPU'}
    intro = f"""# The Polyglot's Shorthand v1.2 - {spec['title']}

{spec['what']}

**Expected GPU time:** {spec['gpu']}

**Colab (recommended):** Runtime > Change runtime type > **T4 GPU**. Put `bundle.zip` at `MyDrive/polyglot12/bundle.zip`
(or upload it when prompted). Then **Runtime > Run all**. Checkpoints go to `MyDrive/polyglot12/{name}/`; after a
disconnect simply **Run all** again - finished runs are skipped and the interrupted run resumes from its last epoch.

**Kaggle:** Settings > Accelerator **GPU T4 x2** (not P100) and **Internet ON**. Add `bundle.zip` as a dataset
(Add Data > Upload). Run all. Outputs are in `/kaggle/working` (only kept across sessions if you use Save Version).

At the end `outputs.zip` is downloaded (Colab) or appears in the Output panel (Kaggle).
"""
    nb.cells = [nbf.v4.new_markdown_cell(intro),
                nbf.v4.new_code_cell(SETUP.format(name=name)),
                nbf.v4.new_code_cell(GPU),
                nbf.v4.new_code_cell(UNPACK.format()),
                nbf.v4.new_code_cell(TRAIN.format(module=spec['module'], config=spec['config'])),
                nbf.v4.new_code_cell(PACKAGE.format(keep_pt=spec.get('keep_pt', [])))]
    path = V12 / 'notebooks' / f'{name}.ipynb'
    nbf.write(nb, path)
    # Every code cell must at least parse.
    for c in nb.cells:
        if c.cell_type == 'code':
            compile(c.source, f'{name}:cell', 'exec')
    print('wrote', path)


if __name__ == '__main__':
    names = sys.argv[1:] or list(NOTEBOOKS)
    for n in names:
        build(n, NOTEBOOKS[n])
