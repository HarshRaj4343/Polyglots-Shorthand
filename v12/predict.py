"""v1.2 student inference (NumPy + onnxruntime, no torch). Same JSON as v1.1 `solution.py predict`.

    python v12/predict.py "refund kab milega bhai"
    python v12/predict.py --stdin < messages.txt          # one JSON per line; bad input -> {"error": ...}
    python v12/predict.py --attention "cancel mat karna, bas track karo"
    python v12/predict.py --deploy v12/deploy/student_kd_s42 --precision fp32 "..."
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from polyglot12.runtime import StudentRuntime  # noqa: E402

DEFAULT_DEPLOY = HERE / 'deploy' / 'student_kd_s42'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('text', nargs='?')
    ap.add_argument('--stdin', action='store_true')
    ap.add_argument('--deploy', default=str(DEFAULT_DEPLOY))
    ap.add_argument('--precision', choices=['int8', 'fp32'], default='int8')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--attention', action='store_true', help='also return attention-pooling weights per token')
    args = ap.parse_args()
    if not (Path(args.deploy) / 'meta.json').exists():
        ap.error(f'no deployed model at {args.deploy}; run scripts/export_student.py first')
    rt = StudentRuntime(args.deploy, args.precision, args.threads)
    if args.stdin:
        for line in sys.stdin:
            try:
                print(json.dumps(rt.predict(line.rstrip('\n'), args.attention), ensure_ascii=False), flush=True)
            except (ValueError, TypeError) as e:
                print(json.dumps({'error': str(e)}), flush=True)
    elif args.text is not None:
        try:
            print(json.dumps(rt.predict(args.text, args.attention), ensure_ascii=False, indent=2))
        except (ValueError, TypeError) as e:
            ap.error(str(e))
    else:
        ap.error('supply text or --stdin')


if __name__ == '__main__':
    main()
