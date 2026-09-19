# web/ — the frontend

A single static page plus a small FastAPI backend. Replaces `app.py` (Streamlit); the
Streamlit app still works and is untouched.

    web/server.py      FastAPI: serves the page, /api/meta and /api/predict
    web/static/        index.html, style.css, app.js (no build step, no framework)

Five models are selectable from the picker above the demo — the same set the Streamlit
sidebar offers:

| Key | Source | Test accuracy (intent / sentiment) |
| --- | --- | --- |
| `v1.2 student + KD (ONNX INT8)` | `v12/deploy/student_kd_s42` | 92.1% / 91.6% |
| `full.npz` | `artifacts/` | 90.5% / 84.7% |
| `char_word.npz` | `artifacts/` | 90.5% / 83.2% |
| `strip_symbols.npz` | `artifacts/` | 88.4% / 76.3% |
| `word_only.npz` | `artifacts/` | 85.3% / 82.6% |

The v1.2 student is the default and is warmed at startup; the rest load on first use and are
then cached. The v1.1 models need only NumPy, so they still work if `onnxruntime` is missing —
the student is simply dropped from the list.

## Run locally

    pip install -r requirements-web.txt
    uvicorn web.server:app --reload --port 8000

Then open <http://localhost:8000>. Interactive API docs: <http://localhost:8000/api/docs>.

## API

`GET /api/meta` — every selectable model with its parameter count, feature mode, test accuracy,
temperatures and review thresholds, plus `default` and the label sets.

`POST /api/predict` — body `{"texts": [...], "model": "<key>"}`, up to 200 messages.
`model` is optional and falls back to the default; an unknown key returns 400.

    curl -s localhost:8000/api/predict \
      -H 'Content-Type: application/json' \
      -d '{"texts":["refund kab milega bhai"], "model":"full.npz"}'

Each result carries `intent` and `sentiment` (label, confidence, `review_required`, and the
probability of every class) plus `attention`, the pooling weight per token.

## Deploy

The model files are committed to the repo (~36 MB), so any git-based host works with no extra
setup. `Procfile` at the repo root holds the start command.

**Render** (free tier) — New → Web Service → connect the repo:

    Build command:  pip install -r requirements-web.txt
    Start command:  uvicorn web.server:app --host 0.0.0.0 --port $PORT

**Railway / Heroku** — detected from `Procfile`; nothing else to configure.

**Fly.io** — `fly launch` picks up the `Procfile`; the image needs ~512 MB RAM.

Note: Streamlit Community Cloud only runs Streamlit, so it cannot host this app —
point it at `app.py` instead, or use one of the hosts above.
