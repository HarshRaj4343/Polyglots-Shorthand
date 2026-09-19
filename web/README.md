# web/ — the frontend

A single static page plus a small FastAPI backend. Replaces `app.py` (Streamlit); the
Streamlit app still works and is untouched.

    web/server.py      FastAPI: serves the page, /api/meta and /api/predict
    web/static/        index.html, style.css, app.js (no build step, no framework)

The model is the v1.2 distilled student, loaded once at startup from
`v12/deploy/student_kd_s42` (INT8 ONNX encoder + NumPy piece embeddings and linear branch).

## Run locally

    pip install -r requirements-web.txt
    uvicorn web.server:app --reload --port 8000

Then open <http://localhost:8000>. Interactive API docs: <http://localhost:8000/api/docs>.

## API

`GET /api/meta` — model name, parameter count, test accuracy, temperatures, thresholds, label sets.

`POST /api/predict` — body `{"texts": ["refund kab milega"]}`, up to 200 messages.

    curl -s localhost:8000/api/predict \
      -H 'Content-Type: application/json' \
      -d '{"texts":["refund kab milega bhai"]}'

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
