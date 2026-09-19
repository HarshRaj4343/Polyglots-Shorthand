# ============================================================================
# app.py - STREAMLIT WEBSITE FOR THE CLASSIFIER
# ----------------------------------------------------------------------------
# Run:  .venv/bin/streamlit run app.py
# Type a Hinglish customer message and see the predicted intent, sentiment,
# confidence for every class, and whether a human should review it.
# Default model: the v1.2 distilled student (ONNX INT8, v12/deploy/student_kd_s42),
# the most accurate model with weights available locally on BOTH heads
# (test accuracy intent 0.921 / sentiment 0.916 vs 0.905 / 0.847 for v1.1).
# The v1.1 models in artifacts/*.npz stay selectable in the sidebar.
# ============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import INTENTS, SENTIMENTS, MAX_CHARS, Model, features

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
V12_DEPLOY = ROOT / 'v12' / 'deploy' / 'student_kd_s42'
V12_KEY = 'v1.2 student + KD (ONNX INT8)'

# Test-set accuracy (190 rows) from the evaluation reports, shown in the sidebar.
def _load_accuracy():
    acc = {}
    try:
        r = json.loads((ART / 'results.json').read_text())['models']
        for mode, v in r.items():
            acc[f'{mode}.npz'] = (v['test']['intent']['accuracy'], v['test']['sentiment']['accuracy'])
    except (OSError, KeyError, ValueError):
        pass
    try:
        c = json.loads((ROOT / 'v12' / 'results' / 'phase5_comparison.json').read_text())['full_reports']['deploy_int8']['test']
        acc[V12_KEY] = (c['intent']['accuracy'], c['sentiment']['accuracy'])
    except (OSError, KeyError, ValueError):
        pass
    return acc


ACCURACY = _load_accuracy()
# Order and short descriptions for the models produced by `solution.py reproduce`.
MODEL_INFO = {
    V12_KEY:             'v1.2: hashed word/alias/char pieces -> 4 transformer blocks + v1.0 linear branch, '
                         'distilled from hing-roberta-mixed; 11.9M parameters, INT8 ONNX. Best available accuracy.',
    'full.npz':          'All features: words, char n-grams, aliases, emoji context + attention',
    'strip_symbols.npz': 'Like full, but emojis and punctuation are removed first',
    'word_only.npz':     'Words and word pairs only (no char n-grams / aliases) + attention',
    'char_word.npz':     'Words, word pairs and character n-grams + attention',
}

# Friendly display names for each label.
INTENT_INFO = {
    'cancel_order': 'Cancel order',
    'refund':       'Refund',
    'track_order':  'Track order',
    'not_received': 'Not received',
    'damaged_item': 'Damaged item',
    'feedback':     'Feedback',
}
SENTIMENT_INFO = {
    'negative': 'Negative',
    'neutral':  'Neutral',
    'positive': 'Positive',
}
EXAMPLES = [
    'refund kab milega bhai',
    'mera order kahan hai',
    'item delivered bol raha hai but mila hi nahi',
    'box khola toh item toota hua hai',
    'bahut acchi service thank you',
    'paise abhi tak wapas nahi aaye',
    'order cancel krdo plz',
]

st.set_page_config(page_title="Polyglot's Shorthand", layout='wide')


# ---------------------------------------------------------------------------
# Load the model once and reuse it for every request.
# ---------------------------------------------------------------------------
class StudentAdapter:
    """Gives the v1.2 ONNX student the same interface the page uses for v1.1 models."""

    def __init__(self, deploy_dir):
        sys.path.insert(0, str(ROOT / 'v12'))
        from polyglot12.runtime import StudentRuntime
        self.rt = StudentRuntime(deploy_dir, precision='int8', threads=2)
        cal = self.rt.meta['calibration']
        self.mode = 'full (hashed pieces + transformer)'
        self.att = True
        self.temperature = [cal['intent']['temperature'], cal['sentiment']['temperature']]
        self.threshold = [cal['intent']['review_threshold'], cal['sentiment']['review_threshold']]

    def parameter_count(self):
        return self.rt.meta['params']['total']

    def predict(self, text):
        return self.rt.predict(text)

    def probabilities(self, text):
        z = self.rt.logits(text)[0]
        soft = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        return [soft(z[:6] / self.temperature[0]), soft(z[6:] / self.temperature[1])]

    def attention_weights(self, text):
        _, toks, alpha = self.rt.logits(text)
        return list(zip(toks, (float(a) for a in alpha)))


def student_available():
    if not (V12_DEPLOY / 'meta.json').exists() or not (V12_DEPLOY / 'encoder.int8.onnx').exists():
        return False, 'v1.2 student not exported (run `make -C v12 deploy`).'
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, 'v1.2 student needs onnxruntime (`pip install -r requirements-app.txt`).'
    return True, ''


@st.cache_resource
def load_model(choice):
    if choice == V12_KEY:
        return StudentAdapter(V12_DEPLOY)
    return Model.load(ART / choice)


# ---------------------------------------------------------------------------
# Run the model on one message and return prediction + all probabilities.
# ---------------------------------------------------------------------------
def analyse(model, text):
    pred = model.predict(text)
    p_intent, p_sentiment = model.probabilities(text)
    n_features = len(features(text, model.mode)[0])
    return pred, p_intent, p_sentiment, n_features, model.attention_weights(text)


# ---------------------------------------------------------------------------
# One result card: label, confidence, review flag and a probability chart.
# ---------------------------------------------------------------------------
def result_card(title, result, labels, probs, info, threshold):
    name = info[result['label']]
    with st.container(border=True):
        st.caption(title)
        st.markdown(f'### {name}')
        st.progress(result['confidence'], text=f"Confidence: {result['confidence']:.1%}")
        if result['review_required']:
            st.warning(f'Needs human review: confidence is below the {threshold:.0%} threshold.')
        else:
            st.success('Confident enough to handle automatically.')
        chart = pd.DataFrame({'class': [info[l] for l in labels],
                              'probability': [float(p) for p in probs]})
        st.bar_chart(chart, x='class', y='probability', horizontal=True, height=60 + 32 * len(labels))


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("The Polyglot's Shorthand")
st.markdown('Classify **Hinglish customer-support messages** by *what the customer wants* '
            '(intent) and *how they feel* (sentiment).')

ok_student, student_note = student_available()
available = ([V12_KEY] if ok_student else []) + sorted(
    (p.name for p in ART.glob('*.npz')),
    key=lambda n: (list(MODEL_INFO).index(n) if n in MODEL_INFO else len(MODEL_INFO), n))
if not available:
    st.error('No trained model found in `artifacts/`. '
             'Run `python solution.py reproduce` (or `./run.sh`) first.')
    st.stop()

# Sidebar: model picker, model facts and honest limitations.
with st.sidebar:
    st.header('Model')
    choice = st.selectbox('Run inference with', available, index=0,
                          help='Default: the most accurate model available locally. v1.1 models are trained by '
                               '`solution.py reproduce`; the v1.2 student by the v12 notebooks.')
    if choice in MODEL_INFO:
        st.caption(MODEL_INFO[choice])
    if not ok_student:
        st.info(student_note)
    if choice in ACCURACY:
        a, b = ACCURACY[choice]
        c1, c2 = st.columns(2)
        c1.metric('Test acc. intent', f'{a:.1%}')
        c2.metric('Test acc. sentiment', f'{b:.1%}')
    try:
        model = load_model(choice)
    except (ValueError, KeyError, OSError, ImportError) as e:
        st.error(f'Could not load `{choice}`: {e}')
        st.stop()
    st.divider()
    st.header('About the model')
    st.metric('Parameters', f'{model.parameter_count():,}')
    st.metric('Feature mode', model.mode)
    st.metric('Attention layer', 'yes' if model.att is not None else 'no')
    st.write(f'Temperature — intent **{model.temperature[0]:.2f}**, sentiment **{model.temperature[1]:.2f}**')
    st.write(f'Review threshold — intent **{model.threshold[0]:.2f}**, sentiment **{model.threshold[1]:.2f}**')
    st.divider()

single_tab, batch_tab = st.tabs(['Single message', 'Batch (many messages)'])

# ----------------------------- single message -------------------------------
with single_tab:
    if 'message' not in st.session_state:
        st.session_state.message = EXAMPLES[0]

    st.write('Try an example:')
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f'ex{i}', width='stretch'):
            st.session_state.message = ex

    text = st.text_area('Customer message', key='message', height=110, max_chars=MAX_CHARS,
                        placeholder='e.g. mera refund kab aayega')

    if st.button('Analyse', type='primary'):
        try:
            pred, p_int, p_sent, n_feat, att = analyse(model, text)
        except (ValueError, TypeError) as e:   # empty or over-long input
            st.error(f'Cannot analyse this message: {e}')
        else:
            left, right = st.columns(2)
            with left:
                result_card('INTENT — what the customer wants', pred['intent'], INTENTS, p_int,
                            INTENT_INFO, model.threshold[0])
            with right:
                result_card('SENTIMENT — how the customer feels', pred['sentiment'], SENTIMENTS, p_sent,
                            SENTIMENT_INFO, model.threshold[1])
            if att:
                with st.expander('Attention: which words the model focused on', expanded=True):
                    st.bar_chart(pd.DataFrame({'token': [f'{i+1}. {t}' for i, (t, _) in enumerate(att)],
                                               'attention': [a for _, a in att]}),
                                 x='token', y='attention', height=240)
            with st.expander('Raw JSON output'):
                st.json({'model': choice, **pred})
            st.caption(f'Model `{choice}` - {n_feat} active hashed features were used for this message.')

# ------------------------------- batch mode ---------------------------------
with batch_tab:
    st.write('Paste one message per line, or upload a CSV with a `text` column.')
    pasted = st.text_area('Messages', height=180, key='batch_text',
                          placeholder='refund kab milega\nmera order kahan hai\nservice bahut acchi thi')
    uploaded = st.file_uploader('…or upload CSV', type='csv')

    if st.button('Analyse all', type='primary'):
        messages = [line for line in pasted.splitlines() if line.strip()]
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            if 'text' not in df_in.columns:
                st.error('The CSV needs a column named `text`.')
                st.stop()
            messages += [str(t) for t in df_in['text'].dropna()]

        if not messages:
            st.info('Add at least one message.')
        else:
            rows = []
            for msg in messages:
                try:
                    p = model.predict(msg)
                    rows.append({'text': msg, 'model': choice,
                                 'intent': p['intent']['label'],
                                 'intent_conf': round(p['intent']['confidence'], 3),
                                 'sentiment': p['sentiment']['label'],
                                 'sentiment_conf': round(p['sentiment']['confidence'], 3),
                                 'needs_review': p['intent']['review_required'] or p['sentiment']['review_required'],
                                 'error': ''})
                except (ValueError, TypeError) as e:   # one bad line must not stop the batch
                    rows.append({'text': msg, 'model': choice, 'intent': None, 'intent_conf': None, 'sentiment': None,
                                 'sentiment_conf': None, 'needs_review': None, 'error': str(e)})
            df = pd.DataFrame(rows)
            ok = df[df['error'] == '']

            # Summary numbers and distributions.
            m1, m2, m3 = st.columns(3)
            m1.metric('Messages', len(df))
            m2.metric('Need human review', int(ok['needs_review'].sum()) if len(ok) else 0)
            m3.metric('Errors', int((df['error'] != '').sum()))
            if len(ok):
                c1, c2 = st.columns(2)
                c1.subheader('Intents')
                c1.bar_chart(ok['intent'].value_counts())
                c2.subheader('Sentiments')
                c2.bar_chart(ok['sentiment'].value_counts())

            st.dataframe(df, width='stretch', hide_index=True)
            st.download_button('Download results as CSV', df.to_csv(index=False).encode('utf-8'),
                               'predictions.csv', 'text/csv')
