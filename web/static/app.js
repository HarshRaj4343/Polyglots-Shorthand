// Frontend logic: calls /api/meta once, then /api/predict for single + batch runs.

const LABELS = {
  cancel_order: 'Cancel order', refund: 'Refund', track_order: 'Track order',
  not_received: 'Not received', damaged_item: 'Damaged item', feedback: 'Feedback',
  negative: 'Negative', neutral: 'Neutral', positive: 'Positive',
};
const EXAMPLES = [
  'refund kab milega bhai',
  'mera order kahan hai',
  'item delivered bol raha hai but mila hi nahi',
  'box khola toh item toota hua hai',
  'bahut acchi service thank you',
  'order cancel krdo plz',
];
const $ = (id) => document.getElementById(id);
const pct = (x) => (x * 100).toFixed(1) + '%';

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `request failed (${res.status})`);
  return data;
}

// ---------------------------------------------------------------- meta + chips
(async function init() {
  EXAMPLES.forEach((ex) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = ex;
    b.onclick = () => { $('msg').value = ex; updateCount(); };
    $('chips').append(b);
  });

  try {
    const m = await api('/api/meta');
    $('stats').innerHTML = [
      [pct(m.accuracy.intent), 'Intent accuracy'],
      [pct(m.accuracy.sentiment), 'Sentiment accuracy'],
      [(m.parameters / 1e6).toFixed(1) + 'M', 'Parameters'],
      ['3.4 MB', 'INT8 ONNX encoder'],
    ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join('');

    $('facts').innerHTML = [
      ['Model', m.model],
      ['Parameters', m.parameters.toLocaleString()],
      ['Intents', m.intents.length + ' classes'],
      ['Sentiments', m.sentiments.length + ' classes'],
      ['Temperature', `intent ${m.temperature.intent.toFixed(2)} · sentiment ${m.temperature.sentiment.toFixed(2)}`],
      ['Review threshold', `intent ${m.review_threshold.intent.toFixed(2)} · sentiment ${m.review_threshold.sentiment.toFixed(2)}`],
    ].map(([l, v]) => `<div class="fact"><span>${l}</span><b>${v}</b></div>`).join('');
  } catch (e) {
    $('stats').innerHTML = `<div class="stat"><span>Model unavailable: ${e.message}</span></div>`;
  }
})();

// ------------------------------------------------------------- single message
function updateCount() { $('count').textContent = $('msg').value.length; }
$('msg').addEventListener('input', updateCount);
$('msg').addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') run();
});
$('run').onclick = run;

function bars(probs, winner) {
  return Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="bar ${k === winner ? 'top' : ''}">
        <span>${LABELS[k] || k}</span>
        <span class="track"><span class="fill" style="width:${Math.max(v * 100, 1.5)}%"></span></span>
        <span class="val">${pct(v)}</span>
      </div>`).join('');
}

function resultCard(el, kicker, r) {
  el.innerHTML = `
    <p class="kicker">${kicker}</p>
    <h3>${LABELS[r.label] || r.label}</h3>
    <p class="conf">Confidence ${pct(r.confidence)}</p>
    <div class="bars">${bars(r.probabilities, r.label)}</div>
    <span class="flag ${r.review_required ? 'review' : 'ok'}">
      ${r.review_required ? 'Needs human review' : 'Confident enough to handle automatically'}
    </span>`;
}

async function run() {
  const text = $('msg').value.trim();
  $('error').hidden = true;
  if (!text) { showError('Type a message first.'); return; }

  $('run').disabled = true; $('run').textContent = 'Analysing…';
  try {
    const { results } = await api('/api/predict', { texts: [text] });
    const r = results[0];
    resultCard($('card-intent'), 'Intent — what the customer wants', r.intent);
    resultCard($('card-sentiment'), 'Sentiment — how the customer feels', r.sentiment);

    const max = Math.max(...r.attention.map((a) => a.weight), 1e-6);
    $('card-attention').innerHTML = `<p class="kicker">Attention — where the model looked</p>
      <div class="att">${r.attention.map((a) => `
        <span class="tok" style="background:rgba(244,81,30,${(a.weight / max * 0.55).toFixed(3)})"
              title="${a.weight.toFixed(4)}">${escapeHtml(a.token)}</span>`).join('')}</div>`;
    $('results').hidden = false;
  } catch (e) {
    showError(e.message);
  } finally {
    $('run').disabled = false; $('run').textContent = 'Analyse';
  }
}

function showError(msg) { $('error').textContent = msg; $('error').hidden = false; }
function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// -------------------------------------------------------------------- batch
let lastRows = [];
$('run-batch').onclick = async () => {
  const texts = $('batch-in').value.split('\n').map((s) => s.trim()).filter(Boolean);
  if (!texts.length) { return; }

  $('run-batch').disabled = true; $('run-batch').textContent = 'Analysing…';
  try {
    const { results } = await api('/api/predict', { texts: texts.slice(0, 200) });
    lastRows = results.map((r) => r.error
      ? { text: r.text, intent: '—', intent_conf: '', sentiment: '—', sentiment_conf: '', review: r.error }
      : {
          text: r.text,
          intent: LABELS[r.intent.label], intent_conf: pct(r.intent.confidence),
          sentiment: LABELS[r.sentiment.label], sentiment_conf: pct(r.sentiment.confidence),
          review: (r.intent.review_required || r.sentiment.review_required) ? 'yes' : 'no',
        });

    const needs = lastRows.filter((r) => r.review === 'yes').length;
    $('batch-out').innerHTML = `
      <p class="lede" style="margin:22px 0 0">${lastRows.length} messages · ${needs} flagged for review</p>
      <div class="table-wrap"><table>
        <thead><tr><th>Message</th><th>Intent</th><th>Conf.</th><th>Sentiment</th><th>Conf.</th><th>Review</th></tr></thead>
        <tbody>${lastRows.map((r) => `<tr>
          <td>${escapeHtml(r.text)}</td><td>${r.intent}</td><td class="num">${r.intent_conf}</td>
          <td>${r.sentiment}</td><td class="num">${r.sentiment_conf}</td><td class="num">${r.review}</td>
        </tr>`).join('')}</tbody>
      </table></div>`;
    $('batch-out').hidden = false;
    $('download').hidden = false;
  } catch (e) {
    $('batch-out').innerHTML = `<div class="error">${e.message}</div>`;
    $('batch-out').hidden = false;
  } finally {
    $('run-batch').disabled = false; $('run-batch').textContent = 'Analyse all';
  }
};

$('download').onclick = () => {
  const head = ['text', 'intent', 'intent_confidence', 'sentiment', 'sentiment_confidence', 'needs_review'];
  const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const csv = [head.join(','), ...lastRows.map((r) =>
    [r.text, r.intent, r.intent_conf, r.sentiment, r.sentiment_conf, r.review].map(esc).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: 'predictions.csv' });
  a.click(); URL.revokeObjectURL(url);
};
