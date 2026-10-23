// The one place the UI gets data from.
//
// Overview, Leads, Conversation and Script insights read the REAL pipeline outputs through the
// data server in inclination_score/api/server.py (start it with `make api`). In development,
// Vite forwards every /api request to http://localhost:8000 (see vite.config.js).
//
// "Analyze new chat" scores with a trained model through POST /api/analyze. Only when the server
// has no trained model (or isn't running) does it fall back to stand-in keyword rules in the browser.
import { scoreTurns, summarise, tagTurn, turningPoints } from '../data/scoring';

export const API_BASE = import.meta.env.VITE_API_URL || '/api';

async function get(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`${API_BASE}${path}${query ? `?${query}` : ''}`);
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    throw new Error(detail || `${path} returned ${response.status}`);
  }
  return response.json();
}

/** GET /health → { ok, outputs, missing[] }. Throws if the server isn't running. */
export const getHealth = () => get('/health');

/** GET /options → { [dataset]: [models...] } for what was actually exported. */
export const getOptions = () => get('/options');

/** GET /ui-config → dashboard-only thresholds and choices mirrored from config.yaml. */
export const getUIConfig = () => get('/ui-config');

/** GET /leads?dataset=&model=&split= → leads.parquet rows plus `curve` (scores in turn order). */
export const getLeads = ({ dataset, model, split }) => get('/leads', { dataset, model, split });

/** GET /search?dataset=&q= → ids of conversations whose messages contain q. */
export const searchMessages = async (dataset, q) => new Set(await get('/search', { dataset, q }));

/** GET /conversations/:id → selected model, optional overlays, cues and turning points. */
export const getConversation = (id, model, { method = 'delta', tagger = 'rules', models = [] } = {}) => (
  get(`/conversations/${encodeURIComponent(id)}`, { model, method, tagger, models: models.join(',') })
);

/** GET /insights?dataset=&model= → insights.parquet rows. */
export const getInsights = ({ dataset, model }) => get('/insights', { dataset, model });

/** GET /insight-examples → paired transcript excerpts from insight_examples.parquet. */
export const getInsightExamples = ({ dataset, model, move, position }) => (
  get('/insight-examples', { dataset, model, move, position })
);

/** GET /metrics?dataset=&model= → { roc_auc, roc_auc_ci } from metrics.json, or null. */
export const getMetrics = ({ dataset, model }) => get('/metrics', { dataset, model });

/** GET /annotations?annotator=&dataset= → fixed annotation sample and latest saved labels. */
export const getAnnotations = (annotator, dataset) => get('/annotations', { annotator, dataset });

/** POST /annotations → append one conversation to annotation/labels/{annotator}.jsonl. */
export async function saveAnnotations(submission) {
  const response = await fetch(`${API_BASE}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(submission),
  });
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    throw new Error(detail || `Saving annotations returned ${response.status}`);
  }
  return response.json();
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** GET /analyze/models → { [dataset]: [models...] } that are trained and can score a new chat. */
export const getAnalyzeModels = () => get('/analyze/models');

/**
 * Analyze a new conversation: POST /analyze { title, dataset, model, turns: [{ role, speaker, text }] },
 * returning the same shape as GET /conversations/{id}. With preview: true, scores in the browser
 * with a keyword tagger and a toy score instead (no trained model available).
 */
export async function analyzeConversation({ title, dataset, model, turns, preview = false }) {
  if (preview) {
    await wait(900);
    const tagged = turns.map((t, i) => ({ turn: i, role: t.role, speaker: t.speaker, text: t.text, cues: tagTurn(t.role, t.text) }));
    const scores = scoreTurns(tagged, { prior: 0.05 });
    const points = turningPoints(tagged, scores);
    return {
      conversation_id: 'uploaded', title: title || 'Uploaded conversation', dataset: 'uploaded',
      model: 'preview', outcome: null, n_turns: tagged.length, turns: tagged, scores, turning_points: points,
      ...summarise(tagged, scores, points),
    };
  }
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, dataset, model, turns }),
  });
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join('; ');
    throw new Error(detail || `Analysis failed (${response.status}). Check the data server's terminal.`);
  }
  return response.json();
}
