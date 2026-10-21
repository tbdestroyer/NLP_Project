// Client-side stand-ins for the Python pipeline: a keyword cue tagger, a toy score walk,
// turning points, lead status and template rationales. Thresholds come from config.yaml.
// Used only for the "Analyze new chat" preview until the data server can score new chats.
import { AGENT_MOVES, CUSTOMER_CUES } from './taxonomy';

export const CONFIG = {
  deltaThreshold: 0.15,
  hotFinalScore: 0.7,
  risingTrend: 0.05,
  coolingTrend: -0.05,
  trendWindow: 3,
};

const RULES = {
  // customer cues
  commitment: /\b(i'?ll take it|i'?m in|deal\b|let'?s do it|sign (me )?up|i'?d like to donate|i will donate|count me in)/i,
  agreement: /\b(ok(ay)?|sure|that works|fair enough|sounds fair|agreed|yes)\b/i,
  positive_sentiment: /\b(great|love|nice|awesome|perfect|sounds good|helpful|impressive|wonderful)\b/i,
  asks_process: /\b(how (do|can) i (pay|sign|apply|pick|get)|next steps?|pick (it )?up|when can|onboarding|trial)\b/i,
  asks_features: /\b(does it|is it|what about|how (well|long|many)|condition|features?|integrat|work with|support)\b/i,
  asks_price_neutral: /\b(how much|what'?s the price|price\?|cost\?)/i,
  price_objection: /\b(too (much|expensive|high)|lower|cheaper|out of (my|our) budget|can you do|best price|discount\?)/i,
  competitor_or_alternative: /\b(another (seller|listing|charity|vendor)|competitor|other offers?|elsewhere|already use)\b/i,
  delay: /\b(think about it|maybe later|get back to you|not right now|next (month|quarter)|need time)\b/i,
  doubt_or_distrust: /\b(not sure|legit|scam|really\?|skeptic|doubt|trust)\b/i,
  explicit_refusal: /\b(no thanks|not interested|pass\b|i'?ll pass|won'?t|can'?t afford)\b/i,
  // agent moves
  gives_discount: /\b(discount|knock off|lower it to|throw in|for free|\d+% off)\b/i,
  pushes_close: /\b(shall we|ready to|do we have a deal|can i count on|sign today|would you like to (donate|proceed))\b/i,
  states_price: /\$\s?\d+|\b\d+ dollars\b/i,
  creates_urgency: /\b(today only|ends (today|friday|soon)|limited|other buyers|before it'?s gone|this week)\b/i,
  mentions_cost_or_fee: /\b(fee|subscription|per (seat|month|user)|shipping cost|setup cost)\b/i,
  provides_evidence: /\b(\d+%|studies|research|customers like|case study|receipts?|proven)\b/i,
  addresses_objection: /\b(i understand|i hear you|that'?s fair|good question|to be clear)\b/i,
  explains_benefit: /\b(saves?|helps?|benefit|you('ll| will) get|includes|barely used|like new|makes? a difference)\b/i,
  asks_question: /\?\s*$/,
  builds_rapport: /\b(hi|hello|hey|thanks|thank you|appreciate|how are you)\b/i,
};

export function tagTurn(role, text) {
  const pool = role === 'customer' ? CUSTOMER_CUES : AGENT_MOVES;
  const cues = Object.keys(pool).filter((name) => RULES[name]?.test(text));
  if (role === 'customer' && cues.length === 0 && text.trim().split(/\s+/).length <= 2) {
    cues.push('disengagement');
  }
  return cues;
}

const WEIGHTS = {
  commitment: 1.6, agreement: 0.5, positive_sentiment: 0.6, asks_process: 0.7, asks_features: 0.35,
  asks_price_neutral: 0.15, price_objection: -0.55, competitor_or_alternative: -0.6, delay: -0.8,
  doubt_or_distrust: -0.6, disengagement: -0.45, explicit_refusal: -1.8,
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Toy per-turn score: logistic of accumulated customer cue weights (decayed). */
export function scoreTurns(turns, { prior = 0.1, noise = () => 0 } = {}) {
  let logit = prior;
  return turns.map((turn) => {
    logit *= 0.92;
    if (turn.role === 'customer') {
      for (const cue of turn.cues) logit += WEIGHTS[cue] ?? 0;
    }
    logit += noise();
    return Math.min(0.99, Math.max(0.01, sigmoid(logit)));
  });
}

function previousAgentTurn(turns, index) {
  for (let i = index - 1; i >= 0; i -= 1) if (turns[i].role === 'agent') return i;
  return -1;
}

/** Mirrors explain.template_rationale: names the customer cue and the agent move before it. */
export function templateRationale(direction, turn, previous) {
  const moved = direction === 'rise' ? 'Interest rose' : 'Interest dropped';
  const cue = turn.role === 'customer'
    ? turn.cues.find((c) => (CUSTOMER_CUES[c]?.direction ?? 0) === (direction === 'rise' ? 1 : -1))
    : null;
  const move = previous?.cues?.[0];
  const trigger = move ? AGENT_MOVES[move].phrase : cue ? CUSTOMER_CUES[cue].phrase : 'this message';
  let sentence;
  if (cue && move) sentence = `${moved} with ${CUSTOMER_CUES[cue].phrase}, right after ${AGENT_MOVES[move].phrase}.`;
  else if (cue) sentence = `${moved} with ${CUSTOMER_CUES[cue].phrase}.`;
  else if (move) sentence = `${moved} after ${AGENT_MOVES[move].phrase}.`;
  else sentence = `${moved} here, with no clear cue in the message.`;
  return { sentence, trigger };
}

export function turningPoints(turns, scores) {
  const points = [];
  for (let i = 1; i < scores.length; i += 1) {
    const delta = scores[i] - scores[i - 1];
    if (Math.abs(delta) < CONFIG.deltaThreshold) continue;
    const direction = delta > 0 ? 'rise' : 'drop';
    const prev = previousAgentTurn(turns, i);
    const { sentence, trigger } = templateRationale(direction, turns[i], prev >= 0 ? turns[prev] : null);
    const faithfulness = Math.round(delta * (0.55 + ((i * 37) % 40) / 100) * 1000) / 1000;
    points.push({
      turn: turns[i].turn, method: 'delta', direction, delta, role: turns[i].role,
      cues: turns[i].cues, rationale: sentence, trigger,
      prev_agent_turn: prev >= 0 ? turns[prev].turn : -1,
      faithfulness, weak: Math.abs(faithfulness) < 0.05, rationale_llm: null,
    });
  }
  return points;
}

export function trendSlope(scores) {
  const tail = scores.slice(-CONFIG.trendWindow);
  if (tail.length < 2) return 0;
  const n = tail.length;
  const meanX = (n - 1) / 2;
  const meanY = tail.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  tail.forEach((y, x) => { num += (x - meanX) * (y - meanY); den += (x - meanX) ** 2; });
  return num / den;
}

export function leadStatus(final, trend) {
  if (final >= CONFIG.hotFinalScore) return 'Hot';
  if (trend > CONFIG.risingTrend) return 'Rising';
  if (trend < CONFIG.coolingTrend) return 'Cooling';
  return 'Stalled';
}

export function statusReason(status, lastPoint) {
  if (!lastPoint) return 'No sharp changes';
  const agrees = (status === 'Cooling' && lastPoint.direction === 'drop')
    || (['Hot', 'Rising'].includes(status) && lastPoint.direction === 'rise');
  return agrees
    ? `${status} after ${lastPoint.trigger}`
    : `Last sharp ${lastPoint.direction} after ${lastPoint.trigger}`;
}

export function summarise(turns, scores, points) {
  const trend = trendSlope(scores);
  const finalScore = scores[scores.length - 1];
  const status = leadStatus(finalScore, trend);
  return {
    final_score: finalScore,
    max_score: Math.max(...scores),
    trend,
    status,
    status_reason: statusReason(status, points[points.length - 1]),
    n_turning_points: points.length,
  };
}
