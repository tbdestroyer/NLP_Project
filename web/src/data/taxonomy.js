// Mirrors inclination_score/src/inclination/cues/taxonomy.py. Keep the two in sync.

export const CUSTOMER_CUES = {
  asks_features: { direction: 1, phrase: 'a question about the product' },
  asks_process: { direction: 1, phrase: 'a question about next steps' },
  asks_price_neutral: { direction: 1, phrase: 'a neutral question about the price' },
  positive_sentiment: { direction: 1, phrase: 'a positive reaction' },
  commitment: { direction: 1, phrase: 'a commitment to go ahead' },
  agreement: { direction: 1, phrase: 'an agreement' },
  price_objection: { direction: -1, phrase: 'a price objection' },
  competitor_or_alternative: { direction: -1, phrase: 'a mention of an alternative' },
  delay: { direction: -1, phrase: 'a request for more time' },
  doubt_or_distrust: { direction: -1, phrase: 'an expression of doubt' },
  disengagement: { direction: -1, phrase: 'a very short, disengaged reply' },
  explicit_refusal: { direction: -1, phrase: 'an explicit refusal' },
};

export const AGENT_MOVES = {
  gives_discount: { phrase: 'offering a discount' },
  pushes_close: { phrase: 'pushing to close' },
  states_price: { phrase: 'stating the price' },
  creates_urgency: { phrase: 'creating urgency' },
  mentions_cost_or_fee: { phrase: 'mentioning a cost or fee' },
  provides_evidence: { phrase: 'providing evidence' },
  addresses_objection: { phrase: 'addressing an objection' },
  explains_benefit: { phrase: 'explaining a benefit' },
  asks_question: { phrase: 'asking a question' },
  builds_rapport: { phrase: 'building rapport' },
};

export const DATASETS = {
  craigslist: { label: 'CraigslistBargains', customer: 'Buyer', agent: 'Seller', synthetic: false,
    about: 'Real people bargaining over a Craigslist listing. Bought = the deal was accepted.' },
  p4g: { label: 'Persuasion for Good', customer: 'Persuadee', agent: 'Persuader', synthetic: false,
    about: 'Real people; one persuades the other to donate. Bought = they actually donated.' },
  saas: { label: 'SaaS', displayLabel: 'SaaS (synthetic)', customer: 'Customer', agent: 'Sales rep', synthetic: true,
    about: 'Synthetic sales chats generated with GPT-4. Used for development, not as evidence.' },
};

export const MODELS = {
  v1: { label: 'v1 · cue model', short: 'v1' },
  v3: { label: 'v3 · transformer', short: 'v3' },
  v2: { label: 'v2 · LLM rubric', short: 'v2', pending: true },
};

export const STATUSES = ['Hot', 'Rising', 'Stalled', 'Cooling'];

export function pretty(name) {
  const text = name.replace(/_/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function isCustomerCue(name) {
  return name in CUSTOMER_CUES;
}
