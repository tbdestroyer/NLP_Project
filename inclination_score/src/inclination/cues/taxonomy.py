"""Cue taxonomy: the single source of truth for cue names, directions and display phrases."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    name: str
    direction: int   # +1 raises inclination, -1 lowers it, 0 neutral (agent moves)
    strength: str    # "weak", "normal" or "strong"
    phrase: str      # used in template rationales, e.g. "Interest dropped after {phrase}"


CUSTOMER_CUES = [
    Cue("asks_features", +1, "normal", "a question about the product"),
    Cue("asks_process", +1, "normal", "a question about next steps"),
    Cue("asks_price_neutral", +1, "weak", "a neutral question about the price"),
    Cue("positive_sentiment", +1, "normal", "a positive reaction"),
    Cue("commitment", +1, "strong", "a commitment to go ahead"),
    Cue("agreement", +1, "normal", "an agreement"),
    Cue("price_objection", -1, "normal", "a price objection"),
    Cue("competitor_or_alternative", -1, "normal", "a mention of an alternative"),
    Cue("delay", -1, "normal", "a request for more time"),
    Cue("doubt_or_distrust", -1, "normal", "an expression of doubt"),
    Cue("disengagement", -1, "normal", "a very short, disengaged reply"),
    Cue("explicit_refusal", -1, "strong", "an explicit refusal"),
]

# Ordered from most to least specific; the first match is the one named in a rationale.
AGENT_MOVES = [
    Cue("gives_discount", 0, "normal", "offering a discount"),
    Cue("pushes_close", 0, "normal", "pushing to close"),
    Cue("states_price", 0, "normal", "stating the price"),
    Cue("creates_urgency", 0, "normal", "creating urgency"),
    Cue("mentions_cost_or_fee", 0, "normal", "mentioning a cost or fee"),
    Cue("provides_evidence", 0, "normal", "providing evidence"),
    Cue("addresses_objection", 0, "normal", "addressing an objection"),
    Cue("explains_benefit", 0, "normal", "explaining a benefit"),
    Cue("asks_question", 0, "normal", "asking a question"),
    Cue("builds_rapport", 0, "normal", "building rapport"),
]

# What each name means, as given to the LLM tagger and the LLM rubric.
DESCRIPTIONS = {
    "asks_features": "asks about features, rewards, condition, specs or benefits",
    "asks_process": "asks how to apply, buy, pay, pick up, sign up, or about next steps",
    "asks_price_neutral": "asks the price neutrally, e.g. \"how much is it?\"",
    "positive_sentiment": "reacts positively, e.g. \"great\", \"sounds good\", \"I love that\"",
    "commitment": "commits to go ahead, e.g. \"I'll take it\", \"I'm in\", \"I'd like to donate\"",
    "agreement": "agrees or accepts a point, e.g. \"ok\", \"deal\", \"that works\"",
    "price_objection": "objects to the price, e.g. \"too expensive\", \"can you go lower\"",
    "competitor_or_alternative": "mentions other offers, sellers, charities or products",
    "delay": "postpones, e.g. \"let me think about it\", \"maybe later\"",
    "doubt_or_distrust": "expresses doubt or distrust, e.g. \"is this legit?\", \"I'm not sure\"",
    "disengagement": "very short or vague reply, topic change, loss of interest",
    "explicit_refusal": "refuses, e.g. \"no thanks\", \"not interested\"",
    "gives_discount": "offers a lower price, a discount or something free",
    "pushes_close": "asks for the sale, the deal or the donation",
    "states_price": "states a price or amount",
    "creates_urgency": "creates time pressure or scarcity",
    "mentions_cost_or_fee": "mentions a cost, fee, subscription or other charge",
    "provides_evidence": "gives facts, statistics or proof",
    "addresses_objection": "responds to a concern the customer raised",
    "explains_benefit": "explains a benefit or positive quality",
    "asks_question": "asks the customer a question",
    "builds_rapport": "greets, thanks, empathises or makes small talk",
}

CUSTOMER_CUE_NAMES = [cue.name for cue in CUSTOMER_CUES]
AGENT_MOVE_NAMES = [cue.name for cue in AGENT_MOVES]
ALL_NAMES = CUSTOMER_CUE_NAMES + AGENT_MOVE_NAMES
BY_NAME = {cue.name: cue for cue in CUSTOMER_CUES + AGENT_MOVES}


def names_for_role(role: str) -> list[str]:
    return CUSTOMER_CUE_NAMES if role == "customer" else AGENT_MOVE_NAMES
