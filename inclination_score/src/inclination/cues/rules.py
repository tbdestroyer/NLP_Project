"""Rule-based cue tagger: regex and keyword lists plus simple features. Always available.

A turn's tags depend only on that turn's role and text, so they can never see later turns.
"""
import re

import numpy as np

from inclination.config import load_config
from inclination.cues.taxonomy import AGENT_MOVE_NAMES, ALL_NAMES, CUSTOMER_CUE_NAMES

_QUESTION = r"\?"

CUSTOMER_PATTERNS = {
    "asks_features": (
        r"(?=.*\?).*\b(features?|condition|specs?|warranty|size|colou?r|model|year|mileage|"
        r"how old|what kind|what type|does it|do they|is it|are they|is there|are there|"
        r"how does|what does|what do|what is|what are|tell me more|more about|details|"
        r"integrat\w*|include[sd]?|come with|benefits?|work with|who (are|is)|where does)\b"
    ),
    "asks_process": (
        r"\b(how (do|can|would|could|should) (i|we)|where (do|can|would) (i|we)|"
        r"how to (donate|buy|apply|pay|sign|order|start)|next steps?|get started|sign up|"
        r"pick ?(it )?up|deliver\w*|shipping|payment|pay (by|with|in)|cash|venmo|paypal|"
        r"when can (i|we|you)|available (today|tomorrow|this|now)|can (i|we) (come|see|meet|try)|"
        r"free trial|trial period|demo|onboarding|implementation|contract|set ?up)\b"
    ),
    "asks_price_neutral": (
        r"\b(how much (is|are|does|do|would|for)|what('?s| is| are) the (price|cost|pricing|rate)|"
        r"what does it cost|what would it cost|pricing|price range|asking price)\b|\bprice\s*\?"
    ),
    "positive_sentiment": (
        r"\b(great|awesome|perfect|love|sounds good|sounds great|sounds nice|wonderful|excellent|"
        r"nice|amazing|fantastic|cool|sweet|interested|interesting|good to know|helpful|"
        r"impressive|appreciate|glad|happy|like that|i like)\b"
    ),
    "commitment": (
        r"\b(i('?ll| will) (take|buy|donate|do|give|go with|sign|get|pay)|let'?s do (it|this|that)|"
        r"i'?m in|i('?d| would) (like|love) to (donate|buy|purchase|sign|move forward|proceed|give)|"
        r"sign me up|count me in|i (can|could) donate|willing to donate|we have a deal|"
        r"you('?ve| have)? got a deal|it'?s a deal|sold|i('?ll| will) (come )?pick it up|"
        r"let'?s (move forward|get started|proceed|go ahead)|ready to (buy|sign|start|move|donate)|"
        r"i'?ll go ahead|send (me|over) the (contract|paperwork|invoice))\b"
    ),
    "agreement": (
        r"^\W*(ok|okay|sure|yes|yeah|yep|yup|deal|alright|all right|fine|agreed)\b|"
        r"\b(that works|works for me|sounds (fair|reasonable)|fair enough|i (can|could) do that|"
        r"i agree|deal|you'?re right|makes sense|good point)\b"
    ),
    "price_objection": (
        r"\b(too (expensive|much|high|pricey|steep)|(go|do|come|get) (any |a (bit|little) )?"
        r"(lower|down|less|cheaper)|that'?s a lot|lower (the|your) price|"
        r"out of (my|our) (budget|price range)|over (my|our) budget|can'?t afford|cannot afford|"
        r"budgets? (is|are) (really |pretty |very )?tight|tight budget|pricey|steep|cheaper|"
        r"best price|lowest|a bit (much|high)|(cost|price)s? (is|are|seems?) (too |pretty |really )?high|"
        r"only (have|afford|pay)|kinda (steep|pricey|expensive)|expensive)\b"
    ),
    "competitor_or_alternative": (
        r"\b(other (offers?|sellers?|listings?|options?|charit\w+|tools?|vendors?|products?|"
        r"solutions?|organi[sz]ations?|places?|providers?)|elsewhere|competitors?|"
        r"another (seller|one|charity|tool|vendor|option|place)|alternatives?|"
        r"similar (ones?|products?|tools?|items?)|already (use|using|have|donate|give)|"
        r"currently us(e|ing)|compar(e|ed|ing) (it )?(to|with)|seen (it|one|them|these) for|"
        r"shop(ping)? around)\b"
    ),
    "delay": (
        r"\b(let me think|think (about|on) it|think it over|maybe later|get back to you|"
        r"not (right )?now|need (some |more |a little )*time|talk (to|with) my|check with|"
        r"discuss (it |this )?with|sleep on it|some other time|in the future|another time|"
        r"run it by|convince (the|my)|not ready|hold off|down the road|next (quarter|year|month))\b"
    ),
    "doubt_or_distrust": (
        r"\b(is (this|it|that) (legit|real|a scam|genuine|safe|true)|not (so |too |really )?sure|"
        r"unsure|how (do|can|would) i know|scams?|skeptical|sceptical|doubt\w*|trust|"
        r"worr(y|ied)|concern(s|ed)?|hesitant|hard to believe|guarantees?|what if|"
        r"too good to be true|risky?|suspicious|i don'?t know|idk|not convinced)\b"
    ),
    "explicit_refusal": (
        r"\b(no,? thanks?|no thank you|not interested|i (do not|don'?t) want|"
        r"i (can'?t|cannot|won'?t|will not) (donate|do|buy|pay|go|give|accept)|"
        r"i('?ll| will) pass|i'?m (going to|gonna) pass|no deal|not for me|"
        r"i('?d| would) rather not|forget it|no way|not (going to|gonna) (work|happen|donate|buy)|"
        r"i decline|i refuse|not today|i (do not|don'?t) think so|i'?m out|never ?mind)\b"
    ),
}

AGENT_PATTERNS = {
    "gives_discount": (
        r"\b(discount\w*|\d+ ?% off|knock \w* ?off|take \$?\d+ off|lower (it|the price)|"
        r"special (offer|price|deal|rate)|free (trial|shipping|delivery|month|onboarding)|"
        r"throw in|waive\w*|promo\w*|come down|meet (you )?(in the middle|halfway)|"
        r"i (can|could|will|would|'?ll|'?d) (do|go|take|accept|let it go for|sell it for) "
        r"(it for |as low as |down to )?\$?\d)"
    ),
    "pushes_close": (
        r"\b(would you (like|be willing|want) to (donate|buy|purchase|sign|proceed|move forward|"
        r"get started|make a donation)|ready to (buy|sign|get started|move forward|donate)|"
        r"shall we|can i (count on|sign you|put you down)|do we have a deal|is (that|it) a deal|"
        r"how much (would|will|do|can) you (like to |want to |be willing to )?(donate|give)|"
        r"will you (donate|take it|buy)|are you (willing|interested) (to|in) (donat|buy|purchas)\w*|"
        r"want to (go ahead|move forward|get started|donate)|"
        r"let'?s (get you started|set (it |that )?up|schedule)|when can you pick|shake on)\b|"
        r"\bdeal\s*\?"
    ),
    "states_price": (
        r"\$\s?\d|\b\d[\d,.]* ?(dollars|bucks|usd|/ ?month|per month|a month|per user)\b|"
        r"\b(price is|asking|priced at|it'?s listed)\b"
    ),
    "creates_urgency": (
        r"\b(today only|right now|limited|only \d+ left|won'?t last|"
        r"other (buyers|people|offers)|someone else|ends? (soon|today|this week|friday)|hurry|"
        r"before (it'?s|its) gone|as soon as possible|asap|deadline|running out|"
        r"every (day|minute|second)|urgent\w*|by (the )?end of (the )?(day|week|month)|expires?)\b"
    ),
    "mentions_cost_or_fee": (
        r"\b(fees?|shipping (cost|charge)s?|delivery (cost|charge|fee)s?|tax(es)?|deposit|"
        r"subscription|per (month|user|seat|year)|monthly|annual(ly)?|up ?front|set ?up cost|"
        r"(additional|extra) (cost|charge)s?|investment|costs?)\b"
    ),
    "provides_evidence": (
        r"\d+ ?%|\b(percent|studies|study|research|statistics?|according to|data shows?|proven|"
        r"customers? (have|saw|see|report)|case stud\w+|reviews?|testimonials?|rated|"
        r"millions?|thousands?|billions?|certified|awards?|compliant|website|url)\b|\.(org|com)\b"
    ),
    "addresses_objection": (
        r"\b(i (totally |completely |really )?(understand|hear you|get (that|it))|"
        r"totally (get|understand)|understandable|(valid|fair|good) (concern|point|question)|"
        r"don'?t worry|no worries|rest assured|that said|however|"
        r"i know it (seems|sounds|feels)|but (think|consider|keep in mind))\b"
    ),
    "explains_benefit": (
        r"\b(sav(e|es|ing)|helps? (you|your|children|kids|them)|benefits?|improv\w+|reduc\w+|"
        r"increas\w+|streamlin\w+|(great|excellent|good|perfect|mint) condition|like new|"
        r"brand new|barely used|works? (great|perfectly|fine|well)|easy to|features?|"
        r"includes?|comes with|quality|reliable|efficien\w+|boost\w*|"
        r"make a (big |huge |real )?difference|impact)\b"
    ),
    "asks_question": _QUESTION,
    "builds_rapport": (
        r"^\W*(hi|hello|hey|good (morning|afternoon|evening))\b|"
        r"\b(how are you|hope you|nice to (meet|talk|chat)|thank(s| you)|great question|glad|"
        r"no problem|you'?re welcome|have a (great|good|nice|wonderful)|appreciate)\b"
    ),
}

_COMPILED = {
    "customer": [(name, re.compile(CUSTOMER_PATTERNS[name], re.I)) for name in CUSTOMER_CUE_NAMES
                 if name in CUSTOMER_PATTERNS],
    "agent": [(name, re.compile(AGENT_PATTERNS[name], re.I)) for name in AGENT_MOVE_NAMES],
}
_COLUMN = {name: i for i, name in enumerate(ALL_NAMES)}


def tag_turn(role: str, text: str) -> list[str]:
    """Cue names for one turn. Customer turns get customer cues, agent turns get agent moves."""
    tags = [name for name, pattern in _COMPILED[role] if pattern.search(text)]
    if role == "customer" and not tags and "?" not in text:
        if len(text.split()) <= load_config()["cues"]["disengagement_max_words"]:
            tags.append("disengagement")
    return tags


def tag_turns(roles, texts) -> list[list[str]]:
    return [tag_turn(role, text) for role, text in zip(roles, texts)]


def tags_to_matrix(tags: list[list[str]]) -> np.ndarray:
    """One row per turn, one 0/1 column per name in taxonomy.ALL_NAMES."""
    matrix = np.zeros((len(tags), len(ALL_NAMES)), dtype=np.float32)
    for row, names in enumerate(tags):
        for name in names:
            matrix[row, _COLUMN[name]] = 1.0
    return matrix
