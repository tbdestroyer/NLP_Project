"""v2 scorer: Gemini reads the transcript so far and returns a 0-100 score from a fixed rubric.

One call per turn, each seeing turns 0..k only, so a score never depends on later turns. Scores
are NOT calibrated: v2 is only run on a small test sample, so there is no validation set to fit
a calibrator on.
"""
from pydantic import BaseModel, Field

from inclination import llm
from inclination.config import load_config
from inclination.cues.taxonomy import CUSTOMER_CUES, DESCRIPTIONS

NAME = "v2"
JOB = "rubric_score"
YES_MEANS = {
    "craigslist": "the buyer (CUSTOMER) and the seller (AGENT) agree on a deal",
    "p4g": "the CUSTOMER actually donates to the charity",
    "saas": "the CUSTOMER buys or signs up for the product",
}


class RubricScore(BaseModel):
    score: int = Field(ge=0, le=100)
    reason: str


def _signals(direction: int) -> str:
    return "\n".join(
        f"- {DESCRIPTIONS[cue.name]}" + (" (strong signal)" if cue.strength == "strong" else "")
        for cue in CUSTOMER_CUES if cue.direction == direction
    )


def build_prompt(dataset: str, roles: list[str], texts: list[str]) -> str:
    """Prompt for the prefix given by `roles` and `texts` (the turns seen so far)."""
    transcript = "\n".join(
        f"[{turn}] {role.upper()}: {text}" for turn, (role, text) in enumerate(zip(roles, texts))
    )
    return (
        "You estimate how likely a CUSTOMER is to say yes, from a conversation that is still in "
        f"progress. Here, \"yes\" means: {YES_MEANS[dataset]}.\n\n"
        "Signals that raise the likelihood (customer):\n"
        f"{_signals(+1)}\n\n"
        "Signals that lower the likelihood (customer):\n"
        f"{_signals(-1)}\n\n"
        "Scale: 0 = certain no, 25 = unlikely, 50 = undecided or too early to tell, "
        "75 = likely, 100 = certain yes.\n"
        "Rules:\n"
        "- Use only the turns shown. The conversation may continue after them.\n"
        "- Weigh recent customer turns more than early ones.\n"
        "- Give `score` as a whole number from 0 to 100 and `reason` as one sentence of at most "
        "25 words about what in the conversation drives the score.\n\n"
        f"Conversation so far:\n{transcript}"
    )


def score_conversation(client: llm.GeminiClient, dataset: str, roles: list[str], texts: list[str]):
    """One (score in [0, 1], reason) per turn. A turn whose answer is unusable gets the
    previous turn's score (0.5 at turn 0) and an empty reason."""
    model = load_config()["llm"]["model"]
    results, previous = [], 0.5
    for k in range(len(roles)):
        parsed, _ = client.generate(build_prompt(dataset, roles[: k + 1], texts[: k + 1]), model, RubricScore, JOB)
        score, reason = (parsed.score / 100, parsed.reason.strip()) if parsed else (previous, "")
        results.append((score, reason))
        previous = score
    return results
