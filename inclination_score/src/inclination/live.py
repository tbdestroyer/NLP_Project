"""Live mode: transcribe a sales call as it happens, re-score it after every message and coach the
salesperson on what to say next. Used by the /api/live/* endpoints.

Three parts:
  * `Transcriber`: local Whisper (transformers + torch) turns one spoken utterance into text.
  * `build_playbook`: from held-out scored conversations, how the customer's score moved after
    each agent move *in reply to each kind of customer cue* (a price objection, a question about
    next steps, ...). Same effect as script insights (score of the next customer turn minus the
    score after the agent turn), but conditioned on what the customer had just said.
  * `coach`: reads the live conversation's latest customer cues and recommends the moves with the
    best record in that situation, with a line to say, real lines from the data and moves to avoid.

The effects are associations in past conversations, not causes, and the page says so.
"""
import re
import threading
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from inclination import analyze as analysis
from inclination.cues import rules
from inclination.cues.taxonomy import AGENT_MOVE_NAMES, BY_NAME, CUSTOMER_CUE_NAMES
from inclination.load.standardise import clean_text

SAMPLE_RATE = 16_000
OPENING = "opening"   # the customer has not spoken yet
NO_CUE = "no_cue"     # the customer's last message had no cue
ANY = "any"           # every agent reply, whatever the customer said

# The cue that decides the situation when a message has several: objections first, because they
# need an answer before anything else.
SITUATION_PRIORITY = [
    "explicit_refusal", "price_objection", "doubt_or_distrust", "competitor_or_alternative",
    "delay", "disengagement", "commitment", "asks_process", "asks_price_neutral", "asks_features",
    "positive_sentiment", "agreement",
]
SITUATION_LABELS = {
    OPENING: "Opening the call",
    NO_CUE: "No clear signal yet",
    ANY: "Any point in the call",
    "explicit_refusal": "The customer said no",
    "price_objection": "Price objection",
    "doubt_or_distrust": "The customer has doubts",
    "competitor_or_alternative": "The customer mentioned an alternative",
    "delay": "The customer wants more time",
    "disengagement": "The customer is disengaging",
    "commitment": "The customer is ready to commit",
    "asks_process": "The customer asked about next steps",
    "asks_price_neutral": "The customer asked about price",
    "asks_features": "The customer asked about the product",
    "positive_sentiment": "The customer reacted positively",
    "agreement": "The customer agreed",
}
MOVE_ACTIONS = {
    "gives_discount": "Offer a concession",
    "pushes_close": "Ask for the next step",
    "states_price": "Give a clear price",
    "creates_urgency": "Give a reason to act now",
    "mentions_cost_or_fee": "Be upfront about costs",
    "provides_evidence": "Back it up with proof",
    "addresses_objection": "Answer the concern directly",
    "explains_benefit": "Explain what's in it for them",
    "asks_question": "Ask a question",
    "builds_rapport": "Build rapport",
}
# What to say, in brackets where the salesperson fills in specifics. A (situation, move) line wins
# over the move's default line.
SAY = {
    "gives_discount": "If we can make the numbers work, would you be ready to go ahead? I can offer [concession] on [plan].",
    "pushes_close": "Shall we get [the next step] set up today?",
    "states_price": "For [what they need], it's [price], and that includes [what's included].",
    "creates_urgency": "[Offer] is available until [date], so now is a good time to lock it in.",
    "mentions_cost_or_fee": "To be upfront: there's [cost], and it covers [what it pays for].",
    "provides_evidence": "[A customer like them] saw [specific result] within [timeframe].",
    "addresses_objection": "That's a fair concern. Here's how we handle it: [answer].",
    "explains_benefit": "For your team, that means [concrete benefit tied to their goal].",
    "asks_question": "What matters most to you in [area]?",
    "builds_rapport": "Thanks for walking me through that, it really helps me understand your situation.",
}
SITUATION_SAY = {
    (OPENING, "asks_question"): "Thanks for making the time. What made you take this call, and what would a great outcome look like?",
    (OPENING, "addresses_objection"): "Before we dive in: is there anything that has held you back from solving [problem] so far?",
    (OPENING, "explains_benefit"): "In short, we help [teams like theirs] [main outcome]. Is that what you're looking for?",
    (OPENING, "states_price"): "It's [price], and [why it's good value].",
    (OPENING, "pushes_close"): "Are you interested in [the product]? I can tell you everything you need to decide today.",
    ("price_objection", "addresses_objection"): "I understand the budget concern. Most teams look at it against [cost it saves or replaces]. Can I show you that comparison?",
    ("price_objection", "provides_evidence"): "Teams your size usually recover the cost in [timeframe] through [saving]. Here's an example: [customer result].",
    ("price_objection", "explains_benefit"): "What you'd get for that is [benefit], which replaces [current cost or effort].",
    ("price_objection", "gives_discount"): "If budget is the blocker, I can do [concession]. Would that work for you?",
    ("price_objection", "asks_question"): "What budget did you have in mind, so I can find the right option for you?",
    ("doubt_or_distrust", "provides_evidence"): "Fair question. [Similar customer] had the same concern; here's what happened after [timeframe]: [result].",
    ("doubt_or_distrust", "addresses_objection"): "I hear you. Let's take that worry head on: [specific answer]. What would you need to see to be comfortable?",
    ("competitor_or_alternative", "explains_benefit"): "[Alternative] is a solid option. Where we're different is [differentiator that matters to them].",
    ("competitor_or_alternative", "asks_question"): "What do you like about [alternative] so far, and what's missing?",
    ("competitor_or_alternative", "provides_evidence"): "Teams that compared us with [alternative] picked us for [reason]; [customer] is a good example.",
    ("delay", "asks_question"): "Of course. What would you need to see to make a decision?",
    ("delay", "pushes_close"): "Totally fair. Shall we book a 15-minute follow-up on [date] so you have what you need?",
    ("delay", "creates_urgency"): "Makes sense. Just so you know, [offer] ends on [date], so it's worth deciding before then.",
    ("disengagement", "asks_question"): "I've been talking a lot. What's the biggest challenge you're trying to solve right now?",
    ("disengagement", "builds_rapport"): "I appreciate your time. Is this still a good moment, or would another time work better?",
    ("explicit_refusal", "asks_question"): "Understood. Can I ask what made it a no, so I don't take more of your time on the wrong thing?",
    ("explicit_refusal", "addresses_objection"): "Thanks for being straight with me. If [main concern] were solved, would it be worth another look?",
    ("commitment", "pushes_close"): "Great! I'll send [the agreement] now. Who else should be on it?",
    ("asks_process", "pushes_close"): "The next step is [step]. Shall I set that up for you today?",
    ("asks_process", "explains_benefit"): "Getting started takes [time], and you'll have [first benefit] from day one.",
    ("asks_price_neutral", "states_price"): "For [what they need], it's [price], and that includes [what's included].",
    ("asks_features", "explains_benefit"): "Yes, and the reason teams like it is [benefit tied to their need].",
    ("asks_features", "provides_evidence"): "Yes. [Customer] uses it to [result].",
    ("positive_sentiment", "pushes_close"): "Glad that fits! Shall we get [the next step] going?",
    ("agreement", "pushes_close"): "Great. Shall I send over [the next step] today?",
}

_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪+")
# Whisper's well-known inventions on silence or noise (from YouTube captions in its training data).
_HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "please subscribe", "subscribe to my channel",
    "like and subscribe", "see you in the next video",
}


# ---- speech to text -----------------------------------------------------------------------------

def clean_transcript(text: str) -> str:
    """Drops non-speech tags ([BLANK_AUDIO], (music), ...) and Whisper's silence hallucinations."""
    text = clean_text(_NON_SPEECH.sub(" ", text or ""))
    words = re.sub(r"[^a-z' ]+", "", text.lower()).strip()
    if not words or any(phrase in words for phrase in _HALLUCINATIONS):
        return ""
    return text


def decode_pcm(body: bytes) -> np.ndarray:
    """Little-endian float32 mono samples, as the Live page sends them."""
    if len(body) % 4:
        raise ValueError("Audio must be float32 samples (a multiple of 4 bytes).")
    audio = np.frombuffer(body, dtype="<f4").astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid samples.")
    return audio


class Transcriber:
    """Whisper loaded once and shared by every request; one transcription at a time."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["live"]
        self.name = self.cfg["whisper_model"]
        self.processor = self.model = self.device = None
        self.error = ""
        self._lock = threading.Lock()

    @staticmethod
    def installed() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def status(self) -> dict:
        return {"installed": self.installed(), "loaded": self.loaded, "model": self.name,
                "device": self.device, "error": self.error}

    def load(self) -> None:
        """Downloads the model the first time (into the project's Hugging Face cache), moves it to
        the GPU when there is one, and runs it once: the first call is several seconds slower."""
        with self._lock:
            if self.loaded:
                return
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                wanted = self.cfg["device"]
                device = ("cuda" if torch.cuda.is_available() else "cpu") if wanted == "auto" else wanted
                processor = WhisperProcessor.from_pretrained(self.name)
                model = WhisperForConditionalGeneration.from_pretrained(self.name).to(device).eval()
                self.processor, self.model, self.device = processor, model, device
                self._generate(np.zeros(SAMPLE_RATE, dtype=np.float32))
                self.error = ""
            except Exception as error:  # reported to the page, which explains what to do
                self.processor = self.model = self.device = None
                self.error = f"{type(error).__name__}: {error}"
                raise

    def _generate(self, audio: np.ndarray) -> str:
        import torch

        features = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = self.model.generate(features.to(self.device), max_new_tokens=self.cfg["max_new_tokens"])
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def transcribe(self, audio: np.ndarray) -> dict:
        """Text of one utterance (16 kHz mono float32). Quiet audio is not sent to the model."""
        duration = len(audio) / SAMPLE_RATE
        if duration > self.cfg["max_utterance_s"]:
            raise ValueError(f"An utterance can be at most {self.cfg['max_utterance_s']} s long.")
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if duration < self.cfg["min_utterance_s"] or rms < self.cfg["min_rms"]:
            return {"text": "", "duration_s": duration, "elapsed_s": 0.0}
        self.load()
        started = time.perf_counter()
        with self._lock:
            text = self._generate(audio)
        return {"text": clean_transcript(text), "duration_s": duration,
                "elapsed_s": time.perf_counter() - started}


# ---- the playbook ----------------------------------------------------------------------------------

def situations_of(cues) -> list[str]:
    """The customer cues of a message, most urgent first, or [NO_CUE]."""
    present = set() if cues is None else set(cues)  # a list, or a numpy array from parquet
    found = [cue for cue in SITUATION_PRIORITY if cue in present]
    return found or [NO_CUE]


def build_playbook(frame: pd.DataFrame, cfg: dict) -> dict:
    """`frame`: one dataset and model's scored turns, columns conversation_id, turn, role, text,
    cues (list) and score, for the held-out conversations.

    Returns {"effects": rows of (situation, move, mean_delta, ci_low, ci_high, count, share_rise),
    "baseline": {situation: {mean_delta, count}}, "examples": {(situation, move): [...]},
    "n_conversations"}.
    """
    settings = cfg["live"]
    frame = frame.sort_values(["conversation_id", "turn"]).reset_index(drop=True)
    roles = frame["role"].to_numpy()
    conversation = frame["conversation_id"].to_numpy()
    score = frame["score"].to_numpy(dtype=float)
    texts = frame["text"].to_numpy()
    cues = frame["cues"].to_numpy()

    records = []  # (situation, move or None, delta, previous customer row, agent row, reply row)
    last_customer = -1
    pending = []  # agent rows waiting for the next customer reply
    for row in range(len(frame)):
        if row == 0 or conversation[row] != conversation[row - 1]:
            last_customer, pending = -1, []
        if roles[row] == "agent":
            pending.append((row, last_customer))
            continue
        for agent_row, previous in pending:
            situations = [OPENING] if previous < 0 else situations_of(cues[previous])
            delta = score[row] - score[agent_row]
            moves = [cue for cue in cues[agent_row] if cue in AGENT_MOVE_NAMES] or [None]
            for situation in [*situations, ANY]:
                for move in moves:
                    records.append((situation, move, delta, previous, agent_row, row))
        last_customer, pending = row, []

    columns = ["situation", "move", "delta", "previous", "agent", "reply"]
    replies = pd.DataFrame(records, columns=columns)
    z = norm.ppf(0.5 + cfg["insights"]["ci_level"] / 2)
    # An agent turn with two moves appears twice; the baseline counts it once.
    unique = replies.drop_duplicates(["situation", "agent"])
    baseline = {situation: {"mean_delta": float(group["delta"].mean()), "count": int(len(group))}
                for situation, group in unique.groupby("situation")}

    effects, examples = [], {}
    for (situation, move), group in replies.dropna(subset=["move"]).groupby(["situation", "move"]):
        values = group["delta"].to_numpy()
        if len(values) < settings["min_count"]:
            continue
        mean = float(values.mean())
        half = float(z * values.std(ddof=1) / np.sqrt(len(values)))
        effects.append({
            "situation": situation, "move": move, "mean_delta": mean, "ci_low": mean - half,
            "ci_high": mean + half, "count": int(len(values)), "share_rise": float((values > 0).mean()),
        })
        lengths = np.array([len(texts[row]) for row in group["agent"]])
        good = group[(group["delta"].to_numpy() > 0) & (lengths <= settings["max_example_chars"])]
        examples[(situation, move)] = [
            {"customer": None if example.previous < 0 else texts[example.previous],
             "agent": texts[example.agent], "reply": texts[example.reply], "delta": float(example.delta)}
            for example in good.nlargest(settings["n_examples"], "delta").itertuples(index=False)
        ]
    return {"effects": pd.DataFrame(effects, columns=[
                "situation", "move", "mean_delta", "ci_low", "ci_high", "count", "share_rise"]),
            "baseline": baseline, "examples": examples,
            "n_conversations": int(frame["conversation_id"].nunique())}


# ---- coaching ----------------------------------------------------------------------------------

def _pts(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value) * 100:.1f} pts"


def _why(situation: str, move: str, effect: dict, baseline: float) -> str:
    where = ("at the start of a call" if situation == OPENING
             else "when the customer gave no clear signal" if situation == NO_CUE
             else "anywhere in a call" if situation == ANY
             else f"after {BY_NAME[situation].phrase}")
    return (f"In past conversations, {BY_NAME[move].phrase} {where} was followed by "
            f"{_pts(effect['mean_delta'])} in the customer's next reply on average, against "
            f"{_pts(baseline)} for a typical reply (95% CI {_pts(effect['ci_low'])} to "
            f"{_pts(effect['ci_high'])}, {effect['count']:,} replies).")


def _headline(situation: str, score, status: str | None, waiting_for: str) -> str:
    if score is None:
        return ("Open the call: find out what they need before you pitch." if situation == OPENING
                else "Keep the customer talking; the score appears once both sides have spoken.")
    if waiting_for == "customer":
        return "Let the customer answer. The next suggestion follows their reply."
    if situation in ("commitment", "asks_process") or (status == "Hot" and situation not in SITUATION_PRIORITY[:6]):
        return "They're warm. Make it easy to say yes and ask for the next step."
    if situation in SITUATION_PRIORITY[:6]:
        return f"{SITUATION_LABELS[situation]}: handle it before moving on."
    if status == "Cooling":
        return "Interest is cooling. Re-engage with a question about what matters to them."
    return "Build on the momentum: connect the product to what they told you."


def coach(turns: list[dict], scores: list[float] | None, status: str | None, playbook: dict | None,
          cfg: dict) -> dict:
    """Suggestions for the salesperson's next message. `turns` carry their cue tags."""
    settings = cfg["live"]
    customer_turns = [turn for turn in turns if turn["role"] == "customer"]
    last_customer = customer_turns[-1] if customer_turns else None
    cues = [cue for cue in (last_customer or {}).get("cues", []) if cue in CUSTOMER_CUE_NAMES]
    situation = OPENING if last_customer is None else situations_of(cues)[0]
    waiting_for = "customer" if turns and turns[-1]["role"] == "agent" else "agent"
    score = scores[-1] if scores else None
    result = {
        "situation": situation,
        "situation_label": SITUATION_LABELS[situation],
        "cues": cues,
        "quote": last_customer["text"] if last_customer else None,
        "waiting_for": waiting_for,
        "headline": _headline(situation, score, status, waiting_for),
        "suggestions": [],
        "avoid": [],
        "evidence_from": None,
        "n_conversations": playbook["n_conversations"] if playbook else 0,
    }
    if not playbook or playbook["effects"].empty:
        return result

    effects = playbook["effects"]
    used = situation
    rows = effects[effects["situation"] == situation]
    if len(rows) < 2:  # too rare in the data to say: use every reply
        used, rows = ANY, effects[effects["situation"] == ANY]
    result["evidence_from"] = used
    baseline = playbook["baseline"].get(used, {}).get("mean_delta", 0.0)
    last_agent = next((turn for turn in reversed(turns) if turn["role"] == "agent"), None)
    just_used = set(last_agent.get("cues", [])) if last_agent else set()

    # Moves are compared with the typical reply in this situation: in some situations (the
    # opening, in SaaS) the score drops after almost any reply, and the useful question is which
    # reply does best. Rare moves are pulled towards the typical reply, so a lucky handful of
    # replies does not outrank a move seen hundreds of times.
    rows = rows.assign(lift=(rows["mean_delta"] - baseline) * rows["count"] / (rows["count"] + settings["shrinkage"]))
    ranked = rows.sort_values("lift", ascending=False).to_dict("records")
    # Suggested: the best-ranked moves that are not clearly worse than a typical reply.
    helpful = [row for row in ranked if row["ci_high"] >= baseline]
    for row in helpful[: settings["max_suggestions"]]:
        move = row["move"]
        result["suggestions"].append({
            "move": move,
            "action": MOVE_ACTIONS[move],
            "say": SITUATION_SAY.get((situation, move), SAY[move]),
            "why": _why(used, move, row, baseline),
            # strong: better than a typical reply beyond the 95% interval; suggestive: better on
            # average; typical: about as good as any other reply
            "evidence": ("strong" if row["ci_low"] > baseline else "suggestive" if row["mean_delta"] > baseline
                         else "typical"),
            "mean_delta": row["mean_delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "baseline": baseline, "count": row["count"], "share_rise": row["share_rise"],
            "just_used": move in just_used,
            "examples": playbook["examples"].get((used, move), []),
        })
    suggested = {s["move"] for s in result["suggestions"]}
    harmful = [row for row in reversed(ranked)
               if row["ci_high"] < baseline and row["mean_delta"] < 0 and row["move"] not in suggested]
    for row in harmful[: settings["max_avoid"]]:
        result["avoid"].append({
            "move": row["move"], "action": MOVE_ACTIONS[row["move"]],
            "why": _why(used, row["move"], row, baseline),
            "mean_delta": row["mean_delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "baseline": baseline, "count": row["count"], "just_used": row["move"] in just_used,
        })
    return result


# ---- one live update ---------------------------------------------------------------------------

def live_update(turns: list[dict], dataset: str, backend, playbook: dict | None, cfg: dict,
                method: str = "delta") -> dict:
    """Scores the call so far and coaches the next message.

    Before both sides have spoken there is no score yet: `ready` is False and the transcript still
    comes back with its cues, so the page can show them. Long calls keep their latest
    `analyze.max_turns` messages.
    """
    limits = cfg["analyze"]
    rows = []
    for raw in turns:
        role = str(raw.get("role", "")).lower()
        if role not in analysis.ROLES:
            raise analysis.InvalidConversation(f"Every message needs a role of 'agent' or 'customer', got {role!r}.")
        text = clean_text(raw.get("text"))[: limits["max_chars_per_turn"]]
        if text:
            rows.append({"role": role, "text": text, "speaker": clean_text(raw.get("speaker")) or role})
    dropped = max(0, len(rows) - limits["max_turns"])
    rows = rows[dropped:]

    ready = len(rows) >= limits["min_turns"] and {row["role"] for row in rows} == set(analysis.ROLES)
    if ready:
        detail = analysis.analyze(rows, dataset, backend, cfg, title="Live call", method=method)
        tagged, scores, status = detail["turns"], detail["scores"], detail["status"]
    else:
        tags = rules.tag_turns([row["role"] for row in rows], [row["text"] for row in rows])
        tagged = [{**row, "turn": index, "cues": list(tag)} for index, (row, tag) in enumerate(zip(rows, tags))]
        detail, scores, status = None, None, None
    return {
        "ready": ready,
        "message": None if ready else "The score appears once both the salesperson and the customer have spoken.",
        "dropped_turns": dropped,
        "turns": tagged,
        "detail": detail,
        "coach": coach(tagged, scores, status, playbook, cfg),
    }
