"""Explanations for turning points: trigger, cues, template rationale and faithfulness."""
import numpy as np

from inclination.cues.taxonomy import AGENT_MOVES, BY_NAME, CUSTOMER_CUES


def previous_agent_rows(roles: np.ndarray, position: np.ndarray) -> np.ndarray:
    """For each row, the row of the latest agent turn strictly before it in the same
    conversation, or -1. Rows are sorted by conversation and turn."""
    result = np.full(len(roles), -1, dtype=np.int64)
    last_agent = -1
    for row in range(len(roles)):
        if position[row] == 0:
            last_agent = -1
        result[row] = last_agent
        if roles[row] == "agent":
            last_agent = row
    return result


def _pick_customer_cue(cues: list[str], direction: str):
    """Prefers a cue whose direction matches the change, strongest first."""
    wanted = 1 if direction == "rise" else -1
    present = [cue for cue in CUSTOMER_CUES if cue.name in cues]
    matching = [cue for cue in present if cue.direction == wanted]
    matching.sort(key=lambda cue: cue.strength != "strong")
    return (matching or present or [None])[0]


def _pick_agent_move(moves: list[str]):
    return next((move for move in AGENT_MOVES if move.name in moves), None)


def template_rationale(direction: str, role: str, cues: list[str], previous_moves: list[str]):
    """Returns (one-sentence rationale, short trigger phrase)."""
    verb = "rose" if direction == "rise" else "dropped"
    if role == "agent":
        move = _pick_agent_move(cues)
        trigger = f"the agent {move.phrase}" if move else "the agent's message"
        return f"Interest {verb} right after {trigger}.", trigger
    cue = _pick_customer_cue(cues, direction)
    trigger = cue.phrase if cue else "the customer's reply"
    sentence = f"Interest {verb} after {trigger}"
    move = _pick_agent_move(previous_moves)
    if move:
        sentence += f", following the agent {move.phrase}"
    return sentence + ".", trigger


def removal_variants(rows: np.ndarray, starts: np.ndarray, sizes: np.ndarray, position: np.ndarray):
    """For each trigger row: the conversation up to the turn after it, without the trigger.

    Returns (variant_rows, variant_groups, evaluation_rows). The faithfulness of a turning point
    is score[evaluation_row] minus the score of the last prefix of its variant. When the trigger
    is the final turn, the evaluation row is the trigger itself.
    """
    variant_rows, variant_groups, evaluation_rows = [], [], []
    for i, row in enumerate(rows):
        start = row - position[row]
        end = min(row + 1, start + sizes[row] - 1)
        kept = np.concatenate([np.arange(start, row), np.arange(row + 1, end + 1)])
        variant_rows.append(kept)
        variant_groups.append(np.full(len(kept), i))
        evaluation_rows.append(end)
    if not variant_rows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(variant_rows), np.concatenate(variant_groups), np.array(evaluation_rows)


def cue_phrase(name: str) -> str:
    return BY_NAME[name].phrase
