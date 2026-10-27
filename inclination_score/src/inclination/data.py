"""Reads the standard tables written by inclination.load.standardise."""
import numpy as np
import pandas as pd

from inclination.config import path


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(turns, conversations), both sorted so each conversation's turns are contiguous."""
    folder = path("processed")
    turns = pd.read_parquet(folder / "turns.parquet")
    conversations = pd.read_parquet(folder / "conversations.parquet")
    turns = turns.sort_values(["dataset", "conversation_id", "turn"]).reset_index(drop=True)
    conversations = conversations.sort_values(["dataset", "conversation_id"]).reset_index(drop=True)
    return turns, conversations


def group_codes(conversation_ids) -> np.ndarray:
    """0-based id of the contiguous run each row belongs to."""
    ids = np.asarray(conversation_ids)
    if len(ids) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(ids[1:] != ids[:-1])]).astype(np.int64)
