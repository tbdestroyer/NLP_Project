"""Shared helpers for the dashboard pages. The app only reads files in outputs/."""
import html
import json

import pandas as pd
import streamlit as st

from inclination.config import load_config, path
from inclination.cues.taxonomy import BY_NAME

# Palette: cream surfaces, light green accents. The base colours live in .streamlit/config.toml.
CREAM, CARD, MINT, SAGE = "#FBF8EE", "#FFFDF7", "#E4F0D8", "#CFE3BF"
INK, MUTED, LINE = "#24332A", "#6F7C6F", "#DCE6CE"
FOREST = "#2F6B4F"                       # score curve, headings
GREEN, RED, GREY = "#3E9B5F", "#D2604D", "#9A9F92"   # rise / drop / neutral
MODEL_COLORS = {"v1": FOREST, "v2": "#D08A3C", "v3": "#4A7FB5"}
MODEL_LABELS = {"v1": "v1 · cue model", "v2": "v2 · LLM rubric", "v3": "v3 · transformer"}
STATUS_ICONS = {"Hot": "◆", "Rising": "▲", "Cooling": "▼", "Stalled": "–"}
STATUS_COLORS = {"Hot": GREEN, "Rising": GREEN, "Cooling": RED, "Stalled": GREY}
SPLITS = ["test", "val", "train", "all"]
CONVERSATION_PAGE = "pages/2_Conversation_View.py"
_PERSISTED = ("dataset", "model", "split", "eval_mode")

_CSS = f"""
<style>
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {FOREST} !important; letter-spacing: -0.015em; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {MUTED} !important; opacity: 1 !important;
}}
[data-testid="stWidgetLabel"] p {{ color: {INK}; font-weight: 500; }}
h1 {{ font-weight: 700; font-size: 2.1rem; padding-bottom: 0.2rem; }}
h3 {{ font-weight: 650; font-size: 1.15rem; }}
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(47, 107, 79, 0.05);
}}
[data-testid="stMetricLabel"] p {{
    color: {MUTED}; font-size: 0.74rem; letter-spacing: 0.06em; text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 650; font-size: 1.65rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-radius: 16px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius: 999px; font-weight: 600; }}
hr {{ border-color: {LINE}; }}
.is-brand {{ display: flex; align-items: center; gap: 10px; margin: 0 0 14px 2px; }}
.is-brand .dot {{
    width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
    background: {FOREST}; color: {CREAM}; font-weight: 700;
}}
.is-brand .name {{ font-weight: 700; color: {FOREST}; font-size: 1.05rem; }}
.is-hero {{
    background: linear-gradient(120deg, {MINT} 0%, {CREAM} 78%); border: 1px solid {LINE};
    border-radius: 22px; padding: 30px 34px; margin-bottom: 8px;
}}
.is-hero .eyebrow {{
    color: {GREEN}; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.is-hero .title {{
    color: {FOREST}; font-size: 2.3rem; font-weight: 750; letter-spacing: -0.02em;
    line-height: 1.15; margin: 4px 0 8px 0;
}}
.is-hero .lead {{ color: {INK}; font-size: 1.02rem; max-width: 780px; opacity: 0.86; margin: 0; }}
.is-pill {{
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; border: 1px solid {LINE}; background: {CARD}; color: {INK};
}}
.is-pill.warn {{ background: #FBEFD3; border-color: #EBD49A; color: #7A5A12; }}
.is-card-title {{ color: {FOREST}; font-weight: 650; font-size: 1.02rem; margin-bottom: 2px; }}
.is-note {{
    background: {MINT}; border: 1px solid {SAGE}; border-radius: 14px; padding: 10px 16px;
    color: {INK}; font-size: 0.9rem;
}}
</style>
"""


@st.cache_data(show_spinner="Loading dashboard data…")
def _read_parquet(name: str, modified: float) -> pd.DataFrame:
    return pd.read_parquet(path("outputs") / name)


def _require(name: str):
    target = path("outputs") / name
    if not target.exists():
        st.error(
            f"`outputs/{name}` is missing. Run `make all` (or `.\\make.ps1 all`) to build the "
            "dashboard files, then reload this page."
        )
        st.stop()
    return target


def load(name: str) -> pd.DataFrame:
    """Cached read of an exported parquet file; stops the page with a clear message if absent."""
    target = _require(name)
    return _read_parquet(name, target.stat().st_mtime)


CUE_TAGGERS = {"rules": "Rules", "llm": "Gemini"}


@st.cache_data(show_spinner="Loading dashboard data…")
def _read_cues(tagger: str, modified: float) -> pd.DataFrame:
    cues = pd.read_parquet(path("outputs") / "cues.parquet")
    return cues[cues["tagger"] == tagger].reset_index(drop=True)


def load_cues(tagger: str = "rules") -> pd.DataFrame:
    """Cue tags from one tagger: "rules" covers every turn, "llm" (Gemini) only a sample."""
    return _read_cues(tagger, _require("cues.parquet").stat().st_mtime)


def load_metrics() -> dict:
    target = path("outputs") / "metrics.json"
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def dataset_label(dataset: str) -> str:
    return load_config()["viz"]["dataset_labels"].get(dataset, dataset)


def apply_style() -> None:
    st.html(_CSS)


def brand() -> None:
    st.markdown(
        '<div class="is-brand"><div class="dot">↗</div><div class="name">Inclination Score</div></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    """Draws the global controls and returns the current selection."""
    apply_style()
    cfg = load_config()
    leads = load("leads.parquet")
    for key in _PERSISTED:  # keeps widget values when moving between pages
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]

    with st.sidebar:
        brand()
        present = set(leads["dataset"])
        datasets = [d for d in cfg["data"]["datasets"] if d in present]
        dataset = st.selectbox("Dataset", datasets, format_func=dataset_label, key="dataset")
        models = sorted(leads.loc[leads["dataset"] == dataset, "model"].unique())
        if st.session_state.get("model") not in models:
            st.session_state["model"] = models[0]
        model = st.selectbox("Model", models, format_func=lambda m: MODEL_LABELS.get(m, m), key="model")
        split = st.selectbox("Split", SPLITS, key="split",
                             help="The test split is the honest view: the model never saw it.")
        st.divider()
        eval_mode = st.toggle("Show true outcomes (evaluation mode)", key="eval_mode",
                              help="Off in product mode: outcomes are never shown.")

    selection = leads[(leads["dataset"] == dataset) & (leads["model"] == model)]
    is_synthetic = bool(selection["is_synthetic"].any())
    if split != "all":
        selection = selection[selection["split"] == split]
    return {
        "dataset": dataset, "model": model, "split": split, "eval_mode": eval_mode,
        "leads": selection, "is_synthetic": is_synthetic, "models": models,
    }


def synthetic_badge(state: dict) -> None:
    if state["is_synthetic"]:
        st.markdown(
            '<span class="is-pill warn">⚗ Synthetic data · these conversations were generated '
            "by an LLM</span>",
            unsafe_allow_html=True,
        )


def page_header(title: str, state: dict, caption: str = "") -> None:
    st.title(title)
    if caption:
        st.caption(caption)
    synthetic_badge(state)


def pills(*labels: str) -> None:
    """A row of small neutral pills, for facts that are too long for a metric card."""
    st.markdown(
        " ".join(f'<span class="is-pill">{html.escape(label)}</span>' for label in labels),
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="is-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_figure(figure, height: int) -> None:
    """Common chart look: transparent surface, hairline grid, ink text."""
    figure.update_layout(
        height=height, margin={"l": 64, "r": 20, "t": 36, "b": 56},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "size": 13}, hovermode="closest",
        hoverlabel={"bgcolor": CARD, "bordercolor": LINE, "font": {"color": INK}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    axis = {"gridcolor": LINE, "linecolor": SAGE, "zerolinecolor": SAGE, "automargin": True,
            "tickfont": {"color": MUTED}, "title": {"font": {"color": MUTED, "size": 12}, "standoff": 12}}
    figure.update_xaxes(**axis)
    figure.update_yaxes(**axis)


def open_conversation(conversation_id: str) -> None:
    st.session_state["pending_conversation"] = conversation_id
    st.switch_page(CONVERSATION_PAGE)


def conversation_rows(frame: pd.DataFrame, conversation_id: str) -> pd.DataFrame:
    return frame[frame["conversation_id"] == conversation_id].sort_values("turn")


def pretty(name: str) -> str:
    return name.replace("_", " ")


def chip(name: str) -> str:
    """Small cue chip coloured by the cue's direction (green +, red -, grey neutral)."""
    direction = BY_NAME[name].direction if name in BY_NAME else 0
    color = GREEN if direction > 0 else RED if direction < 0 else GREY
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin:6px 5px 0 0;'
        f'padding:1px 9px;border-radius:999px;font-size:0.7rem;color:{INK};'
        f'border:1px solid {color}55;background:{color}1c;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{color};"></span>'
        f'{html.escape(pretty(name))}</span>'
    )


def bubble(role: str, text: str, turn: int, cues, highlight: str | None = None,
           is_turning_point: bool = True) -> str:
    """One chat bubble. Customer on the left, agent on the right. `highlight` (rise/drop) draws
    a green or red outline, labelled as a turning point unless `is_turning_point` is False."""
    customer = role == "customer"
    align = "flex-start" if customer else "flex-end"
    background = MINT if customer else CARD
    corner = "16px 16px 16px 4px" if customer else "16px 16px 4px 16px"
    border = {"rise": GREEN, "drop": RED}.get(highlight)
    outline = f"border:2px solid {border};" if border else f"border:1px solid {LINE};"
    word = "turning point" if is_turning_point else "score"
    marker = {"rise": f" · ▲ {word}", "drop": f" · ▼ {word}"}.get(highlight, "")
    chips = "".join(chip(name) for name in cues)
    return (
        f'<div style="display:flex;justify-content:{align};margin:8px 0;">'
        f'<div style="max-width:84%;padding:9px 14px 10px;border-radius:{corner};'
        f'background:{background};{outline}">'
        f'<div style="font-size:0.68rem;letter-spacing:0.04em;color:{MUTED};">'
        f'{"CUSTOMER" if customer else "AGENT"} · turn {turn}'
        f'<span style="color:{border or MUTED};font-weight:700;">{marker}</span></div>'
        f'<div style="font-size:0.93rem;line-height:1.45;color:{INK};overflow-wrap:anywhere;">'
        f'{html.escape(text)}</div>{chips}</div></div>'
    )
