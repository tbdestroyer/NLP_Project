"""View 1: conversations ranked by score."""
import pandas as pd
import streamlit as st

import common
from inclination.config import load_config

st.set_page_config(page_title="Lead List · Inclination Score", page_icon="🌿", layout="wide")
state = common.sidebar()
cfg = load_config()
common.page_header("Lead List", state, "Conversations ranked by the latest inclination score.")
leads = state["leads"]
if leads.empty:
    st.info("No conversations in this selection. Pick another dataset or split in the sidebar.")
    st.stop()


@st.cache_data(show_spinner=False)
def score_curves(dataset: str, model: str, modified: float) -> pd.Series:
    scores = common.load("scores.parquet")
    scores = scores[(scores["dataset"] == dataset) & (scores["model"] == model)]
    return scores.sort_values("turn").groupby("conversation_id")["score"].agg(list)


@st.cache_data(show_spinner=False)
def conversations_mentioning(dataset: str, query: str) -> set:
    turns = common.load("turns.parquet")
    turns = turns[turns["dataset"] == dataset]
    return set(turns.loc[turns["text"].str.contains(query, case=False, regex=False), "conversation_id"])


filters = st.columns([2, 2, 1, 3])
statuses = filters[0].multiselect("Status", list(common.STATUS_ICONS), placeholder="All statuses")
low, high = filters[1].slider("Score range (%)", 0, 100, (0, 100), step=5)
min_turns = filters[2].number_input("Min turns", min_value=1, value=1, step=1)
query = filters[3].text_input("Search title or messages", placeholder="e.g. discount")

view = leads
if statuses:
    view = view[view["status"].isin(statuses)]
view = view[view["final_score"].between(low / 100, high / 100) & (view["n_turns"] >= min_turns)]
if query.strip():
    in_title = view["title"].str.contains(query.strip(), case=False, regex=False)
    in_text = view["conversation_id"].isin(conversations_mentioning(state["dataset"], query.strip()))
    view = view[in_title | in_text]
view = view.sort_values("final_score", ascending=False).reset_index(drop=True)

if state["eval_mode"]:
    top_k = max(cfg["eval"]["precision_at"])
    top = view.head(top_k)
    if len(top):
        panel = st.columns(3)
        panel[0].metric(f"Top {len(top)} leads that converted",
                        f"{int(top['outcome'].sum())} of {len(top)}")
        panel[1].metric("Conversion in the top leads", f"{top['outcome'].mean():.0%}")
        panel[2].metric("Base rate in this selection", f"{leads['outcome'].mean():.0%}")
        st.caption(
            f"Evaluation mode: of the top {len(top)} leads shown, {int(top['outcome'].sum())} "
            "actually converted."
        )

st.caption(f"{len(view):,} of {len(leads):,} conversations")
if view.empty:
    st.info("No conversations match these filters.")
    st.stop()

curves = score_curves(state["dataset"], state["model"],
                      common._require("scores.parquet").stat().st_mtime)
table = pd.DataFrame({
    "Title": view["title"],
    "Score": view["final_score"],
    "Score curve": view["conversation_id"].map(curves),
    "Status": view["status"].map(lambda s: f"{common.STATUS_ICONS[s]} {s}"),
    "Why": view["status_reason"],
    "Turning points": view["n_turning_points"],
    "Turns": view["n_turns"],
})
if state["eval_mode"]:
    table["Outcome"] = view["outcome"].map({1: "● converted", 0: "○ did not"})

event = st.dataframe(
    table,
    hide_index=True,
    height=520,
    on_select="rerun",
    selection_mode="single-row",
    width="stretch",
    column_config={
        "Title": st.column_config.TextColumn(width="large"),
        "Score": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0,
                                                 width="small"),
        "Score curve": st.column_config.LineChartColumn(y_min=0.0, y_max=1.0, width="small"),
        "Status": st.column_config.TextColumn(width="small"),
        "Why": st.column_config.TextColumn(width="medium"),
        "Turning points": st.column_config.NumberColumn(width="small"),
        "Turns": st.column_config.NumberColumn(width="small"),
    },
)

selected = event.selection.rows
if selected:
    row = view.iloc[selected[0]]
    action = st.columns([5, 1])
    action[0].markdown(f"Selected: **{row['title']}** · score {row['final_score']:.0%}")
    if action[1].button("View conversation", type="primary", width="stretch"):
        common.open_conversation(row["conversation_id"])
else:
    st.caption("Select a row to open the conversation.")
