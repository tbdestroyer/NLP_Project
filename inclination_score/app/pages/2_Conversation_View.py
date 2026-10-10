"""View 2: one transcript next to its score curve, with turning points and replay."""
import json

import plotly.graph_objects as go
import streamlit as st

import common

st.set_page_config(page_title="Conversation · Inclination Score", page_icon="🌿", layout="wide")
METHODS = {"delta": "Score change", "pelt": "Change point"}

# A conversation opened from another page (or a ?conversation= link) decides the dataset/split.
all_leads = common.load("leads.parquet")
pending = st.session_state.pop("pending_conversation", None)
if pending is None and "conv_select" not in st.session_state:
    pending = st.query_params.get("conversation")
if pending is not None:
    match = all_leads[all_leads["conversation_id"] == pending]
    if len(match):
        st.session_state["dataset"] = match["dataset"].iloc[0]
        if st.session_state.get("split", "test") not in ("all", match["split"].iloc[0]):
            st.session_state["split"] = match["split"].iloc[0]
        st.session_state["conv_select"] = pending

state = common.sidebar()
leads = state["leads"].sort_values("final_score", ascending=False)
if leads.empty:
    common.page_header("Conversation View", state)
    st.info("No conversations in this selection. Pick another dataset or split in the sidebar.")
    st.stop()

ids = leads["conversation_id"].tolist()
titles = dict(zip(ids, leads["title"]))
finals = dict(zip(ids, leads["final_score"]))
if st.session_state.get("conv_select") not in titles:
    st.session_state["conv_select"] = ids[0]
conversation_id = st.selectbox(
    "Conversation (sorted by score)", ids, key="conv_select",
    format_func=lambda cid: f"{finals[cid]:.0%} · {titles[cid]}",
)
st.query_params["conversation"] = conversation_id
lead = leads[leads["conversation_id"] == conversation_id].iloc[0]

turns = common.conversation_rows(common.load("turns.parquet"), conversation_id)
cues_by_tagger = {
    tagger: common.conversation_rows(common.load_cues(tagger), conversation_id)
    for tagger in common.CUE_TAGGERS
}
scores = common.conversation_rows(common.load("scores.parquet"), conversation_id)
points = common.conversation_rows(common.load("turning_points.parquet"), conversation_id)
points = points[points["model"] == state["model"]]
methods = [m for m in METHODS if m in set(common.load("turning_points.parquet")["method"])]
n_turns = len(turns)

common.page_header(lead["title"], state)
common.pills(common.dataset_label(state["dataset"]), f"{n_turns} turns",
             common.MODEL_LABELS.get(state["model"], state["model"]), f"{state['split']} split")
header = st.columns(3)
header[0].metric("Final score", f"{lead['final_score']:.0%}")
header[1].metric("Status", f"{common.STATUS_ICONS[lead['status']]} {lead['status']}",
                 help=lead["status_reason"])
if state["eval_mode"]:
    header[2].metric("True outcome", "Converted" if lead["outcome"] == 1 else "Did not convert")
else:
    header[2].metric("Turning points", int(lead["n_turning_points"]))

controls = st.columns([3, 2, 2])
method = "delta"
if len(methods) > 1:
    method = controls[2].segmented_control(
        "Turning-point method", methods, default="delta", format_func=METHODS.get,
        help="Score change: the score jumps by 15 points or more in one turn. Change point: "
             "the curve shifts to a new level that lasts at least two turns.",
    ) or "delta"
points = points[points["method"] == method]
tagger = "rules"
if len(cues_by_tagger["llm"]):  # Gemini tagged this conversation too
    tagger = controls[2].segmented_control(
        "Cue chips from", list(common.CUE_TAGGERS), default="rules",
        format_func=common.CUE_TAGGERS.get,
        help="Rules: keyword patterns, available for every conversation. Gemini: an LLM's tags, "
             "available for a sample of conversations.",
    ) or "rules"
cues = cues_by_tagger[tagger]
cues_by_turn = {int(t): list(c) for t, c in zip(cues["turn"], cues["cues"])}
if state["model"] == "v2":
    st.caption("v2 scores come from Gemini reading the conversation so far. They are not "
               "calibrated and exist for a small test sample only.")
upto = controls[0].slider(
    "Replay: show conversation up to turn", 0, n_turns - 1, n_turns - 1,
    key=f"replay_{conversation_id}", help="Shows what a live user would have seen at that turn.",
) if n_turns > 1 else 0
overlay = []
if len(state["models"]) > 1:
    overlay = controls[1].multiselect(
        "Overlay other models", [m for m in state["models"] if m != state["model"]],
        format_func=lambda m: common.MODEL_LABELS.get(m, m),
    )

visible_points = points[points["turn"] <= upto]
chart_column, transcript_column = st.columns([3, 2], gap="large")

with chart_column:
    mine = scores[(scores["model"] == state["model"]) & (scores["turn"] <= upto)]
    current = float(mine["score"].iloc[-1])
    previous = float(mine["score"].iloc[-2]) if len(mine) > 1 else None
    st.metric(
        f"Score at turn {upto}", f"{current:.0%}",
        delta=None if previous is None else f"{(current - previous) * 100:+.0f} pts vs previous turn",
    )

    figure = go.Figure()
    for model in [state["model"], *overlay]:
        curve = scores[(scores["model"] == model) & (scores["turn"] <= upto)]
        figure.add_trace(go.Scatter(
            x=curve["turn"], y=curve["score"], mode="lines+markers",
            name=common.MODEL_LABELS.get(model, model),
            line={"color": common.MODEL_COLORS.get(model, common.GREY), "width": 2.5},
            marker={"size": 7, "line": {"width": 1.5, "color": common.CREAM}},
            fill="tozeroy" if model == state["model"] else None,
            fillcolor="rgba(79, 154, 106, 0.10)",
            hovertemplate="turn %{x}<br>score %{y:.0%}<extra></extra>",
        ))
    for direction, symbol, color, label in (
        ("rise", "triangle-up", common.GREEN, "Rise"), ("drop", "triangle-down", common.RED, "Drop"),
    ):
        subset = visible_points[visible_points["direction"] == direction]
        if subset.empty:
            continue
        y = mine.set_index("turn")["score"].reindex(subset["turn"]).to_numpy()
        figure.add_trace(go.Scatter(
            x=subset["turn"], y=y, mode="markers", name=f"{label} (turning point)",
            marker={"symbol": symbol, "size": 17, "color": color,
                    "line": {"width": 2, "color": common.CREAM}},
            customdata=list(zip(subset["delta"] * 100, subset["rationale"])),
            hovertemplate="turn %{x} · %{customdata[0]:+.0f} pts<br>%{customdata[1]}<extra></extra>",
        ))
    common.style_figure(figure, height=420)
    figure.update_layout(showlegend=len(figure.data) > 1)
    figure.update_xaxes(title_text="Turn", range=[-0.4, n_turns - 0.6],
                        dtick=1 if n_turns <= 25 else None, showgrid=False)
    figure.update_yaxes(title_text="Inclination score", range=[0, 1.02], tickformat=".0%", dtick=0.25)
    event = st.plotly_chart(figure, on_select="rerun", selection_mode="points", theme=None,
                            key=f"chart_{conversation_id}")
    clicked = [p["x"] for p in event.selection.points] if event and event.selection else []

    st.subheader("Turning points")
    if visible_points.empty:
        st.info(
            "No turning points up to this turn: the score never moved sharply between two "
            "consecutive turns."
        )
    else:
        options = visible_points["turn"].tolist()
        labels = {
            int(p.turn): f"Turn {p.turn} · {'▲ rise' if p.direction == 'rise' else '▼ drop'} "
                         f"{p.delta * 100:+.0f} pts"
            for p in visible_points.itertuples()
        }
        clicked_points = [t for t in clicked if t in options]
        if clicked_points:
            st.session_state[f"point_{conversation_id}"] = clicked_points[0]
        if st.session_state.get(f"point_{conversation_id}") not in options:
            st.session_state[f"point_{conversation_id}"] = options[0]
        chosen = st.selectbox("Explain turning point", options, format_func=labels.get,
                              key=f"point_{conversation_id}",
                              help="You can also click a triangle on the chart.")
        point = visible_points[visible_points["turn"] == chosen].iloc[0]
        text_by_turn = dict(zip(turns["turn"], turns["text"]))
        role_by_turn = dict(zip(turns["turn"], turns["role"]))
        with st.container(border=True):
            written = point.get("rationale_llm")
            if isinstance(written, str) and written.strip():
                st.markdown(f"**{written}**")
                st.caption(f"Written by Gemini. Rule-based version: {point['rationale']}")
            else:
                st.markdown(f"**{point['rationale']}**")
            facts = st.columns(2)
            facts[0].metric("Change", f"{point['delta'] * 100:+.0f} pts")
            if point["faithfulness"] == point["faithfulness"]:  # not NaN
                facts[1].metric(
                    "Faithfulness", "weak" if point["weak"] else "strong",
                    help="How much the score changes when the trigger message is removed and the "
                         f"conversation is re-scored: {abs(point['faithfulness']) * 100:.0f} pts. "
                         "A small change means the explanation is weak.",
                )
            else:
                facts[1].metric("Faithfulness", "n/a",
                                help="Only computed for the score-change method.")
            st.caption(f"Detected by the {METHODS[method].lower()} method.")
            if point["prev_agent_turn"] >= 0:
                st.caption(f"Agent message just before (turn {point['prev_agent_turn']})")
                st.markdown(common.bubble(
                    "agent", text_by_turn[point["prev_agent_turn"]], int(point["prev_agent_turn"]),
                    cues_by_turn.get(int(point["prev_agent_turn"]), []),
                ), unsafe_allow_html=True)
            st.caption(f"Trigger message (turn {chosen})")
            st.markdown(common.bubble(
                role_by_turn[chosen], text_by_turn[chosen], int(chosen),
                cues_by_turn.get(int(chosen), []), point["direction"],
            ), unsafe_allow_html=True)
            features = json.loads(point["top_features"]) if point["top_features"] else []
            if features:
                st.caption(
                    "Largest model contributions: "
                    + ", ".join(f"`{f['feature']}` ({f['contribution']:+.2f})" for f in features)
                )

with transcript_column:
    st.subheader("Transcript")
    direction_by_turn = dict(zip(points["turn"], points["direction"]))
    bubbles = [
        common.bubble(row.role, row.text, row.turn, cues_by_turn.get(int(row.turn), []),
                      direction_by_turn.get(row.turn))
        for row in turns[turns["turn"] <= upto].itertuples()
    ]
    with st.container(height=760):
        st.markdown("".join(bubbles), unsafe_allow_html=True)
    st.caption("Chips are detected cues: green raises inclination, red lowers it, grey marks an "
               "agent move.")
