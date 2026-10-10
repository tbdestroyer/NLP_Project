"""View 3: which agent moves come before rises or drops."""
import plotly.graph_objects as go
import streamlit as st

import common
from inclination.config import load_config

st.set_page_config(page_title="Script Insights · Inclination Score", page_icon="🌿", layout="wide")
state = common.sidebar()
cfg = load_config()["insights"]
common.page_header(
    "Script Insights", state,
    "Average change in the customer's score on their next turn after each agent move.",
)
common.note("These are associations, not proven causes.")

effects = common.load("insights.parquet")
effects = effects[(effects["dataset"] == state["dataset"]) & (effects["model"] == state["model"])]
if effects.empty:
    st.info("No agent move occurs often enough in this dataset to report "
            f"(minimum {cfg['min_count']} occurrences).")
    st.stop()

position = st.segmented_control(
    "Position in the conversation", ["all", *cfg["position_buckets"]], default="all",
    help="Early, middle and late thirds of each conversation, to show how timing matters.",
) or "all"
view = effects[effects["position"] == position].sort_values("mean_delta")
st.caption(f"Computed on the test split of {common.dataset_label(state['dataset'])}, whatever "
           f"split is selected. Moves with fewer than {cfg['min_count']} occurrences are left out.")
if view.empty:
    st.info("No agent move occurs often enough at this position.")
    st.stop()

points = view["mean_delta"] * 100
figure = go.Figure(go.Bar(
    x=points, y=view["move"].map(common.pretty), orientation="h",
    marker={"color": [common.GREEN if v >= 0 else common.RED for v in points]},
    error_x={
        "type": "data", "symmetric": False, "color": common.GREY, "thickness": 1.5,
        "array": (view["ci_high"] - view["mean_delta"]) * 100,
        "arrayminus": (view["mean_delta"] - view["ci_low"]) * 100,
    },
    customdata=list(zip(view["count"], view["share_rise"], view["share_drop"],
                        view["ci_low"] * 100, view["ci_high"] * 100)),
    hovertemplate=(
        "<b>%{y}</b><br>mean change %{x:+.1f} pts (95% CI %{customdata[3]:+.1f} to "
        "%{customdata[4]:+.1f})<br>%{customdata[0]:,} occurrences<br>followed by a rise "
        "%{customdata[1]:.0%}, a drop %{customdata[2]:.0%}<extra></extra>"
    ),
))
common.style_figure(figure, height=90 + 46 * len(view))
figure.update_traces(marker_cornerradius=6)
figure.update_layout(bargap=0.42)
figure.update_xaxes(title_text="Mean score change on the next customer turn (percentage points)",
                    zeroline=True, zerolinewidth=1.5, zerolinecolor=common.GREY)
figure.update_yaxes(showgrid=False, tickfont={"color": common.INK})
st.plotly_chart(figure, theme=None)

with st.expander("Table view"):
    table = view.sort_values("mean_delta", ascending=False)[
        ["move", "mean_delta", "ci_low", "ci_high", "count", "share_rise", "share_drop"]
    ].copy()
    table["move"] = table["move"].map(common.pretty)
    for column in ("mean_delta", "ci_low", "ci_high"):
        table[column] = table[column] * 100
    st.dataframe(table, hide_index=True, column_config={
        "move": "Agent move",
        "mean_delta": st.column_config.NumberColumn("Mean change (pts)", format="%+.2f"),
        "ci_low": st.column_config.NumberColumn("CI low", format="%+.2f"),
        "ci_high": st.column_config.NumberColumn("CI high", format="%+.2f"),
        "count": st.column_config.NumberColumn("Occurrences"),
        "share_rise": st.column_config.NumberColumn("Followed by a rise", format="percent"),
        "share_drop": st.column_config.NumberColumn("Followed by a drop", format="percent"),
    })

st.subheader("Examples")
moves = view.sort_values("mean_delta", ascending=False)["move"].tolist()
move = st.selectbox("Agent move", moves, format_func=common.pretty)
examples = common.load("insight_examples.parquet")
examples = examples[
    (examples["dataset"] == state["dataset"]) & (examples["model"] == state["model"])
    & (examples["move"] == move) & (examples["position"] == position)
]
turns = common.load("turns.parquet")
cues = common.load_cues("rules")
for example in examples.itertuples():
    rows = common.conversation_rows(turns, example.conversation_id).set_index("turn")
    tagged = common.conversation_rows(cues, example.conversation_id).set_index("turn")["cues"]
    with st.container(border=True):
        top = st.columns([5, 1])
        top[0].markdown(f"Score change **{example.delta * 100:+.0f} pts** on the customer's reply")
        if top[1].button("Open", key=f"open_{example.conversation_id}_{example.agent_turn}",
                         width="stretch"):
            common.open_conversation(example.conversation_id)
        st.markdown(
            common.bubble("agent", rows.at[example.agent_turn, "text"], example.agent_turn,
                          list(tagged.get(example.agent_turn, [])))
            + common.bubble("customer", rows.at[example.customer_turn, "text"],
                            example.customer_turn, list(tagged.get(example.customer_turn, [])),
                            "rise" if example.delta > 0 else "drop", is_turning_point=False),
            unsafe_allow_html=True,
        )
