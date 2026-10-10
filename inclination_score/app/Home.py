"""Inclination Score dashboard. Run: streamlit run app/Home.py"""
import streamlit as st

import common

st.set_page_config(page_title="Inclination Score", page_icon="🌿", layout="wide")
state = common.sidebar()
leads = state["leads"]

st.markdown(
    '<div class="is-hero"><div class="eyebrow">Sales conversation intelligence</div>'
    '<div class="title">Inclination Score</div>'
    '<p class="lead">Shows, message by message, how likely each customer is to buy and what '
    "changed their mind, so a sales team knows whom to follow up with first and which parts of "
    "the script work.</p></div>",
    unsafe_allow_html=True,
)
common.synthetic_badge(state)

metrics = common.load_metrics().get("outcome", {}).get(state["dataset"], {}).get(state["model"], {})
st.subheader(f"{common.dataset_label(state['dataset'])} · {state['split']} split")
columns = st.columns(4)
columns[0].metric("Conversations scored", f"{len(leads):,}")
if len(leads):
    columns[1].metric("Average score", f"{leads['final_score'].mean():.0%}")
    columns[2].metric("Hot leads", f"{(leads['status'] == 'Hot').sum():,}")
else:
    st.info("No conversations in this selection.")
if metrics:
    low, high = metrics["roc_auc_ci"]
    columns[3].metric(
        "Test ROC AUC", f"{metrics['roc_auc']:.3f}",
        help=f"How well the final score ranks buyers above non-buyers on the test split. "
             f"0.5 is chance. 95% interval {low:.3f} to {high:.3f}.",
    )
else:
    columns[3].metric("Test ROC AUC", "n/a", help="Run `make eval` to compute it.")

st.subheader("Views")
VIEWS = [
    ("pages/1_Lead_List.py", "Lead List", ":material/format_list_bulleted:",
     "All conversations ranked by score, with status and trend, so you know whom to follow up "
     "with first."),
    ("pages/2_Conversation_View.py", "Conversation View", ":material/forum:",
     "One transcript next to its score curve, with turning points, explanations and a "
     "turn-by-turn replay."),
    ("pages/3_Script_Insights.py", "Script Insights", ":material/insights:",
     "Which agent moves tend to come before a rise or a drop in customer inclination, across "
     "all conversations."),
]
for column, (page, label, icon, text) in zip(st.columns(3), VIEWS):
    with column.container(border=True):
        st.page_link(page, label=f"**{label}**", icon=icon)
        st.caption(text)

st.caption(
    "The datasets are proxies for sales: price negotiation (CraigslistBargains), charity donation "
    "(Persuasion for Good) and LLM-generated SaaS sales chats. Everything shown is precomputed."
)
