"""Internal labelling tool for the hand-labelled validation set. Not one of the three views.

Model scores, cues and outcomes are deliberately not loaded here.
"""
import json

import streamlit as st

import common
from inclination import annotation
from inclination.config import load_config

st.set_page_config(page_title="Annotate · Inclination Score", page_icon="🌿", layout="wide")
common.apply_style()
cfg = load_config()
DIRECTIONS = cfg["annotation"]["directions"]
DIRECTION_LABELS = {"more": "More inclined", "same": "Same", "less": "Less inclined"}

if not annotation.sample_path().exists():
    st.error("`annotation/sample.json` is missing. Run `make export` (or `.\\make.ps1 export`) first.")
    st.stop()
sample = json.loads(annotation.sample_path().read_text(encoding="utf-8"))
turns = common.load("turns.parquet")

with st.sidebar:
    common.brand()
    annotator = annotation.safe_name(st.text_input(
        "Your annotator name", key="annotator", placeholder="e.g. shantanu",
        help="Labels are saved to annotation/labels/<name>.jsonl. Use the same name every time.",
    ))
    scope = st.selectbox("Dataset", ["all", *sample], key="annotate_dataset",
                         format_func=lambda d: "All datasets" if d == "all" else common.dataset_label(d))

st.title("Annotate")
st.caption("Internal tool. Label how the customer's inclination changes, turn by turn. "
           "Model scores are hidden here on purpose.")
if not annotator:
    st.info("Enter your annotator name in the sidebar to start.")
    st.stop()

queue = [(d, cid) for d, ids in sample.items() if scope in ("all", d) for cid in ids]
saved = annotation.load_labels(annotator)
done = [cid in saved for _, cid in queue]

position_key = f"annotate_position_{scope}"
if position_key not in st.session_state:
    st.session_state[position_key] = done.index(False) if False in done else 0
position = min(st.session_state[position_key], len(queue) - 1)


def go_to(index: int) -> None:
    st.session_state[position_key] = max(0, min(index, len(queue) - 1))


def next_unlabelled(start: int) -> int:
    order = list(range(start + 1, len(queue))) + list(range(0, start + 1))
    return next((i for i in order if not done[i]), start)


with st.sidebar:
    st.divider()
    st.progress(sum(done) / len(queue), text=f"{sum(done)} of {len(queue)} labelled")

dataset, conversation_id = queue[position]
rows = common.conversation_rows(turns, conversation_id)
previous = {label["turn"]: label for label in saved.get(conversation_id, {}).get("labels", [])}

navigation = st.columns([1, 1, 2, 4])
navigation[0].button("← Previous", on_click=go_to, args=(position - 1,), disabled=position == 0,
                     width="stretch")
navigation[1].button("Next →", on_click=go_to, args=(position + 1,),
                     disabled=position == len(queue) - 1, width="stretch")
navigation[2].button("Next unlabelled", on_click=go_to, args=(next_unlabelled(position),),
                     width="stretch")
status = "labelled" if done[position] else "not labelled yet"
st.markdown(
    f'<span class="is-pill">Conversation {position + 1} of {len(queue)}</span> '
    f'<span class="is-pill">{common.dataset_label(dataset)}</span> '
    f'<span class="is-pill">{status}</span>',
    unsafe_allow_html=True,
)

with st.expander("How to label", expanded=not saved):
    st.markdown(
        "- For **every customer turn**, decide whether the customer is **more**, the **same** or "
        "**less** inclined to buy (accept the deal, donate, sign up) than just before that turn.\n"
        "- Tick **turning point** on any turn, customer or agent, where you feel the conversation "
        "clearly changed direction. Most conversations have zero to two.\n"
        "- Judge only from what has been said so far. Do not read ahead before deciding.\n"
        "- Full instructions and worked examples: `annotation/guidelines.md`."
    )

with st.form(f"labels_{annotator}_{conversation_id}"):
    answers = {}
    for row in rows.itertuples():
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown(common.bubble(row.role, row.text, row.turn, []), unsafe_allow_html=True)
        earlier = previous.get(row.turn, {})
        direction = None
        if row.role == "customer":
            direction = right.radio(
                f"Turn {row.turn}: inclination", DIRECTIONS, horizontal=True,
                index=DIRECTIONS.index(earlier["direction"]) if earlier.get("direction") else None,
                format_func=DIRECTION_LABELS.get, key=f"direction_{annotator}_{conversation_id}_{row.turn}",
                label_visibility="collapsed",
            )
        turning_point = right.checkbox(
            "Turning point", value=bool(earlier.get("turning_point", False)),
            key=f"turning_{annotator}_{conversation_id}_{row.turn}",
        )
        answers[row.turn] = (row.role, direction, turning_point)
    submitted = st.form_submit_button("Save and go to the next unlabelled", type="primary")

if submitted:
    missing = [turn for turn, (role, direction, _) in answers.items()
               if role == "customer" and direction is None]
    if missing:
        st.error("Choose more / same / less for every customer turn. Missing turns: "
                 + ", ".join(map(str, missing)))
    else:
        labels = [
            {"turn": int(turn), "role": role, "direction": direction, "turning_point": bool(point)}
            for turn, (role, direction, point) in answers.items()
        ]
        annotation.save_labels(annotator, conversation_id, dataset, labels)
        done[position] = True
        go_to(next_unlabelled(position))
        st.toast("Saved")
        st.rerun()
