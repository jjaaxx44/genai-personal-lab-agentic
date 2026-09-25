"""The Swarm page: the standard demo layout, plus a custody timeline between the
trajectory track and the final output -- the claim this demo makes (whoever holds
the task owns it until it hands off, with no router deciding for them) is only
checkable if a reader can see who held the task, for how long, and why they gave
it up.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.ui import (
    budget_strip,
    clear_data_button,
    demo_header,
    final_output,
    graph_map,
    provider_note,
    readme_and_trace_tabs,
    sidebar_budget_controls,
    task_row,
    trajectory_track,
)

# Absolute, not relative: Streamlit runs a page as a script, so `from . import`
# has no parent package to resolve against.
from demos.agents.swarm import agent
from demos.agents.swarm.agent import ANALYST_TOOLS, DEMO, GRAPH, NAME, PEERS, PRESETS, RESEARCHER_TOOLS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    f"Handoff cap: {settings.swarm_max_handoffs}. Reaching it does not stop the run -- a "
    "refused transfer comes back to the holding peer as a tool error it has to recover "
    "from, the same as any other tool failure. The shared step cap is the hard stop."
)

st.sidebar.subheader("Peers")
st.sidebar.caption(f"researcher: {', '.join(RESEARCHER_TOOLS)}")
st.sidebar.caption(f"analyst: {', '.join(ANALYST_TOOLS)}")
st.sidebar.caption("writer: no work tools -- drafts the answer, or hands off for what's missing")

st.sidebar.subheader("Settings")
entry_peer = st.sidebar.selectbox(
    "Entry peer",
    PEERS,
    key=f"{DEMO}_entry_peer",
    help="Who starts holding the task. Preset 1 pairs with researcher; preset 2 with analyst.",
)
force_no_answer = st.sidebar.toggle(
    "Nobody may answer",
    key=f"{DEMO}_force_no_answer",
    help=(
        "Every peer is told it may never reply with a final answer, only hand off -- "
        "shows the task ping-ponging between peers until a budget cap catches it, "
        "the swarm equivalent of Step 9's 'Supervisor can't declare done'."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Every peer reads the same shared conversation and decides for itself whether "
    "to keep working, hand off, or answer -- inside the same call that does the "
    "work. Step 9 (Supervisor-worker) is the opposite choice: one router decides "
    "every hop, in a separate call, from a reports board built fresh each turn."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task with a two-peer split")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Working...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(
                task, run_budget, on_step=on_step, entry_peer=entry_peer, force_no_answer=force_no_answer
            )
        except Exception as exc:  # nothing below core is allowed to reach the reader
            status.update(label="The run failed.", state="error")
            st.error(f"The run could not finish: {type(exc).__name__}: {exc}")
            result = None
        else:
            status.update(
                label=f"{result.status.replace('_', ' ')} · {len(result.steps)} steps",
                state="complete",
                expanded=False,
            )

    if result is not None:
        st.session_state[RESULT] = {"run": result, "budget": run_budget}
        # Explain first, then get out of the way: the moment a run finishes the
        # reader wants the trace, not the README. Set above the st.tabs call below.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.rerun()

stored = st.session_state.get(RESULT)
result = stored["run"] if stored else None
run_budget = stored["budget"] if stored else None

with graph_slot:
    graph_map(GRAPH, result.visited if result is not None else None)

with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)

        rows = agent.custody_table(result.steps)
        if rows:
            st.subheader("Custody", anchor=False)
            st.dataframe(
                rows,
                hide_index=True,
                column_order=["holder", "from_step", "to_step", "steps", "tool_calls", "tokens", "handoff_reason"],
                column_config={
                    "holder": st.column_config.TextColumn("Held by", width="small"),
                    "from_step": st.column_config.NumberColumn("From step", width="small"),
                    "to_step": st.column_config.NumberColumn("To step", width="small"),
                    "steps": st.column_config.NumberColumn("Steps", width="small"),
                    "tool_calls": st.column_config.NumberColumn("Tool calls", width="small"),
                    "tokens": st.column_config.NumberColumn("Tokens", width="small"),
                    "handoff_reason": st.column_config.TextColumn("Handed off because"),
                },
                width="stretch",
            )
            st.caption(agent.contrast_sentence(rows))

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if force_no_answer:
        st.caption("Every peer was told it could never answer directly for this run.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
