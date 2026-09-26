"""The Supervisor-worker page: the standard demo layout, plus a routing table
between the trajectory track and the final output -- the claim this demo makes
(a supervisor routes each turn to a named specialist, for a stated reason, and
decides when to stop) is only checkable if every hop's target and reason sit
next to each other, not scattered across the track.
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
from demos.agents.supervisor import agent
from demos.agents.supervisor.agent import (
    ANALYST_TOOLS,
    DEMO,
    GRAPH,
    NAME,
    PRESETS,
    RESEARCHER_TOOLS,
    SENTENCE,
)

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    f"Hop cap: {settings.supervisor_max_hops}. A hop costs at least two steps "
    "(supervisor + worker), so the step cap is usually hit first."
)

st.sidebar.subheader("Workers")
st.sidebar.caption(f"researcher: {', '.join(RESEARCHER_TOOLS)}")
st.sidebar.caption(f"analyst: {', '.join(ANALYST_TOOLS)}")
st.sidebar.caption("writer: no tools -- drafts the answer from the reports")

st.sidebar.subheader("Settings")
force_no_finish = st.sidebar.toggle(
    "Supervisor can't declare done",
    key=f"{DEMO}_force_no_finish",
    help=(
        "Removes 'finish' from the supervisor's options, so it keeps handing off "
        "until the step cap stops the loop."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Workers report only to the supervisor and never talk to each other. "
    "Step 10 (Swarm) is the opposite: no supervisor, workers hand off directly."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task with a two-specialist split")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Routing...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, force_no_finish=force_no_finish)
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

        rows = agent.routing_table(result.steps)
        if rows:
            st.subheader("Routing", anchor=False)
            st.dataframe(
                rows,
                hide_index=True,
                column_order=["hop", "next", "reason", "instruction"],
                column_config={
                    "hop": st.column_config.NumberColumn("Hop", width="small"),
                    "next": st.column_config.TextColumn("Routed to", width="small"),
                    "reason": st.column_config.TextColumn("Reason"),
                    "instruction": st.column_config.TextColumn("Instruction"),
                },
                width="stretch",
            )

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if force_no_finish:
        st.caption("'Finish' was removed from the supervisor's options for this run.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace, GRAPH)
