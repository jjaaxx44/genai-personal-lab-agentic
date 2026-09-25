"""The sub-agent delegation page: the standard demo layout, plus two sidebar
toggles that make the technique's two claims checkable rather than merely
described -- "isolate sub-agent context" controls whether the parent can
delegate at all, and "force the sub-agent to fail" makes the recovery path
reproducible on demand instead of hoping a task happens to trigger it.

The page remembers the last run of each mode (isolated / shared) across toggle
flips, so a reader who tries both on the same task can read the token contrast
off the page without having to hold the first number in their head.
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
from demos.agents.subagents import agent
from demos.agents.subagents.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE, TOOL_NAMES

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"
LAST_ISOLATED = f"{DEMO}_last_isolated"
LAST_SHARED = f"{DEMO}_last_shared"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())

st.sidebar.subheader("Delegation")
isolated = st.sidebar.toggle(
    "Isolate sub-agent context",
    value=True,
    key=f"{DEMO}_isolated",
    help=(
        "On: the parent may hand a self-contained subtask to a sub-agent and gets "
        "back only its final reply. Off: delegate_subagent isn't offered at all -- "
        "the parent does the whole task itself in one shared context, so every "
        "tool result stays in its own history for the rest of the run."
    ),
)
if isolated:
    force_failure = st.sidebar.toggle(
        "Force the sub-agent to fail",
        value=False,
        key=f"{DEMO}_force_failure",
        help=(
            "Clamps the sub-agent's own step cap to zero, so a delegated call "
            "stops before its first LLM call -- shows the parent recovering from "
            "a failed delegation instead of the run crashing."
        ),
    )
else:
    force_failure = False
    st.sidebar.caption("Hidden in shared-context mode -- there's no delegated call to fail.")

st.sidebar.caption(
    f"Sub-agent step cap: {settings.subagent_max_steps}. Tools ({', '.join(TOOL_NAMES)}) are "
    "the same set the parent already has -- delegating trades context, not capability."
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.session_state.pop(LAST_ISOLATED, None)
    st.session_state.pop(LAST_SHARED, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Each preset pairs a corpus fact with an independent SQL fact -- a natural "
    "split for the parent to hand one half off. Run a preset with isolation on, "
    "then flip the toggle and run it again to compare the token counts below."
)
st.caption(
    "The parent and every sub-agent here share the exact same tools -- "
    "delegating only keeps a piece of work out of the parent's own context, it "
    "never hands the sub-agent a capability the parent lacked. Giving each "
    "sub-agent its own dedicated tools, chosen by a router, is Step 9 "
    "(Supervisor–worker), not this one."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task with a self-contained piece worth delegating")
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
            tag = "subagent" if agent_label == "subagent" else "parent"
            marker = f"`{step.index:02d}` [{tag}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, isolated=isolated, force_failure=force_failure)
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
        stored = {"run": result, "budget": run_budget}
        st.session_state[RESULT] = stored
        st.session_state[LAST_ISOLATED if isolated else LAST_SHARED] = stored
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
        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])

    last_isolated = st.session_state.get(LAST_ISOLATED)
    last_shared = st.session_state.get(LAST_SHARED)
    if last_isolated and last_shared:
        iso_run, shared_run = last_isolated["run"], last_shared["run"]
        st.caption(
            f"Isolated: {iso_run.tokens:,} tokens, {len(iso_run.steps)} steps · "
            f"Shared: {shared_run.tokens:,} tokens, {len(shared_run.steps)} steps"
        )


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
