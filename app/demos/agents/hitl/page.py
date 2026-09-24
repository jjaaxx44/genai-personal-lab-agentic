"""The HITL page: the standard demo layout, plus the gate panel wherever a run --
fresh or restored after a refresh or a restart -- is waiting for a decision.

Submitting the task box starts a brand-new run, same as every non-memory demo.
What's different here is that a run can come back `needs_human`: the page then
shows the gate instead of (or alongside, on the trace) the trajectory so far, and
the reader's decision resumes the same run rather than starting another one.
"""

import json
from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.types import AgentRun
from core.ui import (
    budget_strip,
    clear_data_button,
    demo_header,
    final_output,
    gate_panel,
    graph_map,
    provider_note,
    readme_and_trace_tabs,
    sidebar_budget_controls,
    task_row,
    trajectory_track,
)

# Absolute, not relative: Streamlit runs a page as a script, so `from . import`
# has no parent package to resolve against.
from demos.agents.hitl import agent
from demos.agents.hitl.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE, TOOL_NAMES

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"
GATE_CALL_ID_KEY = f"{DEMO}_gate_call_id"
GATE_EDIT_KEY = f"{DEMO}_gate_edit"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())

st.sidebar.subheader("Gated tools")
default_gated = [t.strip() for t in settings.hitl_gated_tools.split(",") if t.strip()]
gated_tools = st.sidebar.multiselect(
    "Paused for a person",
    list(TOOL_NAMES),
    default=default_gated,
    key=f"{DEMO}_gated_tools",
    help="Every call to one of these tools stops the run until a person decides.",
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.session_state.pop(GATE_CALL_ID_KEY, None)
    st.session_state.pop(GATE_EDIT_KEY, None)
    st.rerun()

# --- restore a paused run on load -----------------------------------------------

if RESULT not in st.session_state:
    restored = agent.pending()
    if restored is not None:
        st.session_state[RESULT] = {"run": restored["run"], "budget": restored["budget"]}
        st.info(
            f"Restored a run that was waiting for a decision (`{restored['run'].run_id}`) "
            "-- it survived the refresh/restart."
        )

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Try a rejection on preset 1: *\"File it as supply-risks-2026Q3.md, and name who "
    "owns each mitigation -- say 'unassigned' if the policy doesn't.\"* The agent's "
    "next proposal should use the new name and the extra content, not repeat the call."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task that may need a write, in a sentence")
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
            result = agent.run(task, run_budget, on_step=on_step, gated_tools=gated_tools)
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
        # budget_for() reads the checkpoint's own budget field rather than trusting
        # run_budget as tracked here: it's the same object run() charged against, so
        # it happens to already be correct for a completed run, but reading it back
        # is what keeps this identical to the post-resume path below, where it isn't.
        display_budget = agent.budget_for(result.thread_id) or run_budget
        st.session_state[RESULT] = {"run": result, "budget": display_budget}
        st.session_state.pop(GATE_CALL_ID_KEY, None)
        st.session_state.pop(GATE_EDIT_KEY, None)
        # Explain first, then get out of the way: the moment a run finishes (or
        # pauses) the reader wants the trace, not the README. Set above st.tabs.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.rerun()

stored = st.session_state.get(RESULT)
result: AgentRun | None = stored["run"] if stored else None
run_budget: Budget | None = stored["budget"] if stored else None

with graph_slot:
    graph_map(GRAPH, result.visited if result is not None else None)

with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)

        if result.status == "needs_human":
            pending_now = agent.pending(thread_id=result.thread_id)
            payload = pending_now["payload"] if pending_now else None
            if payload is None:
                st.warning(
                    "This run says it's waiting for a decision, but the checkpoint no "
                    "longer has one -- it may have been cleared."
                )
            else:
                call_id = payload.get("call_id")
                if st.session_state.get(GATE_CALL_ID_KEY) != call_id:
                    st.session_state[GATE_CALL_ID_KEY] = call_id
                    st.session_state[GATE_EDIT_KEY] = json.dumps(payload.get("args", {}), indent=2)

                decision = gate_panel(
                    DEMO,
                    proposal={"tool": payload.get("tool"), "args": payload.get("args")},
                    prompt=payload.get("why") or "The agent proposed this action.",
                )
                if decision is not None:
                    with st.status("Resuming...", expanded=True) as status:

                        def on_step(agent_label: str, step: Any) -> None:
                            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
                            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

                        try:
                            resumed = agent.resume(result.thread_id, decision, on_step=on_step)
                        except Exception as exc:
                            status.update(label="The run failed.", state="error")
                            st.error(f"The run could not finish: {type(exc).__name__}: {exc}")
                            resumed = None
                        else:
                            status.update(
                                label=f"{resumed.status.replace('_', ' ')} · {len(resumed.steps)} steps",
                                state="complete",
                                expanded=False,
                            )

                    if resumed is not None:
                        display_budget = agent.budget_for(resumed.thread_id) or run_budget
                        st.session_state[RESULT] = {"run": resumed, "budget": display_budget}
                        st.session_state.pop(GATE_CALL_ID_KEY, None)
                        st.session_state.pop(GATE_EDIT_KEY, None)
                        st.session_state.pop(f"{DEMO}_gate_edit_input", None)
                        st.session_state.pop(f"{DEMO}_gate_reason", None)
                        st.session_state[f"{DEMO}_tabs"] = "Trace"
                        st.rerun()

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`. Checkpoints live in `{DEMO}_checkpoints`.")


readme_and_trace_tabs(DEMO, README, trace)
