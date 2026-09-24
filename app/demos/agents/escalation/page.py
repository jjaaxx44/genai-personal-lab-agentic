"""The escalation page: the standard demo layout, plus a handoff panel wherever a
run -- fresh or restored after a refresh or a restart -- is waiting for a person to
answer its one question or close the case.

The handoff panel isn't `core.ui.gate_panel`: a gate asks approve/edit/reject on a
proposed action, and this asks a question. Nothing else in the app needs that shape
yet, so it lives here rather than in `core/` (CLAUDE.md rule 2).
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.types import AgentRun, HumanDecision
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
from demos.agents.escalation import agent
from demos.agents.escalation.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())

st.sidebar.subheader("Confidence threshold")
threshold = st.sidebar.slider(
    "Escalate below",
    0.0,
    1.0,
    settings.escalation_confidence_threshold,
    step=0.05,
    key=f"{DEMO}_threshold",
    help="The judge's confidence has to reach this before the draft is returned as the answer.",
)
st.sidebar.caption(
    f"Also escalates after {settings.escalation_max_tool_errors} tool call(s) fail "
    "in a row without recovering."
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
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
    "Preset 1 has no lamp unit price anywhere in the corpus, so it should escalate. "
    "Try answering with *\"£1,150 each, quoted by Kelbrook Scientific\"* -- the "
    "resumed run should land on £23,000 and the two-quote approval tier, both "
    "checkable against procurement-policy.md."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task that may be beyond what the evidence supports")
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
            result = agent.run(task, run_budget, on_step=on_step, threshold=threshold)
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
        display_budget = agent.budget_for(result.thread_id) or run_budget
        st.session_state[RESULT] = {"run": result, "budget": display_budget}
        # Explain first, then get out of the way: the moment a run finishes (or
        # escalates) the reader wants the trace, not the README. Set above st.tabs.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.rerun()

stored = st.session_state.get(RESULT)
result: AgentRun | None = stored["run"] if stored else None
run_budget: Budget | None = stored["budget"] if stored else None

with graph_slot:
    graph_map(GRAPH, result.visited if result is not None else None)


def _handoff_panel(demo: str, payload: dict[str, Any]) -> HumanDecision | None:
    """A rule across the track, the packet in full, then an answer box with two
    ways forward. Returns the decision on the run it's submitted, else None --
    same shape as `core.ui.gate_panel`, but for a question rather than a verdict."""
    st.divider()
    st.subheader("Handed to a person", anchor=False)
    st.write(payload.get("trigger") or "The agent could not complete this task on its own.")

    with st.container(border=True):
        st.markdown("**Asked**")
        st.write(payload.get("asked", ""))
        st.markdown("**Tried**")
        for line in payload.get("tried") or []:
            st.write(f"- {line}")
        st.markdown("**Found**")
        for line in payload.get("found") or []:
            st.write(f"- {line}")
        st.markdown("**Needs from you**")
        st.info(payload.get("needs", ""))
        st.markdown("**Recommendation**")
        st.write(payload.get("recommendation", ""))
        assessment = payload.get("assessment")
        if assessment:
            st.caption(
                f"Judge: confidence {assessment.get('confidence', 0):.2f} -- "
                f"{assessment.get('reason', '')}"
            )

    with st.expander(f"Trajectory summary ({len((payload.get('trail') or '').splitlines())} steps)"):
        st.code(payload.get("trail") or "(none)", language=None)

    with st.form(key=f"{demo}_handoff_form"):
        answer = st.text_area("Your answer", key=f"{demo}_handoff_answer", placeholder="Answer the question above")
        col1, col2 = st.columns(2)
        resume_clicked = col1.form_submit_button("Answer and resume", type="primary")
        close_clicked = col2.form_submit_button("Close the case")

    if resume_clicked:
        if not answer.strip():
            st.warning("Enter an answer first, or use \"Close the case\" instead.")
            return None
        return HumanDecision(verdict="approve", reason=answer.strip())
    if close_clicked:
        return HumanDecision(verdict="reject", reason=answer.strip() or "Closed without an answer")
    return None


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
                decision = _handoff_panel(DEMO, payload)
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
                        st.session_state.pop(f"{DEMO}_handoff_answer", None)
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
