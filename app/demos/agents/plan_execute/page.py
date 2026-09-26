"""The Plan-and-Execute page: the standard demo layout, plus a plan-versions
panel between the trajectory track and the final output -- because the claim
this demo makes (the plan gets revised, and the answer follows the revision) is
only visible if the reader can see every version of the plan side by side.
"""

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
    graph_map,
    provider_note,
    readme_and_trace_tabs,
    sidebar_budget_controls,
    task_row,
    trajectory_track,
)

# Absolute, not relative: Streamlit runs a page as a script, so `from . import`
# has no parent package to resolve against.
from demos.agents.plan_execute import agent
from demos.agents.plan_execute.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

STATUS_LABELS = {"executed": "✓ executed", "pending": "○ pending", "dropped": "✗ dropped"}

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

# This demo's own step-budget default, not the shared AGENT_MAX_STEPS: a replan
# costs one model call per executed step (plan + N x (execute + replan) + respond),
# which runs well past the shared default on anything but a one-step task.
_shared_defaults = settings.budget_defaults().to_budget()
_defaults = Budget(
    max_steps=settings.plan_execute_max_steps,
    max_tokens=_shared_defaults.max_tokens,
    deadline_s=_shared_defaults.deadline_s,
)
budget_template = sidebar_budget_controls(DEMO, _defaults)
st.sidebar.caption(
    "Every model call is one step: the plan, each executed step, each replan "
    "check, and the answer. One revision easily spends 8-12, so a low cap often "
    "stops the run early. The footer counts trajectory rows, not steps spent."
)

st.sidebar.subheader("Settings")
inject_fault = st.sidebar.toggle(
    "Break the first tool call",
    key=f"{DEMO}_fault",
    help=(
        "Sends the first tool call to a tool that always fails, so the "
        "replanner has a real error to react to."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task to work through, in a sentence")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Planning and executing...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, inject_fault=inject_fault)
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
        # Explain first, then get out of the way: the moment a run finishes the reader
        # wants the trace, not the README. Set above the st.tabs call below.
        st.session_state[f"{DEMO}_tabs"] = "Trace"

stored = st.session_state.get(RESULT)
result: AgentRun | None = stored["run"] if stored else None
run_budget: Budget | None = stored["budget"] if stored else None

with graph_slot:
    graph_map(GRAPH, result.visited if result is not None else None)


def _executed_step_texts(run: AgentRun) -> list[dict[str, Any]]:
    """Plan step text and trajectory row for each step actually executed, in
    execution order. `plan_step` is recorded once per execution on the executor's
    `think` row (see `execute_node` in agent.py) -- the shared `Step` shape from
    `core/types.py` already carries everything the panel needs, so nothing
    plan-specific is added to `AgentRun` itself."""
    rows = []
    pending_step_text = None
    for step in run.steps:
        if step.agent != "executor":
            continue
        if step.kind == "think" and step.args and "plan_step" in step.args:
            pending_step_text = step.args["plan_step"]
        elif step.kind == "observe" and pending_step_text is not None:
            rows.append({"step": pending_step_text, "row_index": step.index})
            pending_step_text = None
    return rows


def _plan_versions_from_steps(run: AgentRun) -> list[dict[str, Any]]:
    """Rebuilds the plan versions from the saved `decide` rows."""
    return [
        {
            "version": step.args["plan_version"],
            "steps": step.args["plan"],
            "reason": step.args.get("reason", step.text),
        }
        for step in run.steps
        if step.agent in ("planner", "replanner") and step.args and "plan_version" in step.args
    ]


with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)

        plan_versions = _plan_versions_from_steps(result)
        if plan_versions:
            st.subheader("Plan versions", anchor=False)
            executed = _executed_step_texts(result)
            columns = agent.plan_diff(plan_versions, executed)
            cols = st.columns(len(columns), gap="small")
            for col, version in zip(cols, columns):
                with col:
                    st.markdown(f"**v{version['version']}**")
                    st.caption(version["reason"])
                    for row in version["rows"]:
                        label = STATUS_LABELS[row["status"]]
                        if row["status"] == "executed":
                            label += f" (row {row['row_index']:02d})"
                        if row["added"]:
                            label += " · added"
                        st.caption(label)
                        st.write(row["step"])
            st.caption(agent.plan_summary(plan_versions, executed))

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if inject_fault:
        st.caption("Fault injection was on: the first tool call was diverted to `boom`.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace, GRAPH)
