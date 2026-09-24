"""The Reflexion page: the standard demo layout, plus an attempts panel between the
trajectory track and the final output -- because the claim this demo makes (a
critique makes the next attempt better, or an unanswerable task keeps failing for
the same honest reason) is only visible if every attempt sits side by side with
its evaluation and the reflection that followed it.
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
from demos.agents.reflexion import agent
from demos.agents.reflexion.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

OUTCOME_LABELS = {"passed": "✓ passed", "retry": "↻ retrying", "gave_up": "✗ retry cap reached"}

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    f"Every model call spends one step: the actor's tool decision, its compose call "
    f"when a tool ran, the evaluator, and the reflector. Up to "
    f"`reflexion_max_attempts` ({settings.reflexion_max_attempts}) attempts run before "
    "an unresolved task gives up."
)

st.sidebar.subheader("Settings")
force_memory_only = st.sidebar.toggle(
    "Force the first attempt to answer from memory",
    key=f"{DEMO}_force_memory",
    help=(
        "No tools are bound on attempt 1, so it must answer from what the model "
        "already knows. Pairs with the first preset: the evaluator can then fail an "
        "ungrounded claim on attempt 1 and pass a grounded one on attempt 2."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task worth a second attempt, in a sentence")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Attempting, evaluating, reflecting...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, force_memory_only=force_memory_only)
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

with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)

        attempts = agent.attempt_summary(result.steps)
        if attempts:
            st.subheader("Attempts", anchor=False)
            cols = st.columns(len(attempts), gap="small")
            for col, entry in zip(cols, attempts):
                with col:
                    st.markdown(f"**Attempt {entry['attempt']}**")
                    st.caption(f"Tool: {entry['tool'] or 'none used'}")
                    st.write(entry["text"] or "(no attempt text recorded)")
                    if entry["outcome"]:
                        label = OUTCOME_LABELS.get(entry["outcome"], entry["outcome"])
                        st.caption(f"{label} · score {entry['score']}/5")
                    if entry["reflection"]:
                        st.caption("Reflection for the next attempt:")
                        st.write(entry["reflection"])

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if force_memory_only:
        st.caption("Attempt 1 had no tools bound: it had to answer from memory alone.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
