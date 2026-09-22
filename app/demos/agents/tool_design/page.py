"""The tool design page: the same task against two toolsets, side by side.

The standard demo page from `DESIGN.md`, split below the task row into two
columns -- one first draft, one revised -- because the claim worth proving here
is comparative: same task, same model, same budget, only the tool surface
differs. A comparison table underneath turns "one felt worse" into numbers.
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
from demos.agents.tool_design import agent, toolsets
from demos.agents.tool_design.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE, Side

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())

st.sidebar.subheader("Settings")
inject_fault = st.sidebar.toggle(
    "Break the first tool call",
    key=f"{DEMO}_fault",
    help=(
        "Diverts the first tool call on both sides to a tool that always raises, so the "
        "error each toolset actually shows the model can be compared side by side."
    ),
)
crowd = st.sidebar.toggle(
    "Crowd both toolboxes",
    key=f"{DEMO}_crowd",
    help=(
        "Adds six near-duplicate tools that do nothing to both toolsets, so what extra "
        "tool count does to selection can be seen on its own."
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
table_slot = st.container()

if task:
    expected = next(
        (preset["expected"] for preset in PRESETS if preset["task"] == task),
        list(toolsets.CAPABILITIES),
    )
    with run_slot, st.status("Running both toolsets...", expanded=True) as status:

        def on_step(label: str, step: Any) -> None:
            """Named sub-steps as they land, tagged by side -- never an anonymous spinner."""
            tag = "draft" if label == "first draft" else "revised"
            marker = f"`{step.index:02d}` [{tag}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            first, revised = agent.run_pair(
                task, budget_template, on_step=on_step, inject_fault=inject_fault, crowd=crowd
            )
        except Exception as exc:  # nothing below core is allowed to reach the reader
            status.update(label="The run failed.", state="error")
            st.error(f"The run could not finish: {type(exc).__name__}: {exc}")
            first = revised = None
        else:
            statuses = {first.run.status, revised.run.status}
            worst = (
                "failed"
                if "failed" in statuses
                else "stopped_on_budget" if "stopped_on_budget" in statuses else "completed"
            )
            total_steps = len(first.run.steps) + len(revised.run.steps)
            status.update(
                label=f"{worst.replace('_', ' ')} · {total_steps} steps across both sides",
                state="complete",
                expanded=False,
            )

    if first is not None:
        st.session_state[RESULT] = {"first": first, "revised": revised, "expected": expected}
        # Explain first, then get out of the way: the moment a run finishes the reader
        # wants the trace, not the README. Set above the st.tabs call below.
        st.session_state[f"{DEMO}_tabs"] = "Trace"

stored = st.session_state.get(RESULT)
first: Side | None = stored["first"] if stored else None
revised: Side | None = stored["revised"] if stored else None
expected: list[str] = stored["expected"] if stored else list(toolsets.CAPABILITIES)

with graph_slot:
    visited: list[str] = []
    for side in (first, revised):
        if side is not None:
            visited.extend(v for v in side.run.visited if v not in visited)
    graph_map(GRAPH, visited or None)


def _empty_budget() -> Budget:
    return Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )


def _render_side(container: Any, heading: str, tools: list, side: Side | None) -> None:
    with container:
        st.subheader(heading, anchor=False)
        with st.expander(f"Tool schemas sent to the model ({len(tools)})"):
            st.caption("The exact name, description and JSON schema `bind_tools()` builds.")
            for spec in toolsets.schema_summaries(tools):
                st.markdown(f"**`{spec['name']}`** — {spec['description']}")
                st.json(spec["schema"])

        budget_strip(side.budget if side is not None else _empty_budget())
        trajectory_track(side.run.steps if side is not None else [])
        if side is not None:
            final_output(side.run)


preview_first, preview_revised = toolsets.preview_tools(crowd=crowd)

with body_slot:
    col_first, col_revised = st.columns(2, gap="medium")
    _render_side(col_first, "First draft", preview_first, first)
    _render_side(col_revised, "Revised", preview_revised, revised)

if first is not None and revised is not None:
    with table_slot:
        st.subheader("Comparison", anchor=False)
        st.caption(
            "Off-task = a call to a tool outside this task's expected set. Recoveries = an "
            "error observation that was not the run's last step."
        )
        st.dataframe(
            agent.comparison_rows(first, revised, expected), width="stretch", hide_index=True
        )


def trace() -> None:
    """The stored run record for each side -- what the page above renders from."""
    if first is None or revised is None:
        st.caption("Nothing has run yet.")
        return
    if inject_fault:
        st.caption(
            "Fault injection was on: the first tool call on both sides was diverted to `boom`."
        )
    if crowd:
        st.caption("Crowding was on: six decoy tools were added to both toolsets.")
    for side in (first, revised):
        with st.expander(f"{side.label} · run `{side.run.run_id}`"):
            st.json(side.run.model_dump())
    st.caption(f"Both runs stored as separate records in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
