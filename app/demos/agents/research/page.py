"""The Deep research page: the standard demo layout, plus the panels that make
"every claim links to a step" checkable by eye -- sub-questions with their
answered/gap status, the notes table, the source registry against the "at least
three" target, the brief itself with its citation markers, a claim-to-step table,
and the recovery line. Every one of those panels is reconstructed straight from
`AgentRun.steps` by the functions `agent.py` builds for exactly that (see its
"reconstructing the panels" section) -- nothing demo-specific is stored anywhere
but the trajectory itself.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.tools import is_error
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
from demos.agents.research import agent
from demos.agents.research.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

STATUS_LABELS = {"answered": "✓ answered", "gap": "○ gap"}

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

# This demo's own step-budget default, not the shared AGENT_MAX_STEPS: decompose
# (1) + each sub-question's researcher turns (up to RESEARCH_MAX_STEPS_PER_QUESTION
# each) + synthesise (1) runs well past the shared default for anything but a
# single-sub-question task.
_shared_defaults = settings.budget_defaults().to_budget()
_defaults = Budget(
    max_steps=settings.research_max_steps,
    max_tokens=settings.research_max_tokens,
    deadline_s=_shared_defaults.deadline_s,
)
budget_template = sidebar_budget_controls(DEMO, _defaults)
st.sidebar.caption(
    f"Each model call is one step: decompose, each researcher turn (max "
    f"{settings.research_max_steps_per_question} per sub-question), and synthesise. "
    "A sub-question that hits its cap moves on; only the step count above stops the run."
)

st.sidebar.subheader("Settings")
inject_fault = st.sidebar.toggle(
    "Break the first search",
    key=f"{DEMO}_fault",
    help=(
        "Sends the first search (search_corpus or web_search) to a tool that always "
        "fails, so you can watch the researcher rephrase, switch tool, or admit a gap."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A research question to break down")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Decomposing and researching...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(
                task, run_budget, on_step=on_step, break_first_search=inject_fault
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

        subquestions = agent.subquestions_from_steps(result)
        notes = agent.notes_from_steps(result)
        registry = agent.sources_from_steps(result)
        summary, verified_claims, distinct_sources = agent.brief_from_steps(result)

        if subquestions:
            st.subheader("Sub-questions", anchor=False)
            for row in subquestions:
                label = STATUS_LABELS[row["status"]]
                st.caption(f"{label} · {row['note_count']} note(s)")
                st.write(f"{row['index']}. {row['text']}")

        if notes:
            st.subheader("Notes", anchor=False)
            st.dataframe(
                [
                    {
                        "id": n["id"],
                        "sub-question": n["subquestion"],
                        "source": n["source"],
                        "claim": n["claim"],
                        "quote": n["quote"],
                        "found on step": n["step_index"],
                        "recorded on step": n["note_step_index"],
                    }
                    for n in notes
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Sources", anchor=False)
        target_met = "≥ 3 target met" if distinct_sources >= 3 else "below the ≥ 3 target"
        st.caption(f"{distinct_sources} distinct source(s) cited in the brief — {target_met}.")
        if registry:
            st.dataframe(
                [
                    {"source": key, "title": entry["title"], "first found on step": entry["first_step_index"]}
                    for key, entry in registry.items()
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No search returned a source this run.")

        if summary or verified_claims:
            st.subheader("Brief", anchor=False)
            st.write(summary)
            for i, claim in enumerate(verified_claims, start=1):
                marker = "".join(f"`[{nid}]`" for nid in claim["note_ids"])
                line = f"{i}. {claim['text']} {marker}"
                if claim["unsupported"]:
                    st.warning(line + " — unsupported: cites no known note.")
                else:
                    st.write(line)

            table = agent.citations_table(verified_claims, notes)
            if table:
                with st.expander("Citations → steps"):
                    st.dataframe(table, width="stretch", hide_index=True)

        recovered = agent.recoveries(result.steps)
        failed_searches = sum(
            1
            for s in result.steps
            if s.kind == "observe" and s.tool in agent.SEARCH_TOOL_NAMES and is_error(s.text)
        )
        st.caption(f"{failed_searches} failed search(es), {recovered} recovered.")

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
        st.caption("Fault injection was on: the first search call was diverted to `boom`.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace, GRAPH)
