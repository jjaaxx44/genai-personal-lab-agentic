"""The Autonomous goal loop page: the standard demo layout, plus the spin tally
and objective-log panels between the trajectory track and the final output --
because the claim this demo makes (a loop with no external stopping signal spins,
and the budget is the real stop) is only visible if the reader can see the
repeats and near-duplicates piling up next to the objectives that produced them.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.embeddings import embed
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
from demos.agents.autonomous import agent
from demos.agents.autonomous.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

PROGRESS_LABELS = {"advanced": "↑ advanced", "no_change": "→ no change", "regressed": "↓ regressed"}

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

# This demo's own step-budget default, not the shared AGENT_MAX_STEPS: three
# model calls per objective (propose, act, critique) means the objective cap
# (autonomous_max_objectives x 3) needs headroom to actually be the thing that
# stops an open-ended run, rather than the shared default stopping it first.
_shared_defaults = settings.budget_defaults().to_budget()
_defaults = Budget(
    max_steps=settings.autonomous_max_steps,
    max_tokens=_shared_defaults.max_tokens,
    deadline_s=_shared_defaults.deadline_s,
)
budget_template = sidebar_budget_controls(DEMO, _defaults)
st.sidebar.caption(
    f"Every model call spends one step: propose, act, critique -- three per "
    f"objective. The demo's own cap, `autonomous_max_objectives` "
    f"({settings.autonomous_max_objectives}), stops an open-ended goal on its own "
    "objective count; lower the step cap above to see *that* cap named instead."
)

st.sidebar.subheader("Settings")
allow_web_search = st.sidebar.toggle(
    "Allow web search",
    key=f"{DEMO}_web_search",
    help=(
        "Off by default so the presets stay reproducible -- with web search on, "
        "what the agent finds (and how much it spins looking) can change run to run."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A goal, not a task -- what should be true when this is done?")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Setting objectives, acting, critiquing...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, allow_web_search=allow_web_search)
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

        rows = agent.objective_rows(result.steps)
        objective_texts = [r["objective"] for r in rows]
        dup_pairs = agent.near_duplicates(objective_texts, embed, settings.autonomous_dup_threshold)
        dup_by_b = {p["b"]: p for p in dup_pairs}
        repeat_pairs = agent.repeated_actions(result.steps)
        streak = agent.progress_streak(rows)

        st.subheader("Spin tally", anchor=False)
        st.caption(
            "Repeated actions: the same tool called with the same arguments twice. "
            "Near-duplicate objectives: cosine similarity above "
            f"`autonomous_dup_threshold` ({settings.autonomous_dup_threshold}). "
            "No-progress streak: consecutive critiques that found no advancement. "
            "None of these stop the run -- the budget does; this is what the spin "
            "looks like while it is still spending."
        )
        cols = st.columns(3)
        cols[0].metric("Repeated actions", len(repeat_pairs))
        cols[1].metric("Near-duplicate objectives", len(dup_pairs))
        cols[2].metric("No-progress streak", streak)

        pair_rows = [
            {"kind": "repeated action", "detail": f"step {p['repeat_index']:02d} repeats step {p['original_index']:02d} ({p['tool']})"}
            for p in repeat_pairs
        ] + [
            {"kind": "near-duplicate objective", "detail": f"objective {p['b']} ~ objective {p['a']} ({p['similarity']:.2f})"}
            for p in dup_pairs
        ]
        if pair_rows:
            st.dataframe(
                pair_rows,
                hide_index=True,
                column_config={
                    "kind": st.column_config.TextColumn("Kind", width="small"),
                    "detail": st.column_config.TextColumn("Detail"),
                },
                width="stretch",
            )
        else:
            st.caption("No repeats or near-duplicates were detected this run.")

        if rows:
            st.subheader("Objective log", anchor=False)
            table = []
            for r in rows:
                dup = dup_by_b.get(r["index"])
                table.append(
                    {
                        "#": r["index"],
                        "objective": r["objective"],
                        "tool": r["tool"] or "(none)",
                        "result": r["result_preview"],
                        "progress": PROGRESS_LABELS.get(r["progress"], r["progress"] or "—"),
                        "duplicate": f"of #{dup['a']} ({dup['similarity']:.2f})" if dup else "—",
                    }
                )
            st.dataframe(
                table,
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "objective": st.column_config.TextColumn("Objective"),
                    "tool": st.column_config.TextColumn("Tool", width="small"),
                    "result": st.column_config.TextColumn("Result"),
                    "progress": st.column_config.TextColumn("Progress", width="small"),
                    "duplicate": st.column_config.TextColumn("Duplicate?", width="small"),
                },
                width="stretch",
            )

        st.subheader("Stop", anchor=False)
        if result.status == "completed":
            st.write("The agent argued its own goal was met -- see the output below.")
        elif result.status == "stopped_on_budget":
            st.write(result.stop_reason or "A budget cap stopped the run.")
        elif result.status == "failed":
            st.write(result.stop_reason or "The run failed before it could stop on its own terms.")

        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])
        st.subheader("Spin tally", anchor=False)
        st.caption(
            "Once a run finishes, this panel counts repeated tool calls, "
            "near-duplicate objectives (by embedding similarity) and any streak of "
            "critiques that found no progress -- the loop's own record of starting "
            "to spin."
        )


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if allow_web_search:
        st.caption("Web search was allowed this run.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
