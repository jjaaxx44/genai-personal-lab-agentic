"""The Text-to-SQL analyst page: the standard demo layout, plus a query panel
between the trajectory track and the final output -- one SQL statement, its
result table and (where the shape suits) a chart, for every successful `run_sql`
call in the run, so the reader can check the agent's numbers without re-running
anything themselves.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.tools import is_error
from core.types import AgentRun, Step
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
from demos.agents.sql_analyst import agent
from demos.agents.sql_analyst.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    f"SQL row cap: {settings.sql_row_limit} · Max SQL retries: "
    f"{settings.sql_analyst_max_sql_retries}"
)

st.sidebar.subheader("Settings")
skip_schema = st.sidebar.toggle(
    "Skip the schema tool",
    key=f"{DEMO}_skip_schema",
    help=(
        "Removes describe_schema from this run's tools, so a wrong column or table "
        "name can only be learned from the SQL error SQLite returns, rather than "
        "avoided by inspecting the schema first."
    ),
)

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A question the sales database can answer")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Inspecting, querying, reading rows...", expanded=True) as status:

        def on_step(step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(task, run_budget, on_step=on_step, skip_schema=skip_schema)
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


def _render_chart(spec: dict[str, Any]) -> None:
    """Theme colours only -- `st.bar_chart`/`st.line_chart` read the app's own
    chart palette, so no colour literal is written here."""
    if spec["kind"] == "bar":
        st.bar_chart(spec["df"], x=spec["x"], y=spec["y"], horizontal=True)
    else:
        st.line_chart(spec["df"], x=spec["x"], y=spec["y"])


def _query_panel(steps: list[Step]) -> None:
    """The SQL, the table and (where it suits) a chart for every successful
    run_sql call, in order. Built from the observation text already on the
    trajectory -- parse_rows never re-runs the query."""
    successful = [
        step
        for step in steps
        if step.kind == "observe" and step.tool == "run_sql" and not is_error(step.text)
    ]
    if not successful:
        return

    parsed = [(step, agent.parse_rows(step.text)) for step in successful]
    specs = [(step, df, agent.chart_spec(df)) for step, df in parsed]
    last_chartable = max((i for i, (_, _, spec) in enumerate(specs) if spec), default=None)

    st.subheader("Queries", anchor=False)
    for i, (step, df, spec) in enumerate(specs):
        sql = (step.args or {}).get("sql", "")
        st.code(sql, language="sql")

        if df is None:
            st.caption("Could not be parsed back into a table.")
            continue
        st.dataframe(df, width="stretch", hide_index=True)

        if spec is None:
            st.caption(agent.chart_skip_reason(df))
            continue
        if i == last_chartable:
            _render_chart(spec)
        else:
            with st.expander("Show chart"):
                _render_chart(spec)


with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)
        _query_panel(result.steps)
        final_output(result)
    else:
        budget_strip(budget_template)
        trajectory_track([])


def trace() -> None:
    """The stored run record -- what the page above renders from."""
    if result is None:
        st.caption("Nothing has run yet.")
        return
    if skip_schema:
        st.caption("The schema tool was removed from this run's tools.")
    with st.expander(f"run `{result.run_id}`"):
        st.json(result.model_dump())
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
