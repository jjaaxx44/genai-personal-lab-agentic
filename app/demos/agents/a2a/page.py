"""The A2A page: the standard demo layout, plus the raw protocol log beside the
trajectory track -- the protocol is what this demo is about, so it gets equal
billing with the track rather than living in a collapsed expander. One status
chip strip and one (unclaimed) budget line per remote agent sit above the two,
because "the client decided to delegate" and "the remote actually did the work"
are two different claims and the page should let a reader check both, per agent.
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
from demos.agents.a2a import agent
from demos.agents.a2a.agent import DEMO, GRAPH, NAME, PRESETS, REMOTE_KEYS, REMOTE_NAMES, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    f"Each remote's own step cap: {settings.a2a_remote_max_steps}. Every remote agent has "
    "its own Budget, built from its own config -- never the client's -- and its spend is "
    "shown but never charged to the client's budget above."
)

st.sidebar.subheader("Remote agents")
offline_agents = st.sidebar.multiselect(
    "Offline",
    REMOTE_KEYS,
    format_func=lambda k: REMOTE_NAMES[k],
    key=f"{DEMO}_offline_agents",
    help=(
        "Every request to these hosts fails at the transport layer, as if they were "
        "down. Their cards are never read, so the coordinator doesn't know what they "
        "could have done -- it has to route around them or say what it couldn't answer."
    ),
)
failing_agent = st.sidebar.selectbox(
    "Fails mid-task",
    [None, *REMOTE_KEYS],
    format_func=lambda k: "Nobody" if k is None else REMOTE_NAMES[k],
    key=f"{DEMO}_failing_agent",
    help=(
        "That agent's executor raises after its first tool call. It catches its own "
        "exception and publishes a failed status over the protocol rather than "
        "crashing the connection -- so the reader sees a real failed task arrive, "
        "and the coordinator still answers from the other agents."
    ),
)
if failing_agent is not None and failing_agent in offline_agents:
    st.sidebar.caption(f"{REMOTE_NAMES[failing_agent]} is offline, so it is never reached to fail mid-task.")

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "The client (\"coordinator\") and three remote agents -- Corpus researcher, SQL "
    "analyst, Web researcher -- share no memory and no Python objects. The coordinator "
    "knows only their addresses; what each can do it learns from the card it fetches, "
    "and everything that crosses between them is an A2A protocol message. Contrast with MCP, where an agent calls a *tool*, and with "
    "Sub-agent delegation (Step 8), where a parent and its sub-agent share one "
    "runtime and one process; here the two agents could be two different processes "
    "on two different hosts, built by two different teams, and neither would know."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A task one, several or none of the cards cover")
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
            result = agent.run(
                task,
                run_budget,
                on_step=on_step,
                offline_agents=offline_agents,
                failing_agent=failing_agent,
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

        chips = agent.status_chips(result.steps)
        spends = agent.remote_spend(result.steps)
        if chips:
            st.subheader("Remote tasks", anchor=False)
        for key, states in chips.items():
            line = f"**{REMOTE_NAMES[key]}**: " + " → ".join(f"`{c}`" for c in states)
            spend = spends.get(key)
            if spend:
                line += (
                    f"  \n:gray[own budget, not charged above: {int(spend.get('tokens', 0)):,} tokens · "
                    f"{int(spend.get('llm_calls', 0))} LLM call(s) · {int(spend.get('tool_calls', 0))} tool call(s) · "
                    f"{int(spend.get('steps_used', 0))}/{settings.a2a_remote_max_steps} step(s)]"
                )
            st.markdown(line)

        track_col, protocol_col = st.columns(2, gap="medium")
        with track_col:
            st.subheader("Trajectory", anchor=False)
            trajectory_track(result.steps)
        with protocol_col:
            st.subheader("Protocol log", anchor=False)
            entries = agent.protocol_log(result.steps)
            if not entries:
                st.caption("No exchange was recorded for this run.")
            for i, entry in enumerate(entries):
                direction = entry.get("direction", "?")
                kind = entry.get("kind", "")
                with st.container(border=True):
                    if direction == "request":
                        st.markdown(f"`{i:02d}` **-> {entry.get('method')}** `{entry.get('path')}`")
                        if entry.get("body"):
                            with st.expander("Request body"):
                                st.json(entry["body"])
                    elif kind == "sse":
                        st.markdown(f"`{i:02d}` **<- SSE**, {len(entry.get('frames', []))} frame(s)")
                        for j, frame in enumerate(entry.get("frames", [])):
                            with st.expander(f"Frame {j}"):
                                st.json(frame)
                    elif kind == "error":
                        st.markdown(f"`{i:02d}` **<- transport error**")
                        st.error(entry.get("error", ""))
                    else:
                        st.markdown(f"`{i:02d}` **<- {entry.get('status')}** {kind}")
                        if entry.get("body"):
                            with st.expander("Response body"):
                                st.json(entry["body"])

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
    st.caption(f"Stored as a record in `{DEMO}_runs`.")


readme_and_trace_tabs(DEMO, README, trace)
