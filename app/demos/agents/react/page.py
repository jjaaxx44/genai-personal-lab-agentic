"""The ReAct page: the prompt on the left, what was parsed out of it on the right.

The standard demo page from `DESIGN.md` -- header, graph map, task row, budget
strip, trajectory, output, detail tabs -- with one addition this demo needs. The
raw prompt is shown beside the trajectory, growing by one Thought / Action /
Observation block per step, because the claim worth proving here is that the
agent *is* that string and a `while` loop. Reading the two side by side is the
whole argument.
"""

from pathlib import Path

import streamlit as st

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
from demos.agents.react import agent
from demos.agents.react.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"
SPENT = f"{DEMO}_spent"
PROMPTS = f"{DEMO}_prompts"
REPLIES = f"{DEMO}_replies"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()
budget = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())

st.sidebar.subheader("Settings")
st.sidebar.caption(f"Tools: {', '.join(agent.TOOL_NAMES)}")
inject_malformed = st.sidebar.toggle(
    "Break the first reply",
    key=f"{DEMO}_fault",
    help=(
        "Substitutes prose with no Action line for the model's first reply, so the "
        "parser fails and the loop re-prompts. The model is not called for that step."
    ),
)

if clear_data_button(DEMO):
    for key in (RESULT, SPENT, PROMPTS, REPLIES):
        st.session_state.pop(key, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)

graph_slot = st.container()
task = task_row(DEMO, PRESETS, placeholder="A task to work through, in a sentence")
run_slot = st.container()
strip_slot = st.container()
body_slot = st.container()

if task:
    prompts: list[str] = []
    replies: list[str] = []
    with run_slot, st.status("Running the loop...", expanded=True) as status:

        def on_step(step) -> None:
            """Named sub-steps as they land -- never an anonymous spinner (DESIGN.md)."""
            label = f"`{step.index:02d}` {step.kind}"
            status.write(f"{label} **{step.tool}**" if step.tool else label)

        try:
            result = agent.run(
                task,
                budget,
                on_step=on_step,
                prompt_log=prompts,
                reply_log=replies,
                inject_malformed=inject_malformed,
            )
        except Exception as exc:  # nothing below core is allowed to reach the reader
            status.update(label="The run failed.", state="error")
            st.error(f"The run could not finish: {type(exc).__name__}: {exc}")
            result = None
        else:
            status.update(label=f"{result.status.replace('_', ' ')} · {len(result.steps)} steps",
                          state="complete", expanded=False)

    if result is not None:
        st.session_state[RESULT] = result
        st.session_state[SPENT] = budget
        st.session_state[PROMPTS] = prompts
        st.session_state[REPLIES] = replies
        # Explain first, then get out of the way: the moment a run finishes the
        # reader wants the trace, not the README. Set above the st.tabs call below.
        st.session_state[f"{DEMO}_tabs"] = "Trace"

result = st.session_state.get(RESULT)
prompts = st.session_state.get(PROMPTS, [])
replies = st.session_state.get(REPLIES, [])

with graph_slot:
    graph_map(GRAPH, result.visited if result else None)

budget_strip(st.session_state.get(SPENT, budget), container=strip_slot)

with body_slot:
    left, right = st.columns(2, gap="medium")

    with left:
        st.subheader("The prompt", anchor=False)
        if not prompts:
            st.caption(
                "The prompt appears here. It starts as the task plus the tools written "
                "out as text, and grows by one Thought / Action / Observation block per "
                "step — there is nothing else holding the agent's memory."
            )
        else:
            which = st.select_slider(
                "LLM call",
                options=list(range(1, len(prompts) + 1)),
                value=len(prompts),
                key=f"{DEMO}_prompt_pick",
                help="Every call sends the whole thing again. Slide back to watch it grow.",
            )
            shown = prompts[which - 1]
            st.caption(f"Call {which} of {len(prompts)} · {len(shown):,} characters")
            st.code(shown, language="text", height=520, wrap_lines=True)

    with right:
        st.subheader("The trajectory", anchor=False)
        st.caption("What the hand-written parser made of each reply.")
        trajectory_track(result.steps if result else [])

if result:
    final_output(result)


def trace() -> None:
    """Raw replies and the stored run record -- what the page above renders from."""
    if not result:
        st.caption("Nothing has run yet.")
        return
    st.caption(
        "The model's replies exactly as they arrived, before parsing. Everything on the "
        "track was read out of these by `parse_reply()`."
    )
    for i, reply in enumerate(replies, start=1):
        with st.expander(f"Reply {i} · {len(reply):,} characters"):
            st.code(reply, language="text", wrap_lines=True)
    st.caption(f"Run record `{result.run_id}` in `{DEMO}_runs`.")
    with st.expander("The stored run record"):
        st.json(result.model_dump())


readme_and_trace_tabs(DEMO, README, trace)
