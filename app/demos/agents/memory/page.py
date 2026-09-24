"""The Agent memory page: the standard demo layout, plus a tiers panel between the
trajectory track and the final output -- because the claim this demo makes (a fact
survives to a *new* thread only because it made it into long-term memory, not because
the session happened to still be warm) is only visible if the three tiers sit side by
side, updating turn by turn.

Unlike every other demo page so far, submitting the task box doesn't start a fresh,
isolated run -- it sends one more turn in the *current* thread. The sidebar's thread
controls are what start a genuinely new one, or resume an earlier one after a fresh
browser session has no `thread_id` left in `st.session_state`.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from core.budget import Budget
from core.config import get_settings
from core.mongo import recent_runs
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
from demos.agents.memory import agent
from demos.agents.memory.agent import DEMO, GRAPH, NAME, PRESETS, SENTENCE

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"
THREAD_KEY = f"{DEMO}_thread_id"

settings = get_settings()

# --- sidebar -------------------------------------------------------------------

provider_note()

budget_template = sidebar_budget_controls(DEMO, settings.budget_defaults().to_budget())
st.sidebar.caption(
    "Two model calls spend a step each turn: the reply, and the extraction call that "
    "decides what -- if anything -- is worth writing to long-term memory. Recall and "
    "window trimming cost no steps; they're a local embedding search and a list "
    "operation, not model calls."
)

st.sidebar.subheader("Thread")
current_thread = st.session_state.get(THREAD_KEY)
if current_thread:
    st.sidebar.caption(f"Current: `{current_thread}`")
else:
    st.sidebar.caption("No thread yet -- the first message starts one.")

if st.sidebar.button("New thread", key=f"{DEMO}_new_thread"):
    st.session_state.pop(THREAD_KEY, None)
    st.session_state.pop(RESULT, None)
    st.rerun()

_last = recent_runs(DEMO, limit=1)
_last_thread = _last[0].get("thread_id") if _last else None
if _last_thread and _last_thread != current_thread:
    st.sidebar.caption(
        "A thread from an earlier session is still stored -- resuming proves the "
        "checkpoint survived a restart or a fresh browser session, not just this rerun."
    )
    if st.sidebar.button("Resume last thread", key=f"{DEMO}_resume_thread"):
        st.session_state[THREAD_KEY] = _last_thread
        st.session_state.pop(RESULT, None)
        st.rerun()

if clear_data_button(DEMO):
    st.session_state.pop(RESULT, None)
    st.session_state.pop(THREAD_KEY, None)
    st.rerun()

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Try the two-click test: send the first preset, click **New thread** in the "
    "sidebar, then ask something that needs what you just said. The recall step "
    "should find it even though the session that stated it is gone."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(
    DEMO, preset_labels, placeholder="Say something, or ask a follow-up", label="Message"
)
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Recalling, replying, extracting...", expanded=True) as status:

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(
                task, run_budget, on_step=on_step, thread_id=st.session_state.get(THREAD_KEY)
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
        st.session_state[THREAD_KEY] = result.thread_id
        # Explain first, then get out of the way: the moment a run finishes the reader
        # wants the trace, not the README. Set above the st.tabs call below.
        st.session_state[f"{DEMO}_tabs"] = "Trace"
        st.rerun()  # picks up the thread id in the sidebar caption immediately

stored = st.session_state.get(RESULT)
result: AgentRun | None = stored["run"] if stored else None
run_budget: Budget | None = stored["budget"] if stored else None

with graph_slot:
    graph_map(GRAPH, result.visited if result is not None else None)

with body_slot:
    if result is not None and run_budget is not None:
        budget_strip(run_budget)
        trajectory_track(result.steps)

        tiers = agent.turn_summary(result.steps)
        st.subheader("Memory tiers", anchor=False)
        short_col, session_col, long_col = st.columns(3, gap="medium")

        with short_col:
            st.markdown("**Short-term — message window**")
            st.caption(f"Cap: {settings.memory_window_messages} messages")
            for entry in tiers["window"]:
                st.caption(entry["role"])
                st.write(entry["text"])
            if tiers["dropped"]:
                with st.expander(f"Dropped this turn ({len(tiers['dropped'])})"):
                    for entry in tiers["dropped"]:
                        st.caption(entry["role"])
                        st.write(entry["text"])

        with session_col:
            st.markdown("**Session — checkpointed thread**")
            st.caption("New thread" if not tiers["resumed"] else "Resumed from an earlier turn")
            st.code(result.thread_id or "(none)", language=None)
            st.caption(
                f"~{len(tiers['window']) // 2} exchange(s) in the current window. Stored in "
                f"`{DEMO}_checkpoints` — survives a rerun and an app restart."
            )

        with long_col:
            st.markdown("**Long-term — recalled and stored facts**")
            if tiers["recalled"]:
                st.caption("Recalled this turn:")
                for hit in tiers["recalled"]:
                    st.write(f"{hit['text']} · score {hit['score']:.3f}")
            else:
                st.caption("Nothing recalled this turn.")
            if tiers["written"]:
                st.caption("Written this turn:")
                for fact in tiers["written"]:
                    st.write(fact["text"])

            memories = agent.list_memories()
            with st.expander(f"All stored memories ({len(memories)})"):
                if not memories:
                    st.caption("Nothing stored yet.")
                for doc in memories:
                    memory_id = str(doc["_id"])
                    row = st.container(horizontal=True, gap="small")
                    row.write(doc.get("text", ""))
                    if row.button("Delete", key=f"{DEMO}_delete_{memory_id}"):
                        agent.delete_memory(memory_id)
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
    st.caption(f"Stored as a record in `{DEMO}_runs`. Long-term memory lives in `{DEMO}_memory`.")


readme_and_trace_tabs(DEMO, README, trace)
