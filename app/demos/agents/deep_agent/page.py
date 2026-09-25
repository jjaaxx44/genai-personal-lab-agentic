"""The Deep agents page: the standard demo layout, plus a to-do panel (with a
version slider over every `write_todos` snapshot), a files panel over
`data/vfs/<run_id>/`, and a harness panel naming exactly what `deepagents`
0.7.15 gave each agent -- because the whole point of building on a library
here (rather than hand-writing the loop, as Steps 3 and 8 do) is that the
technique still has to stay visible through it.
"""

import threading
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

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
    sidebar_subagent_step_cap,
    task_row,
    trajectory_track,
)

# Absolute, not relative: Streamlit runs a page as a script, so `from . import`
# has no parent package to resolve against.
from demos.agents.deep_agent import agent
from demos.agents.deep_agent.agent import (
    ANALYST_TOOLS,
    ANALYST_SYSTEM_PROMPT,
    DEMO,
    GRAPH,
    MAIN_SYSTEM_PROMPT,
    NAME,
    PRESETS,
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_TOOLS,
    SAFE_FS_TOOLS,
    SENTENCE,
)

README = str(Path(__file__).with_name("README.md"))

RESULT = f"{DEMO}_result"

settings = get_settings()

HARNESS_TOOLS = {
    "main": [*SAFE_FS_TOOLS, "write_todos", "task"],
    "researcher": [*SAFE_FS_TOOLS, *RESEARCHER_TOOLS],
    "analyst": [*SAFE_FS_TOOLS, *ANALYST_TOOLS],
}

# --- sidebar -------------------------------------------------------------------

provider_note()
demo_defaults = Budget(
    max_steps=settings.deep_agent_max_steps,
    max_tokens=settings.deep_agent_max_tokens,
    deadline_s=settings.agent_deadline_s,
)
budget_template = sidebar_budget_controls(DEMO, demo_defaults)
subagent_max_steps = sidebar_subagent_step_cap(DEMO, settings.subagent_max_steps)
st.sidebar.caption(
    f"Sub-agent launches: capped at {settings.deep_agent_max_subagents} per run. Each "
    "launch that hits its own step cap above stops gracefully and reports back what it had."
)

pending_run_ids = agent.run_ids_for_clear()
if clear_data_button(DEMO):
    deleted = agent.clear_run_directories(pending_run_ids)
    st.session_state.pop(RESULT, None)
    st.session_state[f"{DEMO}_vfs_cleared"] = deleted
    st.rerun()
if f"{DEMO}_vfs_cleared" in st.session_state:
    deleted = st.session_state.pop(f"{DEMO}_vfs_cleared")
    st.sidebar.caption(f"Also removed {deleted} run director(y/ies) under data/vfs/.")

# --- page ----------------------------------------------------------------------

demo_header(NAME, SENTENCE)
st.caption(
    "Built with `deepagents` 0.7.15. Step 3 (Plan-and-Execute) builds the plan by hand; "
    "Step 8 (Sub-agent delegation) builds the sub-agents by hand -- this page shows the "
    "same two ingredients, plus a file system, as the library packages them."
)

graph_slot = st.container()
preset_labels = [preset["task"] for preset in PRESETS]
task = task_row(DEMO, preset_labels, placeholder="A multi-part task that needs research and analysis")
run_slot = st.container()
body_slot = st.container()

if task:
    run_budget = Budget(
        max_steps=budget_template.max_steps,
        max_tokens=budget_template.max_tokens,
        deadline_s=budget_template.deadline_s,
    )
    with run_slot, st.status("Working...", expanded=True) as status:
        # LangGraph's ToolNode runs tool calls on a thread pool, and the `task`
        # tool runs a whole sub-agent there, so most steps land on a worker
        # thread. Streamlit keeps its session on the script thread only; without
        # attaching it here, `status.write` raises NoSessionContext.
        script_ctx = get_script_run_ctx()

        def on_step(agent_label: str, step: Any) -> None:
            """Named sub-steps as they land -- never an anonymous spinner."""
            if get_script_run_ctx(suppress_warning=True) is None and script_ctx is not None:
                add_script_run_ctx(threading.current_thread(), script_ctx)
            marker = f"`{step.index:02d}` [{agent_label}] {step.kind}"
            status.write(f"{marker} **{step.tool}**" if step.tool else marker)

        try:
            result = agent.run(
                task, run_budget, on_step=on_step, subagent_max_steps=subagent_max_steps
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
        trajectory_track(result.steps)

        # --- to-do panel: every write_todos snapshot, newest last -------------
        st.subheader("To-do list", anchor=False)
        todo_rows = [
            s for s in result.steps if s.kind == "decide" and s.tool == "write_todos" and s.args
        ]
        if not todo_rows:
            st.caption("write_todos was never called on this run.")
        else:
            labels = [f"v{i + 1} (step {row.index:02d})" for i, row in enumerate(todo_rows)]
            picked = (
                st.select_slider("Version", options=labels, value=labels[-1], key=f"{DEMO}_todo_version")
                if len(labels) > 1
                else labels[0]
            )
            chosen = todo_rows[labels.index(picked)]
            if chosen is todo_rows[-1] and result.status != "completed":
                open_count = len(agent.open_todo_items(chosen.args.get("todos", [])))
                if open_count:
                    st.caption(
                        f"The run stopped with {open_count} item(s) still open -- "
                        f"{result.stop_reason or result.status}. The list shows where it was "
                        "left, not work still going on."
                    )
            for item in chosen.args.get("todos", []):
                status_text = item.get("status", "pending")
                content = item.get("content", "")
                symbol = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}.get(status_text, "[ ]")
                if content.upper().startswith("ABANDONED:") or "ABANDONED:" in content:
                    st.markdown(f"`{symbol}` **abandoned** -- {content}")
                else:
                    st.markdown(f"`{symbol}` **{status_text}** -- {content}")

        # --- files panel: data/vfs/<run_id>/, assemble's reads marked ----------
        st.subheader("Files", anchor=False)
        assemble_row = next(
            (s for s in result.steps if s.kind == "think" and s.agent == "assemble"), None
        )
        files_read = set((assemble_row.args or {}).get("files_read", [])) if assemble_row else set()
        run_dir = Path(settings.vfs_root)
        if not run_dir.is_absolute():
            from core.config import REPO_ROOT

            run_dir = REPO_ROOT / run_dir
        run_dir = run_dir / result.run_id
        files = sorted(run_dir.rglob("*")) if run_dir.is_dir() else []
        files = [p for p in files if p.is_file()]
        if not files:
            st.caption("No files were written on this run.")
        else:
            for path in files:
                name = str(path.relative_to(run_dir))
                size_kb = path.stat().st_size / 1024
                mark = " · read by assemble" if name in files_read else ""
                with st.expander(f"{name} ({size_kb:.1f} KB){mark}"):
                    try:
                        content = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        st.caption("(binary or unreadable file)")
                    else:
                        if name.endswith(".md"):
                            st.markdown(content)
                        else:
                            st.text(content)

        # --- system prompt + harness panel -------------------------------------
        with st.expander("System prompt (main agent)"):
            st.text(MAIN_SYSTEM_PROMPT)
        with st.expander("System prompts (sub-agents)"):
            st.markdown("**researcher**")
            st.text(RESEARCHER_SYSTEM_PROMPT)
            st.markdown("**analyst**")
            st.text(ANALYST_SYSTEM_PROMPT)
        with st.expander("The harness: tools each agent actually had"):
            st.caption(
                "From the same lists this run's build used to construct each agent's "
                "middleware -- not restated by hand. `execute` (the library's shell tool) "
                "is never in any of them: no sandbox backend is configured, so "
                "`deepagents` never registers it, and each `FilesystemMiddleware` below "
                "is additionally built with an explicit tool list that omits it."
            )
            for agent_name, tools in HARNESS_TOOLS.items():
                st.markdown(f"**{agent_name}**: {', '.join(sorted(tools))}")
            st.caption(
                "The library also auto-adds a general-purpose sub-agent unless one is "
                "supplied under that name; this demo's own task-tool guard refuses any "
                "subagent_type other than researcher/analyst before dispatch, so it is "
                "registered but never actually run (verified LLM-free -- see the README)."
            )

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
    st.caption(f"Stored as a record in `{DEMO}_runs`. Files stored under `data/vfs/{result.run_id}/`.")


readme_and_trace_tabs(DEMO, README, trace)
