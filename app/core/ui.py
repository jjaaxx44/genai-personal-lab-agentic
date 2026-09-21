"""The shared page furniture: graph map, task row, budget strip, trajectory track, gate panel.

`DESIGN.md` is the specification for all of it. The demo page's job is to wire
these together and add whatever its own technique needs; the order on the page is
fixed, because a reader who has seen one demo should be able to read the next one
without relearning where anything is.

Two devices carry meaning and are worth stating here, because a demo can break
them by accident: **row order** on the track is execution order, and **row indent**
means a sub-agent's step under the agent that delegated it. Retries stack under
the step they replace with a primed index (`04`, `04′`).
"""

from typing import Any, Callable

import streamlit as st

from .budget import CAP_LABELS, Budget
from .llm import PROVIDER_LABELS, configured_providers
from .mongo import clear_demo_data
from .types import AgentRun, HumanDecision, Step

# Kind chips. Text as well as position, because meaning is never carried by colour
# alone -- and these are the six kinds Step allows, so an unknown one is a bug.
KIND_LABELS = {
    "think": "think",
    "act": "act",
    "observe": "observe",
    "decide": "decide",
    "gate": "gate",
    "handoff": "handoff",
}

PREVIEW_CHARS = 280


def demo_header(name: str, sentence: str) -> None:
    st.title(name, anchor=False)
    st.caption(sentence)


# --- graph map -----------------------------------------------------------------


def graph_map(mermaid_source: str, visited: list[str] | None = None) -> None:
    """The demo's control flow, drawn in full, with the route actually taken picked out.

    Before a run this is the empty state and the explanation at once; after one it
    is the record of which way the agent went. This replaces the sibling RAG lab's
    pipeline ribbon, which assumes a fixed chain of stages -- there isn't one here.

    Colour literals live here and nowhere else in the app: mermaid cannot read the
    Streamlit theme, and LangGraph's own generated diagrams hardcode classDefs that
    leave labels near-white on near-white in dark mode. The values are the DESIGN.md
    tokens; a later classDef for the same class wins in mermaid, so these are
    appended last.
    """
    dark = st.context.theme.type != "light"
    surface = "#1D1F26" if dark else "#E9EBF0"
    text = "#E7E9EE" if dark else "#14161B"
    border = "#2E323B" if dark else "#D2D6DF"
    accent = "#8E82F5" if dark else "#4B3FC7"

    theme_override = "\n".join(
        [
            f"classDef default fill:{surface},stroke:{border},color:{text},line-height:1.2",
            f"classDef first fill-opacity:0,stroke:{border},color:{text}",
            f"classDef last fill:{border},stroke:{border},color:{text}",
        ]
    )
    seen = list(dict.fromkeys(n for n in (visited or []) if n))
    highlight = "\n".join(
        f"style {node} stroke:{accent},stroke-width:3px,color:{text}" for node in seen
    )
    st.mermaid_chart(f"{mermaid_source}\n{theme_override}" + (f"\n{highlight}" if highlight else ""))


# --- task row ------------------------------------------------------------------


def task_row(
    demo: str, presets: list[str], *, placeholder: str = "Describe the task", label: str = "Task"
) -> str | None:
    """Preset pills above a form. Returns the task when it is submitted, else None.

    The presets exist because a reader should not have to invent a task that
    exercises the technique -- picking one fills the box, which stays editable.
    """
    picked = st.pills(
        "Presets", presets, key=f"{demo}_preset", label_visibility="collapsed", selection_mode="single"
    )
    if picked and st.session_state.get(f"{demo}_last_preset") != picked:
        st.session_state[f"{demo}_last_preset"] = picked
        st.session_state[f"{demo}_task"] = picked

    with st.form(key=f"{demo}_task_form"):
        task = st.text_area(label, key=f"{demo}_task", placeholder=placeholder, height=80)
        submitted = st.form_submit_button("Run", type="primary")

    if not submitted:
        return None
    if not task or not task.strip():
        st.warning("Enter a task first.")
        return None
    return task.strip()


# --- budget strip --------------------------------------------------------------


def budget_strip(budget: Budget, *, container: Any = None) -> None:
    """Steps, tokens, elapsed and tool calls against their caps — always visible.

    It sits above the trajectory rather than in a metrics row at the bottom
    because with agents the cost is the thing that surprises people.
    """
    target = container or st
    fractions = budget.fractions()
    cols = target.columns(4)

    cols[0].metric("Steps", f"{budget.steps_used}/{budget.max_steps}")
    cols[0].progress(fractions["max_steps"])

    cols[1].metric("Tokens", f"{budget.tokens_used:,}/{budget.max_tokens:,}")
    cols[1].progress(fractions["max_tokens"])

    cols[2].metric("Elapsed", f"{budget.elapsed_s:.1f}s/{budget.deadline_s:.0f}s")
    cols[2].progress(fractions["deadline_s"])

    cols[3].metric("Tool calls", budget.tool_calls)
    cols[3].caption(f"{budget.llm_calls} LLM call(s)")


def sidebar_budget_controls(demo: str, defaults: Budget) -> Budget:
    """Per-run overrides for the three caps. Every demo has these, in the same place."""
    st.sidebar.subheader("Budget")
    max_steps = st.sidebar.slider(
        "Step cap", 1, max(defaults.max_steps * 2, 24), defaults.max_steps, key=f"{demo}_max_steps"
    )
    max_tokens = st.sidebar.slider(
        "Token budget",
        5_000,
        max(defaults.max_tokens * 2, 120_000),
        defaults.max_tokens,
        step=5_000,
        key=f"{demo}_max_tokens",
    )
    deadline_s = st.sidebar.slider(
        "Deadline (s)",
        30,
        max(int(defaults.deadline_s) * 2, 600),
        int(defaults.deadline_s),
        step=10,
        key=f"{demo}_deadline",
    )
    return Budget(max_steps=max_steps, max_tokens=max_tokens, deadline_s=deadline_s)


# --- trajectory track ----------------------------------------------------------


def _step_index_label(step: Step) -> str:
    """`04`, or `04′` for a retry stacked under the attempt it replaces."""
    return f"{step.index:02d}" + "′" * (step.attempt - 1)


def step_row(step: Step) -> None:
    """One bordered row on the track. No `height` anywhere: a card that scrolls its own
    text hides the thing it exists to show (learned the hard way in the RAG lab)."""
    with st.container(border=True):
        head = st.container(horizontal=True, gap="small")
        head.markdown(f"`{_step_index_label(step)}`")
        head.markdown(f":gray[{KIND_LABELS.get(step.kind, step.kind)}]")
        if step.agent and step.agent != "agent":
            head.markdown(f"`{step.agent}`")
        if step.tool:
            head.markdown(f"**{step.tool}**")
        head.markdown(
            f":gray[{step.tokens:,} tok · {step.latency_ms:.0f} ms]"
            if step.tokens or step.latency_ms
            else ""
        )

        text = step.text or ""
        if len(text) <= PREVIEW_CHARS:
            st.write(text)
        else:
            st.write(text[:PREVIEW_CHARS].rstrip() + "…")
            with st.expander(f"View the full {step.kind} ({len(text):,} chars)"):
                st.write(text)

        if step.args:
            with st.expander("Arguments"):
                st.json(step.args)


def trajectory_track(steps: list[Step]) -> None:
    """The signature element: a numbered rail down the page, one row per step.

    Sub-agent steps (`parent_index` set) are indented under the handoff that
    spawned them, so the shape of the run is readable without reading the text.
    """
    if not steps:
        st.caption("Nothing has run yet. The trajectory appears here, one row per step.")
        return

    for step in steps:
        if step.parent_index is None:
            step_row(step)
        else:
            spacer, row = st.columns([1, 11], gap="small")
            with row:
                step_row(step)


# --- gate panel ----------------------------------------------------------------


def gate_panel(demo: str, proposal: dict[str, Any], *, prompt: str) -> HumanDecision | None:
    """A rule across the track, the proposed action in full, then the three verdicts.

    Everything above the rule is committed; nothing below it has run. Returns the
    decision on the run it is submitted, else None.
    """
    st.divider()
    st.subheader("Waiting for a decision", anchor=False)
    st.write(prompt)
    st.json(proposal)

    with st.form(key=f"{demo}_gate_form"):
        verdict = st.radio(
            "Verdict",
            ["approve", "edit", "reject"],
            format_func=lambda v: {
                "approve": "Approve — run it as proposed",
                "edit": "Edit — change the arguments, then run it",
                "reject": "Reject — do not run it",
            }[v],
            key=f"{demo}_gate_verdict",
        )
        edited_text = st.text_area(
            "Edited arguments (JSON)",
            value=st.session_state.get(f"{demo}_gate_edit", ""),
            key=f"{demo}_gate_edit_input",
            help="Used only when the verdict is Edit.",
        )
        reason = st.text_area("Reason", key=f"{demo}_gate_reason", placeholder="Shown to the agent")
        submitted = st.form_submit_button("Submit decision", type="primary")

    if not submitted:
        return None

    edited = None
    if verdict == "edit":
        import json

        try:
            edited = json.loads(edited_text)
        except json.JSONDecodeError as exc:
            # Rejected input names the limit and the actual value, and the run stays
            # paused rather than resuming on arguments nobody meant.
            st.error(f"That is not valid JSON ({exc.msg}). The run is still waiting.")
            return None

    return HumanDecision(verdict=verdict, edited=edited, reason=reason)


# --- outcome -------------------------------------------------------------------


def stop_note(run: AgentRun) -> None:
    """A run that hit a cap ends in a plain sentence naming the cap, never an exception."""
    if run.status == "stopped_on_budget":
        st.warning(run.stop_reason or "The run stopped on a budget cap.")
    elif run.status == "needs_human":
        st.info(run.stop_reason or "The run is waiting for a person.")
    elif run.status == "failed":
        st.error(run.stop_reason or "The run failed.")


def final_output(run: AgentRun) -> None:
    st.subheader("Output", anchor=False)
    stop_note(run)
    if run.output:
        st.write(run.output)
    elif run.status != "needs_human":
        st.caption("The run produced no output.")
    st.caption(
        f"`{run.run_id}` · {len(run.steps)} step(s) · {run.tool_calls} tool call(s) · "
        f"{run.llm_calls} LLM call(s) · {run.tokens:,} tokens · {run.latency_ms / 1000:.1f}s"
    )


def readme_and_trace_tabs(demo: str, readme_path: str, trace_renderer: Callable[[], None]) -> None:
    """`How it works` first, `Trace` second — explain, then get out of the way.

    The page moves the reader to `Trace` the moment a run finishes by setting
    `st.session_state[f"{demo}_tabs"] = "Trace"` immediately after storing the
    result, which is always above this call.
    """
    how, trace = st.tabs(["How it works", "Trace"], key=f"{demo}_tabs", on_change="rerun")
    with how:
        try:
            with open(readme_path, encoding="utf-8") as handle:
                st.markdown(handle.read())
        except OSError:
            st.caption("This demo has no README yet.")
    with trace:
        trace_renderer()


# --- sidebar -------------------------------------------------------------------


def provider_note() -> None:
    """Which providers will answer, in fallback order. A demo that degrades to the local
    model should say so before the reader wonders why it got slower and worse."""
    providers = configured_providers()
    if not providers:
        st.sidebar.error("No LLM provider is configured. Runs will fail until one is set.")
        return
    st.sidebar.caption("Providers: " + " → ".join(PROVIDER_LABELS[p] for p in providers))


def clear_data_button(demo: str) -> bool:
    """Deletes this demo's runs, checkpoints and memory. Returns True on the run it is
    clicked, so the page can drop its own session state; the rerun is the caller's."""
    flash_key = f"{demo}_cleared"
    if flash_key in st.session_state:
        counts = st.session_state.pop(flash_key)
        total = sum(counts.values())
        st.sidebar.success(f"Cleared {total} record(s) from {len(counts)} collection(s).")

    if st.sidebar.button("Clear my data", key=f"{demo}_clear"):
        st.session_state[flash_key] = clear_demo_data(demo)
        return True
    return False


def cap_label(cap: str) -> str:
    return CAP_LABELS.get(cap, cap)
