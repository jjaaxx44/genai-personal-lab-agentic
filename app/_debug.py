"""Temporary: a one-glance health check of the shared `core/` plumbing.

Started life as Step 0's "done when" list, when there were no demos to check it
against. Now it answers one question for a visitor -- is the plumbing working? --
with a status row and a single pass/fail run. Raw tool output and the two
maintenance actions sit in collapsed sections for whoever needs them.

**This page is deleted once the demos it stands in for exist.** Nothing in
`core/` may import it, and it is not in the catalogue.
"""

import time
from dataclasses import dataclass

import streamlit as st

from core.budget import Budget, BudgetExceeded
from core.checkpoint import get_checkpointer, has_checkpoint, thread_config
from core.config import get_settings
from core.corpus import CORPUS_DIR, build_index, corpus_files, reset_index_cache
from core.llm import PROVIDER_LABELS, complete, configured_providers
from core.mongo import CORPUS_COLLECTION, get_collection, ping
from core.tools import DEFAULT_TOOLS, Toolbox
from core.types import new_run_id, thread_id_for

DEMO = "debug"
RESULTS = "debug_results"
settings = get_settings()

st.title("Core checks", anchor=False)
st.caption(
    "A quick health check of the shared plumbing every demo runs on. "
    "Nothing here is needed to use the demos."
)

# --- status --------------------------------------------------------------------

providers = configured_providers()
reachable, mongo_message = ping()
try:
    chunks = get_collection(CORPUS_COLLECTION).count_documents({}) if reachable else None
except Exception:
    chunks = None


def status_tile(column, label: str, ok: bool, detail: str) -> None:
    with column.container(border=True):
        st.markdown(f"{':green[:material/check_circle:]' if ok else ':red[:material/error:]'} **{label}**")
        st.caption(detail)


cols = st.columns(3)
status_tile(
    cols[0],
    "Language models",
    bool(providers),
    " → ".join(PROVIDER_LABELS[p] for p in providers) if providers else "No provider key is set.",
)
status_tile(cols[1], "Database", reachable, mongo_message)
status_tile(
    cols[2],
    "Document index",
    bool(chunks),
    f"{chunks} chunks from {len(corpus_files())} files." if chunks else "Empty or unreachable.",
)

# --- checks --------------------------------------------------------------------


@dataclass
class Check:
    label: str
    passed: bool | None  # None: optional and unavailable, not a failure
    detail: str


# (what a visitor reads, tool, args, should the call succeed?, optional?, what a pass means)
TOOL_CHECKS: list[tuple[str, str, dict, bool, bool, str]] = [
    ("Searches the documents", "search_corpus", {"query": "what does the corpus say", "top_k": 3}, True, False, "Found matching passages."),
    ("Reads the database", "run_sql", {"sql": "SELECT Name FROM Genre LIMIT 3"}, True, False, "Returned rows."),
    ("Blocks a database write", "run_sql", {"sql": "UPDATE Genre SET Name = 'x'"}, False, False, "Refused; the database is read-only."),
    ("Explains a bad query instead of crashing", "run_sql", {"sql": "SELECT Nmae FROM Genre LIMIT 1"}, False, False, "Returned an error the agent can learn from."),
    ("Writes and reads a file", "write_file", {"path": "note.md", "content": "# A note\nWritten by a tool."}, True, False, "Wrote a note and read it back."),
    ("Blocks a file outside its folder", "read_file", {"path": "../../samples/chinook.db"}, False, False, "Refused; files stay in the run's own folder."),
    ("Rejects invalid arguments", "search_corpus", {"query": "x", "top_k": 99}, False, False, "Refused with a hint on what is allowed."),
    ("Survives a tool that crashes", "boom", {}, False, False, "The crash came back as a message, not a stack trace."),
    ("Searches the web", "web_search", {"query": "langgraph checkpointing"}, True, True, "Returned results."),
]


def _checkpoint_graph():
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        count: int

    builder = StateGraph(State)
    builder.add_node("bump", lambda state: {"count": state.get("count", 0) + 1})
    builder.add_edge(START, "bump")
    builder.add_edge("bump", END)
    return builder.compile(checkpointer=get_checkpointer(DEMO))


def run_checks() -> tuple[list[Check], dict[str, str]]:
    checks: list[Check] = []
    raw: dict[str, str] = {}

    # "boom" is not in the default toolbox; it is added to show that a tool which
    # raises comes back to the agent as an observation.
    box = Toolbox(new_run_id(), names=[*DEFAULT_TOOLS, "boom"])
    for label, tool, args, should_succeed, optional, meaning in TOOL_CHECKS:
        result = box.call(tool, args)
        if tool == "write_file" and result.ok:
            result = box.call("read_file", {"path": "note.md"})
        raw[label] = f"{tool}({args})\n\n{result.output[:2000]}"
        if result.ok == should_succeed:
            checks.append(Check(label, True, meaning))
        elif optional:
            checks.append(Check(label, None, "Unavailable right now; demos carry on without it."))
        else:
            checks.append(Check(label, False, "Unexpected result — see the raw output below."))

    budget = Budget(max_steps=2, max_tokens=60_000, deadline_s=180).start()
    try:
        for _ in range(3):
            budget.check()
            budget.charge(steps=1)
        checks.append(Check("Stops at the step limit", False, "Ran three steps under a cap of two."))
    except BudgetExceeded as exc:
        checks.append(Check("Stops at the step limit", exc.cap == "max_steps", exc.reason))

    budget = Budget(max_steps=50, max_tokens=60_000, deadline_s=1).start()
    try:
        while True:
            budget.check()
            time.sleep(0.2)
            budget.charge(steps=1)
    except BudgetExceeded as exc:
        checks.append(Check("Stops at the time limit", exc.cap == "deadline_s", exc.reason))

    label = "Saves and restores agent state"
    thread = thread_id_for(DEMO, "fixed-thread")
    try:
        graph = _checkpoint_graph()
        written = graph.invoke({"count": 0}, config=thread_config(thread))["count"]
        restored = graph.get_state(thread_config(thread)).values.get("count")
        ok = has_checkpoint(DEMO, thread) and restored == written
        raw[label] = f"written: {written}\nrestored: {restored}"
        checks.append(Check(label, ok, "Saved to the database and read back." if ok else "Read back a different value."))
    except Exception as exc:
        raw[label] = f"{type(exc).__name__}: {exc}"
        checks.append(Check(label, False, "Could not reach the checkpoint store."))

    return checks, raw


if st.button("Run all checks", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Running checks..."):
        st.session_state[RESULTS] = run_checks()

if RESULTS in st.session_state:
    checks, raw = st.session_state[RESULTS]
    failed = [c for c in checks if c.passed is False]
    if failed:
        st.error(f"{len(failed)} of {len(checks)} checks failed.")
    else:
        st.success(f"All {len(checks)} checks passed.")

    for check in checks:
        icon = {True: ":green[:material/check:]", False: ":red[:material/close:]", None: ":orange[:material/remove:]"}[check.passed]
        st.markdown(f"{icon} **{check.label}** — {check.detail}")

    with st.expander("Raw output", expanded=False):
        for label, output in raw.items():
            st.caption(label)
            st.code(output, language=None)
else:
    st.caption("Takes a few seconds. Uses no language-model quota.")

# --- maintenance ---------------------------------------------------------------

with st.expander("Maintenance", expanded=False):
    st.caption(f"Documents are read from `{CORPUS_DIR}` into the `{settings.mongodb_db}` database.")
    left, right = st.columns(2)
    if left.button("Send a test prompt", disabled=not providers):
        with st.spinner("Asking the first available provider..."):
            started = time.monotonic()
            try:
                answer = complete("Reply with exactly: the chain works.")
                st.success(f"{answer.strip()}  ·  {(time.monotonic() - started):.1f}s")
            except Exception as exc:
                st.error(f"No provider answered: {type(exc).__name__}: {exc}")
    if right.button("Rebuild the document index"):
        with st.spinner("Chunking, embedding and waiting for the index to become queryable..."):
            try:
                stats = build_index(force=True)
                reset_index_cache()
                st.success(
                    f"{stats.files} file(s), {stats.chunks} chunk(s), "
                    f"{stats.indexed} re-embedded, {stats.latency_ms / 1000:.1f}s."
                )
            except Exception as exc:
                st.error(f"Indexing failed: {type(exc).__name__}: {exc}")
