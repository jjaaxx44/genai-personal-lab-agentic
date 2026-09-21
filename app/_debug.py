"""Temporary: the checks in Step 0's "done when" list, in one page.

Everything here exercises `core/` directly — the provider chain, each sandboxed
tool and its guards, the error protocol, the budget caps, the MongoDB checkpoint
and the shared page furniture. It exists because Step 0 ships no demos, so
without it there is nothing in the running app to check.

**This page is deleted once the demos it stands in for exist.** Nothing in
`core/` may import it, and it is not in the catalogue.
"""

import time

import streamlit as st

from core.budget import Budget, BudgetExceeded
from core.checkpoint import get_checkpointer, has_checkpoint, thread_config
from core.config import get_settings
from core.corpus import CORPUS_DIR, build_index, corpus_files, reset_index_cache
from core.llm import PROVIDER_LABELS, complete, configured_providers
from core.mongo import CORPUS_COLLECTION, get_collection, ping
from core.tools import DEFAULT_TOOLS, Toolbox
from core.types import AgentRun, Step, new_run_id, thread_id_for
from core.ui import budget_strip, gate_panel, graph_map, trajectory_track

DEMO = "debug"
settings = get_settings()

st.title("Core checks", anchor=False)
st.caption(
    "Step 0 has no demos yet, so its checks live here. Each section is one line from the "
    "step's done-when list. This page goes when the demos replace it."
)

# --- 1. Providers --------------------------------------------------------------

st.header("Providers", anchor=False, divider="gray")

providers = configured_providers()
if providers:
    st.write("Fallback order: " + " → ".join(PROVIDER_LABELS[p] for p in providers))
    st.caption(
        "Each provider is tried in turn; a rate limit or an outage on one falls through to "
        "the next. Remove a key from the environment file and this list shortens."
    )
else:
    st.error("No provider is configured. Every run will fail until one is set.")

if st.button("Send a test prompt", disabled=not providers):
    with st.spinner("Asking the first available provider..."):
        started = time.monotonic()
        try:
            answer = complete("Reply with exactly: the chain works.")
            st.success(f"{answer.strip()}  ·  {(time.monotonic() - started):.1f}s")
        except Exception as exc:
            st.error(f"No provider answered: {type(exc).__name__}: {exc}")

# --- 2. MongoDB and the corpus -------------------------------------------------

st.header("MongoDB and the corpus", anchor=False, divider="gray")

reachable, message = ping()
(st.success if reachable else st.error)(message)

cols = st.columns(3)
cols[0].metric("Corpus files", len(corpus_files()))
try:
    cols[1].metric("Indexed chunks", get_collection(CORPUS_COLLECTION).count_documents({}))
except Exception:
    cols[1].metric("Indexed chunks", "—")
cols[2].metric("Database", settings.mongodb_db)
st.caption(f"Corpus directory: `{CORPUS_DIR}`")

if st.button("Rebuild the corpus index"):
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

# --- 3. Tools ------------------------------------------------------------------

st.header("Tools and their guards", anchor=False, divider="gray")
st.caption(
    "Every call goes through Toolbox.call(), which validates arguments, enforces a "
    "wall-clock timeout, and turns every failure into an ERROR[...] observation instead "
    "of raising. The refusals below are the sandbox working, not the app breaking."
)

if "debug_run_id" not in st.session_state:
    st.session_state["debug_run_id"] = new_run_id()
run_id = st.session_state["debug_run_id"]
# "boom" is not in the default toolbox; it is added here to show what a tool that
# raises looks like from the agent's side.
box = Toolbox(run_id, names=[*DEFAULT_TOOLS, "boom"])
st.caption(f"Run `{run_id}` · files under `{box.run_dir}`")

CHECKS: list[tuple[str, str, dict]] = [
    ("Search the corpus", "search_corpus", {"query": "what does the corpus say", "top_k": 3}),
    ("Describe the SQL schema", "describe_schema", {}),
    ("Run a SELECT", "run_sql", {"sql": "SELECT Name FROM Genre LIMIT 3"}),
    ("Refuse an UPDATE", "run_sql", {"sql": "UPDATE Genre SET Name = 'x'"}),
    ("Recover from a bad column", "run_sql", {"sql": "SELECT Nmae FROM Genre LIMIT 1"}),
    ("Search the web", "web_search", {"query": "langgraph checkpointing"}),
    ("Write a file", "write_file", {"path": "note.md", "content": "# A note\nWritten by a tool."}),
    ("Read it back", "read_file", {"path": "note.md"}),
    ("List the files", "list_files", {}),
    ("Refuse a path outside the run", "read_file", {"path": "../../samples/chinook.db"}),
    ("Reject junk arguments", "search_corpus", {"query": "x", "top_k": 99}),
    ("Turn a raised exception into an observation", "boom", {}),
]

for label, tool, args in CHECKS:
    with st.container(border=True):
        head = st.container(horizontal=True, gap="small")
        head.markdown(f"**{label}**")
        head.markdown(f"`{tool}`")
        if head.button("Run", key=f"check_{label}"):
            result = box.call(tool, args)
            st.caption(f"{result.latency_ms:.0f} ms · {'ok' if result.ok else 'error'}")
            st.code(result.output[:2000], language=None)
        st.caption(f"`{args}`" if args else "`{}`")

# --- 4. Budget -----------------------------------------------------------------

st.header("Budget", anchor=False, divider="gray")
st.caption(
    "A three-step loop under a two-step cap. The loop calls check() before each step, so "
    "it stops itself and names the cap rather than being killed from outside."
)

if st.button("Run three steps under a cap of two"):
    budget = Budget(max_steps=2, max_tokens=60_000, deadline_s=180).start()
    log = []
    stopped = None
    for i in range(1, 4):
        try:
            budget.check()
        except BudgetExceeded as exc:
            stopped = exc
            break
        budget.charge(steps=1, tokens=100, llm_calls=1)
        log.append(f"step {i} ran")
    for line in log:
        st.write(line)
    st.warning(stopped.reason if stopped else "The loop finished without hitting a cap.")
    st.caption(f"Cap hit: `{stopped.cap}`" if stopped else "")

st.caption("The deadline cap, checked the same way:")
if st.button("Run under a deadline of one second"):
    budget = Budget(max_steps=50, max_tokens=60_000, deadline_s=1).start()
    steps = 0
    try:
        while True:
            budget.check()
            time.sleep(0.4)
            budget.charge(steps=1)
            steps += 1
    except BudgetExceeded as exc:
        st.warning(exc.reason)
        st.caption(f"{steps} step(s) ran · cap hit: `{exc.cap}`")

# --- 5. Checkpoints ------------------------------------------------------------

st.header("Checkpoints", anchor=False, divider="gray")
st.caption(
    "A counter written through the MongoDB checkpointer under a fixed thread id. It keeps "
    "counting across a rerun, a browser refresh and a container restart, because the state "
    "is in MongoDB rather than in session state."
)

THREAD = thread_id_for(DEMO, "fixed-thread")


def _counter_graph():
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        count: int
        note: str

    def bump(state: State) -> State:
        return {"count": state.get("count", 0) + 1, "note": f"written at {time.strftime('%H:%M:%S')}"}

    builder = StateGraph(State)
    builder.add_node("bump", bump)
    builder.add_edge(START, "bump")
    builder.add_edge("bump", END)
    return builder.compile(checkpointer=get_checkpointer(DEMO))


cols = st.columns(2)
if cols[0].button("Write a checkpoint"):
    try:
        graph = _counter_graph()
        state = graph.invoke({"count": 0, "note": ""}, config=thread_config(THREAD))
        st.success(f"Thread `{THREAD}` now at count {state['count']} ({state['note']}).")
    except Exception as exc:
        st.error(f"Could not write the checkpoint: {type(exc).__name__}: {exc}")

if cols[1].button("Read it back"):
    try:
        if not has_checkpoint(DEMO, THREAD):
            st.info("No checkpoint on this thread yet. Write one first.")
        else:
            state = _counter_graph().get_state(thread_config(THREAD))
            st.json(state.values)
            st.caption(f"Checkpoint id `{state.config['configurable'].get('checkpoint_id', '—')}`")
    except Exception as exc:
        st.error(f"Could not read the checkpoint: {type(exc).__name__}: {exc}")

# --- 6. Page furniture ---------------------------------------------------------

st.header("Page furniture", anchor=False, divider="gray")
st.caption(
    "The shared components every demo page is assembled from, rendered against a made-up "
    "run: the graph map with a route picked out, the budget strip, the trajectory track "
    "with a retry and an indented sub-agent step, and the gate panel."
)

graph_map(
    """graph LR
    start([task]) --> think
    think --> act
    act --> observe
    observe --> think
    observe --> finish([answer])""",
    visited=["start", "think", "act", "observe", "finish"],
)

demo_budget = Budget(max_steps=12, max_tokens=60_000, deadline_s=180).start()
demo_budget.charge(steps=5, tokens=18_400, tool_calls=3, llm_calls=6)
budget_strip(demo_budget)

demo_run = AgentRun(
    demo=DEMO,
    task="Find the throughput figure, then check it against the field data.",
    output="The two documents disagree: the benchmark reports one figure, the field bulletin another.",
    steps=[
        Step(index=1, kind="think", text="The corpus should have a benchmark report. Search it first.", tokens=210, latency_ms=840),
        Step(index=2, kind="act", tool="search_corpus", args={"query": "throughput benchmark", "top_k": 5}, text="Searching the corpus.", tokens=90, latency_ms=120),
        Step(index=3, kind="observe", text="Three passages, the best from the benchmark report.", tokens=640, latency_ms=310),
        Step(index=4, kind="act", tool="run_sql", args={"sql": "SELECT Nmae FROM Genre"}, text="Querying the database.", tokens=80, latency_ms=95),
        Step(index=4, attempt=2, kind="act", tool="run_sql", args={"sql": "SELECT Name FROM Genre"}, text="The column was misspelled; retried with the name from the schema.", tokens=88, latency_ms=101),
        Step(index=5, kind="handoff", agent="parent", text="Delegating the field-data check to a sub-agent with its own budget.", tokens=140, latency_ms=70),
        Step(index=6, kind="act", agent="checker", parent_index=5, tool="search_corpus", args={"query": "field service bulletin"}, text="Sub-agent searching.", tokens=120, latency_ms=180),
        Step(index=7, kind="decide", text="The figures conflict. Report both rather than picking one.", tokens=260, latency_ms=900),
    ],
    status="completed",
    visited=["start", "think", "act", "observe", "finish"],
    llm_calls=6,
    tool_calls=3,
    tokens=18_400,
    latency_ms=12_400,
)
trajectory_track(demo_run.steps)

decision = gate_panel(
    DEMO,
    {"tool": "write_file", "args": {"path": "brief.md", "content": "…"}},
    prompt="The agent proposes to write a file. Nothing below this line has run.",
)
if decision is not None:
    st.success(f"Decision: {decision.verdict}. Reason: {decision.reason or '—'}")
    if decision.edited is not None:
        st.json(decision.edited)
