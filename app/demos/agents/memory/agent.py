"""Agent memory: three tiers, each with a different reason a fact does or doesn't
survive to the next moment it's needed.

**Short-term** is the message window inside one graph run: capped, and trimmed at the
end of every turn -- after the reply has been added to it, so what the trim step
reports is exactly what's left persisted going into the next turn, not a snapshot
missing the exchange that just happened -- with LangGraph's own idiom for shrinking
checkpointed history: `RemoveMessage(id=...)`, matched against `add_messages`'s
reducer, rather than just slicing a list in memory. What's dropped is dropped from the
*persisted* state, not merely from what's displayed.

**Session** is that checkpointed thread itself (`MongoDBSaver`, via `core.checkpoint`),
which is what lets the short-term window survive a separate call to `run()`, a
Streamlit rerun, or an app restart -- as long as the reader still has the `thread_id`.
This is the first demo in the app to actually use the checkpointer Step 0 built.

**Long-term** is a fact extracted after each turn, embedded, and written to this demo's
own `memory_memory` collection; recalled by `$vectorSearch` at the start of a *later*
turn, in a different thread than the one that stated it. That's the tier the other two
can't cover on their own -- the whole point of the demo is that a fact survives a new
thread only because it made it into this tier, not because the checkpoint happened to
still be warm.

The graph is hand-built with `langgraph.graph.StateGraph`, the same choice Steps 3-4
made and for the same reason: trim, recall, respond and extract are four distinct
phases, not one `create_agent` loop. No tools are bound here -- unlike every other
StateGraph demo so far, the subject is the memory tiers, not tool use, so keeping the
actor tool-free keeps the trajectory about what this demo is for.

Budget and step accounting: `trim` and `recall` make no model call (free, the same way
Step 3's `revise` node is free) -- trimming is a list operation and recall is a local
embedding plus a vector search, not an LLM call. `act` and `extract` each call
`budget.check()` first and charge one step, their tokens and one LLM call after the
reply comes back. `BudgetExceeded` is not caught inside a node -- it propagates out of
`compiled.invoke()`, and `run()` turns it into `status="stopped_on_budget"`, the same
shape every other demo uses.
"""

import time
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.checkpoint import get_checkpointer, has_checkpoint, thread_config
from core.config import get_settings
from core.embeddings import EMBEDDING_DIMS, embed
from core.llm import agent_models, count_tokens
from core.mongo import ensure_indexes, memory_collection, save_run, vector_search
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id, thread_id_for

DEMO = "memory"
NAME = "Agent memory"
SENTENCE = "Three tiers at once: the message window, the checkpointed thread, and facts recalled from past runs."
SHAPE = "recall → act ↻ → write memory"

# `recall`, `act`, `extract` and `trim` are the real StateGraph nodes; `task` and `cap`
# are pushed onto `visited` at the boundaries, the same convention Steps 3-4 use, so
# the graph map can show where a run started and where a budget cap stopped it.
GRAPH = """flowchart LR
    task[New message] --> recall[Long-term memory recalled by vector search]
    recall --> act[Model replies with window + recalled facts]
    act --> extract[New facts extracted, written to long-term memory]
    extract --> trim[Short-term window trimmed to cap]
    trim -.->|next message, same thread| recall
    act -.->|cap spent| cap[Stopped]
"""

# Preset 1 states a durable, checkable fact -- meant to be followed by "New thread" in
# the sidebar and a related question, which is the two-click version of this demo's
# whole claim. Preset 2 is a plain opener with no engineered payoff, for exploring the
# window cap and recall freely across a longer conversation.
PRESETS: list[dict[str, Any]] = [
    {"task": "I'm allergic to shellfish -- keep that in mind for anything food-related we talk about."},
    {"task": "What's a good way to structure a two-week onboarding plan for a new engineer?"},
]


# --- structured-output schema ---------------------------------------------------


class ExtractedFacts(BaseModel):
    """The extractor's structured reply, made after every turn."""

    facts: list[str] = Field(
        default_factory=list,
        description=(
            "0 or more short, durable facts about the user worth remembering in a later, "
            "unrelated conversation -- a preference, a constraint, a fact about their "
            "situation. Most exchanges have nothing worth keeping. Do not restate the "
            "task itself or the reply just given."
        ),
    )


# --- graph state -------------------------------------------------------------------


class MemoryState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    recalled: list[dict[str, Any]]
    written: list[dict[str, Any]]
    answer: str


ACT_SYSTEM_PROMPT = (
    "You are a helpful assistant with a memory of earlier conversations. Answer the "
    "latest message using the conversation window below and, where relevant, the facts "
    "recalled from earlier sessions. If a recalled fact actually answers the question, "
    "say so plainly rather than hedging -- that is the point of recalling it. Do not "
    "claim to remember something that was not actually given to you below."
)

EXTRACT_SYSTEM_PROMPT = (
    "Given the exchange below, decide what -- if anything -- is worth remembering about "
    "the user for a future, unrelated conversation: a stated preference, a constraint, a "
    "fact about their situation. Most exchanges have nothing worth keeping; return an "
    "empty list rather than restating the task or the reply."
)


def _role_label(message: AnyMessage) -> str:
    return {"human": "user", "ai": "assistant"}.get(message.type, message.type)


def _window_snapshot(messages: list[AnyMessage]) -> list[dict[str, str]]:
    return [{"role": _role_label(m), "text": m.content} for m in messages]


def _recall_lines(recalled: list[dict[str, Any]]) -> str:
    if not recalled:
        return "(nothing recalled)"
    return "\n".join(f"- {r['text']} (score {r['score']:.3f})" for r in recalled)


def _retry_parsed(runnable: Any, messages: list[Any], parsing_error: Any) -> tuple[Any, Any]:
    """One retry on a structured-output parse failure: ask again, plainly, without
    replaying the unparseable reply. Copied from Step 3's helper rather than
    imported -- demos never import from another demo (CLAUDE.md rule 1)."""
    retry_messages = [
        *messages,
        HumanMessage(
            content=(
                f"Your last reply could not be parsed: {parsing_error}. Reply again "
                "with only the required JSON, matching the schema exactly."
            )
        ),
    ]
    result = runnable.invoke(retry_messages, config={"callbacks": get_callbacks()})
    return result["parsed"], result["raw"]


# --- long-term memory store (owned here, not core/ -- see CLAUDE.md rule 1) --------


def _memory_collection():
    return memory_collection(DEMO)


@st.cache_resource(show_spinner="Preparing long-term memory index...")
def _ensure_memory_indexed() -> bool:
    """Creates `memory_memory`'s vector index once per session and waits until it's
    queryable. Unconditional, unlike `core/corpus.py`'s version of this: this
    collection is written only by this demo, so there's no incremental-hash check to
    gate it on. Atlas Local still needs the collection to exist before a search index
    can be created on it -- `list_search_indexes()` raises `NamespaceNotFound`
    otherwise -- so an empty collection is created up front on a brand-new database,
    the one case `core/corpus.py`'s version never hits (it only ever indexes after
    `insert_many` has already created the collection)."""
    collection = _memory_collection()
    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)
    ensure_indexes(collection, vector_dims=EMBEDDING_DIMS)
    return True


def list_memories() -> list[dict[str, Any]]:
    """Every stored long-term memory, newest first -- for the page's memory panel."""
    return list(_memory_collection().find().sort("_id", -1))


def delete_memory(memory_id: str) -> bool:
    """Deletes one memory by id. What 'delete a memory and re-run' calls."""
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        object_id = ObjectId(memory_id)
    except InvalidId:
        return False
    return _memory_collection().delete_one({"_id": object_id}).deleted_count > 0


@observe(name="memory.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `thread_id`.

    `thread_id`, when given, continues that checkpointed thread. `None` mints a new
    one -- which is what makes "a fact recalled in a *new* thread" a real test of the
    long-term tier rather than the session tier quietly carrying it instead.
    """
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    thread_id: str | None = settings.get("thread_id")

    run_id = new_run_id()
    if thread_id is None:
        thread_id = thread_id_for(DEMO, run_id)

    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task, thread_id=thread_id)
    visited: list[str] = ["task"]

    def emit(agent: str, **fields: Any) -> Step:
        step = agent_run.add_step(agent=agent, **fields)
        if on_step is not None:
            on_step(agent, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        resumed = has_checkpoint(DEMO, thread_id)
        _ensure_memory_indexed()

        primary, *fallbacks = agent_models()

        # Bind before wrapping in fallbacks -- RunnableWithFallbacks does not forward
        # bind_tools()/with_structured_output()'s needs, but with_fallbacks() itself
        # works on any Runnable, bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md,
        # "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        act_runnable = _with_fallbacks([primary, *fallbacks])
        extract_models = [
            m.with_structured_output(ExtractedFacts, include_raw=True) for m in [primary, *fallbacks]
        ]
        extract_runnable = _with_fallbacks(extract_models)

        # --- nodes -------------------------------------------------------------
        # `trim` runs last, not first: it acts on the window only after this turn's
        # reply has already been appended to it, so the step it emits -- and the
        # `window` snapshot the page's tiers panel reads back from that step's args
        # -- is exactly what's left persisted going into the next turn, not a
        # snapshot that's missing the exchange that just happened. `recall` and
        # `act` therefore see the window as it stood after the *previous* turn's
        # trim plus this turn's new message -- at most one message over the cap,
        # which is a fine trade for the panel always being exactly right.

        def recall_node(state: MemoryState) -> dict[str, Any]:
            visited.append("recall")
            vector = embed([state["task"]])[0]
            hits = vector_search(_memory_collection(), vector, top_k=settings_obj.memory_recall_k)
            recalled = [{"id": str(h["_id"]), "text": h["text"], "score": h["score"]} for h in hits]
            if recalled:
                text = f"Recalled {len(recalled)} memorie(s):\n{_recall_lines(recalled)}"
            elif resumed:
                text = "Nothing recalled -- nothing stored is relevant to this message."
            else:
                text = "Nothing recalled -- this is a new thread with no matching long-term memory yet."
            emit("memory", kind="observe", text=text, args={"recalled": recalled})
            return {"recalled": recalled}

        def act_node(state: MemoryState) -> dict[str, Any]:
            budget.check()
            visited.append("act")
            messages = [
                SystemMessage(
                    content=(
                        f"{ACT_SYSTEM_PROMPT}\n\nRecalled from earlier sessions:\n"
                        f"{_recall_lines(state['recalled'])}"
                    )
                ),
                *state["messages"],
            ]
            call_started = time.monotonic()
            response = act_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            answer = response.content or ""
            emit(
                "assistant",
                kind="decide",
                text=answer,
                args={"used_recall": bool(state["recalled"])},
                tokens=tokens,
                latency_ms=latency_ms,
            )
            return {"messages": [AIMessage(content=answer)], "answer": answer}

        def extract_node(state: MemoryState) -> dict[str, Any]:
            budget.check()
            visited.append("extract")
            messages = [
                SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"User said: {state['task']}\n\nAssistant replied: {state['answer']}"
                ),
            ]
            call_started = time.monotonic()
            result = extract_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(extract_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The extractor could not produce a valid reply after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            written: list[dict[str, Any]] = []
            if parsed.facts:
                vectors = embed(parsed.facts)
                now = datetime.now(timezone.utc).isoformat()
                docs = [
                    {
                        "text": fact,
                        "embedding": vector,
                        "run_id": run_id,
                        "thread_id": thread_id,
                        "created_at": now,
                    }
                    for fact, vector in zip(parsed.facts, vectors)
                ]
                inserted_ids = _memory_collection().insert_many(docs).inserted_ids
                written = [
                    {"id": str(object_id), "text": doc["text"]}
                    for object_id, doc in zip(inserted_ids, docs)
                ]

            if written:
                joined = "; ".join(w["text"] for w in written)
                text = f"Wrote {len(written)} new memorie(s): {joined}"
            else:
                text = "Nothing new worth remembering from this exchange."
            emit(
                "memory", kind="act", text=text, args={"written": written}, tokens=tokens, latency_ms=latency_ms
            )
            return {"written": written}

        def trim_node(state: MemoryState) -> dict[str, Any]:
            visited.append("trim")
            messages = state["messages"]
            cap = settings_obj.memory_window_messages
            keep = messages[-cap:] if len(messages) > cap else messages
            drop = messages[: len(messages) - len(keep)]
            if drop:
                text = (
                    f"Window cap {cap} message(s): kept the last {len(keep)}, dropped "
                    f"{len(drop)} older one(s)."
                )
            else:
                text = f"Window cap {cap} message(s): {len(keep)} in the window, nothing to drop yet."
            emit(
                "memory",
                kind="decide",
                text=text,
                args={
                    "window": _window_snapshot(keep),
                    "dropped": _window_snapshot(drop),
                    "resumed": resumed,
                },
            )
            return {"messages": [RemoveMessage(id=m.id) for m in drop]}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(MemoryState)
        graph.add_node("recall", recall_node)
        graph.add_node("act", act_node)
        graph.add_node("extract", extract_node)
        graph.add_node("trim", trim_node)

        graph.add_edge(START, "recall")
        graph.add_edge("recall", "act")
        graph.add_edge("act", "extract")
        graph.add_edge("extract", "trim")
        graph.add_edge("trim", END)

        compiled = graph.compile(checkpointer=get_checkpointer(DEMO))

        final_state = compiled.invoke(
            {"task": task, "messages": [HumanMessage(content=task)]},
            config={
                **thread_config(thread_id),
                "callbacks": get_callbacks(),
                "recursion_limit": 4 * budget.max_steps + 10,
            },
        )
        agent_run.output = final_state.get("answer", "")
        agent_run.status = "completed"

    except BudgetExceeded as stop:
        visited.append("cap")
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        emit("agent", kind="decide", text=agent_run.stop_reason)
    except Exception as exc:  # no provider configured, every provider refused, Mongo unreachable
        agent_run.status = "failed"
        agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        emit("agent", kind="decide", text=agent_run.stop_reason)

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run


# --- the tiers panel -----------------------------------------------------------


def turn_summary(steps: list[Step]) -> dict[str, Any]:
    """Rebuilds this turn's tiers from the saved track -- the post-trim window and
    what was dropped, whether the thread was resumed, what was recalled, and what was
    written. Same rebuild-from-the-track approach as Step 3's `plan_diff` and Step 4's
    `attempt_summary`: nothing tier-specific is added to `AgentRun` itself.
    """
    summary: dict[str, Any] = {
        "window": [],
        "dropped": [],
        "resumed": False,
        "recalled": [],
        "written": [],
    }
    for step in steps:
        if step.agent != "memory" or not step.args:
            continue
        if step.kind == "decide" and "window" in step.args:
            summary["window"] = step.args.get("window", [])
            summary["dropped"] = step.args.get("dropped", [])
            summary["resumed"] = step.args.get("resumed", False)
        elif step.kind == "observe" and "recalled" in step.args:
            summary["recalled"] = step.args.get("recalled", [])
        elif step.kind == "act" and "written" in step.args:
            summary["written"] = step.args.get("written", [])
    return summary
