"""Swarm: no supervisor, no router -- three peers (researcher, analyst, writer) that
hand the task to each other directly, and whoever holds it owns it until it hands off.

This is Step 9's mirror image, not its cousin. Supervisor-worker puts one agent in
charge of every routing decision, gives each worker a fresh prompt built from a
shared *reports board*, and pays one extra LLM call per hop for the routing itself.
Here there is no such call: a peer decides whether to keep working, hand off, or
answer, inside the same model call that does the work, and every peer reads the
same shared *conversation* -- not a reports board, the actual messages, including
another peer's tool calls and results. Nothing decides who goes next except the
peer currently holding the task.

The handoff mechanic is a tool: `transfer_to_<peer>`, one per other peer, bound
alongside each peer's own work tools. Calling it is how a peer gives up the task;
the tool's own return value is just "Transferred to X" (a real ToolMessage has to
exist for the call, or the provider rejects the turn), and the actual transfer
happens in code, by the node returning `Command(goto=<peer>, update=...)` --
LangGraph's own primitive for "the next node is not decided by a conditional edge,
it's decided by whichever node just ran." `langgraph-swarm` wraps exactly this
pattern behind `create_handoff_tool()` / `create_swarm()`; it isn't a dependency
here (see the plan's Step 10 entry) because the whole point of this demo is that
the mechanism is three lines of code, not a library call -- the same reasoning
Step 1 (ReAct) used to write its own loop by hand.

The graph is hand-built with `langgraph.graph.StateGraph`, for the same reason as
every other multi-agent demo in this codebase: each peer's own tool loop is its own
node with its own prompt, and Command-based routing has no `create_agent` shape to
fit into.

Budget and step accounting follow the same rule as every demo: every model call
calls `budget.check()` first, then charges one step, its tokens and one LLM call
once the reply comes back; `BudgetExceeded` propagates out of `compiled.invoke()`
uncaught, and `run()` turns it into `status="stopped_on_budget"`. The handoff cap
(`swarm_max_handoffs`) is a *soft* cap, unlike the hop cap in Step 9: reaching it
does not end the run or override a choice in code -- the refused transfer comes
back to the peer as a structured tool error, an observation it has to recover from,
the same shape every tool error takes (ground rule 6). A hard stop still comes from
the shared step cap if the peer cannot recover.
"""

import time
from typing import Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox, tool_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "swarm"
NAME = "Swarm"
SENTENCE = "No router: peers hand the task to each other, and whoever holds it owns it."
SHAPE = "peer ⇄ peer ⇄ peer"

PEERS = ("researcher", "analyst", "writer")

# Node ids double as `visited` entries, exactly like every other hand-built graph in
# this codebase. `cap` is pushed only when the exception handler in `run()` catches
# a spent budget -- it is not a real node.
GRAPH = """flowchart LR
    task[Task] -.->|entry peer| researcher[Researcher searches the corpus]
    task -.->|entry peer| analyst[Analyst queries the sales database]
    task -.->|entry peer| writer[Writer drafts from what peers reported]
    researcher <-->|handoff| analyst
    analyst <-->|handoff| writer
    writer <-->|handoff| researcher
    researcher -->|answers| respond[Answer returned]
    analyst -->|answers| respond
    writer -->|answers| respond
    researcher -.->|cap spent| cap[Stopped]
    analyst -.->|cap spent| cap
    writer -.->|cap spent| cap
"""

RESEARCHER_TOOLS = ("search_corpus",)
ANALYST_TOOLS = ("describe_schema", "run_sql")
WRITER_TOOLS: tuple[str, ...] = ()
PEER_WORK_TOOLS = {"researcher": RESEARCHER_TOOLS, "analyst": ANALYST_TOOLS, "writer": WRITER_TOOLS}
# The set every peer's work tools draw from, used only to size the run's Toolbox --
# same reasoning as Step 9's TOOL_NAMES.
TOOL_NAMES = (*RESEARCHER_TOOLS, *ANALYST_TOOLS)

# Both presets pair a corpus fact with an independent sales-database fact and route
# through a genuine A -> B -> A hand-back, so "passes between peers and returns" is
# always exercised. Facts verified against the bundled data (same ones Step 9
# verified against the same corpus and database): 1,297 Rock tracks falls in
# procurement-policy.md's "up to £2,500: budget holder's approval only" tier when
# read as a sum in pounds; the HX-40's sustained throughput is 840 samples/hour;
# there are 374 Metal tracks in Chinook.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "How many tracks are in the Rock genre in the sales database, and what "
            "does the procurement policy require for approving an order of that "
            "many pounds?"
        ),
        "entry": "researcher",
    },
    {
        "task": (
            "How many tracks are in the Metal genre in the sales database, and does "
            "that number exceed the HX-40's sustained throughput in samples per hour?"
        ),
        "entry": "analyst",
    },
]


# --- handoff tools -----------------------------------------------------------------


class HandoffArgs(BaseModel):
    reason: str = Field(description="One sentence: why this peer, and what it should do.")


def _make_handoff_tool(target: str) -> StructuredTool:
    def _invoke(reason: str) -> str:  # noqa: ARG001 -- reason is read from the tool_call, not the return
        return f"Transferred to {target}."

    return StructuredTool.from_function(
        func=_invoke,
        name=f"transfer_to_{target}",
        description=(
            f"Hand the task to {target}. Use this once {target} is better placed than "
            "you are for what's still missing. Give a one-sentence reason."
        ),
        args_schema=HandoffArgs,
    )


# --- prompts -------------------------------------------------------------------


_ROLE_TEXT = {
    "researcher": (
        "You are the researcher, one of three peers -- researcher, analyst and "
        "writer -- working this task together with no supervisor: whoever holds "
        "the task owns it until they hand it off. Your own tool is search_corpus, "
        "over a corpus of product, incident and policy documents. You also hold "
        "transfer_to_analyst (for anything the sales database would answer) and "
        "transfer_to_writer (once every part of the task has enough in this "
        "conversation for someone to draft the final answer)."
    ),
    "analyst": (
        "You are the analyst, one of three peers -- researcher, analyst and "
        "writer -- working this task together with no supervisor: whoever holds "
        "the task owns it until they hand it off. Your own tools are "
        "describe_schema and run_sql, against the sales database. You also hold "
        "transfer_to_researcher (for anything a written document would answer) "
        "and transfer_to_writer (once every part of the task has enough in this "
        "conversation for someone to draft the final answer)."
    ),
    "writer": (
        "You are the writer, one of three peers -- researcher, analyst and "
        "writer -- working this task together with no supervisor: whoever holds "
        "the task owns it until they hand it off. You have no work tools of your "
        "own -- only transfer_to_researcher and transfer_to_analyst, for when "
        "something the task needs is still missing from this conversation."
    ),
}

_FINISH_CLAUSE = (
    "Once you can fully answer the task yourself from what is already in this "
    "conversation, reply with the final answer in full sentences and call no "
    "tool. Otherwise, either use one of your own tools or call a transfer_to_ "
    "tool to hand the task to whichever peer is better placed for what's still "
    "missing. Before every tool call, write one short sentence saying what you "
    "are about to do and why, then call the tool in the same turn."
)

_NO_ANSWER_CLAUSE = (
    "You may never reply with a final answer this run -- always call a tool: "
    "either one of your own work tools, or a transfer_to_ tool. If you believe "
    "the task is fully answered, hand off to another peer anyway and let them "
    "decide, rather than answering yourself. Before every tool call, write one "
    "short sentence saying what you are about to do and why, then call the tool "
    "in the same turn."
)


def _system_prompt(name: str, force_no_answer: bool) -> str:
    clause = _NO_ANSWER_CLAUSE if force_no_answer else _FINISH_CLAUSE
    return f"{_ROLE_TEXT[name]}\n{clause}"


# --- graph state -------------------------------------------------------------------


class SwarmState(TypedDict):
    task: str
    messages: list[Any]  # shared across every peer -- the thing Step 9 does not have
    active_agent: str
    handoffs: int
    answer: str


def _retry_after_empty(runnable: Any, messages: list[Any]) -> Any:
    """One retry when a model returns neither text nor a tool call -- rare, but a
    plain re-ask is cheaper than treating it as a hard failure."""
    retry_messages = [
        *messages,
        HumanMessage(
            content="Your last reply had no text and no tool call. Reply with either a "
            "final answer or exactly one tool call."
        ),
    ]
    return runnable.invoke(retry_messages, config={"callbacks": get_callbacks()})


@observe(name="swarm.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`; `entry_peer`
    (default "researcher") -- which peer starts holding the task; `force_no_answer`
    (default False) -- every peer is told it may never answer directly, only hand
    off, so the run keeps passing the task around until a cap stops it."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    entry_peer = settings.get("entry_peer", "researcher")
    if entry_peer not in PEERS:
        entry_peer = "researcher"
    force_no_answer = bool(settings.get("force_no_answer", False))
    handoff_cap = settings_obj.swarm_max_handoffs

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["task"]

    def emit(agent: str, **fields: Any) -> Step:
        step = agent_run.add_step(agent=agent, **fields)
        if on_step is not None:
            on_step(agent, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        toolbox = Toolbox(run_id, names=list(TOOL_NAMES))
        primary, *fallbacks = agent_models()

        # Bind before wrapping in fallbacks -- see AGENTIC_IMPLEMENTATION_PLAN.md,
        # "Model access". Copied from Step 9, not imported (CLAUDE.md rule 1).
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        def _peer_tools(name: str) -> list[Any]:
            work = toolbox.as_langchain_tools(names=list(PEER_WORK_TOOLS[name]))
            handoffs = [_make_handoff_tool(target) for target in PEERS if target != name]
            return [*work, *handoffs]

        peer_tools = {name: _peer_tools(name) for name in PEERS}
        peer_runnables = {
            name: _with_fallbacks([m.bind_tools(peer_tools[name]) for m in [primary, *fallbacks]])
            for name in PEERS
        }

        # --- one peer's turn: loop until it hands off or answers ------------------

        def _run_peer_turn(name: str, state: SwarmState) -> Command:
            runnable = peer_runnables[name]
            system_prompt = _system_prompt(name, force_no_answer)
            messages: list[BaseMessage] = list(state["messages"])
            handoffs = state["handoffs"]

            while True:
                budget.check()
                call_messages = [SystemMessage(content=system_prompt), *messages]
                call_started = time.monotonic()
                response = runnable.invoke(call_messages, config={"callbacks": get_callbacks()})
                if not response.tool_calls and not (response.content or "").strip():
                    response = _retry_after_empty(runnable, call_messages)
                latency_ms = (time.monotonic() - call_started) * 1000
                tokens = count_tokens(response)
                budget.charge(steps=1, tokens=tokens, llm_calls=1)

                tool_calls = response.tool_calls or []
                if not tool_calls:
                    text = response.content or "(no result)"
                    emit(name, kind="decide", text=text, tokens=tokens, latency_ms=latency_ms)
                    messages.append(AIMessage(content=text))
                    return Command(
                        goto="respond",
                        update={"messages": messages, "answer": text, "active_agent": name},
                    )

                think_text = response.content or "(no reasoning text with this call)"
                emit(name, kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)
                messages.append(response)

                transfer_target: str | None = None
                transfer_reason = ""
                for call in tool_calls:
                    budget.check()
                    tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]

                    if tool_name.startswith("transfer_to_"):
                        target = tool_name.removeprefix("transfer_to_")
                        reason = tool_args.get("reason", "")
                        if handoffs >= handoff_cap:
                            observation = tool_error(
                                "refused",
                                f"The handoff cap ({handoff_cap}) has been reached this run.",
                                "No more handoffs are available -- answer with what you "
                                "have, or keep working with your own tools.",
                            )
                            emit(name, kind="observe", text=observation, tool=tool_name, args=tool_args)
                            messages.append(ToolMessage(content=observation, tool_call_id=call_id))
                            continue
                        handoffs += 1
                        emit(
                            name,
                            kind="handoff",
                            text=f"Hands off to {target}: {reason}",
                            tool=tool_name,
                            args={"from": name, "to": target, "reason": reason, "handoff_n": handoffs},
                        )
                        messages.append(
                            ToolMessage(content=f"Transferred to {target}.", tool_call_id=call_id)
                        )
                        transfer_target, transfer_reason = target, reason
                        continue

                    emit(name, kind="act", text=f"Calling {tool_name}.", tool=tool_name, args=tool_args)
                    tool_started = time.monotonic()
                    result = toolbox.call(tool_name, tool_args)
                    tool_latency_ms = (time.monotonic() - tool_started) * 1000
                    budget.charge(tool_calls=1)
                    emit(
                        name,
                        kind="observe",
                        text=result.output,
                        tool=tool_name,
                        args=tool_args,
                        latency_ms=tool_latency_ms,
                    )
                    messages.append(ToolMessage(content=result.output, tool_call_id=call_id))

                if transfer_target is not None:
                    return Command(
                        goto=transfer_target,
                        update={"messages": messages, "handoffs": handoffs, "active_agent": transfer_target},
                    )
                # No handoff this turn (either none was called, or every one was
                # refused by the cap) -- this peer keeps the task and loops again.

        # --- nodes -------------------------------------------------------------

        def _make_node(name: str) -> Callable[[SwarmState], Command]:
            def node(state: SwarmState) -> Command:
                visited.append(name)
                return _run_peer_turn(name, state)

            return node

        def respond_node(state: SwarmState) -> dict[str, Any]:
            visited.append("respond")
            return {}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(SwarmState)
        for peer in PEERS:
            graph.add_node(peer, _make_node(peer))
        graph.add_node("respond", respond_node)

        graph.add_conditional_edges(START, lambda state: state["active_agent"], dict(zip(PEERS, PEERS)))
        graph.add_edge("respond", END)

        compiled = graph.compile()

        initial_state: SwarmState = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "active_agent": entry_peer,
            "handoffs": 0,
            "answer": "",
        }
        final_state = compiled.invoke(
            initial_state,
            config={"callbacks": get_callbacks(), "recursion_limit": 4 * budget.max_steps + 10},
        )
        agent_run.output = final_state.get("answer", "")
        agent_run.status = "completed"

    except BudgetExceeded as stop:
        visited.append("cap")
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        emit("agent", kind="decide", text=agent_run.stop_reason)
    except Exception as exc:  # no provider configured, every provider refused, a build failure
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


# --- custody, rebuilt from the track --------------------------------------------


def custody_table(steps: list[Step]) -> list[dict[str, Any]]:
    """One row per holding period -- who had the task, for which steps, and why they
    gave it up. Rebuilt from the track's own `agent` field and its `handoff` rows,
    the same rebuild-from-the-track approach Step 9's `routing_table` uses: nothing
    swarm-specific is added to `AgentRun` itself."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step in steps:
        if step.agent not in PEERS:
            continue
        if current is None or step.agent != current["holder"]:
            if current is not None:
                rows.append(current)
            current = {
                "holder": step.agent,
                "from_step": step.index,
                "to_step": step.index,
                "steps": 0,
                "tool_calls": 0,
                "tokens": 0,
                "handoff_reason": "",
            }
        current["to_step"] = step.index
        current["steps"] += 1
        current["tokens"] += step.tokens
        if step.kind == "act":
            current["tool_calls"] += 1
        if step.kind == "handoff" and step.args:
            current["handoff_reason"] = step.args.get("reason", "")
    if current is not None:
        rows.append(current)
    return rows


def contrast_sentence(rows: list[dict[str, Any]]) -> str:
    """One sentence, computed from this run's own custody table -- swarm cannot read
    Step 9's collections, so the contrast is arithmetic on what actually happened
    here, not a claim about a run that didn't."""
    if not rows:
        return "Nothing has run yet."
    holders = " → ".join(row["holder"] for row in rows)
    handoffs = max(len(rows) - 1, 0)
    return (
        f"Held by {holders}: {handoffs} handoff(s), each decided inside the holder's own "
        "model call, with zero separate routing calls. Under Step 9's supervisor, the "
        f"same route costs one extra LLM call per hop on top of that -- at least "
        f"{handoffs} more here."
    )
