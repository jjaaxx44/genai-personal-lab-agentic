"""Human-in-the-loop gates: the agent proposes a write, and a person approves, edits
or rejects it before it runs -- not after.

`langgraph.types.interrupt()` is what makes this a pause and not a confirmation
dialog bolted on afterwards: the graph itself stops mid-execution, its state is
written to the MongoDB checkpointer, and `compiled.invoke()` returns to the caller
with the pending question rather than blocking. Nothing about the pause survives
in memory -- there is no thread sitting there waiting -- which is exactly what
makes it survive a page reload or a container restart: `resume()` rebuilds the run
from the checkpoint and calls `compiled.invoke(Command(resume=...), config)` on the
same `thread_id`, and LangGraph picks the graph back up at the node that paused.

**The gotcha that shapes every node here:** on resume, the interrupted node
re-executes from its first line -- `interrupt()` isn't a bookmark, it's a call that
simply doesn't return until a value is available. So `gate_node` below does nothing
before `interrupt()` except read state: no model call, no `emit()`, no
`budget.charge()`, no `visited.append()`. Everything with a side effect happens
after it returns, which is also the only code that then runs exactly once.

`langchain.agents.HumanInTheLoopMiddleware` is the packaged version of this same
idea for `create_agent`. This demo hand-builds the interrupt instead, so the
pause is visible as a graph node rather than hidden behind a middleware hook --
consistent with the rest of this app's `StateGraph` demos (Plan-and-Execute,
Reflexion, Agent memory), which exist to show phases `create_agent` doesn't
separate.

**Budget across a pause.** `resume()` gets no `Budget` argument (see the
`AgentDemo` contract), so the run's caps, its spend and its trajectory all travel
through the checkpoint as plain state (`steps`, `visited`, `budget`) rather than
living only in a Python object that a restart would destroy. `Budget.start()`
takes the agent time already spent, so the deadline's clock does not run while a
person is looking at the gate -- see AGENTIC_IMPLEMENTATION_PLAN.md, "Pausing
demos".

Budget and step accounting: `agent` calls `budget.check()` then charges one step,
its tokens and one LLM call. Every tool execution -- whether it ran straight
through or came out of a gate -- calls `budget.check()` then charges one tool
call. The gate itself, and routing, are free.
"""

import time
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from core.budget import Budget, BudgetExceeded
from core.checkpoint import get_checkpointer, thread_config
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import recent_runs, save_run
from core.tools import Toolbox
from core.tracing import get_callbacks, observe
from core.types import AgentRun, HumanDecision, Step, new_run_id, thread_id_for

DEMO = "hitl"
NAME = "Human-in-the-loop gates"
SENTENCE = "The run stops before a write and waits for a person to approve, edit or reject it."
SHAPE = "act → gate ⏸ → act"

# Node ids double as `visited` entries the graph map highlights, so a name changed
# here must change in both places. `respond` and `cap` are not real StateGraph
# nodes -- like every other demo's `cap`, they are pushed onto `visited` at the
# point the run actually ends, so the map can still show where that was.
GRAPH = """flowchart LR
    task[Task] --> agent[Agent proposes the next action]
    agent -->|ungated tool| tools[Tool runs]
    agent -->|gated tool| gate[Gate: paused for a person]
    gate -->|approve / edit| tools
    gate -->|reject + reason| agent
    tools --> agent
    agent -->|no tool call| respond[Answer]
    agent -.->|cap spent| cap[Stopped]
"""

TOOL_NAMES = ("search_corpus", "list_files", "read_file", "write_file")

# Preset 1 pairs with a suggested rejection in the page's caption -- the two-step
# version of this demo's whole claim: reject with a reason, and watch the agent's
# next proposal actually change. Preset 2 is written for Edit: the path or the
# content is worth changing rather than rejecting outright. Preset 3 puts an
# ungated call (list_files) before the gated one, so a reader sees a call run
# straight through before the gate stops the next one.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "Write a short note on our two supply risks -- lamp assembly lead times "
            "and the single-sourced high-sensitivity cartridges -- and save it as "
            "supply-risks.md."
        ),
    },
    {
        "task": (
            "Save the action list from the 2026-02-14 Marbury incident post-mortem, "
            "as a checklist, to incident-actions.md."
        ),
    },
    {
        "task": (
            "List the files already saved in this run, then save the on-call "
            "thirty-minute rule as a checklist in thirty-minute-rule.md."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant working from the bundled document corpus. You can "
    "search it, list and read files already saved in this run, and write new files "
    "into it. A person reviews some of your tool calls before they run: they may "
    "approve a call as proposed, edit its arguments and run the edited version, or "
    "reject it with a reason. When a call comes back rejected, read the reason and "
    "change your next proposal to address it -- do not repeat the same call "
    "unchanged. Once you have done what the task asks, answer in your own words "
    "rather than calling another tool. Before every tool call, write one short "
    "sentence in your reply saying what you are about to do and why, then call "
    "the tool in the same turn."
)


# --- graph state -------------------------------------------------------------------


class HitlState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    queue: list[dict[str, Any]]  # tool calls from the last AIMessage, not yet handled
    gated: list[str]  # frozen at run start; which tool names pause for a person
    answer: str
    # The three fields below are the durable backup of the run: everything `steps`,
    # `visited` and `budget` on the `AgentRun`/`Budget` objects hold, mirrored into
    # graph state so `resume()` can rebuild them after an app restart, when the
    # Python objects `run()` built are long gone.
    steps: list[dict[str, Any]]
    visited: list[str]
    budget: dict[str, Any]
    run_id: str


DISPATCH_MAP = {"agent": "agent", "gate": "gate", "tools": "tools", "end": END}


def dispatch(state: HitlState) -> str:
    """After `agent`, `tools` or `gate`: run the next queued call (gated or not), or --
    once the queue is empty -- go back to `agent` for its next decision, unless the
    queue emptied because `agent` itself just answered directly, in which case the
    last message is that answer, not a tool result, and the run is over."""
    queue = state["queue"]
    if not queue:
        last = state["messages"][-1] if state["messages"] else None
        return "agent" if isinstance(last, ToolMessage) else "end"
    return "gate" if queue[0]["name"] in state["gated"] else "tools"


# --- budget <-> plain-dict snapshot, for the checkpoint -----------------------------


def _budget_snapshot(budget: Budget) -> dict[str, Any]:
    return {
        "max_steps": budget.max_steps,
        "max_tokens": budget.max_tokens,
        "deadline_s": budget.deadline_s,
        "steps_used": budget.steps_used,
        "tokens_used": budget.tokens_used,
        "tool_calls": budget.tool_calls,
        "llm_calls": budget.llm_calls,
        "elapsed_s": budget.elapsed_s,
    }


def _budget_from(snapshot: dict[str, Any]) -> Budget:
    budget = Budget(
        max_steps=snapshot["max_steps"],
        max_tokens=snapshot["max_tokens"],
        deadline_s=snapshot["deadline_s"],
        steps_used=snapshot["steps_used"],
        tokens_used=snapshot["tokens_used"],
        tool_calls=snapshot["tool_calls"],
        llm_calls=snapshot["llm_calls"],
    )
    budget.start(elapsed_s=snapshot["elapsed_s"])
    return budget


def _default_gated_tools() -> list[str]:
    settings = get_settings()
    return [name.strip() for name in settings.hitl_gated_tools.split(",") if name.strip()]


# --- run context: what the graph's node closures read and write --------------------


class _RunContext:
    """A mutable box the node closures read `agent_run`/`budget`/`visited` through,
    rather than capturing those three objects directly. `resume()` needs to read the
    checkpoint *before* it knows what they should be -- building the graph once,
    pointing the closures at a throwaway context, reading `get_state()`, then
    swapping `ctx.agent_run` / `ctx.budget` / `ctx.visited` for the rebuilt ones --
    which only works if the node closures look the values up through `ctx` at call
    time rather than holding their own reference to the originals."""

    __slots__ = ("agent_run", "budget", "visited", "on_step")

    def __init__(
        self,
        agent_run: AgentRun,
        budget: Budget,
        visited: list[str],
        on_step: Callable[[str, Step], None] | None,
    ) -> None:
        self.agent_run = agent_run
        self.budget = budget
        self.visited = visited
        self.on_step = on_step


def _emit(ctx: _RunContext, agent: str, **fields: Any) -> Step:
    step = ctx.agent_run.add_step(agent=agent, **fields)
    if ctx.on_step is not None:
        ctx.on_step(agent, step)
    return step


def _sync(ctx: _RunContext) -> dict[str, Any]:
    """What every node returns alongside its own state changes, so the checkpoint
    written after each step is always a complete backup of the run so far."""
    return {
        "steps": [s.model_dump() for s in ctx.agent_run.steps],
        "visited": list(ctx.visited),
        "budget": _budget_snapshot(ctx.budget),
    }


# --- the graph -----------------------------------------------------------------


def _build_graph(run_id: str, ctx: _RunContext) -> Any:
    toolbox = Toolbox(run_id, names=list(TOOL_NAMES))
    tools = toolbox.as_langchain_tools()

    primary, *fallbacks = agent_models()

    # Bind before wrapping in fallbacks -- RunnableWithFallbacks does not forward
    # bind_tools(), but with_fallbacks() itself works on any Runnable, bound or
    # not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
    def _with_fallbacks(bound: list[Any]) -> Any:
        head, *rest = bound
        return head.with_fallbacks(rest) if rest else head

    act_models = [m.bind_tools(tools) for m in [primary, *fallbacks]]
    act_runnable = _with_fallbacks(act_models)

    def _run_tool(call: dict[str, Any], args: dict[str, Any]) -> ToolMessage:
        ctx.budget.check()
        tool_started = time.monotonic()
        result = toolbox.call(call["name"], args)
        tool_latency_ms = (time.monotonic() - tool_started) * 1000
        ctx.budget.charge(tool_calls=1)
        _emit(
            ctx,
            "agent",
            kind="observe",
            text=result.output,
            tool=call["name"],
            args=args,
            latency_ms=tool_latency_ms,
        )
        return ToolMessage(content=result.output, tool_call_id=call["id"])

    def agent_node(state: HitlState) -> dict[str, Any]:
        ctx.budget.check()
        ctx.visited.append("agent")
        call_started = time.monotonic()
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = act_runnable.invoke(messages, config={"callbacks": get_callbacks()})
        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(response)
        ctx.budget.charge(steps=1, tokens=tokens, llm_calls=1)

        tool_calls = response.tool_calls or []
        if tool_calls:
            think_text = response.content or "(no reasoning text with this call)"
            _emit(ctx, "agent", kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)
            queue = []
            for call in tool_calls:
                tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]
                queue.append({"id": call_id, "name": tool_name, "args": tool_args, "why": think_text})
                _emit(ctx, "agent", kind="act", text=f"Proposed {tool_name}.", tool=tool_name, args=tool_args)
            return {"messages": [response], "queue": queue, **_sync(ctx)}

        answer = response.content or ""
        _emit(ctx, "agent", kind="decide", text=answer, tokens=tokens, latency_ms=latency_ms)
        return {"messages": [response], "answer": answer, "queue": [], **_sync(ctx)}

    def tools_node(state: HitlState) -> dict[str, Any]:
        ctx.visited.append("tools")
        call = state["queue"][0]
        tool_message = _run_tool(call, call["args"])
        return {"messages": [tool_message], "queue": state["queue"][1:], **_sync(ctx)}

    def gate_node(state: HitlState) -> dict[str, Any]:
        call = state["queue"][0]
        payload = {"tool": call["name"], "args": call["args"], "call_id": call["id"], "why": call["why"]}
        decision = interrupt(payload)
        # Nothing above this line has a side effect: this node re-runs from the top
        # on resume, so everything below -- the only code with a side effect -- runs
        # exactly once, when the decision has actually arrived.
        ctx.visited.append("gate")
        verdict = decision.get("verdict", "reject")
        edited = decision.get("edited")
        reason = decision.get("reason", "")

        label = {
            "approve": "Approved by a person",
            "edit": "Edited by a person",
            "reject": "Rejected by a person",
        }.get(verdict, f"Unrecognised verdict '{verdict}', treated as a rejection")
        if reason:
            label += f": {reason}"
        _emit(
            ctx,
            "agent",
            kind="gate",
            text=label,
            tool=call["name"],
            args={"verdict": verdict, "proposed": call["args"], "edited": edited},
        )

        if verdict == "approve":
            tool_message = _run_tool(call, call["args"])
        elif verdict == "edit":
            tool_message = _run_tool(call, edited if edited is not None else call["args"])
        else:
            rejection_text = (
                f"REJECTED by a person: {reason or '(no reason given)'}. The action was "
                "not run. Do not repeat it unchanged -- adjust to the reason."
            )
            tool_message = ToolMessage(content=rejection_text, tool_call_id=call["id"])

        return {"messages": [tool_message], "queue": state["queue"][1:], **_sync(ctx)}

    graph = StateGraph(HitlState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("gate", gate_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", dispatch, DISPATCH_MAP)
    graph.add_conditional_edges("tools", dispatch, DISPATCH_MAP)
    graph.add_conditional_edges("gate", dispatch, DISPATCH_MAP)

    return graph.compile(checkpointer=get_checkpointer(DEMO))


def _apply_result(compiled: Any, config: dict[str, Any], result: dict[str, Any], ctx: _RunContext) -> None:
    """Turns one finished `compiled.invoke()` call into the run's status: paused at a
    gate, or completed. `BudgetExceeded` and any other exception never reach here --
    they propagate out of `invoke()` and are handled by the caller's try/except."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        state = compiled.get_state(config)
        paused_at = state.next[0] if state.next else "gate"
        if paused_at not in ctx.visited:
            ctx.visited.append(paused_at)
        payload = interrupts[0].value
        ctx.agent_run.status = "needs_human"
        ctx.agent_run.stop_reason = (
            f"Waiting for a person to approve, edit or reject a call to {payload.get('tool', 'a tool')}."
        )
        return
    ctx.agent_run.output = result.get("answer", "")
    ctx.agent_run.status = "completed"


@observe(name="hitl.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `gated_tools`
    (defaults to `settings.hitl_gated_tools` from config)."""
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    gated = list(settings.get("gated_tools") or _default_gated_tools())

    run_id = new_run_id()
    thread_id = thread_id_for(DEMO, run_id)
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task, thread_id=thread_id)
    ctx = _RunContext(agent_run=agent_run, budget=budget, visited=["task"], on_step=on_step)

    budget.start()

    try:
        compiled = _build_graph(run_id, ctx)
        initial_state: HitlState = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "queue": [],
            "gated": gated,
            "answer": "",
            "steps": [],
            "visited": ctx.visited,
            "budget": _budget_snapshot(budget),
            "run_id": run_id,
        }
        config = {
            **thread_config(thread_id),
            "callbacks": get_callbacks(),
            "recursion_limit": 4 * budget.max_steps + 10,
        }
        result = compiled.invoke(initial_state, config=config)
        _apply_result(compiled, config, result, ctx)

    except BudgetExceeded as stop:
        ctx.visited.append("cap")
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        _emit(ctx, "agent", kind="decide", text=agent_run.stop_reason)
    except Exception as exc:  # no provider configured, every provider refused, Mongo unreachable
        agent_run.status = "failed"
        agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        _emit(ctx, "agent", kind="decide", text=agent_run.stop_reason)

    agent_run.visited = ctx.visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    # budget.elapsed_s, not a separately tracked wall clock: it already excludes any
    # time spent waiting for a person (Budget.start(elapsed_s=...) backdates it on a
    # resume), so latency_ms stays accurate -- agent time only -- across a pause.
    agent_run.latency_ms = budget.elapsed_s * 1000
    save_run(agent_run)
    return agent_run


@observe(name="hitl.resume")
def resume(thread_id: str, decision: HumanDecision, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract's other half. Rebuilds the run and the budget from the
    checkpoint rather than from anything in memory -- this has to work even after the
    app has restarted and lost every Python object `run()` built."""
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    run_id = thread_id.split(":", 1)[-1]

    placeholder_run = AgentRun(run_id=run_id, demo=DEMO, task="", thread_id=thread_id)
    placeholder_budget = Budget(max_steps=1, max_tokens=1, deadline_s=1).start()
    ctx = _RunContext(agent_run=placeholder_run, budget=placeholder_budget, visited=[], on_step=on_step)

    try:
        compiled = _build_graph(run_id, ctx)
        config = thread_config(thread_id)
        state = compiled.get_state(config)
        if not state.interrupts:
            return AgentRun(
                run_id=run_id,
                demo=DEMO,
                task="",
                thread_id=thread_id,
                status="failed",
                stop_reason="This run is not waiting for a decision -- it may have finished or been cleared.",
            )

        values = state.values
        ctx.agent_run = AgentRun(
            run_id=run_id,
            demo=DEMO,
            task=values.get("task", ""),
            thread_id=thread_id,
            steps=values.get("steps", []),
        )
        ctx.budget = _budget_from(values.get("budget", _budget_snapshot(placeholder_budget)))
        ctx.visited = list(values.get("visited", []))

        result = compiled.invoke(
            Command(resume=decision.model_dump()),
            config={
                **config,
                "callbacks": get_callbacks(),
                "recursion_limit": 4 * ctx.budget.max_steps + 10,
            },
        )
        _apply_result(compiled, config, result, ctx)

    except BudgetExceeded as stop:
        ctx.visited.append("cap")
        ctx.agent_run.status = "stopped_on_budget"
        ctx.agent_run.stop_reason = stop.reason
        _emit(ctx, "agent", kind="decide", text=ctx.agent_run.stop_reason)
    except Exception as exc:
        ctx.agent_run.status = "failed"
        ctx.agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        _emit(ctx, "agent", kind="decide", text=ctx.agent_run.stop_reason)

    ctx.agent_run.visited = ctx.visited
    ctx.agent_run.llm_calls = ctx.budget.llm_calls
    ctx.agent_run.tool_calls = ctx.budget.tool_calls
    ctx.agent_run.tokens = ctx.budget.tokens_used
    ctx.agent_run.latency_ms = ctx.budget.elapsed_s * 1000
    save_run(ctx.agent_run)
    return ctx.agent_run


def pending(thread_id: str | None = None) -> dict[str, Any] | None:
    """The most recent run still waiting for a decision -- `run`, `budget` and
    `payload` (the gate's proposal) -- or `None` if there is nothing waiting.

    Called automatically on every page load with no `thread_id`, so a browser
    refresh or an app restart puts the gate back on screen with no click: that is
    what proves the pause survived rather than just this Streamlit rerun. Soft
    failure on anything that goes wrong -- an unreachable database, same as every
    other demo, but also a build failure (a provider dropped out of the
    environment between runs) -- because this runs on every page load, and ground
    rule 10 says a page load never crashes.
    """
    try:
        if thread_id is None:
            candidates = [r for r in recent_runs(DEMO, limit=20) if r.get("status") == "needs_human"]
            if not candidates:
                return None
            thread_id = candidates[0].get("thread_id")
        if not thread_id:
            return None

        run_id = thread_id.split(":", 1)[-1]
        placeholder_run = AgentRun(run_id=run_id, demo=DEMO, task="", thread_id=thread_id)
        placeholder_budget = Budget(max_steps=1, max_tokens=1, deadline_s=1).start()
        ctx = _RunContext(agent_run=placeholder_run, budget=placeholder_budget, visited=[], on_step=None)
        compiled = _build_graph(run_id, ctx)
        config = thread_config(thread_id)
        state = compiled.get_state(config)
        if not state.interrupts:
            return None

        values = state.values
        agent_run = AgentRun(
            run_id=run_id,
            demo=DEMO,
            task=values.get("task", ""),
            thread_id=thread_id,
            status="needs_human",
            steps=values.get("steps", []),
            visited=values.get("visited", []),
        )
        budget = _budget_from(values.get("budget", _budget_snapshot(placeholder_budget)))
        payload = state.interrupts[0].value
        return {"run": agent_run, "budget": budget, "payload": payload}
    except Exception:
        return None


def budget_for(thread_id: str) -> Budget | None:
    """The accurate `Budget` for a finished or paused thread, read back from the
    checkpoint's own `budget` field rather than tracked by the page -- `resume()`
    builds its own `Budget` internally from the checkpoint and has nowhere in the
    `AgentDemo` contract to hand it back, so the page calls this instead, right
    after `run()` or `resume()` returns, to get the caps and the true spend
    (including anything charged during a resume) for the budget strip. `None` if
    the checkpoint can't be read, in which case the page falls back to the caps it
    already has."""
    try:
        run_id = thread_id.split(":", 1)[-1]
        placeholder_run = AgentRun(run_id=run_id, demo=DEMO, task="", thread_id=thread_id)
        placeholder_budget = Budget(max_steps=1, max_tokens=1, deadline_s=1).start()
        ctx = _RunContext(agent_run=placeholder_run, budget=placeholder_budget, visited=[], on_step=None)
        compiled = _build_graph(run_id, ctx)
        state = compiled.get_state(thread_config(thread_id))
        snapshot = (state.values or {}).get("budget") if state else None
        return _budget_from(snapshot) if snapshot else None
    except Exception:
        return None
