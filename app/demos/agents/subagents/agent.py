"""Sub-agent delegation and context isolation: the parent works the task with the
same bounded tool-calling loop every other demo in this repo uses, plus one extra
tool -- `delegate_subagent` -- that hands a self-contained piece of the task to a
sub-agent instead of doing it inline. The sub-agent gets a fresh message history
(it never sees the parent's conversation), its own tools and its own step budget,
and reports back only its final reply. Everything it thought, called and observed
still lands on the trajectory -- tagged `agent="subagent"` and indented under the
handoff step via `parent_index` -- but none of that detail ever enters the
parent's own `messages`. That is the isolation this demo is about: the parent's
own prompt stays the same size no matter how much work the delegated subtask took.

The "shared context" toggle (`isolated=False`) does not run a second, differently
wired version of the delegation -- it removes `delegate_subagent` entirely. With
no way to hand a subtask off, the model has to do the whole task itself, in the
one loop, so every tool call and its observation sits in `state["messages"]` for
the rest of the run. This is deliberately simpler than running a nested loop and
splicing its messages back into the parent's history: "shared" means there is no
separation at all, not a second copy of one. The measurable difference is exactly
what real context isolation buys -- the isolated run's parent-side prompt only
ever grows by short summaries; the shared run's grows by every raw tool result any
part of the task ever touched, and that larger history is resent on every
subsequent LLM call, so the shared run's total tokens end up visibly higher for
the same task.

Graph shape: two nodes, `act` and `tools`, looping until a final answer -- the
same skeleton Escalation and Plan-and-Execute build on, with no judge/escalate/
wait pieces. `delegate_subagent` is dispatched by hand inside `tools_node`,
exactly like every other tool call in this codebase (none of these demos use
LangGraph's `ToolNode`); it never actually runs the stub function bound to it.

Budget: the sub-agent gets its own `Budget` -- its own step cap
(`subagent_max_steps`, or 0 when the "force failure" toggle is on, so its very
first `check()` raises before any LLM call), and token/deadline caps carved from
whatever the parent has left. Its real spend (tokens, tool calls, LLM calls -- not
steps, which stay a purely parent-side count) is charged onto the parent's own
`Budget` too, so the top-level strip always shows the true total cost of the run,
delegated work included. A sub-agent that hits its own cap doesn't raise out of
`tools_node` -- `BudgetExceeded` is caught right there and turned into a stopped-
early observation the parent reads like any other tool result, which is what
keeps one failed delegation from taking the parent down with it.
"""

import time
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox, is_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "subagents"
NAME = "Sub-agent delegation"
SENTENCE = "Bounded work goes to a sub-agent with its own context and budget; only a summary comes back."
SHAPE = "parent → sub-agent ⤵ → summary"

# `handoff` and `respond`, like every other demo's `cap`, are not real StateGraph
# nodes -- delegate_subagent is dispatched inside `tools`, same as any other tool
# call. They're pushed onto `visited` at the point they actually happen, so the
# graph map can still show where the run went.
GRAPH = """flowchart LR
    task[Task] --> act[Parent works the task]
    act -->|tool call| tools[Tool runs]
    tools --> act
    tools -->|delegate_subagent, isolated mode only| handoff[Sub-agent works the subtask alone]
    handoff -->|summary, or its own stop reason| tools
    act -->|final answer| respond[Answer]
    act -.->|cap spent| cap[Stopped]
"""

# No web_search, same reasoning Reflexion/Escalation/Plan-and-Execute use for the
# same tool list: a missing fact stays missing, which keeps the presets
# deterministic. Both the parent and the sub-agent get this exact set.
TOOL_NAMES = ("search_corpus", "describe_schema", "run_sql")

# Both presets pair a corpus fact with an independent SQL fact -- two genuinely
# self-contained lookups, so there is always something a reader can watch get
# delegated rather than having to hope the model invents a split on its own.
# Verified against the bundled data: the £5,000 order falls in procurement-
# policy.md's £2,501-£25,000 tier: two written quotes, budget holder plus
# finance; Chinook has 1,297 Rock tracks and 13 USA customers; high-sensitivity
# cartridges are single-sourced from Verrall & Sons, 5-week lead time.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "What does the procurement policy say about the approval requirement "
            "for an order over £5,000, and how many tracks are in the Rock genre "
            "in the sales database?"
        ),
    },
    {
        "task": (
            "What does the procurement policy say about the lead time and "
            "supplier for high-sensitivity cartridges, and how many customers "
            "does the sales database have in the USA?"
        ),
    },
]

ACT_SYSTEM_PROMPT_ISOLATED = (
    "You are a helpful assistant working from the bundled tools. You already have "
    "direct access to every tool a sub-agent would use -- delegating never gives "
    "you a capability you lack, it only keeps a piece of work out of your own "
    "context. Part of the task below may be a self-contained lookup or "
    "calculation you can state in one sentence and don't need to see the raw "
    "detail of -- for that part, call delegate_subagent with exactly that "
    "subtask, and use its reply as the answer to that part. Use the other tools "
    "directly for whatever you need to reason about yourself, or that isn't "
    "self-contained. Once every part of the task is covered, answer in full. "
    "Before every tool call, write one short sentence in your reply saying what "
    "you are about to do and why, then call the tool in the same turn."
)

ACT_SYSTEM_PROMPT_SHARED = (
    "You are a helpful assistant working from the bundled tools. Work every part "
    "of the task yourself, calling a tool whenever a specific fact needs "
    "verification. Once every part of the task is covered, answer in full. Before "
    "every tool call, write one short sentence in your reply saying what you are "
    "about to do and why, then call the tool in the same turn."
)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a sub-agent working one bounded, self-contained piece of a larger "
    "task on behalf of a parent agent. You do not see the parent's conversation "
    "-- only the subtask below. Work it using the tools available, then reply "
    "with a short, direct result (no more than two or three sentences) that "
    "fully answers the subtask. The parent will read only this final reply, not "
    "your intermediate tool calls. Before every tool call, write one short "
    "sentence in your reply saying what you are about to do and why, then call "
    "the tool in the same turn."
)


# --- the delegate tool -------------------------------------------------------------


class DelegateArgs(BaseModel):
    subtask: str = Field(
        description=(
            "The self-contained piece of work to hand off, stated as a full "
            "question or instruction the sub-agent can act on with no other "
            "context -- it will not see anything else about this task."
        )
    )


def _delegate_stub(subtask: str) -> str:
    # Never actually called: tools_node dispatches delegate_subagent by hand,
    # exactly like every other tool call in this codebase (see module docstring).
    raise RuntimeError("delegate_subagent is dispatched by hand in tools_node.")


DELEGATE_TOOL = StructuredTool.from_function(
    func=_delegate_stub,
    name="delegate_subagent",
    description=(
        "Hand off a self-contained piece of the task to a sub-agent that works it "
        "alone, with its own tools and its own step budget, and reports back only "
        "its final result -- not its intermediate tool calls or reasoning. Use "
        "this for a lookup or calculation you can state in one sentence and "
        "don't need to see the raw detail of. Do not use it for the part of the "
        "task you need to reason about yourself using what it returns."
    ),
    args_schema=DelegateArgs,
)


# --- graph state -------------------------------------------------------------------


class SubagentState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    pending_calls: list[dict[str, Any]]
    answer: str
    decision: str


# --- the sub-agent's own loop --------------------------------------------------------


def _run_subagent(
    subtask: str,
    handoff_index: int,
    parent_budget: Budget,
    run_id: str,
    force_failure: bool,
    subagent_max_steps: int,
    emit: Callable[..., Step],
) -> str:
    """Runs one bounded, isolated sub-agent turn and returns only its final reply.

    Its own fresh message history, its own `Toolbox`, its own `Budget` -- capped by
    `subagent_max_steps` (or 0 when `force_failure` clamps it, so the very first
    `check()` raises before any LLM call) and by whatever tokens/deadline the
    parent has left. Every thought, call and observation still lands on the shared
    trajectory, tagged `agent="subagent"` and `parent_index=handoff_index` so it
    renders indented under the handoff -- but only the string this returns ever
    reaches the parent's own `messages`.
    """
    remaining_tokens = max(parent_budget.max_tokens - parent_budget.tokens_used, 1)
    remaining_deadline = max(parent_budget.deadline_s - parent_budget.elapsed_s, 1.0)
    sub_budget = Budget(
        max_steps=0 if force_failure else subagent_max_steps,
        max_tokens=remaining_tokens,
        deadline_s=remaining_deadline,
    ).start()

    sub_toolbox = Toolbox(run_id, names=list(TOOL_NAMES))
    sub_tools = sub_toolbox.as_langchain_tools()
    primary, *fallbacks = agent_models()
    sub_models = [m.bind_tools(sub_tools) for m in [primary, *fallbacks]]
    head, *rest = sub_models
    sub_runnable = head.with_fallbacks(rest) if rest else head

    messages: list[Any] = [SystemMessage(content=SUBAGENT_SYSTEM_PROMPT), HumanMessage(content=subtask)]
    findings: list[str] = []

    try:
        while True:
            sub_budget.check()
            call_started = time.monotonic()
            response = sub_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            sub_budget.charge(steps=1, tokens=tokens, llm_calls=1)
            parent_budget.charge(tokens=tokens, llm_calls=1)

            tool_calls = response.tool_calls or []
            if not tool_calls:
                result_text = response.content or "(no result)"
                emit(
                    "subagent",
                    kind="decide",
                    text=result_text,
                    tokens=tokens,
                    latency_ms=latency_ms,
                    parent_index=handoff_index,
                )
                return result_text

            think_text = response.content or "(no reasoning text with this call)"
            emit(
                "subagent",
                kind="think",
                text=think_text,
                tokens=tokens,
                latency_ms=latency_ms,
                parent_index=handoff_index,
            )
            messages.append(response)
            for call in tool_calls:
                sub_budget.check()
                tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]
                emit(
                    "subagent",
                    kind="act",
                    text=f"Calling {tool_name}.",
                    tool=tool_name,
                    args=tool_args,
                    parent_index=handoff_index,
                )
                tool_started = time.monotonic()
                result = sub_toolbox.call(tool_name, tool_args)
                tool_latency_ms = (time.monotonic() - tool_started) * 1000
                sub_budget.charge(tool_calls=1)
                parent_budget.charge(tool_calls=1)
                if not is_error(result.output):
                    findings.append(result.output)
                emit(
                    "subagent",
                    kind="observe",
                    text=result.output,
                    tool=tool_name,
                    args=tool_args,
                    latency_ms=tool_latency_ms,
                    parent_index=handoff_index,
                )
                messages.append(ToolMessage(content=result.output, tool_call_id=call_id))
    except BudgetExceeded as stop:
        text = f"Sub-agent stopped before finishing: {stop.reason}"
        if findings:
            text += " Partial findings: " + "; ".join(f[:200] for f in findings[-2:])
        emit("subagent", kind="observe", text=text, parent_index=handoff_index)
        return text


# --- the parent's run ----------------------------------------------------------------


@observe(name="subagents.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`; `isolated`
    (default True) -- whether delegate_subagent is offered at all, versus the
    parent doing everything itself in one shared context; `force_failure`
    (default False) -- clamps the sub-agent's own step cap to zero so the first
    delegated call fails deterministically, to show the parent recovering from
    it; `subagent_max_steps` (default from config) -- the step cap a launched
    sub-agent gets before it stops on its own cap, overridable from the UI."""
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    isolated = bool(settings.get("isolated", True))
    force_failure = bool(settings.get("force_failure", False))
    raw_cap = settings.get("subagent_max_steps")
    subagent_max_steps = int(raw_cap) if raw_cap is not None else get_settings().subagent_max_steps

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
        tools = toolbox.as_langchain_tools()
        bound_tools = [*tools, DELEGATE_TOOL] if isolated else tools
        act_system_prompt = ACT_SYSTEM_PROMPT_ISOLATED if isolated else ACT_SYSTEM_PROMPT_SHARED

        primary, *fallbacks = agent_models()

        # Bind before wrapping in fallbacks -- RunnableWithFallbacks does not
        # forward bind_tools(), but with_fallbacks() itself works on any Runnable,
        # bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        act_models = [m.bind_tools(bound_tools) for m in [primary, *fallbacks]]
        act_runnable = _with_fallbacks(act_models)

        # --- nodes -------------------------------------------------------------

        def act_node(state: SubagentState) -> dict[str, Any]:
            budget.check()
            visited.append("act")
            call_started = time.monotonic()
            messages = [SystemMessage(content=act_system_prompt), *state["messages"]]
            response = act_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            tool_calls = response.tool_calls or []
            if tool_calls:
                think_text = response.content or "(no reasoning text with this call)"
                emit("agent", kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)
                pending_calls = []
                for call in tool_calls:
                    tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]
                    if tool_name == "delegate_subagent":
                        step = emit(
                            "agent",
                            kind="handoff",
                            text=f"Delegating: {tool_args.get('subtask', '')}",
                            tool=tool_name,
                            args=tool_args,
                        )
                    else:
                        step = emit(
                            "agent", kind="act", text=f"Calling {tool_name}.", tool=tool_name, args=tool_args
                        )
                    pending_calls.append(
                        {"id": call_id, "name": tool_name, "args": tool_args, "step_index": step.index}
                    )
                return {"messages": [response], "pending_calls": pending_calls, "decision": "tools"}

            answer = response.content or ""
            emit("agent", kind="decide", text=answer, tokens=tokens, latency_ms=latency_ms)
            visited.append("respond")
            return {"messages": [response], "answer": answer, "decision": "respond"}

        def tools_node(state: SubagentState) -> dict[str, Any]:
            visited.append("tools")
            tool_messages: list[ToolMessage] = []
            for call in state["pending_calls"]:
                budget.check()
                if call["name"] == "delegate_subagent":
                    visited.append("handoff")
                    summary = _run_subagent(
                        call["args"].get("subtask", ""),
                        call["step_index"],
                        budget,
                        run_id,
                        force_failure,
                        subagent_max_steps,
                        emit,
                    )
                    tool_messages.append(ToolMessage(content=summary, tool_call_id=call["id"]))
                    continue

                tool_started = time.monotonic()
                result = toolbox.call(call["name"], call["args"])
                tool_latency_ms = (time.monotonic() - tool_started) * 1000
                budget.charge(tool_calls=1)
                emit(
                    "agent",
                    kind="observe",
                    text=result.output,
                    tool=call["name"],
                    args=call["args"],
                    latency_ms=tool_latency_ms,
                )
                tool_messages.append(ToolMessage(content=result.output, tool_call_id=call["id"]))

            return {"messages": tool_messages, "pending_calls": [], "decision": "act"}

        graph = StateGraph(SubagentState)
        graph.add_node("act", act_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "act")
        graph.add_conditional_edges("act", lambda s: s["decision"], {"tools": "tools", "respond": END})
        graph.add_edge("tools", "act")

        compiled = graph.compile()

        initial_state: SubagentState = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "pending_calls": [],
            "answer": "",
            "decision": "",
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
