"""Escalation and handoff: the agent works the task, a separate judge scores whether
the draft is actually *supported* by what was retrieved, and -- below a confidence
threshold, or after too many tool calls fail in a row -- the agent stops and writes
a handoff packet instead of guessing. A person reads the packet, answers the one
question it asks, and the run picks up exactly where it left off.

The judge is a second model call, not a field the actor reports about itself: an
actor grading its own homework is the textbook case of overconfidence, and giving
the judge its own node and its own prompt is what makes that grading visible on the
trajectory rather than folded invisibly into the actor's reply. Its rubric is
support, not plausibility -- could every specific claim in the draft be traced back
to a tool result or a person's answer actually retrieved this run? -- which is a
narrower, checkable question than "is this a good answer."

The handoff packet itself carries two different accounts of the same run, on
purpose: the model's own summary of what it tried (`tried`/`found`), and a
mechanically built trail -- one line per trajectory step, built from `Step` data
with no model in the loop -- so a packet that misrepresents what actually happened
is checkable against a record that can't.

Mechanically, this demo pauses the same way Human-in-the-loop gates (Step 6) does:
`interrupt()` at `wait`, with nothing before it and everything with a side effect
after it, because the node re-executes from the top on resume. The run's caps,
spend and trajectory travel through the checkpoint as plain state (`steps`,
`visited`, `budget`) for the same reason -- `resume()` gets no `Budget` argument,
and has to work even after the app restarts -- see
AGENTIC_IMPLEMENTATION_PLAN.md, "Pausing demos". `HumanDecision` is reused
unchanged: `approve` + `reason` is the person's answer, and `reject` + `reason`
closes the case without answering.

Budget and step accounting: `act`, `judge` and `escalate` each call `budget.check()`
then charge one step, their tokens and one LLM call. Every tool call inside `tools`
calls `budget.check()` then charges one tool call. `wait` and routing are free.
"""

import time
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.checkpoint import get_checkpointer, thread_config
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import recent_runs, save_run
from core.tools import Toolbox, is_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, HumanDecision, Step, new_run_id, thread_id_for

DEMO = "escalation"
NAME = "Escalation and handoff"
SENTENCE = "Below its own confidence threshold the agent stops and writes a handoff packet instead of guessing."
SHAPE = "act ↻ → judge → escalate ⏸"

# Node ids double as `visited` entries the graph map highlights, so a name changed
# here must change in both places. `respond`, `closed` and `cap` are not real
# StateGraph nodes -- like every other demo's `cap`, they are pushed onto `visited`
# at the point the run actually ends, so the map can still show where that was.
GRAPH = """flowchart LR
    task[Task] --> act[Agent works the task]
    act -->|tool call| tools[Tool runs]
    tools --> act
    tools -->|too many failed calls| escalate[Handoff packet written]
    act -->|draft answer| judge[Judge scores confidence and coverage]
    judge -->|at or above threshold| respond[Answer]
    judge -->|below threshold| escalate
    escalate --> wait[Waiting for a person]
    wait -->|answered| act
    wait -->|case closed| closed[Handed over, not answered]
    act -.->|cap spent| cap[Stopped]
"""

# No web_search: a missing fact stays missing, which is what keeps the presets
# deterministic (same reasoning Steps 3-4 use for the same tool list decision).
TOOL_NAMES = ("search_corpus", "describe_schema", "run_sql")

# Preset 1 is unanswerable from the corpus alone -- there is no lamp unit price
# anywhere in samples/corpus/ -- so it escalates on the judge; the page's caption
# suggests an answer that's actually checkable against procurement-policy.md, and
# the resumed run should land on the same numbers a reader can verify by hand.
# Preset 2 is answerable outright, to contrast a passing judge against preset 1's
# failing one. Preset 3 points at the wrong data source (Chinook is a music-store
# database, not a sales record of HX-40 analysers), so either the SQL itself
# errors or the judge finds nothing that supports an answer -- either route
# escalates, and the trigger names which one happened.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "How much will 20 HAL-LMP-4419 lamp assemblies cost us, and which "
            "approval tier in the procurement policy does that order fall under?"
        ),
    },
    {
        "task": "What lead time should we plan for lamp assemblies ordered in December?",
    },
    {
        "task": "From the sales database, how many HX-40 analysers did we sell in 2025?",
    },
]

ACT_SYSTEM_PROMPT = (
    "You are a helpful assistant working from the bundled tools. Answer only from "
    "what the tools return this run -- do not state a specific fact (a number, a "
    "name, a date, a policy tier) that a tool call has not actually confirmed. If "
    "part of the task is not covered by what the tools returned, say exactly what "
    "is missing rather than estimating or guessing at it. If a person has already "
    "answered part of an earlier escalation, use their answer together with the "
    "tool results to complete the task. Before every tool call, write one short "
    "sentence in your reply saying what you are about to do and why, then call "
    "the tool in the same turn."
)

JUDGE_SYSTEM_PROMPT = (
    "You are the judge. You have the task, the draft answer, and every piece of "
    "evidence actually retrieved this run -- tool results, and any answer a "
    "person has given. Score whether the draft is SUPPORTED by that evidence, "
    "not whether it sounds plausible: could every specific claim in the draft -- "
    "a number, a name, a date, a policy tier -- be traced back to one of the "
    "observations below? List what the evidence covers, what it doesn't, and give "
    "a confidence from 0 to 1 for how completely and correctly the draft answers "
    "the task using only that evidence."
)

HANDOFF_SYSTEM_PROMPT = (
    "You are handing this task to a person because you cannot complete it "
    "reliably on your own. Write a handoff packet: restate what was asked, say "
    "what you tried in your own words, say what you actually established (with "
    "its source), ask exactly ONE specific question a person can answer that "
    "would let you finish, and say what you would do with that answer."
)


# --- structured-output schemas ---------------------------------------------------


class Assessment(BaseModel):
    """The judge's structured reply."""

    confidence: float = Field(
        ge=0, le=1, description="How well the evidence retrieved this run supports the draft, 0-1."
    )
    covered: list[str] = Field(
        default_factory=list, description="Parts of the task the evidence actually answers."
    )
    missing: list[str] = Field(
        default_factory=list, description="Parts of the task the evidence does not cover."
    )
    reason: str = Field(description="One or two sentences explaining the score.")


class HandoffPacket(BaseModel):
    """The escalation's structured reply -- the model's own account of the run."""

    asked: str = Field(description="The task, restated in your own words.")
    tried: list[str] = Field(description="What you did this run, in your own words.")
    found: list[str] = Field(description="What you actually established, with its source.")
    needs: str = Field(description="ONE specific question a person can answer.")
    recommendation: str = Field(description="What you would do once that question is answered.")


# --- graph state -------------------------------------------------------------------


class EscalationState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    pending_calls: list[dict[str, Any]]  # this turn's tool calls, run in one `tools` pass
    tool_errors: int  # failed tool calls since the last human answer
    trigger: str  # why the run is escalating, set by whichever node decides to
    assessment: dict[str, Any] | None
    packet: dict[str, Any] | None
    draft: str
    answer: str
    threshold: float  # frozen at run start
    decision: str  # this turn's routing choice, read by the conditional edge right after
    # The three fields below are the durable backup of the run -- everything `steps`,
    # `visited` and `budget` hold on the AgentRun/Budget objects, mirrored into graph
    # state so `resume()` can rebuild them after an app restart.
    steps: list[dict[str, Any]]
    visited: list[str]
    budget: dict[str, Any]
    run_id: str


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


# --- run context: what the graph's node closures read and write --------------------


class _RunContext:
    """A mutable box the node closures read `agent_run`/`budget`/`visited` through --
    see hitl/agent.py's identical helper for why (this demo doesn't import from that
    one; the box is copied, not shared, per CLAUDE.md rule 1)."""

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


def _evidence_lines(messages: list[Any]) -> str:
    """Every tool result and every human answer in the message history so far --
    what the judge scores the draft against, and what the handoff packet is
    written from. `messages[0]` is the task itself, never evidence."""
    lines = []
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            lines.append(message.content)
        elif i > 0 and isinstance(message, HumanMessage):
            lines.append(f"A person said: {message.content}")
    return "\n\n".join(lines) if lines else "(nothing retrieved yet)"


def _trail(steps: list[Step]) -> str:
    """The mechanical account of the run, one line per trajectory step -- not
    written by a model, so it can't misstate what actually happened the way the
    packet's own `tried` field, written by a model, in principle could."""
    lines = []
    for step in steps:
        prefix = f"{step.index:02d} {step.kind}"
        if step.tool:
            args_text = ", ".join(f"{k}={v!r}" for k, v in (step.args or {}).items())
            prefix += f" {step.tool}({args_text})"
        snippet = (step.text or "").strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:80].rstrip() + "…"
        lines.append(f"{prefix} → {snippet}" if snippet else prefix)
    return "\n".join(lines) if lines else "(no steps recorded yet)"


def _render_packet_as_answer(packet: dict[str, Any], closing_reason: str) -> str:
    """What `answer` becomes when a case is closed without a person answering --
    the packet, not silence, so the run still has an output to show."""
    return (
        f"Handed to a person, not answered. {packet.get('trigger', '')}\n\n"
        f"Needed from a person: {packet.get('needs', '')}\n"
        f"Recommendation: {packet.get('recommendation', '')}\n\n"
        f"Closed: {closing_reason or '(no reason given)'}"
    )


# --- the graph -----------------------------------------------------------------


def _build_graph(run_id: str, ctx: _RunContext) -> Any:
    settings_obj = get_settings()
    toolbox = Toolbox(run_id, names=list(TOOL_NAMES))
    tools = toolbox.as_langchain_tools()

    primary, *fallbacks = agent_models()

    # Bind before wrapping in fallbacks -- RunnableWithFallbacks does not forward
    # bind_tools()/with_structured_output(), but with_fallbacks() itself works on
    # any Runnable, bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
    def _with_fallbacks(bound: list[Any]) -> Any:
        head, *rest = bound
        return head.with_fallbacks(rest) if rest else head

    act_models = [m.bind_tools(tools) for m in [primary, *fallbacks]]
    act_runnable = _with_fallbacks(act_models)

    judge_models = [m.with_structured_output(Assessment, include_raw=True) for m in [primary, *fallbacks]]
    judge_runnable = _with_fallbacks(judge_models)

    packet_models = [
        m.with_structured_output(HandoffPacket, include_raw=True) for m in [primary, *fallbacks]
    ]
    packet_runnable = _with_fallbacks(packet_models)

    # --- nodes -------------------------------------------------------------

    def act_node(state: EscalationState) -> dict[str, Any]:
        ctx.budget.check()
        ctx.visited.append("act")
        call_started = time.monotonic()
        messages = [SystemMessage(content=ACT_SYSTEM_PROMPT), *state["messages"]]
        response = act_runnable.invoke(messages, config={"callbacks": get_callbacks()})
        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(response)
        ctx.budget.charge(steps=1, tokens=tokens, llm_calls=1)

        tool_calls = response.tool_calls or []
        if tool_calls:
            think_text = response.content or "(no reasoning text with this call)"
            _emit(ctx, "actor", kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)
            pending_calls = []
            for call in tool_calls:
                tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]
                pending_calls.append({"id": call_id, "name": tool_name, "args": tool_args})
                _emit(ctx, "actor", kind="act", text=f"Calling {tool_name}.", tool=tool_name, args=tool_args)
            return {
                "messages": [response],
                "pending_calls": pending_calls,
                "decision": "tools",
                **_sync(ctx),
            }

        draft = response.content or ""
        _emit(ctx, "actor", kind="decide", text=f"Draft: {draft}", tokens=tokens, latency_ms=latency_ms)
        return {"messages": [response], "draft": draft, "decision": "judge", **_sync(ctx)}

    def tools_node(state: EscalationState) -> dict[str, Any]:
        ctx.visited.append("tools")
        tool_messages: list[ToolMessage] = []
        errors = state["tool_errors"]
        for call in state["pending_calls"]:
            ctx.budget.check()
            tool_started = time.monotonic()
            result = toolbox.call(call["name"], call["args"])
            tool_latency_ms = (time.monotonic() - tool_started) * 1000
            ctx.budget.charge(tool_calls=1)
            if is_error(result.output):
                errors += 1
            _emit(
                ctx,
                "actor",
                kind="observe",
                text=result.output,
                tool=call["name"],
                args=call["args"],
                latency_ms=tool_latency_ms,
            )
            tool_messages.append(ToolMessage(content=result.output, tool_call_id=call["id"]))

        base = {"messages": tool_messages, "pending_calls": [], "tool_errors": errors, **_sync(ctx)}
        if errors >= settings_obj.escalation_max_tool_errors:
            plural = "" if errors == 1 else "s"
            return {
                **base,
                "trigger": f"{errors} tool call{plural} failed without recovery.",
                "decision": "escalate",
            }
        return {**base, "decision": "act"}

    def judge_node(state: EscalationState) -> dict[str, Any]:
        ctx.budget.check()
        ctx.visited.append("judge")
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {state['task']}\n\nDraft answer:\n{state['draft']}\n\n"
                    f"Evidence retrieved this run:\n{_evidence_lines(state['messages'])}"
                )
            ),
        ]
        call_started = time.monotonic()
        result = judge_runnable.invoke(messages, config={"callbacks": get_callbacks()})
        parsed, raw = result["parsed"], result["raw"]
        if parsed is None:
            parsed, raw = _retry_parsed(judge_runnable, messages, result.get("parsing_error"))
        if parsed is None:
            raise RuntimeError("The judge could not produce a valid assessment after one retry.")

        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(raw)
        ctx.budget.charge(steps=1, tokens=tokens, llm_calls=1)

        threshold = state["threshold"]
        passed = parsed.confidence >= threshold
        text = f"confidence {parsed.confidence:.2f} {'>=' if passed else '<'} threshold {threshold:.2f}"
        if parsed.missing:
            text += f" — missing: {'; '.join(parsed.missing)}"
        assessment = {
            "confidence": parsed.confidence,
            "covered": parsed.covered,
            "missing": parsed.missing,
            "reason": parsed.reason,
        }
        _emit(ctx, "judge", kind="decide", text=text, args=assessment, tokens=tokens, latency_ms=latency_ms)

        if passed:
            ctx.visited.append("respond")
            return {
                "assessment": assessment,
                "answer": state["draft"],
                "decision": "respond",
                **_sync(ctx),
            }
        return {
            "assessment": assessment,
            "trigger": f"Confidence below threshold -- {parsed.reason}",
            "decision": "escalate",
            **_sync(ctx),
        }

    def escalate_node(state: EscalationState) -> dict[str, Any]:
        ctx.budget.check()
        ctx.visited.append("escalate")
        messages = [
            SystemMessage(content=HANDOFF_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {state['task']}\n\nWhy this is being handed off: {state['trigger']}"
                    f"\n\nDraft so far (may be empty): {state['draft']}\n\n"
                    f"Evidence retrieved this run:\n{_evidence_lines(state['messages'])}"
                )
            ),
        ]
        call_started = time.monotonic()
        result = packet_runnable.invoke(messages, config={"callbacks": get_callbacks()})
        parsed, raw = result["parsed"], result["raw"]
        if parsed is None:
            parsed, raw = _retry_parsed(packet_runnable, messages, result.get("parsing_error"))
        if parsed is None:
            raise RuntimeError("The handoff packet could not be produced after one retry.")

        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(raw)
        ctx.budget.charge(steps=1, tokens=tokens, llm_calls=1)

        packet = {
            "asked": parsed.asked,
            "tried": parsed.tried,
            "found": parsed.found,
            "needs": parsed.needs,
            "recommendation": parsed.recommendation,
            "trigger": state["trigger"],
            "trail": _trail(ctx.agent_run.steps),
            "assessment": state.get("assessment"),
        }
        _emit(ctx, "agent", kind="handoff", text=f"Handoff: {parsed.needs}", args=packet, tokens=tokens, latency_ms=latency_ms)
        return {"packet": packet, **_sync(ctx)}

    def wait_node(state: EscalationState) -> dict[str, Any]:
        decision = interrupt(state["packet"])
        # Nothing above this line has a side effect: this node re-runs from the top
        # on resume, so everything below -- the only code with a side effect -- runs
        # exactly once, when the decision has actually arrived.
        ctx.visited.append("wait")
        verdict = decision.get("verdict", "reject")
        reason = decision.get("reason", "")

        if verdict == "approve":
            _emit(ctx, "agent", kind="gate", text=f"Answered by a person: {reason}")
            human_message = HumanMessage(content=f"A person answered your escalation: {reason}")
            return {
                "messages": [human_message],
                "tool_errors": 0,
                "decision": "act",
                **_sync(ctx),
            }

        _emit(ctx, "agent", kind="gate", text=f"Case closed by a person: {reason}")
        ctx.visited.append("closed")
        return {
            "answer": _render_packet_as_answer(state["packet"], reason),
            "decision": "end",
            **_sync(ctx),
        }

    graph = StateGraph(EscalationState)
    graph.add_node("act", act_node)
    graph.add_node("tools", tools_node)
    graph.add_node("judge", judge_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("wait", wait_node)

    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", lambda s: s["decision"], {"tools": "tools", "judge": "judge"})
    graph.add_conditional_edges("tools", lambda s: s["decision"], {"escalate": "escalate", "act": "act"})
    graph.add_conditional_edges("judge", lambda s: s["decision"], {"respond": END, "escalate": "escalate"})
    graph.add_edge("escalate", "wait")
    graph.add_conditional_edges("wait", lambda s: s["decision"], {"act": "act", "end": END})

    return graph.compile(checkpointer=get_checkpointer(DEMO))


def _apply_result(compiled: Any, config: dict[str, Any], result: dict[str, Any], ctx: _RunContext) -> None:
    """Turns one finished `compiled.invoke()` call into the run's status: handed off
    to a person, or completed. `BudgetExceeded` and any other exception never reach
    here -- they propagate out of `invoke()` and are handled by the caller's
    try/except."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        state = compiled.get_state(config)
        paused_at = state.next[0] if state.next else "wait"
        if paused_at not in ctx.visited:
            ctx.visited.append(paused_at)
        payload = interrupts[0].value
        ctx.agent_run.status = "needs_human"
        ctx.agent_run.stop_reason = f"Handed off to a person: {payload.get('trigger', 'needs a decision')}"
        return
    ctx.agent_run.output = result.get("answer", "")
    ctx.agent_run.status = "completed"


@observe(name="escalation.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `threshold`
    (defaults to `settings.escalation_confidence_threshold` from config)."""
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    threshold = float(settings.get("threshold") or get_settings().escalation_confidence_threshold)

    run_id = new_run_id()
    thread_id = thread_id_for(DEMO, run_id)
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task, thread_id=thread_id)
    ctx = _RunContext(agent_run=agent_run, budget=budget, visited=["task"], on_step=on_step)

    budget.start()

    try:
        compiled = _build_graph(run_id, ctx)
        initial_state: EscalationState = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "pending_calls": [],
            "tool_errors": 0,
            "trigger": "",
            "assessment": None,
            "packet": None,
            "draft": "",
            "answer": "",
            "threshold": threshold,
            "decision": "",
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


@observe(name="escalation.resume")
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
    `payload` (the handoff packet) -- or `None` if there is nothing waiting.

    Called automatically on every page load with no `thread_id`, so a browser
    refresh or an app restart puts the handoff panel back on screen with no click.
    Soft failure on anything that goes wrong -- an unreachable database, same as
    every other demo, but also a build failure (a provider dropped out of the
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
    checkpoint's own `budget` field -- `resume()` builds its own `Budget` internally
    and has nowhere in the `AgentDemo` contract to hand it back, so the page calls
    this right after `run()` or `resume()` returns. `None` if the checkpoint can't
    be read, in which case the page falls back to the caps it already has."""
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
