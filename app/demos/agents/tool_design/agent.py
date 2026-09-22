"""One task, one model, two toolsets -- run through LangChain's `create_agent`.

Everything ReAct wrote by hand (Step 1) is handled by the framework here: the
tool schemas go to the model as JSON, not as prose in a prompt, and the reply
comes back as a structured `tool_calls` field on the `AIMessage`, not text a
parser has to read. What is left to compare, with the parser gone, is the thing
this step is actually about: does the *toolset* -- its names, its argument
shapes, its descriptions, its error text -- change what the agent does.

`_run_side()` is the one loop; `run()` calls it once (the revised toolset, for
the `AgentDemo` contract and Step 16's evaluation) and `run_pair()` calls it
twice (both toolsets, for this page). Budget and step accounting is three
middleware hooks rather than a `while` loop:

- `before_model` calls `budget.check()` before every model call and jumps the
  graph straight to `end` when a cap is spent -- the same check ReAct makes at
  the top of its loop, moved to where `create_agent` expects it.
- `wrap_model_call` times the call, counts its tokens and charges the budget --
  this is also where "Break the first tool call" is disclosed as a possibility,
  though the injection itself happens in `wrap_tool_call` below.
- `wrap_tool_call` charges one tool call, and on the very first call of a run
  with the fault toggle on, diverts it to `core.tools`' `boom` tool instead of
  whatever the model asked for -- the structured error it raises is genuine, not
  fabricated, and each side receives it in its own shape (see `toolsets.py`).
"""

import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage

from core.budget import Budget, BudgetExceeded
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import is_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

from . import toolsets

DEMO = "tool_design"
NAME = "Tool design"
SENTENCE = (
    "One task against two toolsets, so a name, a schema and an error message show up as "
    "behaviour."
)
SHAPE = "act → observe ↻ ×2 toolsets"

GRAPH = """flowchart LR
    task[Task and tool schemas] --> model[Model emits a message]
    model -->|tool call| act[Run the tool]
    act --> observe[Observation appended]
    observe --> model
    model -->|no tool call| done[Answer]
    model -.->|cap spent| cap[Stopped]
"""

SIDES = ("first draft", "revised")

SYSTEM_PROMPT = (
    "You solve the task below by calling tools, one at a time. Before every tool call, "
    "write one short sentence in your reply saying what you are about to do and why, "
    "then call the tool in the same turn. Read each observation before deciding what to "
    "do next -- if it begins with an error, do not repeat the call that just failed. "
    "Answer as soon as you have enough to answer in full sentences, naming the sources "
    "or tables it rests on. Every step is spent from a fixed budget."
)

# One preset per lesson: corpus-only (the first draft has three plausible-sounding
# tools for it), a three-tool chain where `lookup` reliably gets English where SQL
# belongs, and a short task for running with the fault toggle on. `expected` is
# written against the revised names; `comparison_rows()` translates it to the
# first draft's names with `toolsets.REVISED_TO_DRAFT`.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "What sustained throughput does the HX-40 actually reach, and which firmware "
            "release added the indicator for it?"
        ),
        "expected": ["search_corpus"],
    },
    {
        "task": (
            "The procurement policy sets an approval threshold -- find it, then list the "
            "sales-database invoices above that amount."
        ),
        "expected": ["search_corpus", "describe_schema", "run_sql"],
    },
    {
        "task": "How many invoices are in the sales database?",
        "expected": ["describe_schema", "run_sql"],
    },
]


@dataclass
class Side:
    """One toolset's run: the trajectory and the budget it was measured against."""

    label: str
    run: AgentRun
    budget: Budget


def _extract_ai_message(response: Any) -> AIMessage | None:
    """`wrap_model_call`'s handler returns a `ModelResponse` in this LangChain
    version, but the type is documented as `ModelResponse | AIMessage` -- so both
    are handled rather than assumed."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _is_error_text(label: str, text: str) -> bool:
    """The first draft's error text is the fixed string `"Error."`; the revised
    side keeps `core.tools`' own `ERROR[kind]: ...` protocol, which `is_error()`
    already recognises."""
    return text == "Error." if label == "first draft" else is_error(text)


def _run_side(
    label: str,
    task: str,
    budget: Budget,
    *,
    inject_fault: bool,
    crowd: bool,
    on_step: Callable[[str, Step], None] | None,
) -> AgentRun:
    """Runs one toolset against the task until a final answer or a spent cap.
    Never raises: any failure -- no provider configured, every provider refusing,
    an unexpected error building the agent -- lands in the returned `AgentRun` as
    `status="failed"`, so one side blowing up never stops the other from running."""
    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["task"]

    def emit(**fields: Any) -> Step:
        step = agent_run.add_step(agent=label, **fields)
        if on_step is not None:
            on_step(label, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        toolbox = toolsets.build_toolbox(run_id, inject_fault=inject_fault)
        tools = (
            toolsets.first_draft_tools(toolbox)
            if label == "first draft"
            else toolsets.revised_tools(toolbox)
        )
        if crowd:
            tools = [*tools, *toolsets.decoy_tools()]

        primary, *fallbacks = agent_models()

        cap_hit: dict[str, str] = {}
        model_meta: dict[str, Any] = {}
        tool_latency: dict[str, float] = {}
        tool_call_args: dict[str, dict[str, Any]] = {}
        first_tool_call_done = {"value": False}

        @before_model(can_jump_to=["end"])
        def check_budget(state: Any, runtime: Any) -> dict[str, Any] | None:
            try:
                budget.check()
            except BudgetExceeded as stop:
                cap_hit["reason"] = stop.reason
                return {"jump_to": "end"}
            return None

        @wrap_model_call
        def charge_model(request: Any, handler: Callable[[Any], Any]) -> Any:
            call_started = time.monotonic()
            response = handler(request)
            latency_ms = (time.monotonic() - call_started) * 1000
            message = _extract_ai_message(response)
            tokens = count_tokens(message) if message is not None else 0
            budget.charge(steps=1, tokens=tokens, llm_calls=1)
            model_meta["tokens"] = tokens
            model_meta["latency_ms"] = latency_ms
            return response

        @wrap_tool_call
        def run_tool(request: Any, handler: Callable[[Any], Any]) -> Any:
            tool_call_id = request.tool_call["id"]
            tool_call_args[tool_call_id] = dict(request.tool_call.get("args") or {})
            call_started = time.monotonic()
            if inject_fault and not first_tool_call_done["value"]:
                first_tool_call_done["value"] = True
                result = toolbox.call("boom", {})
                text = result.output if label == "revised" else "Error."
                message = ToolMessage(
                    content=text,
                    name=request.tool_call["name"],
                    tool_call_id=tool_call_id,
                )
            else:
                message = handler(request)
            tool_latency[tool_call_id] = (time.monotonic() - call_started) * 1000
            budget.charge(tool_calls=1)
            return message

        middleware: list[Any] = []
        if fallbacks:
            middleware.append(ModelFallbackMiddleware(*fallbacks))
        middleware += [check_budget, charge_model, run_tool]

        compiled_agent = create_agent(
            primary, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=middleware
        )

        stream = compiled_agent.stream(
            {"messages": [{"role": "user", "content": task}]},
            stream_mode="updates",
            config={"callbacks": get_callbacks()},
        )
        for chunk in stream:
            for node, update in chunk.items():
                if node.endswith("before_model"):
                    if isinstance(update, dict) and update.get("jump_to") == "end":
                        visited.append("cap")
                        agent_run.status = "stopped_on_budget"
                        agent_run.stop_reason = cap_hit.get("reason", "Stopped on a budget cap.")
                        emit(kind="decide", text=agent_run.stop_reason)
                    continue

                if node == "model":
                    message = update["messages"][-1]
                    tokens = model_meta.get("tokens", 0)
                    latency_ms = model_meta.get("latency_ms", 0.0)
                    if message.tool_calls:
                        visited.append("model")
                        emit(
                            kind="think",
                            text=message.content or "(no reasoning text with this call)",
                            tokens=tokens,
                            latency_ms=latency_ms,
                        )
                        for call in message.tool_calls:
                            visited.append("act")
                            emit(
                                kind="act",
                                text=f"Calling {call['name']}.",
                                tool=call["name"],
                                args=call.get("args", {}),
                            )
                    else:
                        visited.append("done")
                        emit(
                            kind="think",
                            text=message.content or "(final reply carried no reasoning text)",
                            tokens=tokens,
                            latency_ms=latency_ms,
                        )
                        emit(kind="decide", text="Answered: the task needs no further tool call.")
                        agent_run.output = message.content or ""
                        agent_run.status = "completed"
                    continue

                if node == "tools":
                    visited.append("observe")
                    for message in update["messages"]:
                        emit(
                            kind="observe",
                            text=message.content,
                            tool=message.name,
                            args=tool_call_args.get(message.tool_call_id),
                            latency_ms=tool_latency.get(message.tool_call_id, 0.0),
                        )
                    continue

    except Exception as exc:  # a provider refused, none configured, or the agent failed to build
        agent_run.status = "failed"
        agent_run.stop_reason = f"The {label} run could not finish: {type(exc).__name__}: {exc}"
        emit(kind="decide", text=agent_run.stop_reason)

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run


@observe(name="tool_design.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract: one run, against the revised toolset -- the
    configuration worth comparing other demos to in Step 16. `run_pair()` below
    is what the page itself uses to show both sides at once.

    Settings, all optional: `on_step`, `inject_fault`, `crowd` -- see `run_pair()`.
    """
    return _run_side(
        "revised",
        task,
        budget,
        inject_fault=bool(settings.get("inject_fault")),
        crowd=bool(settings.get("crowd")),
        on_step=settings.get("on_step"),
    )


def run_pair(
    task: str,
    budget_template: Budget,
    **settings: Any,
) -> tuple[Side, Side]:
    """Both toolsets against the same task, sequentially and each in its own
    `try/except` (inside `_run_side`) -- a rate limit on one side still lets the
    other render. Each side gets a fresh `Budget` with the same caps, so "tool
    calls" and "tokens" mean the full budget spent by that side alone, not a
    split of one shared budget.

    Settings, all optional:
      `on_step`       callable(label, Step) -- called as each step lands.
      `inject_fault`  bool -- diverts the first tool call on *both* sides to `boom`.
      `crowd`         bool -- adds the six decoy tools to *both* sides.
    """
    on_step = settings.get("on_step")
    inject_fault = bool(settings.get("inject_fault"))
    crowd = bool(settings.get("crowd"))

    sides = []
    for label in SIDES:
        side_budget = Budget(
            max_steps=budget_template.max_steps,
            max_tokens=budget_template.max_tokens,
            deadline_s=budget_template.deadline_s,
        )
        side_run = _run_side(
            label, task, side_budget, inject_fault=inject_fault, crowd=crowd, on_step=on_step
        )
        sides.append(Side(label=label, run=side_run, budget=side_budget))

    first, revised = sides
    return first, revised


# --- the comparison table --------------------------------------------------------


def comparison_rows(first: Side, revised: Side, expected: list[str]) -> list[dict[str, Any]]:
    """One row per side. Built here, not in `core/` -- this is the one place in
    the lab that needs it (rule 2: something moves to `core/` when a *second*
    demo needs it)."""
    rows = []
    for side in (first, revised):
        expected_names = (
            {toolsets.REVISED_TO_DRAFT.get(name, name) for name in expected}
            if side.label == "first draft"
            else set(expected)
        )
        acts = [s for s in side.run.steps if s.kind == "act"]
        observes = [s for s in side.run.steps if s.kind == "observe"]

        seen: dict[tuple[str, str], int] = {}
        repeated = 0
        for step in acts:
            key = (step.tool or "", str(step.args or {}))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                repeated += 1

        off_task = sum(1 for step in acts if (step.tool or "") not in expected_names)
        errored = sum(1 for step in observes if _is_error_text(side.label, step.text))
        recoveries = sum(
            1
            for i, step in enumerate(observes)
            if _is_error_text(side.label, step.text) and i + 1 < len(observes)
        )

        rows.append(
            {
                "Toolset": side.label,
                "Status": side.run.status,
                "Tool calls": side.budget.tool_calls,
                "LLM calls": side.budget.llm_calls,
                "Steps": side.budget.steps_used,
                "Tokens": side.budget.tokens_used,
                "Elapsed (s)": round(side.budget.elapsed_s, 1),
                "Errored calls": errored,
                "Repeated calls": repeated,
                "Off-task calls": off_task,
                "Recoveries": recoveries,
            }
        )
    return rows
