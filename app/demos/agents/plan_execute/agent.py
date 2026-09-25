"""Plan-and-Execute: a planner writes the whole plan up front, an executor runs it
one step at a time, and a replanner looks at every result and decides whether to
keep going, rewrite what's left, or stop and answer.

Unlike ReAct (Step 1), the model never decides its *next* action from scratch --
it decides only whether the plan it already wrote still holds. That is the
trade this demo exists to show: a global view of the task, bought at the price
of a plan written before any evidence exists to test it, and a replan call after
every single step to keep that plan honest.

The graph is hand-built with `langgraph.graph.StateGraph` rather than
`create_agent`, because the thing being demonstrated -- planning, executing and
replanning as distinct phases -- has no `create_agent` equivalent; that helper
gives one phase (`think -> act -> observe`), not three. Tool execution therefore
happens inside the `execute` node by calling `Toolbox.call()` directly, the same
way ReAct's hand-written loop does, rather than through a `ToolNode`.

Budget and step accounting: every node that calls a model calls `budget.check()`
first, then charges one step, its tokens and one LLM call after the reply comes
back. `revise` makes no model call and is free. `BudgetExceeded` is not caught
inside a node -- it propagates out of `compiled.invoke()`, and `run()` is what
turns it into `status="stopped_on_budget"`, the same shape every other demo uses.
"""

import time
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "plan_execute"
NAME = "Plan-and-Execute"
SENTENCE = "The plan is written first, executed step by step, and revised when a step invalidates it."
SHAPE = "plan → execute ×n → replan ↻"

# Node ids double as the `visited` entries the graph map highlights, so a name
# changed here must be changed in both places at once.
GRAPH = """flowchart LR
    task[Task] --> plan[Planner writes numbered plan]
    plan --> execute[Executor runs the next step, one tool call]
    execute --> replan[Replanner judges the result]
    replan -->|continue| execute
    replan -->|revise| revise[Remaining plan rewritten]
    revise --> execute
    replan -->|finish| respond[Answer from executed steps]
    execute -.->|cap spent| cap[Stopped]
"""

TOOL_NAMES = ("search_corpus", "describe_schema", "run_sql")

# Each preset breaks its own plan on purpose -- the point of this demo is the
# revision, not the first draft. See taxonomy/plans/step-03-plan-execute.md for
# how every fact below was checked against samples/corpus/ and chinook.db.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "A lab needs to process 9,600 samples in an 8-hour shift with HX-40 "
            "analysers. Using the rated throughput figure, work out how many "
            "instruments that needs. Then check that figure against the "
            "sustained-throughput guidance, and name the firmware version that "
            "fixes the fan-curve issue behind it."
        ),
    },
    {
        "task": "Sum the `amount` column of the invoices table by billing country and name the top three.",
    },
    {
        "task": "How many invoices are in the sales database?",
    },
]


# --- structured-output schemas ---------------------------------------------------


class Plan(BaseModel):
    """The planner's structured reply."""

    rationale: str = Field(description="One or two sentences on why this plan answers the task.")
    steps: list[str] = Field(
        description="Each step is one imperative sentence a single tool call can carry out."
    )


class Replan(BaseModel):
    """The replanner's structured reply, made after every executed step."""

    decision: Literal["continue", "revise", "finish"]
    reason: str = Field(description="One sentence explaining the decision.")
    remaining_steps: list[str] = Field(
        default_factory=list,
        description="The corrected remaining steps. Only read when decision is 'revise'.",
    )


# --- graph state -------------------------------------------------------------------


class PlanExecuteState(TypedDict):
    task: str
    plan_versions: list[dict[str, Any]]
    remaining: list[str]
    past_steps: list[dict[str, Any]]
    pending_revision: list[str] | None
    pending_revision_reason: str
    decision: str
    answer: str


PLAN_SYSTEM_PROMPT = (
    "You are the planner. Given a task and the tools available, write a short numbered "
    "plan: each step is one imperative sentence that a single tool call can carry out. "
    "Do not include a step to write the final answer -- that happens separately, after "
    "every step has run. Write at most {max_steps} steps.\n\nAvailable tools:\n{tools}"
)

EXECUTE_SYSTEM_PROMPT = (
    "You are the executor. You are given one step from a plan, the plan it belongs to, "
    "and what has already been done. Call exactly one tool to carry out the step named "
    "below. If the step needs no tool -- pure arithmetic or reasoning over what you "
    "already know -- answer it directly in your reply text instead of calling a tool. "
    "Before calling a tool, write one short sentence in your reply saying what you are "
    "about to do and why, then call the tool in the same turn."
)

REPLAN_SYSTEM_PROMPT = (
    "You are the replanner. After every executed step, decide one of three things: "
    "'continue' -- the remaining plan is still right, run its next step as written; "
    "'revise' -- something just learned invalidates part of the remaining plan, so "
    "write the corrected remaining steps (imperative sentences, at most {max_steps}), "
    "replacing every step not yet run; 'finish' -- enough is known to answer the task "
    "now, even if steps remain. Give a one-sentence reason either way."
)

RESPOND_SYSTEM_PROMPT = (
    "Answer the task using only the steps that were actually executed and their "
    "results. If the plan was revised, follow the latest version -- do not answer from "
    "a plan step that was dropped. Name the sources or tables the answer rests on."
)


def _plan_text(steps: list[str]) -> str:
    return "; ".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def _retry_parsed(
    runnable: Any, messages: list[Any], parsing_error: Any
) -> tuple[Any, Any]:
    """One retry on a structured-output parse failure: ask again, plainly, without
    replaying the unparseable reply (which may carry a dangling tool call some
    providers refuse to see unanswered)."""
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


@observe(name="plan_execute.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `inject_fault`."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    inject_fault = bool(settings.get("inject_fault"))
    max_plan_steps = settings_obj.plan_execute_max_plan_steps

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
        toolbox_names = [*TOOL_NAMES, "boom"] if inject_fault else list(TOOL_NAMES)
        toolbox = Toolbox(run_id, names=toolbox_names)
        tools = toolbox.as_langchain_tools(names=list(TOOL_NAMES))

        primary, *fallbacks = agent_models()

        # Each model is bound (structured output / tools) *before* it is wrapped in
        # fallbacks -- RunnableWithFallbacks does not forward bind_tools(), but
        # with_fallbacks() itself works on any Runnable, bound or not. See
        # taxonomy/plans/AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        plan_models = [m.with_structured_output(Plan, include_raw=True) for m in [primary, *fallbacks]]
        plan_runnable = _with_fallbacks(plan_models)

        replan_models = [
            m.with_structured_output(Replan, include_raw=True) for m in [primary, *fallbacks]
        ]
        replan_runnable = _with_fallbacks(replan_models)

        exec_models = [m.bind_tools(tools) for m in [primary, *fallbacks]]
        exec_runnable = _with_fallbacks(exec_models)

        respond_runnable = _with_fallbacks([primary, *fallbacks])

        fault_used = {"value": False}

        # --- nodes -------------------------------------------------------------

        def plan_node(state: PlanExecuteState) -> dict[str, Any]:
            budget.check()
            visited.append("plan")
            call_started = time.monotonic()
            messages = [
                SystemMessage(
                    content=PLAN_SYSTEM_PROMPT.format(
                        max_steps=max_plan_steps, tools=toolbox.describe()
                    )
                ),
                HumanMessage(content=state["task"]),
            ]
            result = plan_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(plan_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The planner could not produce a valid plan after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            steps = parsed.steps
            truncated = len(steps) > max_plan_steps
            steps = steps[:max_plan_steps]
            text = f"Plan v1: {_plan_text(steps)}"
            if truncated:
                text += f" (truncated to the {max_plan_steps}-step cap)"
            emit(
                "planner",
                kind="decide",
                text=text,
                args={"plan_version": 1, "plan": steps, "reason": parsed.rationale},
                tokens=tokens,
                latency_ms=latency_ms,
            )
            version = {"version": 1, "steps": steps, "reason": parsed.rationale}
            return {"plan_versions": [version], "remaining": list(steps)}

        def execute_node(state: PlanExecuteState) -> dict[str, Any]:
            budget.check()
            visited.append("execute")
            step_text = state["remaining"][0]
            current_plan = state["plan_versions"][-1]["steps"]
            past_lines = (
                "\n".join(
                    f"{i + 1}. {p['step']} -> {p['result']}"
                    for i, p in enumerate(state["past_steps"])
                )
                or "(none yet)"
            )
            messages = [
                SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nCurrent plan:\n{_plan_text(current_plan)}"
                        f"\n\nCompleted steps so far:\n{past_lines}\n\nDo this step now, with "
                        f"one tool call: {step_text}"
                    )
                ),
            ]
            call_started = time.monotonic()
            response = exec_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            tool_calls = response.tool_calls or []
            think_text = response.content or "(no reasoning text with this call)"
            if len(tool_calls) > 1:
                think_text += (
                    f" ({len(tool_calls) - 1} extra tool call(s) in this reply were "
                    "ignored -- only the first runs.)"
                )
            # `plan_step` on this row is the one place the plan step's own text is
            # recorded on the trajectory -- the page's plan-versions panel reads it
            # back to line up an executed step with the plan it came from.
            emit(
                "executor",
                kind="think",
                text=think_text,
                args={"plan_step": step_text},
                tokens=tokens,
                latency_ms=latency_ms,
            )

            if tool_calls:
                call = tool_calls[0]
                tool_name, tool_args = call["name"], call.get("args", {})
                # Decided before the act row is emitted, and said in that row's own
                # text -- not just buried in the error the observe row shows -- so
                # the divert is visible without having to read the error message to
                # notice it happened.
                divert = inject_fault and not fault_used["value"]
                act_text = f"Calling {tool_name}."
                if divert:
                    act_text += " Fault injection is on: this call is diverted to `boom` instead."
                emit("executor", kind="act", text=act_text, tool=tool_name, args=tool_args)

                tool_started = time.monotonic()
                if divert:
                    fault_used["value"] = True
                    result = toolbox.call("boom", {})
                else:
                    result = toolbox.call(tool_name, tool_args)
                tool_latency_ms = (time.monotonic() - tool_started) * 1000
                budget.charge(tool_calls=1)

                observation = result.output
                emit(
                    "executor",
                    kind="observe",
                    text=observation,
                    tool=tool_name,
                    args=tool_args,
                    latency_ms=tool_latency_ms,
                )
                past_entry = {
                    "plan_version": state["plan_versions"][-1]["version"],
                    "step": step_text,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": observation,
                    "ok": result.ok,
                }
            else:
                result_text = response.content or ""
                emit("executor", kind="observe", text=result_text, latency_ms=0.0)
                past_entry = {
                    "plan_version": state["plan_versions"][-1]["version"],
                    "step": step_text,
                    "tool": None,
                    "args": None,
                    "result": result_text,
                    "ok": True,
                }

            return {
                "past_steps": [*state["past_steps"], past_entry],
                "remaining": state["remaining"][1:],
            }

        def replan_node(state: PlanExecuteState) -> dict[str, Any]:
            budget.check()
            visited.append("replan")
            current_plan = state["plan_versions"][-1]["steps"]
            past_lines = (
                "\n".join(
                    f"{i + 1}. {p['step']} -> {p['result']}"
                    for i, p in enumerate(state["past_steps"])
                )
                or "(none yet)"
            )
            remaining_lines = "\n".join(state["remaining"]) or "(none)"
            messages = [
                SystemMessage(content=REPLAN_SYSTEM_PROMPT.format(max_steps=max_plan_steps)),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nCurrent plan:\n{_plan_text(current_plan)}"
                        f"\n\nCompleted steps:\n{past_lines}\n\nSteps not yet run:"
                        f"\n{remaining_lines}\n\nDecide: continue with the plan as is, revise "
                        "the remaining steps, or finish because there's enough to answer."
                    )
                ),
            ]
            call_started = time.monotonic()
            result = replan_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(replan_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The replanner could not produce a valid decision after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            decision = parsed.decision
            note = ""
            if decision == "continue" and not state["remaining"]:
                decision, note = "finish", " (auto: nothing left to run)"
            elif decision == "revise" and not parsed.remaining_steps:
                decision, note = "finish", " (auto: revision produced no steps)"

            if decision == "revise":
                new_steps = parsed.remaining_steps[:max_plan_steps]
                emit(
                    "replanner",
                    kind="decide",
                    text=f"Revise{note}: {parsed.reason}",
                    args={"remaining_steps": new_steps},
                    tokens=tokens,
                    latency_ms=latency_ms,
                )
                return {
                    "decision": decision,
                    "pending_revision": new_steps,
                    "pending_revision_reason": parsed.reason,
                }

            emit(
                "replanner",
                kind="decide",
                text=f"{decision.capitalize()}{note}: {parsed.reason}",
                tokens=tokens,
                latency_ms=latency_ms,
            )
            return {"decision": decision}

        def revise_node(state: PlanExecuteState) -> dict[str, Any]:
            visited.append("revise")
            current_version = state["plan_versions"][-1]["version"]
            executed_texts = [
                p["step"] for p in state["past_steps"] if p["plan_version"] == current_version
            ]
            new_steps = [*executed_texts, *(state["pending_revision"] or [])]
            new_version_number = current_version + 1
            version = {
                "version": new_version_number,
                "steps": new_steps,
                "reason": state["pending_revision_reason"],
            }
            emit(
                "replanner",
                kind="decide",
                text=f"Plan v{new_version_number}: {_plan_text(new_steps)}",
                args={
                    "plan_version": new_version_number,
                    "plan": new_steps,
                    "reason": state["pending_revision_reason"],
                },
            )
            return {
                "plan_versions": [*state["plan_versions"], version],
                "remaining": list(state["pending_revision"] or []),
                "pending_revision": None,
                "pending_revision_reason": "",
            }

        def respond_node(state: PlanExecuteState) -> dict[str, Any]:
            budget.check()
            visited.append("respond")
            latest_version = state["plan_versions"][-1]["version"]
            past_lines = "\n".join(
                f"{i + 1}. {p['step']} -> {p['result']}" for i, p in enumerate(state["past_steps"])
            )
            messages = [
                SystemMessage(content=RESPOND_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nSteps executed (plan v{latest_version}):"
                        f"\n{past_lines}\n\nWrite the final answer now, in full sentences, "
                        "following the latest plan."
                    )
                ),
            ]
            call_started = time.monotonic()
            response = respond_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            answer = response.content or ""
            emit(
                "responder",
                kind="decide",
                text=(
                    f"Answered from {len(state['past_steps'])} executed step(s) under "
                    f"plan v{latest_version}."
                ),
                tokens=tokens,
                latency_ms=latency_ms,
            )
            return {"answer": answer}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(PlanExecuteState)
        graph.add_node("plan", plan_node)
        graph.add_node("execute", execute_node)
        graph.add_node("replan", replan_node)
        graph.add_node("revise", revise_node)
        graph.add_node("respond", respond_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "replan")
        graph.add_conditional_edges(
            "replan",
            lambda state: state["decision"],
            {"continue": "execute", "revise": "revise", "finish": "respond"},
        )
        graph.add_edge("revise", "execute")
        graph.add_edge("respond", END)

        compiled = graph.compile()

        initial_state: PlanExecuteState = {
            "task": task,
            "plan_versions": [],
            "remaining": [],
            "past_steps": [],
            "pending_revision": None,
            "pending_revision_reason": "",
            "decision": "",
            "answer": "",
        }
        final_state = compiled.invoke(
            initial_state,
            config={
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


# --- the plan-versions panel -------------------------------------------------------


def plan_diff(plan_versions: list[dict[str, Any]], past_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per plan version, each of its steps tagged for the page's
    plan-versions panel: `executed` (ran, with the row it ran on), `pending`
    (planned here, not yet run, still present in the latest version), or
    `dropped` (planned here, never run, and absent from the latest version). A
    step absent from v1's own list is additionally flagged `added`.

    Each `past_steps` entry is `{"step": text, "row_index": int}` -- `row_index`
    is the trajectory row the step executed on, if the caller has it (the page
    does; a 1-based execution position is used as a fallback otherwise).
    """
    if not plan_versions:
        return []

    original = set(plan_versions[0]["steps"])
    latest_steps = set(plan_versions[-1]["steps"])
    executed_rows: dict[str, int] = {}
    for i, p in enumerate(past_steps):
        executed_rows.setdefault(p["step"], p.get("row_index", i + 1))

    columns = []
    for version in plan_versions:
        rows = []
        for step_text in version["steps"]:
            if step_text in executed_rows:
                status, row_index = "executed", executed_rows[step_text]
            elif step_text in latest_steps:
                status, row_index = "pending", None
            else:
                status, row_index = "dropped", None
            rows.append(
                {
                    "step": step_text,
                    "status": status,
                    "row_index": row_index,
                    "added": step_text not in original,
                }
            )
        columns.append({"version": version["version"], "reason": version["reason"], "rows": rows})
    return columns


def plan_summary(plan_versions: list[dict[str, Any]], past_steps: list[dict[str, Any]]) -> str:
    """'N of M original steps executed as written; K revisions.'"""
    if not plan_versions:
        return "No plan yet."
    original = plan_versions[0]["steps"]
    executed_texts = {p["step"] for p in past_steps}
    executed_original = sum(1 for s in original if s in executed_texts)
    revisions = len(plan_versions) - 1
    plural = "" if revisions == 1 else "s"
    return (
        f"{executed_original} of {len(original)} original step(s) executed as written; "
        f"{revisions} revision{plural}."
    )
