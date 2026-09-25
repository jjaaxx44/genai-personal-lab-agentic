"""Reflexion: an actor makes an attempt, an evaluator scores it against the task,
and -- only on a failure -- a reflector writes one concrete note the next attempt
carries forward. There is no gradient update anywhere in this loop: the "learning"
between attempt 1 and attempt 2 is entirely the reflection note sitting in the next
prompt. That is the whole mechanism, and it is also the whole limitation.

The graph is hand-built with `langgraph.graph.StateGraph`, the same choice Step 3
made and for the same reason: actor, evaluator and reflector are three distinct
jobs with three distinct prompts, which `create_agent` has no way to express as
separate phases. The actor calls `Toolbox.call()` directly, exactly like
Plan-and-Execute's executor and ReAct's hand-written loop, rather than through a
`ToolNode`.

The evaluator has no more access to ground truth than the actor did -- it is an
LLM judging the attempt against the task and the evidence retrieved *this
attempt*, nothing else. So its rubric is grounding and completeness, not
correctness against a hidden answer key: does the attempt cover the task, is
every specific claim traceable to the evidence shown, and does it admit rather
than guess when the evidence falls short. That is what makes an evidence-free
attempt fail reliably, and what makes an unanswerable task fail forever.

Budget and step accounting: every node that calls a model calls `budget.check()`
first, then charges one step, its tokens and one LLM call after the reply comes
back. `BudgetExceeded` is not caught inside a node -- it propagates out of
`compiled.invoke()`, and `run()` turns it into `status="stopped_on_budget"`, the
same shape every other demo uses.
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

DEMO = "reflexion"
NAME = "Reflexion"
SENTENCE = "An attempt is scored against the task, critiqued in writing, and retried with the critique in hand."
SHAPE = "act → evaluate → reflect ↻"

# `respond` and `giveup`, like Step 3's `cap`, are not real StateGraph nodes -- they
# are pushed onto `visited` at the point `evaluate` decides the run is over, so the
# graph map can still show where the run actually ended.
GRAPH = """flowchart LR
    task[Task] --> act[Actor attempts the task]
    act --> evaluate[Evaluator scores the attempt]
    evaluate -->|passed| respond[Attempt returned as the answer]
    evaluate -->|failed, attempts remain| reflect[Self-reflection note written]
    evaluate -->|failed, retry cap reached| giveup[Retry cap reached, last attempt stands]
    reflect --> act
    act -.->|cap spent| cap[Stopped]
"""

TOOL_NAMES = ("search_corpus", "run_sql", "describe_schema")

# Preset 1 is written to be run with the "Force the first attempt to answer from
# memory" toggle on: it guarantees attempt 1 is evidence-free, so the evaluator's
# fail and the reflector's note are not left to chance. Preset 2 needs no toggle --
# neither fact it asks for exists anywhere in samples/corpus/, so every attempt
# fails for the same reason and the run exhausts reflexion_max_attempts on its own.
# Preset 3 is a plain second-domain task with no engineered failure.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "What is the HX-40's sustained throughput in samples per hour, and which "
            "firmware version fixes the fan-curve issue behind the gap between that "
            "figure and the rated one?"
        ),
    },
    {
        "task": "Who is Halden Instruments' CEO, and what is the company's total headcount?",
    },
    {
        "task": (
            "Which billing country's customers generated the highest total invoice "
            "amount, and how much did they spend?"
        ),
    },
]


# --- structured-output schemas ---------------------------------------------------


class Evaluation(BaseModel):
    """The evaluator's structured reply. No ground truth is available to it -- see
    the module docstring -- so it scores grounding and completeness, not correctness."""

    passed: bool
    score: int = Field(
        ge=1,
        le=5,
        description=(
            "1 = does not address the task at all; 5 = fully addresses it and every "
            "specific claim is backed by the evidence shown."
        ),
    )
    reason: str = Field(description="One or two sentences: why it passed or failed.")


class Reflection(BaseModel):
    """The reflector's structured reply, made only after a failed evaluation."""

    critique: str = Field(
        description=(
            "One or two sentences: concretely what to do differently on the next "
            "attempt. Not a restatement of the task or the evaluator's reason."
        )
    )


# --- graph state -------------------------------------------------------------------


class ReflexionState(TypedDict):
    task: str
    attempt: int
    reflections: list[dict[str, Any]]
    attempt_text: str
    attempt_tool: str | None
    attempt_args: dict[str, Any] | None
    attempt_evidence: str
    evaluation: dict[str, Any] | None
    decision: Literal["retry", "stop", ""]
    answer: str


ACT_SYSTEM_PROMPT = (
    "You are the actor. Make one attempt at the task below. If a specific fact needs "
    "verification -- a number, a name, a version, a date -- call exactly one tool to "
    "retrieve it before stating it; do not state a specific fact you have not just "
    "retrieved with a tool this attempt. If the task needs no tool -- pure reasoning "
    "over what you already know -- answer it directly instead. Before calling a "
    "tool, write one short sentence in your reply saying what you are about to do "
    "and why, then call the tool in the same turn."
)

ACT_NO_TOOLS_PROMPT = (
    "You are the actor. No tools are available to you on this attempt. Answer the "
    "task below using only what you already know, and say plainly which parts you "
    "are not sure of rather than guessing at a specific number, name or version."
)

COMPOSE_SYSTEM_PROMPT = (
    "You just retrieved the evidence below with a tool call. Write your attempt at "
    "the task using only what the evidence actually supports. If the evidence does "
    "not contain something the task needs, say so rather than filling the gap from "
    "memory."
)

EVALUATE_SYSTEM_PROMPT = (
    "You are the evaluator. You have no access to the correct answer -- only the "
    "task, the evidence the actor retrieved this attempt (which may be none), and "
    "the attempt itself. `passed` means the task has actually been answered: every "
    "part of the task has a specific answer, and every specific claim in the "
    "attempt -- a number, a name, a version, a date -- is directly supported by "
    "the evidence shown, not merely plausible. If any part of the task is left "
    "unanswered, or is answered with a claim the evidence does not support, "
    "`passed` is false. This is true even when the attempt is honest about the "
    "gap rather than guessing at it -- admitting the evidence doesn't cover "
    "something is the right thing for the actor to do, but it still does not "
    "complete the task, so it still does not pass. Give a pass/fail, a 1-5 score, "
    "and a one-or-two sentence reason."
)

REFLECT_SYSTEM_PROMPT = (
    "You are the reflector. The attempt below failed evaluation. Given the task, "
    "the attempt, the evaluator's reason, and any earlier reflections, write one "
    "concrete, actionable note for the next attempt: what specifically to do "
    "differently. Do not restate the task or the evaluator's reason back -- say "
    "what to change."
)


def _reflection_lines(reflections: list[dict[str, Any]]) -> str:
    if not reflections:
        return "(first attempt -- no reflections yet)"
    return "\n".join(f"- (after attempt {r['attempt']}) {r['critique']}" for r in reflections)


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


@observe(name="reflexion.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `force_memory_only`."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    force_memory_only = bool(settings.get("force_memory_only"))
    max_attempts = settings_obj.reflexion_max_attempts

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

        primary, *fallbacks = agent_models()

        # Bind each model before wrapping in fallbacks -- RunnableWithFallbacks does
        # not forward bind_tools(), but with_fallbacks() itself works on any
        # Runnable, bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        exec_models = [m.bind_tools(tools) for m in [primary, *fallbacks]]
        exec_runnable = _with_fallbacks(exec_models)

        plain_runnable = _with_fallbacks([primary, *fallbacks])

        eval_models = [m.with_structured_output(Evaluation, include_raw=True) for m in [primary, *fallbacks]]
        eval_runnable = _with_fallbacks(eval_models)

        reflect_models = [
            m.with_structured_output(Reflection, include_raw=True) for m in [primary, *fallbacks]
        ]
        reflect_runnable = _with_fallbacks(reflect_models)

        fault_used = {"value": False}

        # --- nodes -------------------------------------------------------------

        def act_node(state: ReflexionState) -> dict[str, Any]:
            budget.check()
            visited.append("act")
            attempt_num = state["attempt"]
            reflection_lines = _reflection_lines(state["reflections"])

            force_memory = force_memory_only and attempt_num == 1 and not fault_used["value"]
            tool_name: str | None = None
            tool_args: dict[str, Any] | None = None
            evidence = "(no tool used)"

            if force_memory:
                fault_used["value"] = True
                messages = [
                    SystemMessage(content=ACT_NO_TOOLS_PROMPT),
                    HumanMessage(
                        content=f"Task: {state['task']}\n\nReflections from earlier attempts:\n{reflection_lines}"
                    ),
                ]
                call_started = time.monotonic()
                response = plain_runnable.invoke(messages, config={"callbacks": get_callbacks()})
                final_latency_ms = (time.monotonic() - call_started) * 1000
                final_tokens = count_tokens(response)
                budget.charge(steps=1, tokens=final_tokens, llm_calls=1)
                emit(
                    "actor",
                    kind="think",
                    text="No tools are available this attempt (forced, to show an ungrounded first attempt).",
                    attempt=attempt_num,
                )
                attempt_text = response.content or ""
            else:
                messages = [
                    SystemMessage(content=ACT_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Task: {state['task']}\n\nReflections from earlier attempts:\n{reflection_lines}"
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
                emit("actor", kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms, attempt=attempt_num)

                if tool_calls:
                    call = tool_calls[0]
                    tool_name, tool_args = call["name"], call.get("args", {})
                    emit(
                        "actor",
                        kind="act",
                        text=f"Calling {tool_name}.",
                        tool=tool_name,
                        args=tool_args,
                        attempt=attempt_num,
                    )
                    tool_started = time.monotonic()
                    result = toolbox.call(tool_name, tool_args)
                    tool_latency_ms = (time.monotonic() - tool_started) * 1000
                    budget.charge(tool_calls=1)
                    evidence = result.output
                    emit(
                        "actor",
                        kind="observe",
                        text=evidence,
                        tool=tool_name,
                        args=tool_args,
                        latency_ms=tool_latency_ms,
                        attempt=attempt_num,
                    )

                    compose_messages = [
                        SystemMessage(content=COMPOSE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"Task: {state['task']}\n\nEvidence just retrieved:\n{evidence}"
                                "\n\nWrite the attempt now."
                            )
                        ),
                    ]
                    compose_started = time.monotonic()
                    compose_response = plain_runnable.invoke(
                        compose_messages, config={"callbacks": get_callbacks()}
                    )
                    final_latency_ms = (time.monotonic() - compose_started) * 1000
                    final_tokens = count_tokens(compose_response)
                    budget.charge(steps=1, tokens=final_tokens, llm_calls=1)
                    attempt_text = compose_response.content or ""
                else:
                    attempt_text = response.content or ""
                    final_tokens, final_latency_ms = 0, 0.0

            emit(
                "actor",
                kind="decide",
                text=f"Attempt {attempt_num}: {attempt_text}",
                args={"attempt": attempt_num},
                tokens=final_tokens,
                latency_ms=final_latency_ms,
                attempt=attempt_num,
            )
            return {
                "attempt_text": attempt_text,
                "attempt_tool": tool_name,
                "attempt_args": tool_args,
                "attempt_evidence": evidence,
            }

        def evaluate_node(state: ReflexionState) -> dict[str, Any]:
            budget.check()
            visited.append("evaluate")
            attempt_num = state["attempt"]
            messages = [
                SystemMessage(content=EVALUATE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nEvidence available to the actor this "
                        f"attempt:\n{state['attempt_evidence']}\n\nThe attempt:\n{state['attempt_text']}"
                    )
                ),
            ]
            call_started = time.monotonic()
            result = eval_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(eval_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The evaluator could not produce a valid verdict after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            if parsed.passed:
                decision, outcome = "stop", "passed"
            elif attempt_num >= max_attempts:
                decision, outcome = "stop", "gave_up"
            else:
                decision, outcome = "retry", "retry"

            text = (
                f"Attempt {attempt_num}: {'passed' if parsed.passed else 'failed'} "
                f"(score {parsed.score}/5) -- {parsed.reason}"
            )
            if outcome == "gave_up":
                text += (
                    f" Retry cap reached ({max_attempts} attempts); returning this "
                    "attempt as the answer, unresolved."
                )
            emit(
                "evaluator",
                kind="decide",
                text=text,
                args={"attempt": attempt_num, "passed": parsed.passed, "score": parsed.score, "outcome": outcome},
                tokens=tokens,
                latency_ms=latency_ms,
                attempt=attempt_num,
            )

            result_state: dict[str, Any] = {
                "decision": decision,
                "evaluation": {"attempt": attempt_num, "passed": parsed.passed, "score": parsed.score, "reason": parsed.reason},
            }
            if decision == "stop":
                visited.append("respond" if outcome == "passed" else "giveup")
                result_state["answer"] = state["attempt_text"]
            return result_state

        def reflect_node(state: ReflexionState) -> dict[str, Any]:
            budget.check()
            visited.append("reflect")
            attempt_num = state["attempt"]
            reflection_lines = _reflection_lines(state["reflections"])
            evaluation = state["evaluation"] or {}
            messages = [
                SystemMessage(content=REFLECT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nAttempt {attempt_num}:\n{state['attempt_text']}"
                        f"\n\nEvaluator's reason for failing it:\n{evaluation.get('reason', '')}"
                        f"\n\nEarlier reflections:\n{reflection_lines}"
                    )
                ),
            ]
            call_started = time.monotonic()
            result = reflect_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(reflect_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The reflector could not produce a valid critique after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            emit(
                "reflector",
                kind="decide",
                text=f"Reflection after attempt {attempt_num}: {parsed.critique}",
                args={"attempt": attempt_num},
                tokens=tokens,
                latency_ms=latency_ms,
                attempt=attempt_num,
            )

            new_reflection = {"attempt": attempt_num, "critique": parsed.critique}
            return {
                "reflections": [*state["reflections"], new_reflection],
                "attempt": attempt_num + 1,
                "attempt_text": "",
                "attempt_tool": None,
                "attempt_args": None,
                "attempt_evidence": "",
                "evaluation": None,
            }

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(ReflexionState)
        graph.add_node("act", act_node)
        graph.add_node("evaluate", evaluate_node)
        graph.add_node("reflect", reflect_node)

        graph.add_edge(START, "act")
        graph.add_edge("act", "evaluate")
        graph.add_conditional_edges("evaluate", lambda state: state["decision"], {"retry": "reflect", "stop": END})
        graph.add_edge("reflect", "act")

        compiled = graph.compile()

        initial_state: ReflexionState = {
            "task": task,
            "attempt": 1,
            "reflections": [],
            "attempt_text": "",
            "attempt_tool": None,
            "attempt_args": None,
            "attempt_evidence": "",
            "evaluation": None,
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


# --- the attempts panel -------------------------------------------------------------


def attempt_summary(steps: list[Step]) -> list[dict[str, Any]]:
    """One entry per attempt, rebuilt from the saved rows -- the actor's tool (if
    any), the attempt text, the evaluator's verdict, and the reflection that
    followed a failed attempt (absent on the final one, whether it passed or the
    retry cap ended it). Same rebuild-from-the-track approach as Step 3's
    `plan_diff`: nothing attempt-specific is added to `AgentRun` itself.
    """
    attempts: dict[int, dict[str, Any]] = {}

    def row(n: int) -> dict[str, Any]:
        return attempts.setdefault(
            n,
            {
                "attempt": n,
                "tool": None,
                "text": "",
                "passed": None,
                "score": None,
                "outcome": None,
                "reflection": None,
            },
        )

    for step in steps:
        n = step.attempt
        if step.agent == "actor":
            if step.kind == "act" and step.tool:
                row(n)["tool"] = step.tool
            elif step.kind == "decide" and step.args and "attempt" in step.args:
                row(n)["text"] = step.text
        elif step.agent == "evaluator" and step.args and "attempt" in step.args:
            entry = row(n)
            entry["passed"] = step.args.get("passed")
            entry["score"] = step.args.get("score")
            entry["outcome"] = step.args.get("outcome")
        elif step.agent == "reflector" and step.args and "attempt" in step.args:
            row(n)["reflection"] = step.text

    return [attempts[n] for n in sorted(attempts)]
