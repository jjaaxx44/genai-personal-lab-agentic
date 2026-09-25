"""Autonomous goal loop: AutoGPT/BabyAGI lineage. The agent is given a goal, not a
task list -- it proposes its own next objective, acts on it with one tool call,
then critiques its own progress and argues whether the goal is now met. There is
no external signal of done, ever: the only brake is the critique the same model
writes about its own work, and the budget behind it.

This demo exists to show the failure as much as the technique: without a real
stopping signal, this kind of loop spins -- it repeats actions, proposes
near-duplicate objectives, and can churn through a whole budget with no progress.
The page keeps a live tally of exactly that, computed by pure functions in this
file, so a reader watches the spin happen rather than being told it can.

The graph is hand-built with `langgraph.graph.StateGraph`, the same choice Steps
3 and 4 made: propose, act and critique are three distinct prompts with three
distinct jobs. The propose node deliberately does not see the full message
history -- it sees a compressed objective log (one line per past objective: what
it was, a one-line result, the critique's verdict) and the list of files written
so far. That compression is not an implementation shortcut; it is why the loop
drifts and repeats itself, exactly as AutoGPT-style agents do, so the README says
so rather than hiding it behind a bigger context window.

Budget and step accounting: every node that calls a model calls `budget.check()`
first, then charges one step, its tokens and one LLM call after the reply comes
back. `BudgetExceeded` is not caught inside a node -- it propagates out of
`compiled.invoke()`, and `run()` turns it into `status="stopped_on_budget"`, the
same shape every other demo uses. The demo's own `autonomous_max_objectives` cap
is enforced the same way: `critique_node` raises `BudgetExceeded` for it directly,
once an objective that didn't argue the goal was met would otherwise loop back to
`propose` past the cap.
"""

import json
import string
import time
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.embeddings import embed
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "autonomous"
NAME = "Autonomous goal loop"
SENTENCE = (
    "Goal in, objectives it sets itself, critique it writes itself — and a tally "
    "of how it starts to spin."
)
SHAPE = "objective → act → critique ↻"

# Node ids double as the `visited` entries the graph map highlights. "done" and
# "cap" are not real StateGraph nodes -- they are pushed onto `visited` at the
# point the run actually ends, the same convention Steps 3 and 4 use.
GRAPH = """flowchart LR
    goal[Goal] --> propose[Agent sets its next objective]
    propose --> act[Act: one tool call toward it]
    act --> critique[Self-critique: progress and stop argument]
    critique -->|keep going| propose
    critique -->|argues goal met| done[Stops on its own criterion]
    critique -.->|objective cap| cap[Stopped]
    act -.->|cap spent| cap[Stopped]
"""

TOOL_NAMES = ("search_corpus", "describe_schema", "run_sql", "write_file", "read_file", "list_files")

# Preset 1 is checkable end to end against samples/corpus/: FSB-114 gives the
# sustained throughput as 840 samples/hour against a rated 1,200, and firmware
# 4.3.1 is the release that adds the fan-curve fix -- see
# samples/corpus/fsb-114-thermal-derating.md and firmware-changelog.md. Presets 2
# and 3 are open-ended on purpose: "make sure this never happens again" and "find
# every interesting pattern" have no checkable end state, so the critique can
# never honestly argue the goal is met and the run should spin into a cap.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "Find the HX-40's sustained throughput and the firmware release that "
            "fixes it, and save a two-sentence summary to summary.md."
        ),
    },
    {
        "task": "Make sure nothing like the Marbury Clinical Labs incident ever happens again.",
    },
    {
        "task": "Find every interesting pattern in the sales database.",
    },
]


# --- structured-output schemas ---------------------------------------------------


class Objective(BaseModel):
    """The propose step's structured reply: the single next objective, not a plan."""

    objective: str = Field(
        description="The single next objective toward the goal -- concrete enough for one tool call to advance it."
    )
    why: str = Field(description="One sentence: why this objective moves toward the goal.")
    expected_tool: str = Field(description="The tool you expect to use for it, by name.")


class Critique(BaseModel):
    """The self-critique's structured reply, made after every objective is acted on."""

    progress: Literal["advanced", "no_change", "regressed"]
    goal_met: bool
    stop_argument: str = Field(
        default="",
        description=(
            "Only meaningful when goal_met is true: cite specifically what was produced "
            "that satisfies the goal. Leave empty otherwise -- a vague claim with no "
            "citation is treated as not met."
        ),
    )
    next_focus: str = Field(description="One sentence: what the next objective should focus on.")


# --- graph state -------------------------------------------------------------------


class CurrentObjective(TypedDict):
    index: int
    objective: str
    why: str
    expected_tool: str


class AutonomousState(TypedDict):
    goal: str
    objective_log: list[dict[str, Any]]
    file_contents: dict[str, str]
    objectives_set: int
    current_objective: CurrentObjective | None
    current_tool: str | None
    current_result: str
    decision: Literal["continue", "stop", ""]
    answer: str


PROPOSE_SYSTEM_PROMPT = (
    "You are pursuing a goal autonomously, one objective at a time. Decide the single "
    "next objective that moves toward the goal -- concrete enough that one tool call "
    "can make progress on it. You are given a compressed log of every objective "
    "already tried, not the full conversation, so read it carefully: do not propose an "
    "objective that repeats one already logged as no_change, regressed, or already met. "
    "Say which tool you expect to use and why."
)

ACT_SYSTEM_PROMPT = (
    "You are working on one objective toward a larger goal. Call exactly one tool that "
    "advances this objective. You are given the compressed log of earlier objectives "
    "and their results -- use the facts it holds; when you write a file, write the "
    "actual findings from that log, never a placeholder saying information is missing. "
    "If the objective needs no tool -- pure reasoning over what you already know -- "
    "answer it directly in your reply text instead."
)

CRITIQUE_SYSTEM_PROMPT = (
    "You are the critic, self-critiquing the objective just attempted. Judge whether "
    "real progress was made toward the goal (advanced / no_change / regressed), and "
    "whether the GOAL AS A WHOLE is now met -- not the objective just attempted. "
    "`goal_met` may only be true when every part of the goal is satisfied by something "
    "concrete that exists now (a file whose current contents you can see below, or a "
    "fact already in the log), and stop_argument cites it. Finding or listing "
    "information is not the same as achieving a goal that asks for an outcome; a goal "
    "whose end state cannot be checked from what has been produced is not met. A vague "
    "or empty stop_argument means the goal is not met, however confident you are. "
    "Also say what the next objective should focus on, whether or not you are stopping."
)


def _log_lines(entries: list[dict[str, Any]], preview_chars: int = 160) -> str:
    """The compressed memory: one line per objective. `propose` and `critique` read
    the short form; `act` reads a longer result excerpt so it can reuse facts found
    earlier -- still a log, never the conversation."""
    if not entries:
        return "(no objectives attempted yet)"
    lines = []
    for e in entries:
        verdict = "goal met" if e["goal_met"] else e["progress"]
        result = _preview(e.get("result_excerpt", e["result_preview"]), preview_chars)
        lines.append(f"{e['index']}. {e['objective']} -> {result} [{verdict}]")
    return "\n".join(lines)


def _preview(text: str, limit: int = 160) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit].rstrip() + "…"


def _files_summary(file_contents: dict[str, str], limit: int = 400) -> str:
    if not file_contents:
        return "No files were written this run."
    parts = ["Files written this run:"]
    for path, content in file_contents.items():
        body = content if len(content) <= limit else content[:limit].rstrip() + "…"
        parts.append(f"\n### {path}\n{body}")
    return "\n".join(parts)


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


# --- the spin tally: pure helpers, exercised without an LLM ----------------------


def normalise_args(tool: str, args: dict[str, Any]) -> str:
    """Canonical key for one (tool, args) pair: lowercase, whitespace collapsed,
    trailing punctuation stripped off string values, dict keys sorted. Two calls
    that only differ in case, spacing or a trailing period normalise to the same key."""

    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            value = " ".join(value.split()).lower()
            return value.rstrip(string.punctuation)
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [_norm(v) for v in value]
        return value

    normalised = {k: _norm(v) for k, v in sorted((args or {}).items())}
    return f"{tool.strip().lower()}::{json.dumps(normalised, sort_keys=True)}"


def repeated_actions(steps: list[Step]) -> list[dict[str, Any]]:
    """Every `act` step whose (tool, normalised args) equals an earlier `act` step in
    the same run. One entry per repeat, naming the row that repeated and the row it
    repeated -- `step i repeats step j`."""
    seen: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    for step in steps:
        if step.kind != "act" or not step.tool:
            continue
        key = normalise_args(step.tool, step.args or {})
        if key in seen:
            pairs.append({"repeat_index": step.index, "original_index": seen[key], "tool": step.tool})
        else:
            seen[key] = step.index
    return pairs


def near_duplicates(
    objectives: list[str], embed_fn: Callable[[list[str]], list[list[float]]], threshold: float
) -> list[dict[str, Any]]:
    """Pairs of objectives, by 1-based position in `objectives`, whose embeddings are
    cosine-similar at or above `threshold`. bge-small embeddings from `core.embeddings`
    are already L2-normalised, so cosine similarity is a plain dot product."""
    if len(objectives) < 2:
        return []
    vectors = embed_fn(list(objectives))
    pairs: list[dict[str, Any]] = []
    for i in range(1, len(vectors)):
        for j in range(i):
            similarity = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            if similarity >= threshold:
                pairs.append({"a": j + 1, "b": i + 1, "similarity": round(similarity, 4)})
    return pairs


def progress_streak(entries: list[dict[str, Any]]) -> int:
    """The longest run of consecutive `no_change` critiques anywhere in the log --
    used for the spin tally's third metric. Distinct from the pairwise tallies above:
    there is nothing to pair, just a run length."""
    longest = current = 0
    for e in entries:
        if e.get("progress") == "no_change":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# --- the objective-log panel -------------------------------------------------------


def objective_rows(steps: list[Step]) -> list[dict[str, Any]]:
    """One row per objective, rebuilt from the trajectory: the propose `decide` row
    opens it, the `act` row (if any) supplies the tool, the `observe` row supplies the
    result preview, and the critic's `decide` row closes it with the verdict. Same
    rebuild-from-the-track approach as Step 3's `plan_diff` and Step 4's
    `attempt_summary` -- nothing objective-specific is added to `AgentRun` itself."""
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step in steps:
        if step.agent == "agent" and step.kind == "decide" and step.args and "objective_text" in step.args:
            if current is not None:
                rows.append(current)
            current = {
                "index": step.args["objective"],
                "objective": step.args["objective_text"],
                "tool": None,
                "result_preview": "",
                "progress": None,
                "goal_met": None,
            }
        elif current is not None and step.kind == "act" and step.tool:
            current["tool"] = step.tool
        elif current is not None and step.kind == "observe":
            current["result_preview"] = _preview(step.text)
        elif (
            current is not None
            and step.agent == "critic"
            and step.kind == "decide"
            and step.args
            and step.args.get("objective") == current["index"]
        ):
            current["progress"] = step.args.get("progress")
            current["goal_met"] = step.args.get("goal_met")
    if current is not None:
        rows.append(current)
    return rows


@observe(name="autonomous.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. `task` is used as the goal. Settings, all optional:
    `on_step`, `allow_web_search`."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    allow_web_search = bool(settings.get("allow_web_search"))
    max_objectives = settings_obj.autonomous_max_objectives
    dup_threshold = settings_obj.autonomous_dup_threshold

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["goal"]

    def emit(agent: str, **fields: Any) -> Step:
        step = agent_run.add_step(agent=agent, **fields)
        if on_step is not None:
            on_step(agent, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        toolbox_names = [*TOOL_NAMES, "web_search"] if allow_web_search else list(TOOL_NAMES)
        toolbox = Toolbox(run_id, names=toolbox_names)
        tools = toolbox.as_langchain_tools()

        primary, *fallbacks = agent_models()

        # Bind each model before wrapping in fallbacks -- RunnableWithFallbacks does
        # not forward bind_tools(), but with_fallbacks() itself works on any
        # Runnable, bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        propose_models = [m.with_structured_output(Objective, include_raw=True) for m in [primary, *fallbacks]]
        propose_runnable = _with_fallbacks(propose_models)

        critique_models = [m.with_structured_output(Critique, include_raw=True) for m in [primary, *fallbacks]]
        critique_runnable = _with_fallbacks(critique_models)

        exec_models = [m.bind_tools(tools) for m in [primary, *fallbacks]]
        exec_runnable = _with_fallbacks(exec_models)

        # --- nodes -------------------------------------------------------------

        def propose_node(state: AutonomousState) -> dict[str, Any]:
            budget.check()
            visited.append("propose")
            idx = state["objectives_set"] + 1
            log_lines = _log_lines(state["objective_log"])
            files_line = ", ".join(state["file_contents"]) or "(none yet)"
            # The critique -> next objective hand-off, AutoGPT-style: without it the
            # critic can name the missing step every turn and the proposer never hears it.
            last_focus = (
                state["objective_log"][-1].get("next_focus", "") if state["objective_log"] else ""
            ) or "(none yet)"
            messages = [
                SystemMessage(content=PROPOSE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Goal: {state['goal']}\n\nObjective log so far:\n{log_lines}"
                        f"\n\nThe critic's advice after the last objective: {last_focus}"
                        f"\n\nFiles written so far: {files_line}\n\nAvailable tools:\n{toolbox.describe()}"
                    )
                ),
            ]
            call_started = time.monotonic()
            result = propose_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(propose_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The agent could not produce a valid objective after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            emit(
                "agent",
                kind="decide",
                text=f"Objective {idx}: {parsed.objective} -- {parsed.why}",
                args={
                    "objective": idx,
                    "objective_text": parsed.objective,
                    "why": parsed.why,
                    "expected_tool": parsed.expected_tool,
                },
                tokens=tokens,
                latency_ms=latency_ms,
            )

            # Live near-duplicate detection: compare the new objective against every
            # earlier one and announce only pairs that involve it -- earlier pairs
            # were already announced when they were the newest.
            all_texts = [e["objective"] for e in state["objective_log"]] + [parsed.objective]
            for pair in near_duplicates(all_texts, embed, dup_threshold):
                if pair["b"] == len(all_texts):
                    emit(
                        "monitor",
                        kind="decide",
                        text=(
                            f"Objective {pair['b']} is a near-duplicate of objective "
                            f"{pair['a']} ({pair['similarity']:.2f})."
                        ),
                        args=pair,
                    )

            return {
                "objectives_set": idx,
                "current_objective": {
                    "index": idx,
                    "objective": parsed.objective,
                    "why": parsed.why,
                    "expected_tool": parsed.expected_tool,
                },
            }

        def act_node(state: AutonomousState) -> dict[str, Any]:
            budget.check()
            visited.append("act")
            objective = state["current_objective"]
            assert objective is not None
            messages = [
                SystemMessage(content=ACT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Goal: {state['goal']}\n\nObjective log so far:\n"
                        f"{_log_lines(state['objective_log'], preview_chars=400)}"
                        f"\n\nCurrent objective: {objective['objective']}"
                        f"\nWhy: {objective['why']}\nExpected tool: {objective['expected_tool']}"
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
            emit("agent", kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)

            tool_name: str | None = None
            file_contents = state["file_contents"]
            if tool_calls:
                call = tool_calls[0]
                tool_name, tool_args = call["name"], call.get("args", {})
                act_step = emit("agent", kind="act", text=f"Calling {tool_name}.", tool=tool_name, args=tool_args)

                tool_started = time.monotonic()
                result = toolbox.call(tool_name, tool_args)
                tool_latency_ms = (time.monotonic() - tool_started) * 1000
                budget.charge(tool_calls=1)
                result_text = result.output
                emit(
                    "agent",
                    kind="observe",
                    text=result_text,
                    tool=tool_name,
                    args=tool_args,
                    latency_ms=tool_latency_ms,
                )

                # Live repeated-action detection: recompute over every act row so far
                # and announce only a repeat that ends at the row just emitted.
                for pair in repeated_actions(agent_run.steps):
                    if pair["repeat_index"] == act_step.index:
                        emit(
                            "monitor",
                            kind="decide",
                            text=(
                                f"Step {pair['repeat_index']:02d} repeats step "
                                f"{pair['original_index']:02d}'s action ({pair['tool']})."
                            ),
                            args=pair,
                        )

                if tool_name == "write_file" and result.ok:
                    file_contents = {**file_contents, tool_args.get("path", ""): tool_args.get("content", "")}
            else:
                result_text = response.content or ""
                emit("agent", kind="observe", text=result_text)

            return {"current_tool": tool_name, "current_result": result_text, "file_contents": file_contents}

        def critique_node(state: AutonomousState) -> dict[str, Any]:
            budget.check()
            visited.append("critique")
            objective = state["current_objective"]
            assert objective is not None
            idx = objective["index"]
            log_lines = _log_lines(state["objective_log"])
            messages = [
                SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Goal: {state['goal']}\n\nObjective just attempted: {objective['objective']}"
                        f"\nTool used: {state['current_tool'] or 'none'}\nResult:\n{state['current_result']}"
                        f"\n\nObjective log so far:\n{log_lines}"
                        f"\n\nFiles as they stand now:\n{_files_summary(state['file_contents'])}"
                    )
                ),
            ]
            call_started = time.monotonic()
            result = critique_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(critique_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The critic could not produce a valid verdict after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            stop_argument = (parsed.stop_argument or "").strip()
            goal_met = bool(parsed.goal_met and stop_argument)
            note = ""
            if parsed.goal_met and not stop_argument:
                note = " (goal_met asserted with no citation of what was produced -- treated as not met.)"

            log_entry = {
                "index": idx,
                "objective": objective["objective"],
                "result_preview": _preview(state["current_result"]),
                "result_excerpt": _preview(state["current_result"], 400),
                "next_focus": parsed.next_focus,
                "progress": parsed.progress,
                "goal_met": goal_met,
            }
            updated_log = [*state["objective_log"], log_entry]

            text = (
                f"Objective {idx}: {parsed.progress}{note} -- "
                f"{stop_argument if goal_met else parsed.next_focus}"
            )
            emit(
                "critic",
                kind="decide",
                text=text,
                args={"objective": idx, "progress": parsed.progress, "goal_met": goal_met},
                tokens=tokens,
                latency_ms=latency_ms,
            )

            streak = progress_streak(updated_log)
            if streak >= 2 and parsed.progress == "no_change":
                emit(
                    "monitor",
                    kind="decide",
                    text=f"No-progress streak: {streak} consecutive objectives with no advancement.",
                    args={"streak": streak},
                )

            if goal_met:
                visited.append("done")
                answer = f"{stop_argument}\n\n{_files_summary(state['file_contents'])}"
                return {"decision": "stop", "answer": answer, "objective_log": updated_log}

            if state["objectives_set"] >= max_objectives:
                raise BudgetExceeded(
                    "autonomous_max_objectives",
                    f"Stopped on the objective cap: {max_objectives} of {max_objectives} objectives set.",
                )

            return {"decision": "continue", "objective_log": updated_log}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(AutonomousState)
        graph.add_node("propose", propose_node)
        graph.add_node("act", act_node)
        graph.add_node("critique", critique_node)

        graph.add_edge(START, "propose")
        graph.add_edge("propose", "act")
        graph.add_edge("act", "critique")
        graph.add_conditional_edges("critique", lambda state: state["decision"], {"continue": "propose", "stop": END})

        compiled = graph.compile()

        initial_state: AutonomousState = {
            "goal": task,
            "objective_log": [],
            "file_contents": {},
            "objectives_set": 0,
            "current_objective": None,
            "current_tool": None,
            "current_result": "",
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
