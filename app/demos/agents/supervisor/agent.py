"""Supervisor-worker: a supervisor that never touches a tool itself, routing every
turn to exactly one of three specialists -- a researcher (the corpus), an analyst
(the sales database) and a writer (drafts the final answer from what the other two
found) -- and deciding, turn by turn, when the task is done.

This is a different shape from Step 8's sub-agent delegation. There, one agent
decides for itself which self-contained pieces are worth handing off, keeps doing
everything else inline, and gets back a single summary per delegation. Here,
routing *is* the whole job: the supervisor never works the task directly, every
worker turn reports back to it and it alone, and it chooses the next worker on
every hop rather than firing off one bounded piece of work and moving on. Step 10
(Swarm) removes the supervisor entirely and lets the workers hand off to each
other directly -- contrasted on this demo's own page in one sentence.

Workers share no message history with each other or with the supervisor -- each
turn gets a fresh prompt built from the task, the supervisor's instruction for
that turn, and the reports gathered so far. That is a shared *reports board*, not
a shared conversation: closer to Step 8's isolation than to Step 10's shared
state, but built for routing rather than one-off delegation.

The graph is hand-built with `langgraph.graph.StateGraph`: the supervisor's
routing decision, each worker's own tool loop and the final answer are three
distinct jobs with three distinct prompts, the same reason Plan-and-Execute
(Step 3) and Reflexion (Step 4) are hand-built rather than `create_agent`. A
worker's tool calls are dispatched by calling `Toolbox.call()` directly inside
its own node, exactly like every other hand-built graph in this codebase.

Budget and step accounting: every model call -- the supervisor's routing
decision, a worker's tool-decision calls, the writer's draft -- calls
`budget.check()` first, then charges one step, its tokens and one LLM call once
the reply comes back. `BudgetExceeded` is not caught inside a node; it propagates
out of `compiled.invoke()`, and `run()` turns it into `status="stopped_on_budget"`,
the same shape every other demo uses. `supervisor_max_hops` is a second, softer
cap enforced only inside `supervisor_node`: once it is reached the supervisor's
own choice is overridden to `finish` regardless of what it picked, the same
"programmatic override after the model call" shape Reflexion uses for its retry
cap. At the defaults, a hop costs at least two steps (the supervisor's own call
plus one worker call), so with a run that never finishes on its own the shared
step cap is reached first -- the hop cap is a backstop for a cheaper hop shape,
not the mechanism this demo's stuck-loop preset is built to exercise.
"""

import time
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "supervisor"
NAME = "Supervisor-worker"
SENTENCE = "A supervisor routes each turn to one specialist and decides when the task is done."
SHAPE = "supervisor → worker ×3 ↻"

# Node ids double as the `visited` entries the graph map highlights -- a name
# changed here must change in both places. `cap` is not a real node, like every
# other hand-built graph in this codebase: it is pushed onto `visited` only when
# the exception handler in `run()` catches a spent budget.
GRAPH = """flowchart LR
    task[Task] --> supervisor[Supervisor routes the next turn]
    supervisor -->|researcher| researcher[Researcher searches the corpus]
    supervisor -->|analyst| analyst[Analyst queries the sales database]
    supervisor -->|writer| writer[Writer drafts the answer from reports]
    researcher --> supervisor
    analyst --> supervisor
    writer --> supervisor
    supervisor -->|finish| respond[Answer returned]
    supervisor -.->|cap spent| cap[Stopped]
"""

RESEARCHER_TOOLS = ("search_corpus",)
ANALYST_TOOLS = ("describe_schema", "run_sql")
# The set every worker draws from, used only to size the run's Toolbox. Each
# worker is bound to its own subset above -- a router demo is the natural place
# to show tools chosen *per specialist*, not shared wholesale the way every
# earlier demo in this repo hands the same toolbox to one agent.
TOOL_NAMES = (*RESEARCHER_TOOLS, *ANALYST_TOOLS)

# Both presets pair a corpus fact with an independent sales-database fact, so the
# supervisor always has a genuine two-specialist split to route across. Verified
# against the bundled data: 1,297 Rock tracks falls in procurement-policy.md's
# "up to £2,500: budget holder's approval only" tier when read as a sum in
# pounds; the HX-40's sustained throughput is 840 samples/hour; there are 374
# Metal tracks in Chinook.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "How many tracks are in the Rock genre in the sales database, and what "
            "does the procurement policy require for approving an order of that "
            "many pounds?"
        ),
    },
    {
        "task": (
            "What is the HX-40's sustained throughput in samples per hour, and how "
            "many tracks are in the Metal genre in the sales database?"
        ),
    },
]


# --- structured-output schemas ---------------------------------------------------


class Route(BaseModel):
    """The supervisor's structured reply -- who goes next, and why."""

    next: Literal["researcher", "analyst", "writer", "finish"]
    reason: str = Field(description="One sentence: why this worker, or why the task is done.")
    instruction: str = Field(
        description=(
            "What this turn's worker should do, stated so it can act with no other "
            "context than this instruction, the task and the reports gathered so far. "
            "Ignored when next is 'finish'."
        )
    )


class RouteNoFinish(BaseModel):
    """The same schema with `finish` removed -- used only when the "Supervisor
    can't declare done" fault toggle is on, so the run keeps routing between
    workers until a budget cap stops it rather than the supervisor ending it."""

    next: Literal["researcher", "analyst", "writer"]
    reason: str = Field(description="One sentence: why this worker goes next.")
    instruction: str = Field(
        description=(
            "What this turn's worker should do, stated so it can act with no other "
            "context than this instruction, the task and the reports gathered so far."
        )
    )


# --- graph state -------------------------------------------------------------------


class SupervisorState(TypedDict):
    task: str
    reports: list[dict[str, Any]]
    writer_draft: str
    hops: int
    routes: list[dict[str, Any]]
    next: str
    answer: str


SUPERVISOR_SYSTEM_PROMPT = (
    "You are the supervisor. You never work the task yourself -- each turn you "
    "route to exactly one specialist and give it a short instruction, then read "
    "what it reports back before deciding again. The specialists:\n"
    "- researcher: searches a corpus of product, incident and policy documents "
    "(search_corpus). Use it for anything a written document would answer.\n"
    "- analyst: inspects and queries the sales database (describe_schema, "
    "run_sql). Use it for anything that database would answer.\n"
    "- writer: has no tools. It drafts the final answer from the reports "
    "gathered so far. Route to it once every part of the task has a report to "
    "draw on.\n"
    "Route to 'finish' only after the writer has produced a draft that fully "
    "answers the task -- never before it has run. Give a one-sentence reason for "
    "your choice, and, unless you are finishing, an instruction naming exactly "
    "what this turn's worker should do."
)

SUPERVISOR_SYSTEM_PROMPT_NO_FINISH = (
    "You are the supervisor. You never work the task yourself -- each turn you "
    "route to exactly one specialist and give it a short instruction, then read "
    "what it reports back before deciding again. The specialists:\n"
    "- researcher: searches a corpus of product, incident and policy documents "
    "(search_corpus). Use it for anything a written document would answer.\n"
    "- analyst: inspects and queries the sales database (describe_schema, "
    "run_sql). Use it for anything that database would answer.\n"
    "- writer: has no tools. It drafts the final answer from the reports "
    "gathered so far.\n"
    "You do not have a 'finish' option this run -- keep routing between the "
    "three specialists. Give a one-sentence reason for your choice, and an "
    "instruction naming exactly what this turn's worker should do."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are the researcher. You have one tool, search_corpus, and no memory of "
    "anything beyond what is given to you below. Carry out the instruction using "
    "the tool as needed, then reply with a short, direct report (one or two "
    "sentences) that states what you found. Before every tool call, write one "
    "short sentence in your reply saying what you are about to do and why, then "
    "call the tool in the same turn."
)

ANALYST_SYSTEM_PROMPT = (
    "You are the analyst. You have two tools, describe_schema and run_sql, and "
    "no memory of anything beyond what is given to you below. Carry out the "
    "instruction using the tools as needed, then reply with a short, direct "
    "report (one or two sentences) that states what you found. Before every tool "
    "call, write one short sentence in your reply saying what you are about to "
    "do and why, then call the tool in the same turn."
)

WRITER_SYSTEM_PROMPT = (
    "You are the writer. You have no tools. Given the task and the reports the "
    "other specialists gathered, write the final answer in full sentences, "
    "naming which report each part of it rests on. If a part of the task has no "
    "report to draw on, say so rather than guessing."
)

WORKER_PROMPTS = {"researcher": RESEARCHER_SYSTEM_PROMPT, "analyst": ANALYST_SYSTEM_PROMPT}


def _reports_text(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "(none yet)"
    lines = []
    for i, r in enumerate(reports):
        lines.append(f"{i + 1}. [{r['agent']}] instructed: {r['instruction']}\n   reported: {r['report']}")
    return "\n".join(lines)


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


@observe(name="supervisor.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`; `force_no_finish`
    (default False) -- removes `finish` from the supervisor's routing options, so
    the run keeps handing off between workers until a budget cap stops it."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    force_no_finish = bool(settings.get("force_no_finish", False))
    max_hops = settings_obj.supervisor_max_hops

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
        researcher_tools = toolbox.as_langchain_tools(names=list(RESEARCHER_TOOLS))
        analyst_tools = toolbox.as_langchain_tools(names=list(ANALYST_TOOLS))

        primary, *fallbacks = agent_models()

        # Bind before wrapping in fallbacks -- RunnableWithFallbacks does not
        # forward bind_tools(), but with_fallbacks() itself works on any Runnable,
        # bound or not. See AGENTIC_IMPLEMENTATION_PLAN.md, "Model access".
        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        researcher_models = [m.bind_tools(researcher_tools) for m in [primary, *fallbacks]]
        researcher_runnable = _with_fallbacks(researcher_models)

        analyst_models = [m.bind_tools(analyst_tools) for m in [primary, *fallbacks]]
        analyst_runnable = _with_fallbacks(analyst_models)

        writer_runnable = _with_fallbacks([primary, *fallbacks])

        route_cls = RouteNoFinish if force_no_finish else Route
        supervisor_models = [
            m.with_structured_output(route_cls, include_raw=True) for m in [primary, *fallbacks]
        ]
        supervisor_runnable = _with_fallbacks(supervisor_models)

        supervisor_prompt = SUPERVISOR_SYSTEM_PROMPT_NO_FINISH if force_no_finish else SUPERVISOR_SYSTEM_PROMPT

        # --- one tool-calling turn, shared by the researcher and analyst nodes ---

        def _run_tool_worker(
            name: str, tools: list[Any], runnable: Any, instruction: str, reports_text: str
        ) -> str:
            messages: list[Any] = [
                SystemMessage(content=WORKER_PROMPTS[name]),
                HumanMessage(
                    content=(
                        f"Task: {task}\n\nReports gathered so far:\n{reports_text}\n\n"
                        f"Your instruction this turn: {instruction}"
                    )
                ),
            ]
            while True:
                budget.check()
                call_started = time.monotonic()
                response = runnable.invoke(messages, config={"callbacks": get_callbacks()})
                latency_ms = (time.monotonic() - call_started) * 1000
                tokens = count_tokens(response)
                budget.charge(steps=1, tokens=tokens, llm_calls=1)

                tool_calls = response.tool_calls or []
                if not tool_calls:
                    text = response.content or "(no result)"
                    emit(name, kind="decide", text=text, tokens=tokens, latency_ms=latency_ms)
                    return text

                think_text = response.content or "(no reasoning text with this call)"
                emit(name, kind="think", text=think_text, tokens=tokens, latency_ms=latency_ms)
                messages.append(response)
                for call in tool_calls:
                    budget.check()
                    tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]
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

        # --- nodes -------------------------------------------------------------

        def supervisor_node(state: SupervisorState) -> dict[str, Any]:
            budget.check()
            visited.append("supervisor")
            hop_num = state["hops"] + 1
            reports_text = _reports_text(state["reports"])
            messages = [
                SystemMessage(content=supervisor_prompt),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nReports gathered so far:\n{reports_text}\n\n"
                        f"This is hop {hop_num} of at most {max_hops}. Decide the next step."
                    )
                ),
            ]
            call_started = time.monotonic()
            result = supervisor_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                parsed, raw = _retry_parsed(supervisor_runnable, messages, result.get("parsing_error"))
            if parsed is None:
                raise RuntimeError("The supervisor could not produce a valid routing decision after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            next_choice = parsed.next
            note = ""
            if hop_num > max_hops and next_choice != "finish":
                next_choice, note = "finish", " (auto: hop cap reached)"
            elif next_choice == "finish" and not state["writer_draft"]:
                note = " (writer never ran -- answer assembled directly from reports)"

            route_entry = {
                "hop": hop_num,
                "next": next_choice,
                "reason": parsed.reason,
                "instruction": "" if next_choice == "finish" else parsed.instruction,
            }
            # 'handoff' for the hops that hand the turn to a worker, 'decide' only
            # for the one hop that actually ends the run -- Step 8 uses the same
            # 'handoff' kind for its own delegate_subagent call.
            emit(
                "supervisor",
                kind="decide" if next_choice == "finish" else "handoff",
                text=f"Hop {hop_num}: routes to {next_choice}{note} -- {parsed.reason}",
                args=route_entry,
                tokens=tokens,
                latency_ms=latency_ms,
            )
            return {"next": next_choice, "hops": hop_num, "routes": [*state["routes"], route_entry]}

        def researcher_node(state: SupervisorState) -> dict[str, Any]:
            visited.append("researcher")
            instruction = state["routes"][-1]["instruction"]
            report = _run_tool_worker(
                "researcher", researcher_tools, researcher_runnable, instruction, _reports_text(state["reports"])
            )
            new_report = {"agent": "researcher", "instruction": instruction, "report": report}
            return {"reports": [*state["reports"], new_report]}

        def analyst_node(state: SupervisorState) -> dict[str, Any]:
            visited.append("analyst")
            instruction = state["routes"][-1]["instruction"]
            report = _run_tool_worker(
                "analyst", analyst_tools, analyst_runnable, instruction, _reports_text(state["reports"])
            )
            new_report = {"agent": "analyst", "instruction": instruction, "report": report}
            return {"reports": [*state["reports"], new_report]}

        def writer_node(state: SupervisorState) -> dict[str, Any]:
            budget.check()
            visited.append("writer")
            instruction = state["routes"][-1]["instruction"]
            reports_text = _reports_text(state["reports"])
            messages = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n\nReports gathered so far:\n{reports_text}\n\n"
                        f"Instruction: {instruction}\n\nWrite the final answer now."
                    )
                ),
            ]
            call_started = time.monotonic()
            response = writer_runnable.invoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)

            draft = response.content or ""
            emit("writer", kind="decide", text=draft, tokens=tokens, latency_ms=latency_ms)
            new_report = {"agent": "writer", "instruction": instruction, "report": draft}
            return {"reports": [*state["reports"], new_report], "writer_draft": draft}

        def respond_node(state: SupervisorState) -> dict[str, Any]:
            visited.append("respond")
            if state["writer_draft"]:
                return {"answer": state["writer_draft"]}
            joined = " ".join(r["report"] for r in state["reports"])
            answer = joined or "The supervisor finished before any worker ran."
            emit(
                "agent",
                kind="decide",
                text="Finished without a writer draft -- answer assembled directly from worker reports.",
            )
            return {"answer": answer}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(SupervisorState)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)
        graph.add_node("respond", respond_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "finish": "respond"},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("respond", END)

        compiled = graph.compile()

        initial_state: SupervisorState = {
            "task": task,
            "reports": [],
            "writer_draft": "",
            "hops": 0,
            "routes": [],
            "next": "",
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


# --- the routing table ---------------------------------------------------------


def routing_table(steps: list[Step]) -> list[dict[str, Any]]:
    """One row per hop, rebuilt from the supervisor's own `decide` rows -- same
    rebuild-from-the-track approach as Step 3's `plan_diff` and Step 4's
    `attempt_summary`: nothing routing-specific is added to `AgentRun` itself."""
    return [
        {
            "hop": step.args["hop"],
            "next": step.args["next"],
            "reason": step.args["reason"],
            "instruction": step.args.get("instruction", ""),
        }
        for step in steps
        if step.agent == "supervisor" and step.args and "hop" in step.args
    ]
