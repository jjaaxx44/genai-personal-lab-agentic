"""Deep agents, built on LangChain's `deepagents` library: a to-do list held in
graph state, a file system used as external memory, and sub-agents for context
isolation -- packaged behind `create_deep_agent()` rather than hand-built. Step 3
(Plan-and-Execute) hand-builds the plan; Step 8 (Sub-agent delegation) hand-builds
the delegation. This demo shows the same two ingredients as the current idiom
packages them, plus the third ingredient neither of those steps needed: a file
system the agent treats as its own working memory rather than the conversation.

The library gives the model, by default, `write_todos` is *not* included --
0.7.15 moved it out of `deepagents` and into `langchain.agents.middleware`
(`TodoListMiddleware`), so it is wired in explicitly below alongside the
library's own filesystem and sub-agent middleware. See the README's "In this
demo" section for the full list of what the installed 0.7.15 API actually gives
by default versus what the original brief expected from an older/newer version
-- verified by reading the installed source, not the online docs.

Every row on the trajectory comes from middleware hooks, not from parsing the
final message list: `wrap_model_call` emits the `think` row as each model call
returns, and `wrap_tool_call` emits `act`/`observe` (or `decide` for
`write_todos`, or `handoff`/`observe` for `task`) as each tool call returns.
This is the one place in the lab that departs from the outer-stream-loop
pattern (Step 2, Step 9's `.stream(stream_mode="updates")`), because a
sub-agent launched through `task` runs its whole graph *inside* one tool call
of the main agent's own stream -- there is no outer chunk for its individual
steps, so the only place to observe them is the middleware hooks that fire
while it runs. `make_middleware(agent_name, ...)` builds one three-hook bundle
per agent (main, researcher, analyst), each closing over the same shared
`Budget`, `emit()` and trajectory list, so every hop -- however deep -- lands
on one flat, ordered track.
"""

import time
from pathlib import Path
from typing import Any, Callable
from contextvars import ContextVar

from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    TodoListMiddleware,
    after_model,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents import FilesystemMiddleware, SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox, tool_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "deep_agent"
NAME = "Deep agents"
SENTENCE = (
    "A to-do list held in state, a file system for notes and drafts, and sub-agents for "
    "the pieces."
)
SHAPE = "plan → write files ↻ → assemble"

# Node ids double as `visited` entries; a name changed here must change in the
# middleware below too.
GRAPH = """flowchart LR
    task[Task] --> agent[Main agent plans and acts]
    agent -->|write_todos| todos[To-do list updated]
    todos --> agent
    agent -->|task| sub[Sub-agent works in its own context]
    sub -->|summary + file| agent
    agent -->|file tools| vfs[Files written and read]
    vfs --> agent
    agent -->|finish| assemble[Answer assembled from files]
    agent -.->|cap spent| cap[Stopped]
"""

# The library's own file tools, minus `execute` -- rule 5. `create_deep_agent`
# only ever adds `execute` when the backend implements `SandboxBackendProtocol`
# (see `deepagents/middleware/filesystem.py`); `FilesystemBackend` does not, so
# `execute` is never registered in the first place. Passing this explicit list
# to every `FilesystemMiddleware` we build (main, researcher, analyst, and --
# via the library's own name-based inheritance -- the auto-added
# general-purpose sub-agent) is the second, redundant layer: even if a future
# release of the library grew a sandboxing default, this list still excludes
# it by name. Verified LLM-free in the "harness" panel and in the verification
# script recorded in the implementation report.
SAFE_FS_TOOLS: list[str] = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

RESEARCHER_TOOLS = ("search_corpus",)
ANALYST_TOOLS = ("describe_schema", "run_sql")
ALLOWED_SUBAGENTS = ("researcher", "analyst")

PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "Write a short briefing for a lab manager deciding whether to buy HX-40 "
            "analysers: the realistic sustained throughput and why, the firmware that "
            "fixes it, and what the procurement policy requires for an order of three "
            "instruments."
        ),
    },
    {
        "task": (
            "Produce a one-page summary of the sales database: the top three billing "
            "countries by revenue and the top-revenue genre, with a sentence on each."
        ),
    },
    {
        "task": (
            "What went wrong at Marbury Clinical Labs on 2026-02-14, which runbook rule "
            "would have caught it sooner, and how many customers in the sales database "
            "are in the USA?"
        ),
    },
]

MAIN_SYSTEM_PROMPT = (
    "You are the main agent on a multi-part task. Work it like this:\n\n"
    "1. Call write_todos first, breaking the task into concrete items. Mark exactly one "
    "item in_progress at a time -- never more than one, and never zero while work "
    "remains.\n"
    "2. You have no search or database tools of your own -- delegate every lookup with "
    "the task tool. The researcher searches the document corpus: HX-40 product brief, "
    "field service bulletin, benchmark report, firmware changelog, procurement policy, "
    "on-call runbook, incident postmortem. The analyst queries the sales database, a "
    "music store's customers, invoices, tracks and genres -- nothing about instruments "
    "or policy. Give each launch one narrow question.\n"
    "3. A sub-agent's reply names the file it wrote. Read only paths a reply names or ls "
    "shows -- never guess one. If a sub-agent stops on its step cap without writing a "
    "file, do not resend the same brief: split it into a narrower question, or mark the "
    "item abandoned.\n"
    "4. Keep your own notes and every sub-agent's findings in files, not in your "
    "replies -- write_file, read_file, edit_file, ls, glob and grep are your working "
    "memory. Your final deliverable belongs in report.md.\n"
    "5. Before you finish, every to-do item must be completed, or explicitly abandoned: "
    "rewrite an abandoned item's content to start with 'ABANDONED: <reason>' and mark it "
    "completed. Never stop while an item is still pending or in_progress.\n"
    "6. Once every item is completed or abandoned, write report.md (if you have not "
    "already) and stop -- do not call write_todos again just to restate a finished list."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are the researcher sub-agent. You see only the task below, not the parent's "
    "conversation. Use search_corpus to answer it, write your findings to a named file "
    "(for example research-<topic>.md), and reply with one paragraph naming that file "
    "and summarising what it contains.\n\n"
    "The corpus is not in your file system: its documents are reachable only through "
    "search_corpus, and each result already carries the passage text -- read_file, ls "
    "and glob will never find them. You have only a few model calls per launch, so "
    "search at most three times, then write the file, then reply. A file with partial "
    "findings is worth more than none."
)

ANALYST_SYSTEM_PROMPT = (
    "You are the analyst sub-agent. You see only the task below, not the parent's "
    "conversation. Use describe_schema and run_sql to answer it, write your findings to "
    "a named file (for example analysis-<topic>.md), and reply with one paragraph naming "
    "that file and summarising what it contains. If the database cannot answer the "
    "question, say so in the file and the reply -- never fill the gap with invented "
    "figures or generic policy."
)

ASSEMBLE_SYSTEM_PROMPT = (
    "Write the final answer to the task below, using only the files listed -- you have "
    "not seen the conversation that produced them. report.md, if present, is the agent's "
    "own deliverable and should anchor your answer; the other files are supporting "
    "notes. If report.md is missing, say so and answer from whatever files exist. Name "
    "the files your answer rests on."
)

ASSEMBLE_CHAR_CAP = 12_000


# --- small helpers -----------------------------------------------------------------


def _extract_ai_message(response: Any) -> AIMessage | None:
    """`wrap_model_call`'s handler returns a `ModelResponse` in this LangChain
    version, but the type is documented as `ModelResponse | AIMessage` -- so both
    are handled rather than assumed. Mirrors `tool_design/agent.py`."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if text:
        return text
    return str(getattr(message, "content", "") or "")


def _extract_tool_message_text(result: Any) -> str:
    """A `wrap_tool_call` handler's return is a `ToolMessage` for most tools, or a
    `Command` whose `update["messages"]` holds one (`write_todos`, `task`). Both
    shapes are read for the observation text."""
    if isinstance(result, ToolMessage):
        return _message_text(result)
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        for message in reversed(update.get("messages") or []):
            if isinstance(message, ToolMessage):
                return _message_text(message)
    return str(result)


def _todo_summary(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "Todo list updated: 0 items."
    counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for item in todos:
        counts[item.get("status", "pending")] = counts.get(item.get("status", "pending"), 0) + 1
    return (
        f"Todo list updated: {len(todos)} item(s) -- {counts['completed']} completed, "
        f"{counts['in_progress']} in progress, {counts['pending']} pending."
    )


def open_todo_items(todos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`pending` or `in_progress` items -- what the finish guard and the page's
    to-do panel both call "still open"."""
    return [t for t in todos if t.get("status") in ("pending", "in_progress")]


def finish_guard_message(open_items: list[dict[str, Any]]) -> str:
    """The message the finish guard appends when the agent tries to stop with
    open items. A pure function so it can be checked LLM-free against hand-made
    `todos`."""
    lines = "; ".join(f"{t.get('status')}: {t.get('content')}" for t in open_items)
    return (
        f"{len(open_items)} to-do item(s) are still open: {lines}. Mark each completed, "
        "or abandoned with a reason, then finish."
    )


def build_assemble_input(task: str, todos: list[dict[str, Any]], run_dir: Path) -> tuple[str, list[str]]:
    """The plain-text prompt body for `assemble()`, plus which files it read.

    `report.md` is read first, then every other file in name order, up to
    `ASSEMBLE_CHAR_CAP` total characters -- with a note appended when the cap
    cuts a file short or drops later ones entirely. Pure and file-only (no
    message history), so it is checkable against a scratch `run_dir` with no
    model and no graph.
    """
    files = sorted((p for p in run_dir.rglob("*") if p.is_file()), key=lambda p: p.name)
    ordered = sorted(files, key=lambda p: (p.name != "report.md", p.name))

    sections: list[str] = []
    read: list[str] = []
    budget_chars = ASSEMBLE_CHAR_CAP
    truncated = False

    for path in ordered:
        if budget_chars <= 0:
            truncated = True
            break
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name = str(path.relative_to(run_dir))
        block = f"--- {name} ---\n{content}\n"
        if len(block) > budget_chars:
            block = block[:budget_chars].rstrip() + "\n[truncated]\n"
            truncated = True
        sections.append(block)
        read.append(name)
        budget_chars -= len(block)

    todo_lines = "\n".join(f"- [{t.get('status')}] {t.get('content')}" for t in todos) or "(none)"
    body = (
        f"Task:\n{task}\n\nFinal to-do list:\n{todo_lines}\n\nFiles"
        + (" (truncated to fit the budget):" if truncated else ":")
        + "\n\n"
        + ("\n".join(sections) if sections else "(no files were written)")
    )
    if not any(name == "report.md" for name in read):
        body += "\n\n(report.md was not found; answer from whatever files exist above.)"
    return body, read


# --- sub-agent specs ----------------------------------------------------------------


def _subagent_spec(
    name: str,
    description: str,
    system_prompt: str,
    work_tools: list[Any],
    model: Any,
    fallbacks: list[Any],
    backend: FilesystemBackend,
    launch_cap: int,
    shared: dict[str, Any],
) -> SubAgent:
    check_budget, charge_model, run_tool = make_middleware(name, launch_cap=launch_cap, shared=shared)
    middleware: list[Any] = []
    if fallbacks:
        middleware.append(ModelFallbackMiddleware(*fallbacks))
    middleware += [check_budget, charge_model, run_tool]
    # Replaces this sub-agent's own auto-built FilesystemMiddleware by name (see
    # `deepagents/middleware/subagents.py`'s `SubAgent["middleware"]` docstring):
    # same confined `backend`, but an explicit tool list with no `execute`.
    middleware.append(FilesystemMiddleware(backend=backend, tools=SAFE_FS_TOOLS))
    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tools": work_tools,
        "model": model,
        "middleware": middleware,
    }


# --- the shared middleware factory --------------------------------------------------


def make_middleware(
    agent_name: str,
    *,
    launch_cap: int | None,
    shared: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """One `before_model` / `wrap_model_call` / `wrap_tool_call` bundle, closing
    over the run's shared state (`shared`, populated by `run()`). Building this
    fresh per agent name is what lets one shared `Budget` and trajectory list
    absorb rows from the main agent and every sub-agent without the hooks
    needing to know about each other.

    `launch_cap` is `None` for the main agent (its cap is the shared `Budget`'s
    step cap) and `subagent_max_steps` for a sub-agent (its own per-launch cap,
    reset to zero each time `task` dispatches a fresh call -- see the `task`
    branch of `run_tool` below).
    """
    budget: Budget = shared["budget"]
    emit: Callable[..., Step] = shared["emit"]
    visited: list[str] = shared["visited"]
    cap_hit: dict[str, str] = shared["cap_hit"]
    sub_cap_hit: dict[str, str] = shared["sub_cap_hit"]
    launch_steps: dict[str, int] = shared["launch_steps"]
    todo_versions: list[dict[str, Any]] = shared["todo_versions"]
    total_launches: dict[str, int] = shared["total_launches"]
    max_subagents: int = shared["max_subagents"]
    current_parent_index: ContextVar[int | None] = shared["current_parent_index"]
    is_main = agent_name == "main"

    @before_model(can_jump_to=["end"])
    def check_budget(state: Any, runtime: Any) -> dict[str, Any] | None:
        try:
            budget.check()
        except BudgetExceeded as stop:
            if is_main:
                cap_hit["reason"] = stop.reason
                visited.append("cap")
                emit("main", kind="decide", text=stop.reason)
            else:
                sub_cap_hit[agent_name] = stop.reason
            return {"jump_to": "end"}
        if not is_main and launch_cap is not None and launch_steps.get(agent_name, 0) >= launch_cap:
            sub_cap_hit[agent_name] = (
                f"Stopped on its own step cap: {launch_steps[agent_name]} of {launch_cap} "
                "steps used for this launch."
            )
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
        if not is_main:
            launch_steps[agent_name] = launch_steps.get(agent_name, 0) + 1
        visited.append("agent" if is_main else "sub")
        think_text = _message_text(message) if message is not None else ""
        emit(
            agent_name,
            kind="think",
            text=think_text or "(no reasoning text with this call)",
            tokens=tokens,
            latency_ms=latency_ms,
        )
        return response

    @wrap_tool_call
    def run_tool(request: Any, handler: Callable[[Any], Any]) -> Any:
        name = request.tool_call["name"]
        args = dict(request.tool_call.get("args") or {})
        call_id = request.tool_call["id"]
        call_started = time.monotonic()

        if name == "write_todos":
            todos = args.get("todos", [])
            result = handler(request)
            budget.charge(tool_calls=1)
            todo_versions.append({"version": len(todo_versions) + 1, "todos": todos})
            visited.append("todos")
            emit(agent_name, kind="decide", text=_todo_summary(todos), tool=name, args={"todos": todos})
            return result

        if name == "task":
            subagent_type = args.get("subagent_type", "?")
            description = args.get("description", "")

            if subagent_type not in ALLOWED_SUBAGENTS:
                budget.charge(tool_calls=1)
                error = tool_error(
                    "refused",
                    f"'{subagent_type}' is not a sub-agent this demo dispatches.",
                    f"Use one of: {', '.join(ALLOWED_SUBAGENTS)}.",
                )
                emit(agent_name, kind="observe", text=error, tool=name, args=args)
                return ToolMessage(content=error, name=name, tool_call_id=call_id)

            if total_launches["count"] >= max_subagents:
                budget.charge(tool_calls=1)
                error = tool_error(
                    "refused",
                    f"Sub-agent cap reached: {total_launches['count']} of {max_subagents} "
                    "launches already used this run.",
                    "Work the remaining items yourself with the file tools, or mark them "
                    "abandoned with a reason.",
                )
                emit(agent_name, kind="observe", text=error, tool=name, args=args)
                return ToolMessage(content=error, name=name, tool_call_id=call_id)

            total_launches["count"] += 1
            launch_steps[subagent_type] = 0
            handoff_step = emit(
                agent_name,
                kind="handoff",
                text=f"Delegating to {subagent_type}: {description}",
                tool=name,
                args=args,
            )
            visited.append("sub")
            token = current_parent_index.set(handoff_step.index)
            try:
                result = handler(request)
            finally:
                current_parent_index.reset(token)
            budget.charge(tool_calls=1)
            latency_ms = (time.monotonic() - call_started) * 1000
            observation = _extract_tool_message_text(result)
            stop_note = sub_cap_hit.pop(subagent_type, None)
            if stop_note:
                observation = f"{observation}\n\n({stop_note})".strip()
            emit(agent_name, kind="observe", text=observation, tool=name, args=args, latency_ms=latency_ms)
            return result

        # A file tool (ls, read_file, write_file, edit_file, glob, grep) or a
        # sub-agent's own work tool (search_corpus, describe_schema, run_sql).
        emit(agent_name, kind="act", text=f"Calling {name}.", tool=name, args=args)
        visited.append("vfs")
        result = handler(request)
        latency_ms = (time.monotonic() - call_started) * 1000
        budget.charge(tool_calls=1)
        observation = _extract_tool_message_text(result)
        emit(agent_name, kind="observe", text=observation, tool=name, args=args, latency_ms=latency_ms)
        return result

    return check_budget, charge_model, run_tool


# --- run() ---------------------------------------------------------------------


@observe(name="deep_agent.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`;
    `subagent_max_steps` (default from config) -- the step cap each researcher/
    analyst launch gets before it stops on its own cap, overridable from the UI."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    max_subagents = settings_obj.deep_agent_max_subagents
    raw_cap = settings.get("subagent_max_steps")
    launch_cap = int(raw_cap) if raw_cap is not None else settings_obj.subagent_max_steps

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["task"]

    def emit(agent: str, **fields: Any) -> Step:
        if "parent_index" not in fields:
            parent = current_parent_index.get()
            if parent is not None:
                fields["parent_index"] = parent
        step = agent_run.add_step(agent=agent, **fields)
        if on_step is not None:
            on_step(agent, step)
        return step

    current_parent_index: ContextVar[int | None] = ContextVar("deep_agent_parent_index", default=None)
    started = time.monotonic()
    budget.start()

    try:
        toolbox = Toolbox(run_id, names=["search_corpus", "describe_schema", "run_sql"])
        backend = FilesystemBackend(root_dir=toolbox.run_dir, virtual_mode=True, max_file_size_mb=1)

        primary, *fallbacks = agent_models()

        shared: dict[str, Any] = {
            "budget": budget,
            "emit": emit,
            "visited": visited,
            "cap_hit": {},
            "sub_cap_hit": {},
            "launch_steps": {},
            "todo_versions": [],
            "total_launches": {"count": 0},
            "max_subagents": max_subagents,
            "current_parent_index": current_parent_index,
        }

        researcher_spec = _subagent_spec(
            "researcher",
            "Searches the document corpus (HX-40 product brief, service bulletin, benchmark "
            "report, firmware changelog, procurement policy, on-call runbook, incident "
            "postmortem) and writes its findings to a file.",
            RESEARCHER_SYSTEM_PROMPT,
            toolbox.as_langchain_tools(names=list(RESEARCHER_TOOLS)),
            primary,
            fallbacks,
            backend,
            launch_cap,
            shared,
        )
        analyst_spec = _subagent_spec(
            "analyst",
            "Queries the sales database (a music store's customers, invoices, tracks and "
            "genres) and writes its findings to a file.",
            ANALYST_SYSTEM_PROMPT,
            toolbox.as_langchain_tools(names=list(ANALYST_TOOLS)),
            primary,
            fallbacks,
            backend,
            launch_cap,
            shared,
        )

        main_check, main_model, main_tool = make_middleware("main", launch_cap=None, shared=shared)

        @after_model(can_jump_to=["model"])
        def finish_guard(state: Any, runtime: Any) -> dict[str, Any] | None:
            todos = state.get("todos") or []
            open_items = open_todo_items(todos)
            if not open_items:
                return None
            messages = state.get("messages") or []
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and last.tool_calls:
                return None
            text = finish_guard_message(open_items)
            emit("guard", kind="decide", text=text)
            return {"messages": [HumanMessage(content=text)], "jump_to": "model"}

        main_middleware: list[Any] = []
        if fallbacks:
            main_middleware.append(ModelFallbackMiddleware(*fallbacks))
        main_middleware += [TodoListMiddleware(), main_check, main_model, main_tool, finish_guard]
        # Replaces the main agent's own auto-built FilesystemMiddleware by name,
        # and -- via the library's own name-based inheritance into the
        # auto-added general-purpose sub-agent's stack -- replaces its
        # FilesystemMiddleware too. `execute` is absent from both regardless
        # (see SAFE_FS_TOOLS above); the general-purpose sub-agent is also
        # never dispatched at all (see ALLOWED_SUBAGENTS in run_tool).
        main_middleware.append(FilesystemMiddleware(backend=backend, tools=SAFE_FS_TOOLS))

        agent_graph = create_deep_agent(
            primary,
            tools=None,
            system_prompt=MAIN_SYSTEM_PROMPT,
            middleware=main_middleware,
            subagents=[researcher_spec, analyst_spec],
            backend=backend,
            checkpointer=None,
            name=DEMO,
        )

        final_state = agent_graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"callbacks": get_callbacks(), "recursion_limit": 4 * settings_obj.deep_agent_max_steps + 10},
        )
        final_todos = final_state.get("todos") or []

        if "reason" in shared["cap_hit"]:
            agent_run.status = "stopped_on_budget"
            agent_run.stop_reason = shared["cap_hit"]["reason"]
        else:
            agent_run.status = "completed"

        # assemble(): one plain model call with no access to the agent's own
        # message history -- input is the task, the final todos and the run's
        # files, read fresh from disk.
        budget.check()
        body, files_read = build_assemble_input(task, final_todos, toolbox.run_dir)
        assemble_models = [primary, *fallbacks]
        head, *rest = assemble_models
        assemble_runnable = head.with_fallbacks(rest) if rest else head
        call_started = time.monotonic()
        response = assemble_runnable.invoke(
            [SystemMessage(content=ASSEMBLE_SYSTEM_PROMPT), HumanMessage(content=body)],
            config={"callbacks": get_callbacks()},
        )
        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(response)
        budget.charge(steps=1, tokens=tokens, llm_calls=1)
        visited.append("assemble")
        emit(
            "assemble",
            kind="think",
            text=f"Assembled the answer from: {', '.join(files_read) or '(no files)'}.",
            tokens=tokens,
            latency_ms=latency_ms,
            args={"files_read": files_read},
        )
        agent_run.output = response.content or ""

    except BudgetExceeded as stop:
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        # The main agent's before_model hook may already have recorded this same stop
        # (and then the check before assemble raises it again) -- one row, not two.
        if "reason" not in shared["cap_hit"]:
            visited.append("cap")
            emit("main", kind="decide", text=stop.reason)
    except Exception as exc:  # no provider configured, every provider refused, a build failure
        agent_run.status = "failed"
        agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        emit("main", kind="decide", text=agent_run.stop_reason)

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run


# --- clearing this demo's virtual file systems ----------------------------------


def run_ids_for_clear() -> set[str]:
    """Run ids this demo has recorded, read *before* `Clear my data` empties
    `deep_agent_runs` -- the page calls this at the top of the script, ahead of
    `clear_data_button()`, so the ids are still there when the button's own
    click handler (which runs `core.mongo.clear_demo_data` synchronously,
    inside the same rerun) fires."""
    from core.mongo import recent_runs

    return {r["_id"] for r in recent_runs(DEMO, limit=100_000)}


def clear_run_directories(run_ids: set[str]) -> int:
    """Deletes `data/vfs/<run_id>/` for each id in `run_ids` that is actually a
    directory directly under the vfs root -- never a glob-delete of
    `data/vfs/`, and never a directory this demo did not itself record."""
    import shutil

    from core.config import REPO_ROOT

    settings = get_settings()
    root = Path(settings.vfs_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()

    deleted = 0
    for run_id in run_ids:
        run_dir = (root / run_id).resolve()
        if run_dir.parent == root and run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
            deleted += 1
    return deleted
