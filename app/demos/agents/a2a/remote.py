"""The remote agents -- "Corpus researcher", "SQL analyst" and "Web researcher" --
and nothing that belongs to the client. This file is the file-boundary half of
"the agents share no memory and no Python objects": `agent.py` (the client /
coordinator) may import this module only to build the ASGI apps it talks to over
HTTP and to read the directory of hosts, and must never reach past that into any
remote's own state.

Each remote is described once, as a `RemoteSpec`, and built into its own
independent service: its own agent card, its own tools, its own prompt, its own
`Budget`, its own task store and its own Starlette app on its own host name. The
three share this file's code the way three services built from one template
would share a library -- they share no state. None of them knows the other two
exist; only the coordinator reads all three cards.

Each agent is a small hand-built LangGraph tool loop wrapped in an
`a2a.server.agent_execution.AgentExecutor`, the official SDK's seam for "run my
agent's logic and publish protocol events as it goes". Its `Budget` is built only
from its own config cap (`a2a_remote_max_steps`) plus the shared token/deadline
defaults -- never anything the client passes in, because the client has no
channel to pass it on: everything that crosses the boundary is a protocol message.

The A2A protocol types in this SDK version (1.1.5) are protobuf messages, not
the plain-JSON dataclasses older (0.x) examples use -- `a2a.server.apps` does not
exist in 1.x. The high-level constructors in `a2a.helpers` (`new_text_part`,
`new_data_part`, `new_task_from_user_message`, ...) build those messages without
having to touch protobuf directly, and `TaskUpdater` is the SDK's helper for
publishing `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent` in the right shape.
"""

import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from starlette.applications import Starlette

from a2a.helpers import get_message_text, new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.tools import Toolbox, is_error
from core.tracing import get_callbacks

# --- evidence: what each agent reports it relied on --------------------------------

# core.tools._search_corpus prefixes every hit "[<file> · <score>]".
_CORPUS_SOURCE_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
# core.tools._web_search prints each hit as "[i] title\n<url>\nbody".
_URL_RE = re.compile(r"^(https?://\S+)$", re.MULTILINE)


def _corpus_evidence(tool: str, args: dict[str, Any], output: str) -> list[str]:
    sources: list[str] = []
    for match in _CORPUS_SOURCE_RE.finditer(output):
        source = match.group(1).split("·")[0].strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def _web_evidence(tool: str, args: dict[str, Any], output: str) -> list[str]:
    return list(dict.fromkeys(_URL_RE.findall(output)))


def _sql_evidence(tool: str, args: dict[str, Any], output: str) -> list[str]:
    # The queries that actually returned rows are this agent's evidence -- a reader
    # can re-run them; a schema lookup or a refused statement proves nothing.
    if tool == "run_sql" and not is_error(output):
        return [" ".join(str(args.get("sql", "")).split())]
    return []


def _corpus_status(tool: str, args: dict[str, Any]) -> str:
    return f"searching the corpus for {args.get('query', '')!r}"


def _web_status(tool: str, args: dict[str, Any]) -> str:
    return f"searching the web for {args.get('query', '')!r}"


def _sql_status(tool: str, args: dict[str, Any]) -> str:
    if tool == "describe_schema":
        return "reading the database schema"
    return f"running SQL: {' '.join(str(args.get('sql', '')).split())}"


# --- the three specs ----------------------------------------------------------------


@dataclass(frozen=True)
class RemoteSpec:
    """One remote agent, described once: its card, its tools and how it narrates
    its own progress. `key` doubles as the agent label on the trajectory."""

    key: str
    name: str
    host: str
    description: str
    skill_id: str
    skill_name: str
    skill_description: str
    tags: tuple[str, ...]
    examples: tuple[str, ...]
    tools: tuple[str, ...]
    system_prompt: str
    status_text: Callable[[str, dict[str, Any]], str]
    evidence: Callable[[str, dict[str, Any], str], list[str]]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}"


CORPUS = RemoteSpec(
    key="corpus-researcher",
    name="Corpus researcher",
    host="corpus-researcher.a2a.local",
    description=(
        "Searches a bundled corpus of the organisation's own product, incident and policy "
        "documents and answers with the passages and source files it used. Has no "
        "database access and no web access."
    ),
    skill_id="corpus_research",
    skill_name="Corpus research",
    skill_description=(
        "Answers a question from the organisation's own documents -- the HX-40 product "
        "brief, field service bulletins, firmware changelog, incident post-mortems, the "
        "on-call runbook and the procurement policy -- citing the source file(s)."
    ),
    tags=("corpus", "documents", "internal", "policy", "product"),
    examples=(
        "What sustained throughput should a lab plan for with the HX-40?",
        "According to the on-call runbook, what is the thirty-minute rule?",
    ),
    tools=("search_corpus",),
    system_prompt=(
        "You answer questions from a bundled document corpus, using search_corpus. Call "
        "it with a focused query; if the first search does not find what the question "
        "needs, refine the query and call it again. Once you can answer, reply in full "
        "sentences with no further tool call, citing what the passages actually say. "
        "Never guess at anything the corpus does not cover -- say plainly that it is not "
        "there."
    ),
    status_text=_corpus_status,
    evidence=_corpus_evidence,
)

SQL = RemoteSpec(
    key="sql-analyst",
    name="SQL analyst",
    host="sql-analyst.a2a.local",
    description=(
        "Answers questions about the Chinook music-store sales database (customers, "
        "invoices, tracks, genres, artists, employees) by writing read-only SQL. Has no "
        "access to documents and no web access."
    ),
    skill_id="sales_data_analysis",
    skill_name="Sales data analysis",
    skill_description=(
        "Counts, sums and ranks over the sales database -- invoices, invoice lines, "
        "customers, tracks, genres, artists -- and reports the SQL the answer rests on."
    ),
    tags=("sql", "database", "sales", "analytics"),
    examples=(
        "How many invoices are in the sales database?",
        "Which billing country has the highest total sales?",
    ),
    tools=("describe_schema", "run_sql"),
    system_prompt=(
        "You answer questions about a read-only SQLite sales database. Call "
        "describe_schema before your first query -- table and column names are not "
        "guessable. Write one SELECT per call and aggregate in SQL. If a query errors, "
        "read the error and fix it once. Once you can answer, reply in full sentences "
        "with no further tool call, naming the query the answer rests on."
    ),
    status_text=_sql_status,
    evidence=_sql_evidence,
)

WEB = RemoteSpec(
    key="web-researcher",
    name="Web researcher",
    host="web-researcher.a2a.local",
    description=(
        "Searches the public web for current or external information and answers with "
        "the URLs it used. Has no access to the organisation's own documents and no "
        "database access."
    ),
    skill_id="web_research",
    skill_name="Public web research",
    skill_description=(
        "Answers a question from public web sources -- general guidance, standards, "
        "vendor-neutral practice, current events -- citing the URLs. Knows nothing about "
        "the organisation's own products or policies."
    ),
    tags=("web", "public", "external", "guidance"),
    examples=(
        "What does public guidance say about room temperature for lab analysers?",
        "What are general best practices for single-source procurement risk?",
    ),
    tools=("web_search",),
    system_prompt=(
        "You answer questions from the public web, using web_search. Search with a "
        "focused query; if the results are off-topic, rephrase once -- two searches at "
        "most, then answer from what you have. Web results are "
        "snippets, not whole pages -- report what they say, attribute each point to "
        "its URL, and say plainly when the results do not settle the question."
    ),
    status_text=_web_status,
    evidence=_web_evidence,
)

REMOTES: dict[str, RemoteSpec] = {spec.key: spec for spec in (CORPUS, SQL, WEB)}

# The only thing the coordinator is given up front: where to look. What each host
# can actually do it has to learn from the card it fetches there -- the same
# starting point a client has against a real registry of agent URLs.
DIRECTORY: dict[str, str] = {spec.key: spec.base_url for spec in REMOTES.values()}


def build_card(spec: RemoteSpec) -> AgentCard:
    """The public contract: exactly one skill per agent, and a description that says
    what it can't do as well as what it can -- so choosing between agents is something
    the *client* does from the cards, not something a remote refuses at run time."""
    return AgentCard(
        name=spec.name,
        description=spec.description,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id=spec.skill_id,
                name=spec.skill_name,
                description=spec.skill_description,
                tags=list(spec.tags),
                examples=list(spec.examples),
            )
        ],
        supported_interfaces=[
            AgentInterface(
                url=f"{spec.base_url}/",
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
    )


# --- one remote's own tool loop, one hand-built StateGraph per task -----------------


class _RemoteState(TypedDict):
    # add_messages appends each node's returned messages rather than replacing the
    # list -- without it the tool node's ToolMessages would overwrite the AIMessage
    # that asked for them, and the provider rejects an orphaned tool result.
    messages: Annotated[list[Any], add_messages]
    pending_calls: list[dict[str, Any]]
    evidence: list[str]
    answer: str
    decision: str


async def _run_remote_graph(
    spec: RemoteSpec,
    query: str,
    run_id: str,
    budget: Budget,
    updater: TaskUpdater,
    fail_after_first_step: bool,
) -> tuple[str, list[str]]:
    """The hand-built `act`/`tools` loop -- the same skeleton every other demo in
    this codebase uses, running on this agent's own `Budget` and publishing a
    `working` status (through `updater`) before every tool call, so the client sees
    real progress rather than one opaque wait.

    `graph.ainvoke()` rather than `graph.invoke()` in a thread: this coroutine
    already runs on the event loop the ASGI app is served on (same process, same
    loop, in this in-process demo), so awaiting `updater.start_work()` directly
    from inside a node keeps status updates and tool calls in the true order.
    """
    toolbox = Toolbox(run_id, names=list(spec.tools))
    tools = toolbox.as_langchain_tools()
    primary, *fallbacks = agent_models()

    # Bind before wrapping in fallbacks -- see AGENTIC_IMPLEMENTATION_PLAN.md,
    # "Model access". Copied from the other hand-built StateGraph demos, not
    # imported (CLAUDE.md rule 1: no cross-demo imports).
    def _with_fallbacks(bound: list[Any]) -> Any:
        head, *rest = bound
        return head.with_fallbacks(rest) if rest else head

    act_runnable = _with_fallbacks([m.bind_tools(tools) for m in [primary, *fallbacks]])
    # The last step is kept for an answer: the same model with no tools bound, so an
    # agent that keeps searching still reports what it found instead of failing on
    # its own step cap with nothing to show for the calls it made.
    answer_runnable = _with_fallbacks([primary, *fallbacks])
    steps_run = {"n": 0}

    async def act_node(state: _RemoteState) -> dict[str, Any]:
        budget.check()
        last_step = budget.steps_used >= budget.max_steps - 1
        messages = [SystemMessage(content=spec.system_prompt), *state["messages"]]
        if last_step:
            messages.append(
                HumanMessage(
                    content="This is your last step: answer now from what you have found, "
                    "and say plainly what you could not establish."
                )
            )
        runnable = answer_runnable if last_step else act_runnable
        response = await runnable.ainvoke(messages, config={"callbacks": get_callbacks()})
        budget.charge(steps=1, tokens=count_tokens(response), llm_calls=1)

        tool_calls = response.tool_calls or []
        if not tool_calls:
            return {"messages": [response], "answer": response.content or "", "decision": "respond"}
        pending = [{"id": c["id"], "name": c["name"], "args": c.get("args", {})} for c in tool_calls]
        return {"messages": [response], "pending_calls": pending, "decision": "tools"}

    async def tools_node(state: _RemoteState) -> dict[str, Any]:
        tool_messages: list[Any] = []
        evidence = list(state["evidence"])
        for call in state["pending_calls"]:
            budget.check()
            await updater.start_work(
                message=updater.new_agent_message([new_text_part(spec.status_text(call["name"], call["args"]))])
            )
            steps_run["n"] += 1
            if fail_after_first_step and steps_run["n"] == 1:
                raise RuntimeError(f"{spec.name} fails mid-task (forced by the sidebar toggle).")
            result = toolbox.call(call["name"], call["args"])
            budget.charge(tool_calls=1)
            for item in spec.evidence(call["name"], call["args"], result.output):
                if item not in evidence:
                    evidence.append(item)
            tool_messages.append(ToolMessage(content=result.output, tool_call_id=call["id"]))
        return {"messages": tool_messages, "pending_calls": [], "evidence": evidence, "decision": "act"}

    graph = StateGraph(_RemoteState)
    graph.add_node("act", act_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", lambda s: s["decision"], {"tools": "tools", "respond": END})
    graph.add_edge("tools", "act")
    compiled = graph.compile()

    initial_state: _RemoteState = {
        "messages": [HumanMessage(content=query)],
        "pending_calls": [],
        "evidence": [],
        "answer": "",
        "decision": "",
    }
    final_state = await compiled.ainvoke(
        initial_state, config={"callbacks": get_callbacks(), "recursion_limit": 4 * budget.max_steps + 10}
    )
    return final_state.get("answer", ""), final_state.get("evidence", [])


# --- the AgentExecutor: the SDK's seam into one agent's logic -----------------------


class RemoteExecutor(AgentExecutor):
    """Runs one remote's graph for one task and publishes its progress over
    `TaskUpdater` -- `submitted` -> one `working` update per tool call ->
    `completed` with the artefact, or `failed` with a plain reason (a spent budget
    names its own cap; any other exception is reported by type and message).
    Nothing here is visible to the client except what these calls put on the wire.
    """

    def __init__(self, spec: RemoteSpec, *, fail_mid_task: bool = False) -> None:
        self.spec = spec
        # Set once per app build, from the "Fails mid-task" sidebar setting -- makes
        # the failure path reproducible rather than hoping a task triggers one.
        self.fail_mid_task = fail_mid_task

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            return
        task = new_task_from_user_message(message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.submit()

        settings = get_settings()
        # This agent's own Budget: its own step cap, plus the shared token/deadline
        # defaults -- never anything carved from the client's Budget, because the
        # client has no channel to pass it one. See the module docstring.
        budget = Budget(
            max_steps=settings.a2a_remote_max_steps,
            max_tokens=settings.agent_max_tokens,
            deadline_s=settings.agent_deadline_s,
        ).start()

        try:
            answer, evidence = await _run_remote_graph(
                self.spec, get_message_text(message), task.id, budget, updater, self.fail_mid_task
            )
        except BudgetExceeded as stop:
            await updater.failed(
                message=updater.new_agent_message(
                    [new_text_part(f"Stopped on the {stop.cap.replace('_', ' ')}: {stop.reason}")]
                )
            )
            return
        except Exception as exc:  # a genuine bug, or the forced mid-task failure
            await updater.failed(
                message=updater.new_agent_message(
                    [new_text_part(f"The remote agent failed: {type(exc).__name__}: {exc}")]
                )
            )
            return

        # Its own spend, as metadata on the artefact -- read by the client and
        # shown, never charged to the client's own Budget (see the README).
        await updater.add_artifact(
            [new_text_part(answer), new_data_part({"evidence": evidence})],
            name="answer",
            metadata={
                "tokens": budget.tokens_used,
                "llm_calls": budget.llm_calls,
                "tool_calls": budget.tool_calls,
                "steps_used": budget.steps_used,
            },
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("This demo does not support task cancellation.")


def build_app(key: str, *, fail_mid_task: bool = False) -> Starlette:
    """One remote's ASGI app: agent-card routes + JSON-RPC routes over its own
    `InMemoryTaskStore`. Built fresh per client `run()` call, so every task starts
    on a fresh Budget -- what independently deployed services would give you anyway.
    """
    spec = REMOTES[key]
    card = build_card(spec)
    handler = DefaultRequestHandler(
        agent_executor=RemoteExecutor(spec, fail_mid_task=fail_mid_task),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")
    return Starlette(routes=routes)
