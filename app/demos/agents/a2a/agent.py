"""Agent-to-agent (A2A): a client agent -- the "coordinator" -- and three remote
agents it has never seen the code of ("Corpus researcher", "SQL analyst", "Web
researcher"), cooperating over the official A2A protocol (`a2a-sdk` 1.1.5) rather
than by sharing memory or Python objects. `remote.py` holds the remotes end to
end; this file holds only the client and the protocol plumbing to reach them.
Nothing here imports a remote's internals -- only `remote.DIRECTORY` (which hosts
to look at) and `remote.build_app()` (the ASGI app answering on each host), which
this file talks to exactly the way it would talk to three real services.

**In-process transport.** `HostRouterTransport` sends each request to the ASGI
app for its host name, through `httpx.ASGITransport` -- no socket, no port, no
second process. `A2ACardResolver` and the SDK's own `ClientFactory` neither know
nor care that there's no real network underneath. A real deployment puts each
agent on its own host; this demo's "network" is one Python call stack, which is
the one way it differs from a real deployment (see the README).

**The client's own graph.** A hand-built `StateGraph`, `discover -> choose ->
delegate -> respond`: `discover` fetches every card in the directory (no LLM
call); `choose` is one structured-output call that decides, from the cards alone,
which agents to delegate which part of the task to -- none, one or several;
`delegate` sends each message in turn and streams its status/artefact events
until that task reaches a terminal state or the client's own deadline runs out;
`respond` writes the final answer from whatever came back (artefacts, declines,
failures). Delegations run one after another, not concurrently, so each remote's
status rows stay together under its own handoff row on the track.

**Async in a synchronous contract.** `run()` stays synchronous, as
`core.types.AgentDemo` requires, and wraps the whole client graph in
`asyncio.run(...)` -- the Streamlit script thread has no running event loop of
its own for the SDK's async client to attach to.

**The raw protocol log is the subject of this page.** `RecordingTransport`
wraps the router and records every exchange -- method, URL, JSON body, and every
SSE `data:` frame as it streams -- without altering what the SDK actually sees.
An early version of this class drained an SSE response eagerly with
`aiter_lines()` and handed the client a reconstructed, fully-buffered
`httpx.Response`; that silently dropped the final frame (the terminal status
event) because the naive rebuild didn't reproduce SSE framing exactly. The
version below passes bytes through untouched -- recording is a side effect of
iteration, never a replacement for it -- which is why it subclasses
`httpx.AsyncByteStream` rather than duck-typing one: httpx asserts the type
before it will stream a response.
"""

import asyncio
import time
from typing import Any, Callable, TypedDict

import httpx
from google.protobuf.json_format import MessageToDict
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.helpers import get_data_parts, get_text_parts, new_text_message
from a2a.types.a2a_pb2 import AgentCard, Role, SendMessageRequest, TaskState
from a2a.utils.constants import TransportProtocol

from core.budget import Budget, BudgetExceeded
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import tool_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

# Absolute, not relative: Streamlit runs a page as a script (from subagents' and
# swarm's own house note), and this module is also imported directly by scripts.
from demos.agents.a2a import remote

DEMO = "a2a"
NAME = "Agent-to-agent (A2A)"
SENTENCE = "Agents that share no memory cooperate over the protocol: cards, tasks, status, artefacts."
SHAPE = "discover → task → status ↻ → artefact"

# The agent labels the trajectory uses for each remote, in directory order.
REMOTE_KEYS: tuple[str, ...] = tuple(remote.DIRECTORY)
REMOTE_NAMES: dict[str, str] = {key: remote.REMOTES[key].name for key in REMOTE_KEYS}

# The graph map's node id for each remote -- pushed onto `visited` when that agent
# is actually sent a task, so the map shows which of the three a run used.
REMOTE_NODES: dict[str, str] = {"corpus-researcher": "corpus", "sql-analyst": "sql", "web-researcher": "web"}

# Node ids double as the `visited` entries the graph map highlights. `delegate` is
# visited once per delegation; "cap" only on a spent budget (drawn from `respond`
# alone to keep the map readable -- any node can hit a cap). The dotted edge is the
# card fetches; the solid two-way edges are A2A tasks -- the message out, status
# updates and the artefact back.
GRAPH = """flowchart LR
    task[Task] --> discover[Discover: fetch every card in the directory]
    discover --> choose[Choose: which agents, from the cards alone]
    choose -->|one or more| delegate[Delegate: one A2A task per agent]
    choose -->|none fit| respond[Respond: combine the artefacts, or explain]
    delegate --> respond
    subgraph remotes [Remote agents -- each on its own host]
        corpus[Corpus researcher: search_corpus]
        sql[SQL analyst: describe_schema, run_sql]
        web[Web researcher: web_search]
    end
    discover -.->|GET card| remotes
    delegate <-->|task / status / artefact| corpus
    delegate <-->|task / status / artefact| sql
    delegate <-->|task / status / artefact| web
    respond -.->|cap spent| cap[Stopped]
"""

# Facts verified against samples/: FSB-114 gives 840 samples/hour sustained vs. the
# 1,200 rated; chinook.db has 374 Metal tracks, 412 invoices, and USA is the top
# billing country at 523.06. The web halves have no fixed answer -- the point is
# watching the coordinator route that part to the only card that lists web access.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "How many tracks are in the Metal genre in the sales database, and is that "
            "more or fewer than the HX-40's sustained throughput in samples per hour?"
        ),
        "shows": "corpus + SQL",
    },
    {
        "task": (
            "What sustained throughput should a lab plan for with the HX-40 according to "
            "our documents, and what does public guidance say about room temperature for "
            "benchtop lab analysers?"
        ),
        "shows": "corpus + web",
    },
    {
        "task": (
            "Prepare three facts for a planning memo: the HX-40's sustained throughput "
            "from our field service bulletin, the top billing country by total sales in "
            "the sales database, and what public sources recommend as lab room temperature."
        ),
        "shows": "all three",
    },
    {"task": "How many invoices are in the sales database?", "shows": "SQL only"},
]

CHOOSE_SYSTEM_PROMPT = (
    "You are the coordinator agent. Remote agents have published their capabilities as "
    "agent cards, reproduced below verbatim. Split the task into parts and delegate each "
    "part to the one agent whose card covers it -- one delegation per agent at most, and "
    "only to agents listed as reachable. Use only what the cards actually say: never "
    "assume a capability a card doesn't advertise, and never delegate a part no card "
    "covers. A card fits a part only if its skill can actually *do* that part: if the "
    "task asks for an action no card can perform, delegate nothing for it -- do not "
    "send research about the task instead. Each message must be self-contained -- the "
    "remote agent sees nothing but that message. If no card fits any part of the task, "
    "delegate nothing and say why in terms of the cards."
)

RESPOND_SYSTEM_PROMPT = (
    "Answer the task in full sentences, using only what is given below. Never invent a "
    "fact a remote agent did not report, never claim a delegation happened if it didn't, "
    "and say plainly which part of the task could not be answered and why."
)


class Delegation(BaseModel):
    agent: str = Field(description="The exact name of the agent, as its card gives it.")
    skill_id: str = Field(description="The card's skill id to invoke.")
    message: str = Field(description="The exact, self-contained message to send that agent.")
    reason: str = Field(description="Why this agent, referencing what its card lists.")


class ChooseDecision(BaseModel):
    delegations: list[Delegation] = Field(
        default_factory=list, description="Zero or more delegations, one per agent at most."
    )
    reason: str = Field(description="The overall routing decision, in terms of the cards.")


# --- transports: the recorder (the subject of the protocol panel) and the "network" --


class _RecordingByteStream(httpx.AsyncByteStream):
    """Passes SSE bytes through untouched while recording each `data:` frame as it
    arrives. Must subclass httpx.AsyncByteStream -- httpx asserts the type before
    streaming a response, and a plain duck-typed class fails that assertion. See
    the module docstring for why this passes bytes through rather than buffering
    and replaying them."""

    def __init__(self, inner: Any, entry: dict[str, Any]) -> None:
        self._inner = inner.__aiter__()
        self._entry = entry
        self._buffer = b""

    def __aiter__(self) -> "_RecordingByteStream":
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._inner.__anext__()
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace")
            if text.startswith("data:"):
                self._entry["frames"].append(text[len("data:") :].strip())
        return chunk

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


class RecordingTransport(httpx.AsyncBaseTransport):
    """Wraps another transport and records every exchange -- method, URL, JSON
    request body, and every SSE frame -- without altering what the SDK sees. The
    log is what the page's protocol panel renders, via `protocol_log()` below."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.log: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.log.append(
            {
                "direction": "request",
                "method": request.method,
                "path": str(request.url),
                "body": request.content.decode("utf-8", errors="replace") if request.content else None,
            }
        )
        try:
            response = await self.inner.handle_async_request(request)
        except Exception as exc:
            # An offline host's ConnectError, or any other transport failure --
            # recorded so the protocol panel shows the failed request, never a
            # traceback (ground rule 10).
            self.log.append({"direction": "response", "kind": "error", "error": f"{type(exc).__name__}: {exc}"})
            raise

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            sse_entry: dict[str, Any] = {"direction": "response", "kind": "sse", "frames": []}
            self.log.append(sse_entry)
            stream = _RecordingByteStream(response.stream, sse_entry)
            return httpx.Response(response.status_code, headers=response.headers, stream=stream, request=request)

        await response.aread()
        self.log.append(
            {
                "direction": "response",
                "kind": "json",
                "status": response.status_code,
                "body": response.content.decode("utf-8", errors="replace"),
            }
        )
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content, request=request
        )


class OfflineTransport(httpx.AsyncBaseTransport):
    """An offline host: every request fails at the transport layer, exactly like a
    service that's down -- not a mocked response."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"{request.url.host} is offline (toggled off in the sidebar).", request=request)


class HostRouterTransport(httpx.AsyncBaseTransport):
    """The demo's stand-in for DNS: routes each request to the transport for its
    host name. An unknown host fails the same way an unresolvable one would."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]) -> None:
        self.routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        inner = self.routes.get(request.url.host)
        if inner is None:
            raise httpx.ConnectError(f"No host named {request.url.host}.", request=request)
        return await inner.handle_async_request(request)


# --- graph state ---------------------------------------------------------------


class _CoordinatorState(TypedDict):
    task: str
    cards: dict[str, AgentCard]  # remote key -> card, for the reachable ones
    unreachable: dict[str, str]  # remote key -> why the card could not be fetched
    decision: ChooseDecision | None
    results: list[dict[str, Any]]  # one per delegation actually sent
    answer: str
    route: str


def _cards_text(cards: dict[str, AgentCard], unreachable: dict[str, str]) -> str:
    blocks = []
    for card in cards.values():
        skills = "\n".join(
            f"  - skill {s.id}: {s.name} -- {s.description} (tags: {', '.join(s.tags) or 'none'})"
            for s in card.skills
        ) or "  (this card lists no skills)"
        blocks.append(f"Agent: {card.name} (reachable)\n  {card.description}\n{skills}")
    for key in unreachable:
        blocks.append(f"Agent at {remote.DIRECTORY[key]}: unreachable -- its card could not be fetched.")
    return "\n\n".join(blocks) or "(no agents found)"


def _respond_prompt(state: _CoordinatorState) -> str:
    decision = state["decision"]
    parts = [f"Task: {state['task']}"]
    if state["unreachable"]:
        names = ", ".join(remote.DIRECTORY[k] for k in state["unreachable"])
        parts.append(f"These agents could not be reached, so their cards were never read: {names}.")
    if decision is not None:
        parts.append(f"Your routing decision: {decision.reason}")
    if not state["results"]:
        parts.append(
            "Nothing was delegated. Answer the task if you can from your own knowledge; "
            "otherwise say plainly that it needs a capability none of the reachable agents has."
        )
    for result in state["results"]:
        if result["failed"]:
            parts.append(f"{result['agent']} was asked: {result['message']}\nIt did not complete: {result['error']}")
        else:
            evidence = "; ".join(result["evidence"]) or "(none reported)"
            parts.append(
                f"{result['agent']} was asked: {result['message']}\nIt reported:\n{result['text']}"
                f"\nEvidence it cited: {evidence}"
            )
    parts.append("Write the final answer from these results.")
    return "\n\n".join(parts)


@observe(name="a2a.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`; `offline_agents`
    (default none) -- remote keys whose host fails every request, so their card is
    never read; `failing_agent` (default None) -- one remote key whose executor
    raises after its first tool call, which it turns into its own `failed` status
    rather than a transport-level crash (see remote.py)."""
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    offline_agents = {k for k in (settings.get("offline_agents") or []) if k in remote.REMOTES}
    failing_agent = settings.get("failing_agent")

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["task"]
    recorder: RecordingTransport | None = None

    def emit(agent: str, **fields: Any) -> Step:
        step = agent_run.add_step(agent=agent, **fields)
        if on_step is not None:
            on_step(agent, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        primary, *fallbacks = agent_models()

        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        choose_models = [m.with_structured_output(ChooseDecision, include_raw=True) for m in [primary, *fallbacks]]
        choose_runnable = _with_fallbacks(choose_models)
        respond_runnable = _with_fallbacks([primary, *fallbacks])

        routes: dict[str, httpx.AsyncBaseTransport] = {}
        for key, spec in remote.REMOTES.items():
            if key in offline_agents:
                routes[spec.host] = OfflineTransport()
            else:
                app = remote.build_app(key, fail_mid_task=(key == failing_agent))
                routes[spec.host] = httpx.ASGITransport(app=app)
        recorder = RecordingTransport(HostRouterTransport(routes))

        # --- nodes (async: the SDK client and httpx are both async-only) -------

        async def discover_node(state: _CoordinatorState) -> dict[str, Any]:
            visited.append("discover")
            cards: dict[str, AgentCard] = {}
            unreachable: dict[str, str] = {}
            async with httpx.AsyncClient(transport=recorder) as httpx_client:
                for key, base_url in remote.DIRECTORY.items():
                    resolver = A2ACardResolver(httpx_client, base_url)
                    try:
                        card = await resolver.get_agent_card()
                    except Exception as exc:
                        text = tool_error(
                            "unavailable",
                            f"{base_url} could not be reached ({type(exc).__name__}: {exc}).",
                            "Route that part of the task elsewhere, or say it could not be answered.",
                        )
                        emit("coordinator", kind="observe", text=text, tool="discover", args={"host": base_url})
                        unreachable[key] = text
                        continue
                    cards[key] = card
                    emit(
                        "coordinator",
                        kind="observe",
                        text=f"Card from {base_url}: {card.name} -- skill(s): {', '.join(s.id for s in card.skills) or '(none)'}",
                        tool="discover",
                        args=MessageToDict(card),
                    )
            return {"cards": cards, "unreachable": unreachable}

        async def choose_node(state: _CoordinatorState) -> dict[str, Any]:
            visited.append("choose")
            if not state["cards"]:
                decision = ChooseDecision(reason="No agent's card could be fetched, so there is nothing to delegate to.")
                emit("coordinator", kind="decide", text=decision.reason, args=decision.model_dump())
                return {"decision": decision, "route": "respond"}

            budget.check()
            messages = [
                HumanMessage(
                    content=(
                        f"{CHOOSE_SYSTEM_PROMPT}\n\nAgent cards:\n\n"
                        f"{_cards_text(state['cards'], state['unreachable'])}\n\nTask: {state['task']}"
                    )
                )
            ]
            call_started = time.monotonic()
            result = await choose_runnable.ainvoke(messages, config={"callbacks": get_callbacks()})
            parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                retry_messages = [
                    *messages,
                    HumanMessage(
                        content=(
                            f"Your last reply could not be parsed: {result.get('parsing_error')}. Reply "
                            "again with only the required JSON, matching the schema exactly."
                        )
                    ),
                ]
                result = await choose_runnable.ainvoke(retry_messages, config={"callbacks": get_callbacks()})
                parsed, raw = result["parsed"], result["raw"]
            if parsed is None:
                raise RuntimeError("The coordinator could not produce a valid decision after one retry.")

            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(raw)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)
            routed = ", ".join(d.agent for d in parsed.delegations) or "nobody"
            emit(
                "coordinator",
                kind="decide",
                text=f"Route to {routed}. {parsed.reason}",
                tokens=tokens,
                latency_ms=latency_ms,
                args=parsed.model_dump(),
            )
            return {"decision": parsed, "route": "delegate" if parsed.delegations else "respond"}

        async def _delegate_one(key: str, card: AgentCard, delegation: Delegation) -> dict[str, Any]:
            """One task to one remote: send, stream its status and artefact, and
            return what came back. Never raises -- a timeout or a transport error
            becomes a failed result the respond step has to explain."""
            visited.extend(["delegate", REMOTE_NODES[key]])
            handoff_step = emit(
                "coordinator",
                kind="handoff",
                text=f"Delegating to {card.name} ({delegation.skill_id}): {delegation.message}",
                tool=delegation.skill_id or None,
                args={
                    "agent": card.name,
                    "skill": delegation.skill_id,
                    "message": delegation.message,
                    "reason": delegation.reason,
                },
            )
            outcome: dict[str, Any] = {
                "key": key,
                "agent": card.name,
                "message": delegation.message,
                "text": "",
                "evidence": [],
                "spend": {},
                "failed": False,
                "error": "",
            }

            factory = ClientFactory(
                ClientConfig(
                    httpx_client=httpx.AsyncClient(transport=recorder),
                    streaming=True,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            )
            client = factory.create(card)
            request = SendMessageRequest(message=new_text_message(delegation.message, role=Role.ROLE_USER))
            remaining_s = max(budget.deadline_s - budget.elapsed_s, 1.0)

            async def _consume() -> None:
                async for event in client.send_message(request):
                    if event.HasField("status_update"):
                        state_name = TaskState.Name(event.status_update.status.state)
                        status_text = ""
                        if event.status_update.status.HasField("message"):
                            status_text = " ".join(get_text_parts(event.status_update.status.message.parts))
                        text = state_name.removeprefix("TASK_STATE_").replace("_", " ").capitalize()
                        if status_text:
                            text += f": {status_text}"
                        emit(key, kind="observe", text=text, parent_index=handoff_step.index, args={"state": state_name})
                        if state_name == "TASK_STATE_FAILED":
                            outcome["failed"] = True
                            outcome["error"] = status_text or "The remote task reported failure with no message."
                    elif event.HasField("artifact_update"):
                        artifact = event.artifact_update.artifact
                        outcome["text"] = "\n".join(get_text_parts(artifact.parts))
                        for data in get_data_parts(artifact.parts):
                            for item in data.get("evidence", []):
                                if item not in outcome["evidence"]:
                                    outcome["evidence"].append(item)
                        if artifact.HasField("metadata"):
                            outcome["spend"] = MessageToDict(artifact.metadata)
                        emit(
                            key,
                            kind="decide",
                            text=outcome["text"],
                            tokens=int(outcome["spend"].get("tokens", 0)),
                            parent_index=handoff_step.index,
                            args={"evidence": outcome["evidence"], "remote_spend": outcome["spend"]},
                        )

            try:
                await asyncio.wait_for(_consume(), timeout=remaining_s)
            except asyncio.TimeoutError:
                text = tool_error(
                    "timeout",
                    f"No terminal status arrived within the {remaining_s:.0f}s left on the client's deadline.",
                    "Answer from whatever the remote already reported.",
                )
                emit(key, kind="observe", text=text, parent_index=handoff_step.index)
                outcome.update(failed=True, error=text)
            except Exception as exc:
                text = tool_error(
                    "failed", f"The delegated call raised {type(exc).__name__}: {exc}.", "Answer from what is known."
                )
                emit(key, kind="observe", text=text, parent_index=handoff_step.index)
                outcome.update(failed=True, error=text)
            finally:
                await client.close()

            budget.charge(tool_calls=1)
            return outcome

        async def delegate_node(state: _CoordinatorState) -> dict[str, Any]:
            decision = state["decision"]
            assert decision is not None
            by_name = {card.name.casefold(): key for key, card in state["cards"].items()}
            results: list[dict[str, Any]] = []
            used: set[str] = set()
            for delegation in decision.delegations:
                budget.check()
                key = by_name.get(delegation.agent.strip().casefold())
                if key is None or key in used:
                    # A name no reachable card carries, or a second task for the same
                    # agent: refused as an observation, the same shape as a bad tool call.
                    why = "was already given a task this run" if key in used else "is not a reachable agent"
                    emit(
                        "coordinator",
                        kind="observe",
                        text=tool_error(
                            "bad_argument",
                            f"'{delegation.agent}' {why}; that delegation was skipped.",
                            f"Reachable agents: {', '.join(c.name for c in state['cards'].values())}.",
                        ),
                    )
                    continue
                used.add(key)
                results.append(await _delegate_one(key, state["cards"][key], delegation))
            return {"results": results, "route": "respond"}

        async def respond_node(state: _CoordinatorState) -> dict[str, Any]:
            visited.append("respond")
            budget.check()
            messages = [HumanMessage(content=f"{RESPOND_SYSTEM_PROMPT}\n\n{_respond_prompt(state)}")]
            call_started = time.monotonic()
            response = await respond_runnable.ainvoke(messages, config={"callbacks": get_callbacks()})
            latency_ms = (time.monotonic() - call_started) * 1000
            tokens = count_tokens(response)
            budget.charge(steps=1, tokens=tokens, llm_calls=1)
            answer = response.content or ""
            emit("coordinator", kind="decide", text=answer, tokens=tokens, latency_ms=latency_ms)
            return {"answer": answer}

        # --- graph ---------------------------------------------------------------

        graph = StateGraph(_CoordinatorState)
        graph.add_node("discover", discover_node)
        graph.add_node("choose", choose_node)
        graph.add_node("delegate", delegate_node)
        graph.add_node("respond", respond_node)
        graph.add_edge(START, "discover")
        graph.add_edge("discover", "choose")
        graph.add_conditional_edges("choose", lambda s: s["route"], {"delegate": "delegate", "respond": "respond"})
        graph.add_edge("delegate", "respond")
        graph.add_edge("respond", END)
        compiled = graph.compile()

        initial_state: _CoordinatorState = {
            "task": task,
            "cards": {},
            "unreachable": {},
            "decision": None,
            "results": [],
            "answer": "",
            "route": "",
        }
        final_state = asyncio.run(
            compiled.ainvoke(initial_state, config={"callbacks": get_callbacks(), "recursion_limit": 10})
        )
        agent_run.output = final_state.get("answer", "")
        agent_run.status = "completed"

    except BudgetExceeded as stop:
        visited.append("cap")
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        emit("coordinator", kind="decide", text=agent_run.stop_reason)
    except Exception as exc:  # no provider configured, every provider refused, a build failure
        agent_run.status = "failed"
        agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        emit("coordinator", kind="decide", text=agent_run.stop_reason)

    # The raw protocol log: a row of its own on the track (ground rule 4 -- nothing
    # is hidden), recorded even when the run stopped early, and what protocol_log()
    # below pulls out for the page's dedicated panel.
    if recorder is not None:
        emit(
            "coordinator",
            kind="observe",
            text=f"{len(recorder.log)} protocol exchange(s) recorded.",
            tool="protocol_log",
            args={"entries": recorder.log},
        )

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run


# --- rebuilt from the track, like swarm.py's custody_table -------------------------


def status_chips(steps: list[Step]) -> dict[str, list[str]]:
    """Per remote, its status transitions in the order they arrived --
    `submitted → working → completed` -- rebuilt from that remote's own `observe`
    rows rather than stored separately."""
    chips: dict[str, list[str]] = {}
    for step in steps:
        if step.agent in REMOTE_NAMES and step.kind == "observe" and step.args and "state" in step.args:
            chips.setdefault(step.agent, []).append(
                step.args["state"].removeprefix("TASK_STATE_").replace("_", " ").lower()
            )
    return chips


def remote_spend(steps: list[Step]) -> dict[str, dict[str, Any]]:
    """Per remote, its own reported spend (tokens, llm_calls, tool_calls, steps_used)
    -- shown, never added to the client's own Budget."""
    spend: dict[str, dict[str, Any]] = {}
    for step in steps:
        if step.agent in REMOTE_NAMES and step.kind == "decide" and step.args and step.args.get("remote_spend"):
            spend[step.agent] = step.args["remote_spend"]
    return spend


def protocol_log(steps: list[Step]) -> list[dict[str, Any]]:
    """The raw exchange log RecordingTransport captured this run, for the page's
    protocol panel."""
    for step in steps:
        if step.tool == "protocol_log" and step.args:
            return step.args.get("entries", [])
    return []
