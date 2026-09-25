"""Deep research: split a question into sub-questions, research each one with its
own fresh agent, take notes that carry their source, then synthesise a brief whose
every claim cites one of those notes -- checked, not just asked for.

The pipeline is plain Python inside `run()`, not a `StateGraph`: `decompose` and
`synthesise` are single structured-output calls, and `verify` is pure Python with
no model call at all. `create_agent` is used only for `research` -- the one node
that is genuinely an agentic loop, searching, reading and deciding what to keep,
one sub-question at a time. That is the plan's framework choice for this step, and
it means the research loop looks exactly like Step 2's `tool_design` middleware
(`before_model` for the budget check and jump-to-end, `wrap_model_call` to charge
and count tokens, `wrap_tool_call` to charge and time each tool) rather than a
graph node's `budget.check()`, the way `plan_execute`'s hand-built graph does it.

The thing this demo actually has to prove -- "every claim links to a step" -- lives
in three small pieces, not in the model:

- A **source registry**, built as the run goes: every observation from
  `search_corpus` or `web_search` is parsed for the source keys it returned (a
  corpus file name, or a web result's URL), each remembered against the row it
  first appeared on.
- **`record_note`**, a demo-local tool the researcher calls to keep a fact. It
  refuses any `source` that is not already in the registry -- a model cannot cite
  a source it never actually saw a search return, in this run or an earlier
  sub-question's.
- **`verify`**, which runs after synthesis with no model call: a claim's note ids
  are checked against the notes that actually exist, an unknown id is dropped and
  the claim is flagged (shown, never hidden), and the distinct sources behind
  every surviving citation are counted.

Because every one of those three pieces reads only `core.tools`' own observation
strings and the notes list, they are also the pieces this demo's LLM-free
verification exercises directly -- see the bottom of this file's docstring in the
brief for the exact list, and `page.py`'s panels reconstruct the sub-questions,
notes and sources from `AgentRun.steps` afterwards using the same functions,
rather than the run storing them anywhere new.
"""

import re
import time
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox, is_error, tool_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "research"
NAME = "Deep research"
SENTENCE = "Break the question up, search per part, take notes, then synthesise a brief with citations."
SHAPE = "decompose → search ×n → synthesise"

# Node ids double as the `visited` entries the graph map highlights.
GRAPH = """flowchart LR
    question[Question] --> decompose[Split into sub-questions]
    decompose --> research[Researcher searches and takes notes]
    research -->|next sub-question| research
    research --> synthesise[Brief written from notes, with citations]
    synthesise --> verify[Citations checked against notes]
    research -.->|cap spent| cap[Stopped]
"""

SEARCH_TOOL_NAMES = ("search_corpus", "web_search")

# Every fact named below was checked against samples/corpus/ -- see this step's
# report for the exact numbers and which files carry them.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "Why did the HX-40 underperform its rated throughput in the field, what "
            "was done about it, and what should a lab plan for?"
        ),
    },
    {
        "task": (
            "What is the HX-40 thermal derating issue, and what does current public "
            "guidance say about managing heat for benchtop lab analysers?"
        ),
    },
    {
        "task": (
            "What does our procurement policy require for single-sourced instruments, "
            "and what are the general best practices for single-source procurement "
            "risk?"
        ),
    },
]


# --- structured-output schemas ---------------------------------------------------


class SubQuestions(BaseModel):
    """The decomposer's structured reply."""

    questions: list[str] = Field(
        description=(
            "Independent sub-questions that together cover the research question, "
            "each answerable on its own by searching a corpus or the web."
        )
    )
    why: str = Field(description="One or two sentences on why these sub-questions cover it.")


class Claim(BaseModel):
    """One claim in the synthesised brief, tied to the note(s) it rests on."""

    text: str = Field(description="One factual claim, in a full sentence.")
    note_ids: list[int] = Field(description="The id(s) of the note(s) this claim rests on.")


class Brief(BaseModel):
    """The writer's structured reply."""

    summary: str = Field(description="A short opening summary of the findings, two to four sentences.")
    claims: list[Claim] = Field(description="Each claim, citing the note id(s) it rests on.")


# --- prompts -----------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = (
    "You are the decomposer. Given a research question, split it into a short list "
    "of independent sub-questions that together cover it -- each one answerable on "
    "its own by searching a document corpus or the web. Write at most {max_subq} "
    "sub-questions. Say briefly why they cover the question."
)

RESEARCH_SYSTEM_PROMPT = (
    "You are the researcher for one sub-question of a larger research task. Call "
    "search_corpus or web_search to find evidence -- one tool call per turn. Start "
    "with search_corpus: the corpus holds the organisation's own documents, and "
    "product names, incidents and policies in the question usually live there. Use "
    "web_search for public, external context the corpus cannot hold. As soon as an "
    "observation contains a fact that answers part of the sub-question, call "
    "record_note for it before searching again -- a fact you saw but did not record "
    "is lost. Pass top_k=3 to search_corpus -- long result lists cost budget. Two or "
    "three good notes are enough; then stop. Prefer a source the notes so far do not "
    "already cite: a claim two documents agree on is stronger than one document "
    "cited twice. Read "
    "each observation before deciding what to do next: if it is an error or has "
    "nothing useful, rephrase the query or switch tool (search_corpus <-> "
    "web_search) rather than repeating the same call, and never give up on the "
    "sub-question silently. When you find a fact worth keeping, call "
    "record_note(claim, source, quote) -- source must be exactly a source key shown "
    "in an observation you actually received, never invented or guessed. When you "
    "have enough notes to answer this sub-question, reply with a short sentence and "
    "no tool call. If you have genuinely tried and nothing can be found, say that "
    "plainly in your reply instead of calling record_note with a guess."
)

SYNTHESISE_SYSTEM_PROMPT = (
    "You are the writer. You are given the overall research question and a list of "
    "notes, each with an id, its source and a claim already extracted from that "
    "source. Write a short brief: a summary, then a list of specific claims, each "
    "citing the note id(s) it rests on. Use only the notes given -- never introduce "
    "a claim with no note behind it, and never invent a note id."
)


# --- keeping budget back for the brief ------------------------------------------

# Research is open-ended; the brief is the deliverable. Research stops starting new
# model calls once this share of the token budget -- or the last step -- is all that
# is left, so `synthesise` always has room to run. What was skipped becomes a gap.
BRIEF_TOKEN_RESERVE = 0.25

_RESERVE_REASON = (
    "Sub-question {qi}: research stopped to keep budget for the brief "
    "(a quarter of the token budget, or the last step, is held back for synthesise)."
)


def _brief_reserve_reached(budget: Budget) -> bool:
    tokens_left = budget.max_tokens - budget.tokens_used
    steps_left = budget.max_steps - budget.steps_used
    return tokens_left <= BRIEF_TOKEN_RESERVE * budget.max_tokens or steps_left <= 1


# --- source parsing ------------------------------------------------------------

# core.tools._search_corpus joins passages as "[<file> · <score>]<text>"; the score
# is always a fixed-point float ("0.842"), which is what keeps this from matching
# "ERROR[bad_argument]: ..." -- that bracket has no " · " inside it.
_CORPUS_SOURCE_RE = re.compile(r"\[([^\[\]]+) · [0-9]+\.[0-9]+\]")


def _parse_corpus_sources(observation: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip(), m.group(1).strip()) for m in _CORPUS_SOURCE_RE.finditer(observation)]


def _parse_web_sources(observation: str) -> list[tuple[str, str]]:
    # core.tools._web_search joins hits as "[i] title\nurl\nbody", blocks separated
    # by a blank line -- the key is the URL on the second line of each block.
    pairs: list[tuple[str, str]] = []
    for block in observation.split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        head = re.match(r"^\[\d+\]\s*(.*)$", lines[0])
        if not head:
            continue
        url = lines[1].strip()
        if url:
            pairs.append((url, head.group(1).strip() or url))
    return pairs


def parse_sources(tool: str, observation: str) -> list[tuple[str, str]]:
    """Source keys an observation actually returned, as `(key, title)` pairs, deduped
    by first occurrence. `tool` is the name of the tool that produced `observation`
    (`search_corpus` keys by file name, `web_search` by URL); any other tool, an
    error observation, or an empty-result observation yields nothing.

    This is the one place a "source" is ever recognised in this demo -- both the
    run-time registry (below) and the page's post-hoc reconstruction of it call this
    same function against the same `core.tools` output shapes, so there is exactly
    one parser to get right.
    """
    if is_error(observation):
        return []
    if tool == "search_corpus":
        pairs = _parse_corpus_sources(observation)
    elif tool == "web_search":
        pairs = _parse_web_sources(observation)
    else:
        return []
    seen: dict[str, str] = {}
    for key, title in pairs:
        seen.setdefault(key, title)
    return list(seen.items())


# --- the record_note tool -------------------------------------------------------


class RecordNoteArgs(BaseModel):
    claim: str = Field(description="One factual claim this note supports, in a full sentence.")
    source: str = Field(
        description=(
            "The exact source key this claim comes from -- a corpus file name or a "
            "web URL, copied exactly from an observation you actually received in "
            "this run. Never invent one."
        )
    )
    quote: str = Field(description="A short quote or close paraphrase from that source backing the claim.")


def _make_record_note_tool(
    *, qi: int, registry: dict[str, dict], notes: list[dict], next_step_index: Callable[[], int]
) -> StructuredTool:
    """One `record_note` tool. `registry` and `notes` are the run's own -- shared by
    reference across every sub-question's researcher, so a later sub-question can
    cite a source an earlier one turned up, and every note lands in one list in the
    order it was recorded.

    The guard that makes "every claim links to a step" true starts here: a `source`
    not already in `registry` (i.e. not something a search in this run actually
    returned) is refused with a hint naming what would be accepted, in the same
    `ERROR[kind]: ... Hint: ...` shape every other tool in this lab uses.
    """

    def _record_note(claim: str, source: str, quote: str) -> str:
        entry = registry.get(source)
        if entry is None:
            known = ", ".join(sorted(registry)) or "none yet -- search before citing"
            return tool_error(
                "bad_argument",
                f"'{source}' was not returned by any search in this run.",
                f"cite one of: {known}",
            )
        note = {
            "id": len(notes) + 1,
            "subquestion": qi,
            "claim": claim,
            "source": source,
            "quote": quote,
            "step_index": entry["first_step_index"],
            "note_step_index": next_step_index(),
        }
        notes.append(note)
        return f"Recorded note {note['id']}, citing '{source}'."

    return StructuredTool.from_function(
        func=_record_note,
        name="record_note",
        description=(
            "Record one claim you found, with the exact source key it came from and "
            "a short supporting quote. The source must be one already returned by "
            "search_corpus or web_search in this run -- citing anything else is "
            "refused."
        ),
        args_schema=RecordNoteArgs,
    )


# --- verify (no LLM) -------------------------------------------------------------


def verify(claims: list[Claim], notes: list[dict]) -> tuple[list[dict], int]:
    """Checks every claim's note ids against the notes that actually exist. No model
    call: this is the step that makes "every claim links to a step" a checked fact
    rather than a hope. An id the writer invented is dropped from that claim and the
    claim is flagged `unsupported` (rendered, never hidden) rather than dropped
    outright -- a claim that cites nothing real is still something the reader should
    see and can then discount.

    Returns `(verified_claims, distinct_source_count)`, where the count is of
    distinct note sources across every note id actually cited by a surviving
    citation -- the number the page checks against the "at least three" target.
    """
    valid_ids = {n["id"] for n in notes}
    note_by_id = {n["id"]: n for n in notes}
    verified: list[dict] = []
    cited_sources: set[str] = set()
    for claim in claims:
        kept_ids = [nid for nid in claim.note_ids if nid in valid_ids]
        dropped_ids = [nid for nid in claim.note_ids if nid not in valid_ids]
        for nid in kept_ids:
            cited_sources.add(note_by_id[nid]["source"])
        verified.append(
            {
                "text": claim.text,
                "note_ids": kept_ids,
                "dropped_ids": dropped_ids,
                "unsupported": len(kept_ids) == 0,
            }
        )
    return verified, len(cited_sources)


# --- recoveries (no LLM) ---------------------------------------------------------


def recoveries(steps: list[Step]) -> int:
    """Counts a search error observation that is immediately followed, within the
    same sub-question (`step.agent`), by another search observation that succeeded
    -- a search that failed and was then retried into a working one. Steps from
    other kinds (record_note, decompose, synthesise, verify) never count, and an
    error with no later search in its sub-question -- the kind that becomes a gap
    -- never counts either.
    """
    by_subquestion: dict[str, list[Step]] = {}
    for step in steps:
        if step.kind == "observe" and step.tool in SEARCH_TOOL_NAMES:
            by_subquestion.setdefault(step.agent, []).append(step)
    count = 0
    for group in by_subquestion.values():
        for prev, nxt in zip(group, group[1:]):
            if is_error(prev.text) and not is_error(nxt.text):
                count += 1
    return count


# --- rendering the brief as markdown --------------------------------------------


def render_brief(
    summary: str, verified_claims: list[dict], notes: list[dict], registry: dict[str, dict]
) -> str:
    """`AgentRun.output`: the brief as markdown, numbered claims with their citation
    markers, and the sources they actually cite appended below -- self-contained,
    so it reads on its own even without the page's richer panels."""
    note_by_id = {n["id"]: n for n in notes}
    lines = [summary.strip()]
    if not verified_claims:
        lines += ["", "_No claims could be written -- see the sub-question gaps above._"]
    else:
        lines.append("")
        for i, claim in enumerate(verified_claims, start=1):
            marker = "".join(f"[{nid}]" for nid in claim["note_ids"])
            suffix = " _(unsupported — cites no known note)_" if claim["unsupported"] else ""
            lines.append(f"{i}. {claim['text']} {marker}{suffix}".rstrip())

    cited_sources = sorted(
        {note_by_id[nid]["source"] for c in verified_claims for nid in c["note_ids"] if nid in note_by_id}
    )
    if cited_sources:
        lines += ["", "**Sources**"]
        for key in cited_sources:
            entry = registry.get(key, {})
            title = entry.get("title", key)
            step_idx = entry.get("first_step_index")
            step_note = f" → step {step_idx:02d}" if step_idx else ""
            lines.append(f"- [{key}] {title}{step_note}")
    return "\n".join(lines)


# --- the researcher (one sub-question, one create_agent run) --------------------


def _extract_ai_message(response: Any) -> AIMessage | None:
    """`wrap_model_call`'s handler returns a `ModelResponse` in this LangChain
    version, documented as `ModelResponse | AIMessage` -- so both are handled."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _research_subquestion(
    *,
    qi: int,
    subq: str,
    task: str,
    notes: list[dict],
    registry: dict[str, dict],
    toolbox: Toolbox,
    primary: Any,
    fallbacks: list[Any],
    budget: Budget,
    max_steps_per_q: int,
    inject_fault: bool,
    fault_used: dict[str, bool],
    agent_run: AgentRun,
    visited: list[str],
    emit: Callable[..., Step],
) -> str:
    """Runs one sub-question's researcher to a final reply or a spent cap.

    Two caps apply, and they mean different things: the *shared* run budget
    stopping mid-sub-question ends the whole run (raised here as `BudgetExceeded`,
    same as every other demo's shared cap), while this sub-question's own step cap
    (`max_steps_per_q`) stopping it only ends this sub-question -- the run moves on
    to the next one with a `decide` row saying so. Returns `"completed"` or
    `"cap_subq"`.
    """
    agent_label = f"q{qi}"
    notes_context = (
        "\n".join(f"[{n['id']}] ({n['source']}) {n['claim']}" for n in notes) or "(none yet)"
    )
    record_note_tool = _make_record_note_tool(
        qi=qi, registry=registry, notes=notes, next_step_index=lambda: len(agent_run.steps) + 1
    )
    search_tools = toolbox.as_langchain_tools(names=list(SEARCH_TOOL_NAMES))
    tools = [*search_tools, record_note_tool]

    steps_this_subq = {"value": 0}
    cap_hit: dict[str, str] = {}
    model_meta: dict[str, Any] = {}
    tool_latency: dict[str, float] = {}
    tool_call_args: dict[str, dict[str, Any]] = {}

    @before_model(can_jump_to=["end"])
    def check_budget(state: Any, runtime: Any) -> dict[str, Any] | None:
        try:
            budget.check()
        except BudgetExceeded as stop:
            cap_hit["kind"] = "shared"
            cap_hit["cap"] = stop.cap
            cap_hit["reason"] = stop.reason
            return {"jump_to": "end"}
        if _brief_reserve_reached(budget):
            cap_hit["kind"] = "reserve"
            cap_hit["reason"] = _RESERVE_REASON.format(qi=qi)
            return {"jump_to": "end"}
        if steps_this_subq["value"] >= max_steps_per_q:
            cap_hit["kind"] = "subq"
            cap_hit["reason"] = (
                f"Sub-question {qi} step cap reached: {steps_this_subq['value']} of "
                f"{max_steps_per_q} steps used for this sub-question; moving on."
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
        steps_this_subq["value"] += 1
        model_meta["tokens"] = tokens
        model_meta["latency_ms"] = latency_ms
        return response

    @wrap_tool_call
    def run_tool(request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call_id = request.tool_call["id"]
        name = request.tool_call["name"]
        args = dict(request.tool_call.get("args") or {})
        tool_call_args[tool_call_id] = args
        call_started = time.monotonic()

        # "Break the first search": the first search call of either kind, once per
        # run -- not per sub-question. Either kind, because the researcher is told to
        # search the corpus first and may never reach for web_search at all.
        divert = inject_fault and name in SEARCH_TOOL_NAMES and not fault_used["value"]
        if divert:
            fault_used["value"] = True
            result = toolbox.call("boom", {})
            message = ToolMessage(content=result.output, name=name, tool_call_id=tool_call_id)
        else:
            message = handler(request)
        tool_latency[tool_call_id] = (time.monotonic() - call_started) * 1000
        budget.charge(tool_calls=1)

        if name in SEARCH_TOOL_NAMES:
            # Predicts the index this call's own observe row will get: this handler
            # runs before that row is emitted (below, once the "tools" node chunk
            # arrives), and the system prompt asks for one tool call per turn, so
            # nothing else can land between this call and its observation.
            predicted_index = len(agent_run.steps) + 1
            for key, title in parse_sources(name, message.content):
                registry.setdefault(key, {"title": title, "first_step_index": predicted_index})
        elif name == "record_note" and not is_error(message.content) and notes:
            # The note was appended synchronously inside record_note's own call
            # above; folding it into the observe row's args is what lets the page
            # reconstruct the notes table straight from `AgentRun.steps`.
            tool_call_args[tool_call_id] = {**args, **notes[-1]}

        return message

    middleware: list[Any] = []
    if fallbacks:
        middleware.append(ModelFallbackMiddleware(*fallbacks))
    middleware += [check_budget, charge_model, run_tool]

    compiled_agent = create_agent(
        primary, tools=tools, system_prompt=RESEARCH_SYSTEM_PROMPT, middleware=middleware
    )

    human_content = (
        f"Overall research question: {task}\n\nYour sub-question ({qi}): {subq}\n\n"
        "Notes recorded so far, from any earlier sub-question (context only -- do "
        f"not repeat a search this already covers unless you need more):\n{notes_context}"
    )

    stream = compiled_agent.stream(
        {"messages": [{"role": "user", "content": human_content}]},
        stream_mode="updates",
        config={"callbacks": get_callbacks()},
    )

    notes_before = len(notes)
    outcome = "completed"
    for chunk in stream:
        for node, update in chunk.items():
            if node.endswith("before_model"):
                if isinstance(update, dict) and update.get("jump_to") == "end":
                    if cap_hit.get("kind") == "shared":
                        outcome = "cap_shared"
                    else:
                        outcome = "cap_subq"
                        visited.append("research")
                        emit(
                            agent_label,
                            kind="decide",
                            text=cap_hit.get("reason", "Sub-question step cap reached."),
                        )
                continue

            if node == "model":
                message = update["messages"][-1]
                tokens = model_meta.get("tokens", 0)
                latency_ms = model_meta.get("latency_ms", 0.0)
                visited.append("research")
                if message.tool_calls:
                    emit(
                        agent_label,
                        kind="think",
                        text=message.content or "(no reasoning text with this call)",
                        tokens=tokens,
                        latency_ms=latency_ms,
                    )
                    for call in message.tool_calls:
                        emit(
                            agent_label,
                            kind="act",
                            text=f"Calling {call['name']}.",
                            tool=call["name"],
                            args=call.get("args", {}),
                        )
                else:
                    emit(
                        agent_label,
                        kind="think",
                        text=message.content or "(final reply carried no reasoning text)",
                        tokens=tokens,
                        latency_ms=latency_ms,
                    )
                    emit(
                        agent_label,
                        kind="decide",
                        text=f"Done with sub-question {qi}: no further tool call.",
                    )
                continue

            if node == "tools":
                visited.append("research")
                for message in update["messages"]:
                    emit(
                        agent_label,
                        kind="observe",
                        text=message.content,
                        tool=message.name,
                        args=tool_call_args.get(message.tool_call_id),
                        latency_ms=tool_latency.get(message.tool_call_id, 0.0),
                    )
                continue

    if outcome == "cap_shared":
        raise BudgetExceeded(
            cap_hit.get("cap", "max_steps"), cap_hit.get("reason", "Stopped on a budget cap.")
        )

    if len(notes) == notes_before:
        emit(
            agent_label,
            kind="decide",
            text=f"Gap: no notes were recorded for sub-question {qi} ({subq!r}).",
        )

    return outcome


# --- reconstructing the panels from AgentRun.steps -------------------------------
#
# Nothing above stores sub-questions, notes or the source registry anywhere but the
# trajectory itself -- `page.py`'s panels rebuild all of it from `run.steps` after
# the fact, the same way `plan_execute.plan_diff()` rebuilds plan versions from its
# `decide` rows. That keeps `AgentRun` the one shape every demo (and Step 16's
# evaluation) can rely on, with nothing demo-specific bolted onto it.


def subquestions_from_steps(run: AgentRun) -> list[dict[str, Any]]:
    """Each sub-question with its note count and answered/gap status, in the order
    `decompose` produced them."""
    decompose_step = next(
        (s for s in run.steps if s.agent == "decompose" and s.args and "questions" in s.args), None
    )
    if decompose_step is None:
        return []
    counts: dict[int, int] = {}
    for note in notes_from_steps(run):
        counts[note["subquestion"]] = counts.get(note["subquestion"], 0) + 1
    return [
        {
            "index": i,
            "text": question,
            "note_count": counts.get(i, 0),
            "status": "answered" if counts.get(i, 0) else "gap",
        }
        for i, question in enumerate(decompose_step.args["questions"], start=1)
    ]


def notes_from_steps(run: AgentRun) -> list[dict[str, Any]]:
    """Every note actually recorded, in id order -- read back from the `record_note`
    observe rows' `args`, which carry the full note (see `_research_subquestion`)."""
    return [
        step.args
        for step in run.steps
        if step.kind == "observe"
        and step.tool == "record_note"
        and step.args
        and "id" in step.args
        and not is_error(step.text)
    ]


def sources_from_steps(run: AgentRun) -> dict[str, dict[str, Any]]:
    """The source registry, rebuilt by running the same `parse_sources()` the run
    used against every search observation still on the track, first occurrence
    wins -- so this is always in agreement with what `record_note` was allowed to
    cite at the time, not a fresh guess at it."""
    registry: dict[str, dict[str, Any]] = {}
    for step in run.steps:
        if step.kind == "observe" and step.tool in SEARCH_TOOL_NAMES:
            for key, title in parse_sources(step.tool, step.text):
                registry.setdefault(key, {"title": title, "first_step_index": step.index})
    return registry


def brief_from_steps(run: AgentRun) -> tuple[str, list[dict[str, Any]], int]:
    """`(summary, verified_claims, distinct_sources)` from the `synthesise` and
    `verify` decide rows -- the verified claims are what `verify()` actually
    decided, not recomputed here."""
    synth_step = next(
        (s for s in run.steps if s.agent == "synthesise" and s.args and "summary" in s.args), None
    )
    verify_step = next(
        (s for s in run.steps if s.agent == "verify" and s.args and "claims" in s.args), None
    )
    summary = synth_step.args["summary"] if synth_step else ""
    if verify_step is None:
        return summary, [], 0
    return summary, verify_step.args["claims"], verify_step.args.get("distinct_sources", 0)


def citations_table(verified_claims: list[dict[str, Any]], notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per claim-note pair: claim #, note #, source, the observe step the
    source first came from, and the step the note itself was recorded on -- the
    "Citations → steps" link from a claim back to the track, since the track's own
    rows carry no anchors."""
    note_by_id = {n["id"]: n for n in notes}
    rows: list[dict[str, Any]] = []
    for ci, claim in enumerate(verified_claims, start=1):
        if not claim["note_ids"]:
            rows.append(
                {
                    "claim #": ci,
                    "note #": None,
                    "source": None,
                    "observe step #": None,
                    "note step #": None,
                }
            )
            continue
        for nid in claim["note_ids"]:
            note = note_by_id.get(nid)
            if note is None:
                continue
            rows.append(
                {
                    "claim #": ci,
                    "note #": nid,
                    "source": note["source"],
                    "observe step #": note["step_index"],
                    "note step #": note["note_step_index"],
                }
            )
    return rows


def _retry_parsed(runnable: Any, messages: list[Any], parsing_error: Any) -> tuple[Any, Any]:
    """One retry on a structured-output parse failure: ask again, plainly, without
    replaying the unparseable reply."""
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


@observe(name="research.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """The `AgentDemo` contract. Settings, all optional: `on_step`, `break_first_search`."""
    settings_obj = get_settings()
    on_step: Callable[[str, Step], None] | None = settings.get("on_step")
    inject_fault = bool(settings.get("break_first_search"))
    max_subq = settings_obj.research_max_subquestions
    max_steps_per_q = settings_obj.research_max_steps_per_question

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["question"]

    def emit(agent_label: str, **fields: Any) -> Step:
        step = agent_run.add_step(agent=agent_label, **fields)
        if on_step is not None:
            on_step(agent_label, step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        primary, *fallbacks = agent_models()

        def _with_fallbacks(bound: list[Any]) -> Any:
            head, *rest = bound
            return head.with_fallbacks(rest) if rest else head

        # --- decompose -----------------------------------------------------------

        budget.check()
        visited.append("decompose")
        decompose_models = [
            m.with_structured_output(SubQuestions, include_raw=True) for m in [primary, *fallbacks]
        ]
        decompose_runnable = _with_fallbacks(decompose_models)
        decompose_messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT.format(max_subq=max_subq)),
            HumanMessage(content=task),
        ]
        call_started = time.monotonic()
        result = decompose_runnable.invoke(decompose_messages, config={"callbacks": get_callbacks()})
        parsed, raw = result["parsed"], result["raw"]
        if parsed is None:
            parsed, raw = _retry_parsed(decompose_runnable, decompose_messages, result.get("parsing_error"))
        if parsed is None:
            raise RuntimeError("The decomposer could not produce valid sub-questions after one retry.")
        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(raw)
        budget.charge(steps=1, tokens=tokens, llm_calls=1)

        all_questions = parsed.questions
        truncated = len(all_questions) > max_subq
        subquestions = all_questions[:max_subq]
        text = f"Split into {len(subquestions)} sub-question(s): {parsed.why}"
        if truncated:
            text += f" (truncated from {len(all_questions)} to the {max_subq}-question cap)"
        emit(
            "decompose",
            kind="decide",
            text=text,
            args={"questions": subquestions, "why": parsed.why, "truncated": truncated},
            tokens=tokens,
            latency_ms=latency_ms,
        )

        # --- research, one sub-question at a time ---------------------------------

        toolbox_names = [*SEARCH_TOOL_NAMES, "boom"] if inject_fault else list(SEARCH_TOOL_NAMES)
        toolbox = Toolbox(run_id, names=toolbox_names)
        registry: dict[str, dict] = {}
        notes: list[dict] = []
        fault_used = {"value": False}

        for qi, subq in enumerate(subquestions, start=1):
            budget.check()
            if _brief_reserve_reached(budget):
                visited.append("research")
                emit(f"q{qi}", kind="decide", text=_RESERVE_REASON.format(qi=qi))
                emit(
                    f"q{qi}",
                    kind="decide",
                    text=f"Gap: sub-question {qi} ({subq!r}) was not researched -- skipped to keep budget for the brief.",
                )
                continue
            _research_subquestion(
                qi=qi,
                subq=subq,
                task=task,
                notes=notes,
                registry=registry,
                toolbox=toolbox,
                primary=primary,
                fallbacks=fallbacks,
                budget=budget,
                max_steps_per_q=max_steps_per_q,
                inject_fault=inject_fault,
                fault_used=fault_used,
                agent_run=agent_run,
                visited=visited,
                emit=emit,
            )

        # --- synthesise ------------------------------------------------------------

        budget.check()
        visited.append("synthesise")
        synth_models = [m.with_structured_output(Brief, include_raw=True) for m in [primary, *fallbacks]]
        synth_runnable = _with_fallbacks(synth_models)
        notes_text = (
            "\n".join(
                f"[{n['id']}] source: {n['source']} — {n['claim']} (\"{n['quote']}\")" for n in notes
            )
            or "(no notes were recorded -- every sub-question was a gap.)"
        )
        synth_messages = [
            SystemMessage(content=SYNTHESISE_SYSTEM_PROMPT),
            HumanMessage(content=f"Research question: {task}\n\nNotes:\n{notes_text}\n\nWrite the brief now."),
        ]
        call_started = time.monotonic()
        result = synth_runnable.invoke(synth_messages, config={"callbacks": get_callbacks()})
        parsed_brief, raw = result["parsed"], result["raw"]
        if parsed_brief is None:
            parsed_brief, raw = _retry_parsed(synth_runnable, synth_messages, result.get("parsing_error"))
        if parsed_brief is None:
            raise RuntimeError("The writer could not produce a valid brief after one retry.")
        latency_ms = (time.monotonic() - call_started) * 1000
        tokens = count_tokens(raw)
        budget.charge(steps=1, tokens=tokens, llm_calls=1)
        emit(
            "synthesise",
            kind="decide",
            text=f"Synthesised a brief with {len(parsed_brief.claims)} claim(s) from {len(notes)} note(s).",
            args={"summary": parsed_brief.summary, "claims": [c.model_dump() for c in parsed_brief.claims]},
            tokens=tokens,
            latency_ms=latency_ms,
        )

        # --- verify (no LLM) ---------------------------------------------------------

        visited.append("verify")
        verified_claims, distinct_sources = verify(parsed_brief.claims, notes)
        unsupported = sum(1 for c in verified_claims if c["unsupported"])
        emit(
            "verify",
            kind="decide",
            text=(
                f"Verified: {distinct_sources} distinct source(s) across cited notes "
                f"(target ≥3); {unsupported} of {len(verified_claims)} claim(s) flagged unsupported."
            ),
            args={"claims": verified_claims, "distinct_sources": distinct_sources},
        )

        agent_run.output = render_brief(parsed_brief.summary, verified_claims, notes, registry)
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
