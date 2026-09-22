"""ReAct, written out by hand.

Every other demo in this lab hands the loop to a framework. This one does not,
because the loop is the subject: the "agent" here is a string that grows and a
`while` that reads it. Nothing in this folder imports LangChain (plan, Step 1) --
`core.llm` is the only thing that knows a chat model exists, and it hands back a
reply and a token count.

The four moving parts, all visible below:

1. `build_prompt()`  -- the tools documented as *text*, plus the reply format.
2. `parse_reply()`   -- a hand-written parser for `Thought:` / `Action:` /
                        `Action Input:` / `Final Answer:`.
3. `run()`           -- the loop: send the transcript, parse, call the tool,
                        append the observation, repeat.
4. The transcript    -- one string. The agent's entire "memory" for this run.

The things a framework would hide are therefore all here to be read: that a
malformed reply is just a parse failure you re-prompt on, that an observation is
string concatenation, and that the only thing stopping the loop is the budget.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from core.budget import Budget, BudgetExceeded
from core.llm import count_tokens, get_chat_model
from core.mongo import save_run
from core.tools import ToolSpec, Toolbox
from core.tracing import observe
from core.types import AgentRun, Step, new_run_id

DEMO = "react"
NAME = "ReAct"
SENTENCE = (
    "The loop itself, written by hand: think, act on a tool, read what came back, decide again."
)
SHAPE = "think → act → observe ↻"

# A deliberately small toolbox. Two of these (describe_schema, run_sql) are meant
# to be used one after the other, which is what makes a two-tool task natural
# rather than contrived; the file tools are left out because a prompt that
# documents seven tools buries the format instructions this demo is about.
TOOL_NAMES = ["search_corpus", "describe_schema", "run_sql", "web_search"]

# Node ids match the `visited` entries the run records, so the map lights up the
# route actually taken. Unstyled on purpose: it inherits the app theme.
GRAPH = """flowchart LR
    prompt[Prompt] --> think[Think]
    think --> parse[Parse the reply]
    parse -->|action| act[Act]
    act --> observe[Observe]
    observe --> think
    parse -->|unparseable| repair[Re-prompt]
    repair --> think
    parse -->|final answer| done[Answer]
    think -.->|cap spent| cap[Stopped]
"""

# The pill label is the task itself, so these stay short enough to read at a glance.
# One per thing worth watching: two tools in sequence, a tool error to recover from,
# and a task with no end condition, which the step cap has to stop.
PRESETS = [
    "Which five artists have the most tracks in the sales database?",
    "What happens to the HX-40 above 26 °C, and which firmware version addressed it?",
    "Check every HX-40 firmware release against the thermal bulletin until nothing is missing.",
]

# How many consecutive unparseable replies to re-prompt on before giving up. Two is
# enough to show the repair working without letting a confused model burn the budget
# on nothing but format corrections.
MAX_REPAIRS = 2

# Substituted for the model's first reply when the page's fault toggle is on. It is
# plausible prose with no Action and no Final Answer -- the single most common way a
# real ReAct reply fails to parse -- so the repair path can be checked on demand
# instead of waited for. The model is not called for this step; nothing is spent.
FAULT_REPLY = (
    "Sure! I'll start by searching the corpus for the thermal derating bulletin, then "
    "cross-check it against the firmware changelog and report back."
)


# --- 1. the prompt -------------------------------------------------------------


def build_prompt(task: str, toolbox: Toolbox) -> str:
    """The whole agent, as far as the model is concerned.

    `toolbox.describe()` writes the tools out as text -- name, arguments and
    description. A tool-calling API would send the same information as JSON schema
    in a separate field; here it is prose in the prompt, which is exactly what the
    original ReAct paper did and what every framework still does underneath.
    """
    return (
        "You are a ReAct agent. You answer a task by alternating between thinking and "
        "using one tool at a time.\n\n"
        "Tools available to you:\n"
        f"{toolbox.describe()}\n\n"
        "Reply with exactly one block, in this format:\n\n"
        "Thought: what you know so far and what you need next\n"
        "Action: the name of one tool, exactly as written above\n"
        "Action Input: a JSON object of arguments for that tool\n\n"
        "Stop after Action Input. The Observation is given to you -- never write one "
        "yourself, and never put two Actions in one block.\n\n"
        "When you have enough to answer, reply with this instead:\n\n"
        "Thought: why you can answer now\n"
        "Final Answer: the answer in full sentences, naming the sources or tables it "
        "rests on\n\n"
        "Rules:\n"
        '- Action Input is always a JSON object, e.g. {"query": "thermal derating", "top_k": 3}.\n'
        "- Use a tool rather than guessing.\n"
        "- If an Observation begins with ERROR[, read its hint and make a different call. "
        "Repeating the call that just failed will fail again.\n"
        "- The documents and the sales database are unrelated: search_corpus knows nothing "
        "about the database, and run_sql knows nothing about the documents.\n"
        "- Answer as soon as you can. Every step is spent from a fixed budget.\n\n"
        f"Task: {task}\n\n"
    )


# --- 2. the parser -------------------------------------------------------------

ParsedKind = Literal["action", "final", "malformed"]

# "Action Input" is listed before "Action" so the alternation prefers the longer
# label; `[ \t]*` rather than `\s*` so a label only ever matches at a line start.
_LABEL = re.compile(
    r"^[ \t]*(Thought|Action Input|Action|Final Answer|Observation)[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Parsed:
    """What one model reply turned out to be."""

    kind: ParsedKind
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    problem: str = ""  # what to tell the model, when it could not be parsed
    raw: str = ""


def _sections(text: str) -> dict[str, str]:
    """The labelled blocks of a reply, first occurrence of each label winning."""
    matches = list(_LABEL.finditer(text))
    found: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        found.setdefault(match.group(1).lower(), text[match.end() : end].strip())
    return found


def _parse_args(blob: str, spec: ToolSpec | None) -> tuple[dict[str, Any] | None, str]:
    """`Action Input` as a dict, or None and the reason it could not be read.

    Three fallbacks, in order of how often they actually come up. Being forgiving
    here is a deliberate choice: the tolerable near-misses are handled, and what
    is left over is a genuine format failure worth re-prompting on.
    """
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```[A-Za-z]*[ \t]*\n?", "", blob)
        blob = re.sub(r"```\s*$", "", blob).strip()

    if not blob:
        # describe_schema and list_files take no arguments, so an empty input is correct.
        if spec is None or not spec.args_schema.model_fields:
            return {}, ""
        return None, "Action Input was empty"

    try:
        loaded = json.loads(blob)
        if isinstance(loaded, dict):
            return loaded, ""
    except json.JSONDecodeError:
        pass

    # A JSON object wrapped in a sentence ("Action Input: here you go {...}").
    embedded = re.search(r"\{.*\}", blob, re.DOTALL)
    if embedded:
        try:
            loaded = json.loads(embedded.group(0))
            if isinstance(loaded, dict):
                return loaded, ""
        except json.JSONDecodeError:
            pass

    # A bare value for a single-argument tool. The schema says without ambiguity
    # which field it belongs to, so reading it is safe rather than a guess.
    if spec is not None:
        required = [n for n, f in spec.args_schema.model_fields.items() if f.is_required()]
        if len(required) == 1:
            return {required[0]: blob.strip().strip("\"'")}, ""

    return None, f"Action Input was not a JSON object: {blob[:80]!r}"


def parse_reply(text: str, toolbox: Toolbox) -> Parsed:
    """One reply -> an action, an answer, or a format failure to re-prompt on."""
    raw = text

    # Models often carry on and write their own Observation, hallucinating the tool
    # result. Everything from that point on is discarded rather than trusted.
    cut = re.search(r"^[ \t]*Observation[ \t]*:", text, re.IGNORECASE | re.MULTILINE)
    if cut:
        text = text[: cut.start()]

    found = _sections(text)
    thought = found.get("thought", "")

    if found.get("final answer"):
        return Parsed("final", thought=thought, answer=found["final answer"], raw=raw)

    if "action" not in found:
        return Parsed(
            "malformed",
            thought=thought,
            problem="there was no Action line and no Final Answer line",
            raw=raw,
        )

    # `Action: **run_sql**` and `Action: `run_sql`` are both common.
    action = found["action"]
    tool = action.splitlines()[0].strip().strip("`*\"' .") if action else ""
    if not tool:
        return Parsed(
            "malformed", thought=thought, problem="the Action line named no tool", raw=raw
        )

    # An unknown tool name is *not* a parse failure: it is handed to the toolbox,
    # which answers ERROR[unknown_tool] listing the real ones. Recovering from that
    # is the agent's job, and watching it do so is half the point of the demo.
    spec = toolbox.specs.get(tool)
    args, problem = _parse_args(found.get("action input", ""), spec)
    if args is None:
        return Parsed("malformed", thought=thought, tool=tool, problem=problem, raw=raw)

    return Parsed("action", thought=thought, tool=tool, args=args, raw=raw)


# --- 3. the loop ---------------------------------------------------------------


@observe(name="react.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """Think, act, observe, repeat -- until a Final Answer or a spent cap.

    Settings, all optional:
      `on_step`          callable(Step) -- called as each step lands, for streaming.
      `prompt_log`       list -- every prompt sent, appended in order.
      `reply_log`        list -- every raw model reply, appended in order.
      `inject_malformed` bool -- substitute an unparseable first reply (see FAULT_REPLY).
      `tools`            list[str] -- override the toolbox.
    """
    on_step: Callable[[Step], None] | None = settings.get("on_step")
    prompt_log: list[str] | None = settings.get("prompt_log")
    reply_log: list[str] | None = settings.get("reply_log")
    inject_malformed: bool = bool(settings.get("inject_malformed"))

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    toolbox = Toolbox(run_id, names=list(settings.get("tools") or TOOL_NAMES))
    model = get_chat_model()

    started = time.monotonic()
    budget.start()

    header = build_prompt(task, toolbox)
    transcript = ""  # everything the agent has thought, done and seen. The whole memory.
    visited = ["prompt"]
    repairs = 0
    iteration = 0

    def emit(**fields: Any) -> Step:
        step = agent_run.add_step(**fields)
        if on_step is not None:
            on_step(step)
        return step

    try:
        while True:
            # Ground rule 2: the check is inside the loop, before the work, so a run
            # that stops says which cap stopped it and keeps the trajectory so far.
            budget.check()
            visited.append("think")

            prompt = header + transcript
            if prompt_log is not None:
                prompt_log.append(prompt)

            # Counted on iterations rather than LLM calls: the injected reply makes no
            # call, so an llm_calls test would re-fire it on every pass round the loop.
            injected = inject_malformed and iteration == 0
            iteration += 1
            if injected:
                reply, tokens, latency_ms = FAULT_REPLY, 0, 0.0
                budget.charge(steps=1)
            else:
                call_started = time.monotonic()
                try:
                    message = model.invoke(prompt)
                except Exception as exc:  # every provider in the chain refused
                    agent_run.status = "failed"
                    agent_run.stop_reason = (
                        f"No provider answered: {type(exc).__name__}: {exc}"
                    )
                    emit(kind="decide", text=agent_run.stop_reason)
                    break
                latency_ms = (time.monotonic() - call_started) * 1000
                reply = message.text
                tokens = count_tokens(message)
                budget.charge(steps=1, tokens=tokens, llm_calls=1)

            if reply_log is not None:
                reply_log.append(reply)

            visited.append("parse")
            parsed = parse_reply(reply, toolbox)

            if parsed.kind == "malformed":
                repairs += 1
                visited.append("repair")
                emit(
                    kind="decide",
                    attempt=repairs + 1,
                    text=(
                        f"The reply could not be parsed: {parsed.problem}."
                        + (" (Injected fault: the model was not called.)" if injected else "")
                        + (
                            " Re-prompting with the format."
                            if repairs <= MAX_REPAIRS
                            else f" That is {repairs} in a row; giving up."
                        )
                    ),
                    tokens=tokens,
                    latency_ms=latency_ms,
                )
                if repairs > MAX_REPAIRS:
                    agent_run.status = "failed"
                    agent_run.stop_reason = (
                        f"The model returned {repairs} replies in a row that did not follow "
                        "the Thought / Action / Action Input format."
                    )
                    break
                # The correction goes into the transcript, so the next prompt carries
                # both the bad reply and what was wrong with it.
                transcript += (
                    f"{parsed.raw.strip()}\n\n"
                    f"Correction: that reply could not be parsed -- {parsed.problem}. "
                    "Reply again using exactly the format described above: a Thought line, "
                    "then either an Action line with an Action Input line, or a Final "
                    "Answer line.\n\n"
                )
                continue

            repairs = 0

            if parsed.kind == "final":
                if parsed.thought:
                    emit(kind="think", text=parsed.thought, tokens=tokens, latency_ms=latency_ms)
                visited.append("done")
                emit(kind="decide", text="Answered: the task needs no further tool call.")
                agent_run.output = parsed.answer
                agent_run.status = "completed"
                break

            # --- an action ---
            emit(
                kind="think",
                text=parsed.thought or "(the reply carried no Thought line)",
                tokens=tokens,
                latency_ms=latency_ms,
            )
            visited.append("act")
            emit(kind="act", text=f"Calling {parsed.tool}.", tool=parsed.tool, args=parsed.args)

            result = toolbox.call(parsed.tool, parsed.args)
            budget.charge(tool_calls=1)
            visited.append("observe")
            emit(
                kind="observe",
                text=result.output,
                tool=result.tool,
                args=result.args,
                latency_ms=result.latency_ms,
            )

            # The observation is appended as text and nothing else happens to it. This
            # line is the entire mechanism by which a ReAct agent "remembers".
            transcript += (
                f"Thought: {parsed.thought}\n"
                f"Action: {parsed.tool}\n"
                f"Action Input: {json.dumps(parsed.args)}\n"
                f"Observation: {result.output}\n\n"
            )

    except BudgetExceeded as stop:
        visited.append("cap")
        agent_run.status = "stopped_on_budget"
        agent_run.stop_reason = stop.reason
        emit(kind="decide", text=stop.reason)

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run
