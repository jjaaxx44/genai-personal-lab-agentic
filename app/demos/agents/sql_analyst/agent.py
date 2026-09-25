"""Text-to-SQL analyst: inspect the schema, write one SELECT, run it read-only,
read the rows, then answer or write a follow-up query -- through LangChain's
`create_agent`, the same framework Step 2 (`tool_design/`) uses.

Two guards already live in `core.tools._run_sql` (SELECT-only regex, a read-only
SQLite connection) and this demo adds no guard in front of them: a refused write
has to come from the real tool, or the "done when" check proves nothing about the
sandbox and only about this file. What this demo adds on top is a **fix-and-retry-
once** rule, enforced in `wrap_tool_call` rather than in `core.tools`, because it
is specific to this demo's loop (a model gets one chance to read a SQL error and
correct it before further `run_sql` calls are refused outright) rather than a
property of the tool itself.

Budget accounting follows `tool_design/agent.py`'s three middleware hooks:
`before_model` checks the budget and jumps to `end` on a spent cap, `wrap_model_call`
times and charges each model call, `wrap_tool_call` charges each tool call and is
also where the retry rule and its refusal live.
"""

import re
import time
from typing import Any, Callable

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage

from core.budget import Budget, BudgetExceeded
from core.config import get_settings
from core.llm import agent_models, count_tokens
from core.mongo import save_run
from core.tools import Toolbox, is_error, tool_error
from core.tracing import get_callbacks, observe
from core.types import AgentRun, Step, new_run_id

DEMO = "sql_analyst"
NAME = "Text-to-SQL analyst"
SENTENCE = "Inspect the schema, write SQL, run it read-only, read the rows, then answer or query again."
SHAPE = "schema → SQL → rows → answer ↻"

GRAPH = """flowchart LR
    question[Question] --> schema[Inspect the schema]
    schema --> sql[Write one SELECT]
    sql --> run[Run it read-only]
    run -->|rows| read[Read the rows]
    run -->|error| fix[Fix and retry once]
    fix --> run
    read -->|need more| sql
    read -->|enough| answer[Answer with SQL, table and chart]
    run -.->|cap spent| cap[Stopped]
"""

# One preset per lesson: a two-query aggregate-then-drill-down over genres and
# artists, the same shape over countries and customers (with a genuine tie in the
# answer), a wrong column/table name the model has to recover from (paired with
# the "Skip the schema tool" toggle so the recovery lands on the track even when
# the model would otherwise check first), and a write the tool must refuse.
PRESETS: list[dict[str, Any]] = [
    {
        "task": (
            "Which genre brings in the most revenue, and within that genre, which "
            "three artists earn the most?"
        ),
    },
    {
        "task": (
            "Which billing country has the highest total sales, and who are the "
            "top three customers there?"
        ),
    },
    {
        "task": "Sum the `amount` column of the Invoice table by billing country and name the top three.",
    },
    {
        "task": "Set every invoice total for Germany to zero, then tell me the new German total.",
    },
]

SYSTEM_PROMPT = (
    "You are a SQL analyst working against a read-only sales database. Call "
    "describe_schema before writing your first query -- the table and column names "
    "are not guessable. Write one SELECT statement per tool call, and aggregate in "
    "SQL rather than fetching raw rows and summing them yourself. After reading the "
    "rows back, either answer the question in full sentences, naming the query it "
    "rests on, or write one follow-up query if you need more. If a query errors, "
    "read the error and fix the query once -- do not repeat a call that just failed "
    "outright. The database enforces read-only access itself, "
    "below you: if the request asks for a change, send that statement through "
    "run_sql as asked and report exactly what the database says, then answer "
    "whatever part of the request is a read."
)

SYSTEM_PROMPT_NO_SCHEMA = (
    "You are a SQL analyst working against a read-only sales database. No schema "
    "tool is available on this run, so you do not know the table or column names "
    "in advance -- write your best guess for the first query and learn the real "
    "names from the error SQLite returns if it is wrong. Write one SELECT statement "
    "per tool call, and aggregate in SQL rather than fetching raw rows and summing "
    "them yourself. After reading the rows back, either answer the question in full "
    "sentences, naming the query it rests on, or write one follow-up query if you "
    "need more. If a query errors, read the error and fix the query once -- do not "
    "repeat a call that just failed outright. The database enforces read-only "
    "access itself, below you: if the request asks for a change, send that "
    "statement through run_sql as asked and report exactly what the database "
    "says, then answer whatever part of the request is a read."
)


def _system_prompt(skip_schema: bool) -> str:
    return SYSTEM_PROMPT_NO_SCHEMA if skip_schema else SYSTEM_PROMPT


def _extract_ai_message(response: Any) -> AIMessage | None:
    """`wrap_model_call`'s handler returns a `ModelResponse` in this LangChain
    version, but the type is documented as `ModelResponse | AIMessage` -- so both
    are handled rather than assumed. Copied from `tool_design/agent.py`."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None)
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _retry_refused(consecutive_failures: int, max_retries: int) -> bool:
    """Whether the next run_sql call should be blocked rather than executed.

    One real failure is the ordinary cost of a first attempt; the promised retry
    is the *next* call, which must still be allowed to run even though a failure
    already happened -- so the gate is "more than" max_retries, not "at least".
    With the default cap of 1: the original call may fail, one retry may also
    fail, and only the call after *that* is refused outright. A success anywhere
    in the streak resets the caller's counter back to 0 before this is checked
    again, so a fixed query always gets a clean run of retries next time it fails.
    """
    return consecutive_failures > max_retries


# --- table and chart helpers (pure -- used by the page, never touch Streamlit) --

_ROW_COUNT_LINE = re.compile(r"^\d+ row\(s\)\.")


def parse_rows(observation: str) -> pd.DataFrame | None:
    """Turns one `run_sql` observation back into a table, for the page to render.

    Reads only the text already on the trajectory -- the page never re-runs the
    query. Returns `None` for an error observation, or for one with nothing
    tabular in it. `"0 rows."` becomes an empty DataFrame, which is tabular (a
    real, empty result) even though it has nothing to show.
    """
    if not observation or is_error(observation):
        return None
    text = observation.strip()
    if text == "0 rows.":
        return pd.DataFrame()

    lines = text.splitlines()
    # Strips the trailing "N row(s)." line and, when the row cap was hit, the
    # "(N row cap reached -- there are more.)" note after it.
    while lines and (_ROW_COUNT_LINE.match(lines[-1]) or lines[-1].startswith("(")):
        lines.pop()
    if not lines:
        return None

    header = [c.strip() for c in lines[0].split(" | ")]
    rows = [[c.strip() for c in line.split(" | ")] for line in lines[1:]]
    df = pd.DataFrame(rows, columns=header)

    # Numeric columns coerced only where every value in them parses -- a column
    # with one NULL (empty string) or one non-numeric value stays text.
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().all():
            df[col] = coerced
    return df


_DATE_LIKE_NAME = re.compile(r"year|date|month", re.IGNORECASE)
_DATE_LIKE_VALUE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _looks_date_like(name: str, values: "pd.Series") -> bool:
    if _DATE_LIKE_NAME.search(name):
        return True
    text_values = values.dropna().astype(str)
    if text_values.empty:
        return False
    return bool(text_values.map(lambda v: bool(_DATE_LIKE_VALUE.match(v))).all())


def chart_spec(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """What chart (if any) suits this table -- exactly one text column and one
    numeric column decide it; row count and the text column's shape decide which
    kind. Pure: the page passes the DataFrame `parse_rows` already built and
    renders whatever comes back, or nothing at all.

    - one text + one numeric, 2-25 rows -> horizontal bar, sorted by value.
    - one date/year-like column + one numeric -> line, sorted by that column.
    - anything else -> None.
    """
    if df is None or df.empty or len(df.columns) != 2:
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in df.columns if c not in numeric_cols]
    if len(numeric_cols) != 1 or len(text_cols) != 1:
        return None

    numeric_col, text_col = numeric_cols[0], text_cols[0]
    if _looks_date_like(text_col, df[text_col]):
        return {"kind": "line", "x": text_col, "y": numeric_col, "df": df.sort_values(text_col)}
    if 2 <= len(df) <= 25:
        return {
            "kind": "bar",
            "x": text_col,
            "y": numeric_col,
            "df": df.sort_values(numeric_col, ascending=False),
        }
    return None


def chart_skip_reason(df: pd.DataFrame) -> str:
    """Why `chart_spec` returned `None` for this table -- shown as a caption next
    to the table instead of the chart, so the shape rule is legible, not silent."""
    if len(df.columns) != 2:
        return f"no chart: {len(df.columns)} columns"
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) != 1:
        return "no chart: not one text column and one numeric column"
    text_col = next(c for c in df.columns if c not in numeric_cols)
    if not _looks_date_like(text_col, df[text_col]) and not (2 <= len(df) <= 25):
        return f"no chart: {len(df)} row(s)"
    return "no chart: the shape doesn't suit one"


# --- the run ---------------------------------------------------------------------


@observe(name="sql_analyst.run")
def run(task: str, budget: Budget, **settings: Any) -> AgentRun:
    """Question, then schema / SQL / run / read, round and round until a final
    answer or a spent cap. Never raises: any failure -- no provider configured,
    every provider refusing, an unexpected error building the agent -- lands in
    the returned `AgentRun` as `status="failed"`.

    Settings, all optional:
      `on_step`     callable(Step) -- called as each step lands, for streaming.
      `skip_schema` bool -- removes describe_schema from this run's tools, so a
                    wrong column name can only be learned from the SQL error
                    SQLite itself returns, rather than avoided by inspection.
    """
    on_step: Callable[[Step], None] | None = settings.get("on_step")
    skip_schema = bool(settings.get("skip_schema"))

    run_id = new_run_id()
    agent_run = AgentRun(run_id=run_id, demo=DEMO, task=task)
    visited: list[str] = ["question"]

    def emit(**fields: Any) -> Step:
        step = agent_run.add_step(**fields)
        if on_step is not None:
            on_step(step)
        return step

    started = time.monotonic()
    budget.start()

    try:
        cfg = get_settings()
        max_retries = cfg.sql_analyst_max_sql_retries
        tool_names = ["run_sql"] if skip_schema else ["describe_schema", "run_sql"]
        toolbox = Toolbox(run_id, names=tool_names)
        tools = toolbox.as_langchain_tools()

        primary, *fallbacks = agent_models()

        cap_hit: dict[str, str] = {}
        model_meta: dict[str, Any] = {}
        tool_latency: dict[str, float] = {}
        tool_call_args: dict[str, dict[str, Any]] = {}
        call_attempt: dict[str, int] = {}
        consecutive_failures = {"n": 0}

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
            name = request.tool_call["name"]
            args = dict(request.tool_call.get("args") or {})
            tool_call_args[tool_call_id] = args
            call_started = time.monotonic()

            if name == "run_sql" and _retry_refused(consecutive_failures["n"], max_retries):
                # Not executed at all -- the refusal itself is the observation, exactly
                # like core.tools' own ERROR[refused] protocol, so the model reads it
                # the same way it reads any other tool error.
                text = tool_error(
                    "refused",
                    "The retry limit for failed queries is reached.",
                    "answer with what you have, and say which query failed.",
                )
                message = ToolMessage(content=text, name=name, tool_call_id=tool_call_id)
            else:
                message = handler(request)
                if name == "run_sql":
                    if is_error(message.content):
                        consecutive_failures["n"] += 1
                    else:
                        consecutive_failures["n"] = 0

            tool_latency[tool_call_id] = (time.monotonic() - call_started) * 1000
            budget.charge(tool_calls=1)
            return message

        middleware: list[Any] = []
        if fallbacks:
            middleware.append(ModelFallbackMiddleware(*fallbacks))
        middleware += [check_budget, charge_model, run_tool]

        compiled_agent = create_agent(
            primary,
            tools=tools,
            system_prompt=_system_prompt(skip_schema),
            middleware=middleware,
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
                        emit(
                            kind="think",
                            text=message.content or "(no reasoning text with this call)",
                            tokens=tokens,
                            latency_ms=latency_ms,
                        )
                        for call in message.tool_calls:
                            name = call["name"]
                            call_id = call["id"]
                            if name == "run_sql":
                                attempt = 2 if consecutive_failures["n"] >= 1 else 1
                                refused_now = _retry_refused(consecutive_failures["n"], max_retries)
                                visited.append("fix" if attempt == 2 else "sql")
                                if not refused_now:
                                    visited.append("run")
                            else:
                                attempt = 1
                                visited.append("schema")
                            call_attempt[call_id] = attempt
                            emit(
                                kind="act",
                                attempt=attempt,
                                text=f"Calling {name}.",
                                tool=name,
                                args=call.get("args", {}),
                            )
                    else:
                        visited.append("answer")
                        emit(
                            kind="think",
                            text=message.content or "(final reply carried no reasoning text)",
                            tokens=tokens,
                            latency_ms=latency_ms,
                        )
                        emit(kind="decide", text="Answered: the task needs no further query.")
                        agent_run.output = message.content or ""
                        agent_run.status = "completed"
                    continue

                if node == "tools":
                    for message in update["messages"]:
                        name = message.name
                        call_id = message.tool_call_id
                        attempt = call_attempt.get(call_id, 1)
                        if name == "run_sql" and not is_error(message.content):
                            visited.append("read")
                        emit(
                            kind="observe",
                            attempt=attempt,
                            text=message.content,
                            tool=name,
                            args=tool_call_args.get(call_id),
                            latency_ms=tool_latency.get(call_id, 0.0),
                        )
                    continue

    except Exception as exc:  # a provider refused, none configured, or the agent failed to build
        agent_run.status = "failed"
        agent_run.stop_reason = f"The run could not finish: {type(exc).__name__}: {exc}"
        emit(kind="decide", text=agent_run.stop_reason)

    agent_run.visited = visited
    agent_run.llm_calls = budget.llm_calls
    agent_run.tool_calls = budget.tool_calls
    agent_run.tokens = budget.tokens_used
    agent_run.latency_ms = (time.monotonic() - started) * 1000
    save_run(agent_run)
    return agent_run
