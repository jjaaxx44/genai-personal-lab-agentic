"""The sandboxed toolbox every demo draws from.

Two rules shape this file.

**Ground rule 5 — sandboxed tools only.** The corpus is read-only, SQL is
SELECT-only against a bundled SQLite opened read-only and row-capped, web search
is `ddgs` with a result cap and a timeout, and the file tools are confined to one
directory per run under `data/vfs/`. There is no shell tool and there never will
be. Every call is also wrapped in a wall-clock timeout, because a tool that hangs
is a run that ignores its deadline.

**Ground rule 6 — a tool error is data, not a crash.** Nothing here raises into a
demo. A bad argument, a refused statement, an escape attempt, a timeout and an
unexpected exception all come back as a structured `ERROR[...]` string, which the
demo appends to the trajectory as an observation and hands back to the model. The
model recovering from it is part of what these demos teach.

The tools are defined once as plain callables with a pydantic schema each
(`ToolSpec`), and wired up two ways: `Toolbox.call()` for the hand-written ReAct
loop, which must not import LangChain at all, and `Toolbox.as_langchain_tools()`
for everything built on `create_agent` / `StateGraph`. How a demo describes and
binds them is the demo's business — that is the subject of Step 2.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .config import REPO_ROOT, get_settings

CHINOOK_PATH = REPO_ROOT / "samples" / "chinook.db"


# --- the error protocol --------------------------------------------------------

ERROR_KINDS = (
    "unknown_tool",
    "bad_argument",
    "refused",
    "not_found",
    "timeout",
    "unavailable",
    "failed",
)


def tool_error(kind: str, message: str, hint: str = "") -> str:
    """The one shape every tool failure takes.

    Written to be read by a model, not by a person: the kind says what class of
    problem it is, the message says what happened, and the hint says what a
    different call would have to look like. `hint` is what makes recovery on the
    next step possible, so it is filled in wherever there is a next call worth
    suggesting.
    """
    text = f"ERROR[{kind}]: {message}"
    return f"{text} Hint: {hint}" if hint else text


def is_error(observation: str) -> bool:
    """Whether an observation is a tool error. Used by the track (to mark the row) and
    by evaluation (to count recoveries)."""
    return observation.startswith("ERROR[")


# --- tool argument schemas -----------------------------------------------------


class SearchCorpusArgs(BaseModel):
    query: str = Field(description="What to look for, as a phrase or question.")
    top_k: int = Field(default=5, ge=1, le=10, description="How many passages to return.")


class RunSqlArgs(BaseModel):
    sql: str = Field(description="One read-only SQLite SELECT statement.")


class DescribeSchemaArgs(BaseModel):
    pass


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query, as you would type it into a search box.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File name relative to this run's directory, e.g. notes.md")
    content: str = Field(description="The complete text to write. Replaces any existing content.")


class ReadFileArgs(BaseModel):
    path: str = Field(description="File name relative to this run's directory.")


class ListFilesArgs(BaseModel):
    pass


class BoomArgs(BaseModel):
    pass


# --- the tools themselves ------------------------------------------------------


def _search_corpus(*, run_dir: Path, query: str, top_k: int = 5) -> str:
    from pymongo.errors import PyMongoError

    from .corpus import search_corpus as _search  # imported here: indexing pulls in torch

    try:
        hits = _search(query, top_k=top_k)
    except PyMongoError:
        return tool_error(
            "unavailable",
            "The corpus index could not be reached.",
            "Try again, or answer from what you already have.",
        )
    if not hits:
        return "No passages matched. The corpus may be empty, or the wording may be too specific."
    return "\n\n".join(
        f"[{hit['source']} · {hit['score']:.3f}] {hit['text']}" for hit in hits
    )


_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def _get_engine():
    from sqlalchemy import create_engine

    # SQLite's own read-only URI flag blocks writes at the database layer, behind the
    # SELECT-only check below. Two independent guards, because one regex is not a
    # security boundary.
    return create_engine(f"sqlite:///file:{CHINOOK_PATH}?mode=ro&uri=true")


def _run_sql(*, run_dir: Path, sql: str) -> str:
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    settings = get_settings()
    statement = sql.strip().rstrip(";").strip()

    if not statement:
        return tool_error("bad_argument", "No SQL was given.", "Pass one SELECT statement.")
    if ";" in statement:
        return tool_error(
            "refused",
            "More than one statement was passed.",
            "Send a single SELECT statement, with no semicolons inside it.",
        )
    if not re.match(r"^(SELECT|WITH)\b", statement, re.IGNORECASE):
        return tool_error(
            "refused",
            "This database is read-only and only SELECT is allowed.",
            "Rewrite the request as a SELECT, or answer without changing data.",
        )
    if _FORBIDDEN_SQL.search(statement):
        return tool_error(
            "refused",
            "The statement contains a keyword that could modify the database.",
            "Use only SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY and LIMIT.",
        )

    try:
        with _get_engine().connect() as connection:
            result = connection.execute(text(statement))
            columns = list(result.keys())
            rows = result.fetchmany(settings.sql_row_limit + 1)
    except SQLAlchemyError as exc:
        # The database's own message (unknown column, syntax error) is the most useful
        # thing the model can get here, so it is passed through rather than swallowed.
        detail = str(getattr(exc, "orig", exc)).strip().splitlines()[0]
        return tool_error(
            "failed",
            f"SQLite rejected the query: {detail}",
            "Check the table and column names with describe_schema, then try again.",
        )

    truncated = len(rows) > settings.sql_row_limit
    rows = rows[: settings.sql_row_limit]
    if not rows:
        return "0 rows."

    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    note = f"\n({settings.sql_row_limit} row cap reached — there are more.)" if truncated else ""
    return f"{header}\n{body}\n{len(rows)} row(s).{note}"


def _describe_schema(*, run_dir: Path) -> str:
    from sqlalchemy import inspect

    inspector = inspect(_get_engine())
    lines = []
    for table in sorted(inspector.get_table_names()):
        columns = ", ".join(
            f"{c['name']} {c['type']}" for c in inspector.get_columns(table)
        )
        lines.append(f"{table}({columns})")
    return "\n".join(lines)


def _web_search(*, run_dir: Path, query: str) -> str:
    from ddgs import DDGS

    settings = get_settings()
    try:
        hits = DDGS().text(query, max_results=settings.web_search_results)
    except Exception as exc:
        return tool_error(
            "unavailable",
            f"The web search did not answer ({type(exc).__name__}).",
            "Try search_corpus instead, or answer from what you already have.",
        )
    if not hits:
        return "No web results."
    return "\n\n".join(
        f"[{i}] {h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}"
        for i, h in enumerate(hits, start=1)
    )


def _resolve_in_run_dir(run_dir: Path, path: str) -> Path | str:
    """A path inside this run's directory, or an error string explaining the refusal.

    The check is on the *resolved* path, so `../`, a symlink and an absolute path
    are all caught by the same test rather than by three string rules.
    """
    candidate = (run_dir / path).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        return tool_error(
            "refused",
            f"'{path}' is outside this run's directory.",
            "Use a plain file name with no leading slash and no '..'.",
        )
    return candidate


def _write_file(*, run_dir: Path, path: str, content: str) -> str:
    settings = get_settings()
    target = _resolve_in_run_dir(run_dir, path)
    if isinstance(target, str):
        return target

    size_kb = len(content.encode("utf-8")) / 1024
    if size_kb > settings.vfs_max_file_kb:
        return tool_error(
            "refused",
            f"That content is {size_kb:.0f} KB; the limit is {settings.vfs_max_file_kb} KB per file.",
            "Write a shorter note, or split it across two files.",
        )
    existing = [p for p in run_dir.rglob("*") if p.is_file()]
    if len(existing) >= settings.vfs_max_files and not target.exists():
        return tool_error(
            "refused",
            f"This run already has {len(existing)} files; the limit is {settings.vfs_max_files}.",
            "Overwrite one of the files you already wrote instead.",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {path} ({size_kb:.1f} KB)."


def _read_file(*, run_dir: Path, path: str) -> str:
    target = _resolve_in_run_dir(run_dir, path)
    if isinstance(target, str):
        return target
    if not target.is_file():
        names = sorted(p.name for p in run_dir.glob("*") if p.is_file())
        return tool_error(
            "not_found",
            f"There is no file called '{path}' in this run.",
            f"Files in this run: {', '.join(names) or 'none yet'}.",
        )
    return target.read_text(encoding="utf-8")


def _list_files(*, run_dir: Path) -> str:
    files = sorted((p for p in run_dir.rglob("*") if p.is_file()), key=lambda p: p.name)
    if not files:
        return "No files yet."
    return "\n".join(
        f"{p.relative_to(run_dir)} ({p.stat().st_size / 1024:.1f} KB)" for p in files
    )


def _boom(*, run_dir: Path) -> str:
    """Deliberately raises. Not in the default toolbox -- the debug page uses it to show
    that an exception inside a tool becomes an observation, and Step 2 uses it as its
    injected tool error."""
    raise RuntimeError("this tool always fails")


# --- specs ---------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool, described once.

    The description is the selection signal the model actually reads, so it says
    what the tool is for *and* when not to reach for it. Step 2 exists to make
    that visible by breaking it on purpose.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable[..., str]


TOOLBOX: dict[str, ToolSpec] = {
    "search_corpus": ToolSpec(
        name="search_corpus",
        description=(
            "Search the bundled document corpus for passages relevant to a query. Returns "
            "the best-matching passages with their source file and score. Use this for "
            "anything about the corpus's subject matter. Do not use it for questions about "
            "the sales database — that is run_sql — or for current events, which it has none of."
        ),
        args_schema=SearchCorpusArgs,
        func=_search_corpus,
    ),
    "describe_schema": ToolSpec(
        name="describe_schema",
        description=(
            "List every table in the bundled Chinook sales database with its columns and "
            "types. Call this before writing SQL for the first time; the column names are "
            "not guessable."
        ),
        args_schema=DescribeSchemaArgs,
        func=_describe_schema,
    ),
    "run_sql": ToolSpec(
        name="run_sql",
        description=(
            "Run one read-only SELECT statement against the bundled Chinook sales database "
            "and return the rows. SELECT only — anything that would change data is refused. "
            "Results are capped, so aggregate in SQL rather than fetching everything."
        ),
        args_schema=RunSqlArgs,
        func=_run_sql,
    ),
    "web_search": ToolSpec(
        name="web_search",
        description=(
            "Search the public web and return the top results with titles, URLs and "
            "snippets. Use it for current or external facts the corpus cannot hold. It "
            "returns snippets, not whole pages."
        ),
        args_schema=WebSearchArgs,
        func=_web_search,
    ),
    "write_file": ToolSpec(
        name="write_file",
        description=(
            "Write a text file into this run's own directory, replacing it if it exists. "
            "Use it to keep notes and drafts out of the conversation on long tasks."
        ),
        args_schema=WriteFileArgs,
        func=_write_file,
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read back a file this run wrote earlier, by name.",
        args_schema=ReadFileArgs,
        func=_read_file,
    ),
    "list_files": ToolSpec(
        name="list_files",
        description="List the files this run has written, with their sizes.",
        args_schema=ListFilesArgs,
        func=_list_files,
    ),
}

# Available by name, never in the default toolbox. See _boom.
EXTRA_SPECS: dict[str, ToolSpec] = {
    "boom": ToolSpec(
        name="boom",
        description="A tool that always raises, for demonstrating error recovery.",
        args_schema=BoomArgs,
        func=_boom,
    )
}

DEFAULT_TOOLS = tuple(TOOLBOX)


@dataclass
class ToolResult:
    """What one call cost and what it returned. Becomes an `observe` step."""

    tool: str
    args: dict[str, Any]
    output: str
    ok: bool
    latency_ms: float


class Toolbox:
    """The tools one run may use, bound to that run's file-system directory.

    A demo builds one of these per run and chooses which tools to expose:

        box = Toolbox(run_id, names=["search_corpus", "run_sql"])
        result = box.call("run_sql", {"sql": "SELECT 1"})
    """

    def __init__(self, run_id: str, *, names: list[str] | None = None) -> None:
        settings = get_settings()
        self.run_id = run_id
        self.specs: dict[str, ToolSpec] = {}
        for name in names or DEFAULT_TOOLS:
            spec = TOOLBOX.get(name) or EXTRA_SPECS.get(name)
            if spec is None:
                raise KeyError(f"No such tool: {name}")
            self.specs[name] = spec

        root = Path(settings.vfs_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def describe(self) -> str:
        """The toolbox as text, for a prompt that documents its tools by hand (Step 1)."""
        lines = []
        for spec in self.specs.values():
            fields = ", ".join(
                f"{name}: {getattr(field.annotation, '__name__', field.annotation)}"
                for name, field in spec.args_schema.model_fields.items()
            )
            lines.append(f"- {spec.name}({fields}): {spec.description}")
        return "\n".join(lines)

    def call(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        """Runs one tool. Never raises: every failure comes back as an error string."""
        args = dict(args or {})
        start = time.monotonic()

        def finish(output: str) -> ToolResult:
            return ToolResult(
                tool=name,
                args=args,
                output=output,
                ok=not is_error(output),
                latency_ms=(time.monotonic() - start) * 1000,
            )

        spec = self.specs.get(name)
        if spec is None:
            return finish(
                tool_error(
                    "unknown_tool",
                    f"There is no tool called '{name}'.",
                    f"Available tools: {', '.join(self.names)}.",
                )
            )

        try:
            validated = spec.args_schema(**args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'argument'}: {err['msg']}"
                for err in exc.errors()
            )
            return finish(
                tool_error(
                    "bad_argument",
                    f"The arguments for {name} were rejected: {problems}.",
                    f"Expected: {list(spec.args_schema.model_fields)}.",
                )
            )
        except TypeError as exc:
            return finish(tool_error("bad_argument", str(exc)))

        settings = get_settings()
        # A wall-clock timeout per call, so one stuck tool cannot outlive the run's
        # deadline. The worker thread is abandoned rather than killed -- Python cannot
        # kill a thread -- which is acceptable here because every tool is read-only or
        # writes only inside this run's own directory.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(spec.func, run_dir=self.run_dir, **validated.model_dump())
            try:
                output = future.result(timeout=settings.tool_timeout_s)
            except FutureTimeout:
                return finish(
                    tool_error(
                        "timeout",
                        f"{name} did not finish within {settings.tool_timeout_s:.0f}s.",
                        "Try a narrower query, or a different tool.",
                    )
                )
            except Exception as exc:
                # The demo's trajectory shows this; the container log keeps the traceback.
                return finish(
                    tool_error(
                        "failed",
                        f"{name} raised {type(exc).__name__}: {exc}.",
                        "Try different arguments, or another tool.",
                    )
                )

        return finish(output if isinstance(output, str) else str(output))

    def as_langchain_tools(self, names: list[str] | None = None) -> list[Any]:
        """The same tools as LangChain `StructuredTool`s, for the create_agent and
        StateGraph demos. Imported here so that a plain-Python demo importing this
        module never pulls LangChain in behind its back."""
        from langchain_core.tools import StructuredTool

        tools = []
        for name in names or self.names:
            spec = self.specs[name]

            def _make(tool_name: str) -> Callable[..., str]:
                def _invoke(**kwargs: Any) -> str:
                    return self.call(tool_name, kwargs).output

                return _invoke

            tools.append(
                StructuredTool.from_function(
                    func=_make(name),
                    name=spec.name,
                    description=spec.description,
                    args_schema=spec.args_schema,
                )
            )
        return tools
