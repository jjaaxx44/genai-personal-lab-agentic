"""Two toolsets over the same four sandboxed capabilities.

Both sides call the exact same `core.tools` functions through the exact same
`Toolbox` -- nothing here talks to the corpus, the database or the web directly,
and no new tool is added to the shared toolbox (CLAUDE.md rule 5). What differs is
only the *surface* the model sees: the tool's name, its argument shape, its
description, and what a failure looks like when it comes back.

- `revised_tools()` is `core.tools` describing itself: precise names, one typed
  argument per tool, a docstring that says when *not* to reach for the tool, and
  the structured `ERROR[kind]: ... Hint: ...` protocol passed straight through.
- `first_draft_tools()` is what a first pass usually looks like before anyone
  thinks about tool design: a generic one-field `input: str` schema on every
  tool, a name that does not say what the tool does (`lookup` for SQL, `search`
  overlapping `query` by name alone), a description with no guidance on when to
  use it, and `get_data` that accepts an argument and ignores it. Every error
  from `core.tools` is flattened to the single word `"Error."` before it reaches
  the model -- the same failure, with the hint that makes recovery possible
  removed.
- `decoy_tools()` adds six tools that do nothing at all: no sandbox, no I/O, a
  fixed stub reply. The "Crowd both toolboxes" toggle adds the same six to both
  sides, so what extra tool count does to selection is observed on its own,
  never mixed into the naming/schema/error comparison above.
"""

from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.tools import Toolbox

# The four capabilities both toolsets expose, and the core.tools name behind each
# one -- the comparison table and the fault injection both need to find "the SQL
# one" etc. without caring which side's tool name it is wearing.
CAPABILITIES = ("search_corpus", "describe_schema", "run_sql", "web_search")

# Every tool the sandboxed toolbox can dispatch that this demo might call,
# including `boom` for the injected fault -- constructed once per run so the
# Toolbox always has what either side needs.
TOOLBOX_NAMES = (*CAPABILITIES, "boom")

# revised name -> first-draft name, so a preset's expected tools (written once,
# against the revised names) can be checked against either side's trajectory.
REVISED_TO_DRAFT = {
    "search_corpus": "query",
    "describe_schema": "get_data",
    "run_sql": "lookup",
    "web_search": "search",
}

DECOY_NAMES = ("fetch", "check", "find", "read_info", "get_details", "list_all")
_DECOY_REPLY = "No matching data for that request."


class _FreeTextArgs(BaseModel):
    input: str = Field(description="What to look up, as free text.")


def _flatten(toolbox: Toolbox, name: str, args: dict) -> str:
    """Calls a sandboxed tool and collapses whatever it returns to `"Error."` on
    failure. The structured `ERROR[kind]: ... Hint: ...` string is still what
    `core.tools` produced -- it never reaches the model on this side."""
    result = toolbox.call(name, args)
    return result.output if result.ok else "Error."


def first_draft_tools(toolbox: Toolbox) -> list[StructuredTool]:
    """Four tools wired to the same sandboxed functions as `revised_tools()`, with
    a vague name, one free-text argument, no guidance on when not to use it, and
    every structured error flattened."""

    def query(input: str) -> str:
        return _flatten(toolbox, "search_corpus", {"query": input})

    def get_data(input: str) -> str:  # the argument is accepted and ignored
        return _flatten(toolbox, "describe_schema", {})

    def lookup(input: str) -> str:
        return _flatten(toolbox, "run_sql", {"sql": input})

    def search(input: str) -> str:
        return _flatten(toolbox, "web_search", {"query": input})

    specs: list[tuple[str, str, Callable[..., str]]] = [
        ("query", "Queries the knowledge base.", query),
        ("get_data", "Gets data.", get_data),
        ("lookup", "Looks up records.", lookup),
        ("search", "Searches for information.", search),
    ]
    return [
        StructuredTool.from_function(
            func=fn, name=name, description=desc, args_schema=_FreeTextArgs
        )
        for name, desc, fn in specs
    ]


def revised_tools(toolbox: Toolbox) -> list[StructuredTool]:
    """The same four capabilities, described and validated the way `core.tools`
    already does it -- precise names, typed Pydantic arguments, a docstring that
    says when not to use the tool, and the real error protocol."""
    return toolbox.as_langchain_tools(list(CAPABILITIES))


def _decoy_fn(name: str) -> Callable[..., str]:
    def _call(input: str) -> str:
        return _DECOY_REPLY

    _call.__name__ = name
    return _call


def decoy_tools() -> list[StructuredTool]:
    """Six near-duplicate tools that do nothing. Identical on both sides, so the
    effect of tool count is isolated from the naming/schema/error comparison."""
    return [
        StructuredTool.from_function(
            func=_decoy_fn(name),
            name=name,
            description=f"{name.replace('_', ' ').capitalize()} something. General purpose.",
            args_schema=_FreeTextArgs,
        )
        for name in DECOY_NAMES
    ]


def build_toolbox(run_id: str, *, inject_fault: bool) -> Toolbox:
    """One sandboxed `Toolbox` per run, shared by both the wiring above and the
    fault injection in `agent.py`."""
    names = list(TOOLBOX_NAMES) if inject_fault else list(CAPABILITIES)
    return Toolbox(run_id, names=names)


# A fixed run id used only to construct tools for the page's schema preview --
# never passed to `_run_side()`, so it never appears in a stored run record. The
# Toolbox it creates has no tools called against it; it exists so the same
# `first_draft_tools()` / `revised_tools()` used for a real run can also describe
# themselves before one has happened.
PREVIEW_RUN_ID = "tool-design-schema-preview"


def preview_tools(*, crowd: bool) -> tuple[list[StructuredTool], list[StructuredTool]]:
    """The tools exactly as a real run would build them, for showing their JSON
    schemas on the page without waiting for a run."""
    toolbox = build_toolbox(PREVIEW_RUN_ID, inject_fault=False)
    first = first_draft_tools(toolbox)
    revised = revised_tools(toolbox)
    if crowd:
        decoys = decoy_tools()
        first = [*first, *decoys]
        revised = [*revised, *decoys]
    return first, revised


def schema_summaries(tools: list[StructuredTool]) -> list[dict]:
    """The tool schemas exactly as `create_agent` sends them to the model --
    name, description and the JSON schema `bind_tools()` builds from
    `args_schema`. Shown side by side on the page; this is the whole
    schema-design lesson made visible."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.args_schema.model_json_schema()
            if isinstance(tool.args_schema, type)
            else tool.args_schema,
        }
        for tool in tools
    ]
