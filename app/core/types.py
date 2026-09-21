"""The contract every demo implements, and the record every run leaves behind.

`Step` and `AgentRun` are the reason the trajectory is renderable and the reason
the evaluation demo (Step 16) can compare a hand-written loop with a LangGraph
one: both return the same shape. Nothing demo-specific belongs in here.
"""

import secrets
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Budget lives in core.budget (caps + spend accounting) but is part of the demo
# contract, so it is re-exported here: `from core.types import Budget` works, and
# so does `from core.budget import Budget`.
from .budget import Budget, BudgetExceeded  # noqa: F401

StepKind = Literal["think", "act", "observe", "decide", "gate", "handoff"]
RunStatus = Literal["completed", "needs_human", "stopped_on_budget", "failed"]


def new_run_id() -> str:
    """A time-sortable run id, e.g. `20260921T143002-9f21c4`.

    Sortable by creation because the timestamp leads, unique because of the
    suffix. Deliberately not a package: a ULID would buy nothing a lexically
    sorted string does not already give us here (see the plan, Step 0).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def thread_id_for(demo: str, run_id: str) -> str:
    """The checkpoint thread a run resumes from. One namespace per demo."""
    return f"{demo}:{run_id}"


class Step(BaseModel):
    """One row on the trajectory track.

    Every thought, action, observation, decision, gate and handoff gets one --
    including the ones that failed, which are usually the interesting rows.
    """

    index: int  # 1-based, position on the track
    attempt: int = 1  # >1 renders as 04' stacked under 04
    kind: StepKind
    agent: str = "agent"  # which agent produced it (multi-agent demos)
    parent_index: int | None = None  # set on sub-agent steps: indent under the handoff
    text: str  # the thought, decision or observation, as written
    tool: str | None = None
    args: dict | None = None
    tokens: int = 0
    latency_ms: float = 0.0


class AgentRun(BaseModel):
    """Everything one run produced: the answer, and how it got there."""

    run_id: str = Field(default_factory=new_run_id)
    demo: str = ""
    task: str
    output: str = ""
    steps: list[Step] = Field(default_factory=list)
    status: RunStatus = "completed"
    stop_reason: str | None = None  # names the cap or the error, in a sentence
    visited: list[str] = Field(default_factory=list)  # graph node ids, in order, for the map
    thread_id: str | None = None  # checkpoint thread, when the demo is resumable
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0

    def add_step(self, **fields: object) -> Step:
        """Appends a step, numbering it. Demos call this rather than building Step
        themselves, so the index can never drift from the position on the track."""
        step = Step(index=len(self.steps) + 1, **fields)  # type: ignore[arg-type]
        self.steps.append(step)
        return step


class HumanDecision(BaseModel):
    """What a person did at a gate. Carried into `resume()`."""

    verdict: Literal["approve", "edit", "reject"]
    edited: dict | None = None  # the amended tool arguments, when the verdict is "edit"
    reason: str = ""  # shown to the agent, so a rejection can be reacted to


@runtime_checkable
class AgentDemo(Protocol):
    """What the evaluation demo (Step 16) is allowed to assume about a demo.

    `resume()` is implemented only by the demos that pause -- HITL, escalation,
    and any demo whose gate is optional.
    """

    def run(self, task: str, budget: Budget, **settings: object) -> AgentRun: ...

    def resume(self, thread_id: str, decision: HumanDecision) -> AgentRun: ...
