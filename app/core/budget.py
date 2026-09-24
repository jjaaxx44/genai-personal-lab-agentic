"""The three caps every run is bounded by, and the check the loops call.

Ground rule 2: no agent loop ships without a step cap, a token budget and a
wall-clock deadline. They are enforced *inside* the loop rather than by a timeout
wrapper around it, so a run that stops can say which cap stopped it and the
trajectory up to that point survives.

`Budget` is re-exported from `core.types`, which is where the demo contract is
written down; the accounting lives here.
"""

import time

from pydantic import BaseModel, Field, PrivateAttr

# Names used in the stop sentence and in the budget strip, in cap order.
CAP_LABELS = {"max_steps": "step cap", "max_tokens": "token budget", "deadline_s": "deadline"}


class BudgetExceeded(Exception):
    """Raised by Budget.check() when a cap is spent.

    Demos catch this at the top of their loop and turn it into
    `AgentRun(status="stopped_on_budget", stop_reason=str(exc))` -- it is a
    normal outcome of a bounded run, not an error.
    """

    def __init__(self, cap: str, reason: str) -> None:
        super().__init__(reason)
        self.cap = cap
        self.reason = reason


class Budget(BaseModel):
    """Caps plus the spend against them.

    One Budget belongs to one run. `start()` is called once, before the first
    step; `check()` before every step; `charge()` after each step's work.
    """

    max_steps: int
    max_tokens: int
    deadline_s: float

    steps_used: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    llm_calls: int = 0

    # Wall-clock is measured from start() on a monotonic clock, so a system clock
    # change mid-run cannot extend or collapse the deadline. Private, because it
    # is not part of the run record the page renders.
    _started_at: float | None = PrivateAttr(default=None)

    def start(self, elapsed_s: float = 0.0) -> "Budget":
        """Starts (or resumes) the clock. `elapsed_s` is the agent time already spent on
        an earlier leg of this run -- passed by a demo that pauses (HITL, escalation)
        when it resumes from a checkpoint, so the wait for a person's decision is never
        counted against the deadline. Every other caller leaves it at 0.0."""
        self._started_at = time.monotonic() - elapsed_s
        return self

    @property
    def elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def check(self) -> None:
        """Raises BudgetExceeded if any cap is spent. Called before each step."""
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded(
                "max_steps",
                f"Stopped on the step cap: {self.steps_used} of {self.max_steps} steps used.",
            )
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceeded(
                "max_tokens",
                f"Stopped on the token budget: {self.tokens_used:,} of {self.max_tokens:,} tokens used.",
            )
        if self.elapsed_s >= self.deadline_s:
            raise BudgetExceeded(
                "deadline_s",
                f"Stopped on the deadline: {self.elapsed_s:.1f}s of {self.deadline_s:.0f}s elapsed.",
            )

    def charge(
        self, *, steps: int = 0, tokens: int = 0, tool_calls: int = 0, llm_calls: int = 0
    ) -> None:
        """Records spend. Never raises -- the next check() is what stops the loop, so a
        step that goes over a cap still gets recorded on the trajectory before the stop."""
        self.steps_used += steps
        self.tokens_used += tokens
        self.tool_calls += tool_calls
        self.llm_calls += llm_calls

    def fractions(self) -> dict[str, float]:
        """Spend as a 0-1 fraction of each cap, for the budget strip's bars."""
        return {
            "max_steps": min(self.steps_used / self.max_steps, 1.0) if self.max_steps else 0.0,
            "max_tokens": min(self.tokens_used / self.max_tokens, 1.0) if self.max_tokens else 0.0,
            "deadline_s": min(self.elapsed_s / self.deadline_s, 1.0) if self.deadline_s else 0.0,
        }


class BudgetDefaults(BaseModel):
    """The caps as they arrive from config, before a run copies them into a Budget."""

    max_steps: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    deadline_s: float = Field(gt=0)

    def to_budget(self) -> Budget:
        return Budget(
            max_steps=self.max_steps, max_tokens=self.max_tokens, deadline_s=self.deadline_s
        )
