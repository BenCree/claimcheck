"""Best-practice checklists from the literature, as executable checks.

**The contract.** Each module in this package encodes the explicit checklist of
**one** paper — usually a LiveCoMS best-practices article — as runnable code.
The point is not documentation: it is that our adversarial reviewer
(`GOALS.md`, roles table) should compare us against a **published standard**
rather than against its own judgement, and a standard you cannot execute is one
nobody runs.

    from bestpractices import Check, Context, Result, PASS, FAIL, UNTESTABLE, NA

    PAPER = Paper(
        slug="uncertainty",
        title="...",
        authors="...",
        url="https://livecomsjournal.org/...",
        doi="...",
        scope="one sentence: what this paper governs",
    )

    def _has_error_bars(ctx: Context) -> Result:
        ...
        return Result(PASS, "every reported mean carries a CI", evidence="...")

    CHECKS = [
        Check(id="unc-01", question="Does every reported mean carry an uncertainty?",
              severity="must", section="§3.1", fn=_has_error_bars),
        ...
    ]

Run everything with::

    python check_bestpractices.py                  # all papers
    python check_bestpractices.py --paper uncertainty
    python check_bestpractices.py --severity must  # only the hard requirements

### Four outcomes, not two

`UNTESTABLE` is a result in its own right and **must never be reported as a pass**.
This project has already shipped a vacuous PASS — a metal-geometry check that
reported "closest inf Å, PASS" when there were no hydrogens to test
(`FINDINGS` §3.7). A check that cannot fail has not checked anything.

`NA` means the practice genuinely does not apply to us (e.g. a periodic-boundary
requirement for a gas-phase calculation). **`NA` must carry a reason**, because
"not applicable" is the easiest way to make a checklist say what you want.

### Writing a good check

* **Prefer measuring a file over asking a question.** A check that greps our
  actual output is worth ten that ask the operator to confirm something.
* **Say what would fail.** If you cannot describe an input that returns FAIL,
  the check is decorative.
* **Cite the section.** The value of these modules is traceability to a
  published standard; an unattributed rule is just our opinion again.
* Checks must be **cheap and side-effect free** — no cluster jobs, no writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

PASS = "PASS"
FAIL = "FAIL"
UNTESTABLE = "UNTESTABLE"
NA = "N/A"

Status = Literal["PASS", "FAIL", "UNTESTABLE", "N/A"]
Severity = Literal["must", "should", "consider"]


@dataclass(frozen=True)
class Paper:
    """Provenance for the checklist. Every check traces back to this."""
    slug: str
    title: str
    authors: str
    url: str
    scope: str
    doi: str = ""
    year: str = ""


@dataclass
class Result:
    status: Status
    detail: str
    #: The number, path or command the verdict rests on. A Result with no
    #: evidence is an assertion, and assertions are what this package replaces.
    evidence: str = ""

    def __post_init__(self):
        if self.status not in (PASS, FAIL, UNTESTABLE, NA):
            raise ValueError(f"bad status {self.status!r}")
        if self.status == NA and not self.detail:
            raise ValueError("N/A must carry a reason — see the module docstring")


@dataclass
class Check:
    id: str
    question: str
    severity: Severity
    section: str
    fn: Callable[["Context"], Result]
    #: Set when the check inspects our repo rather than asking a question.
    measures: str = ""


@dataclass
class Context:
    """Everything a check may look at. Read-only by contract."""
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def results(self) -> Path:
        return self.root / "results"

    def read(self, rel: str) -> str:
        """Text of a repo file, or '' if absent — so a check can report
        UNTESTABLE rather than raising."""
        p = self.root / rel
        return p.read_text(errors="replace") if p.exists() else ""

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def glob(self, pattern: str):
        return sorted(self.root.glob(pattern))


__all__ = ["Paper", "Check", "Context", "Result",
           "PASS", "FAIL", "UNTESTABLE", "NA"]
