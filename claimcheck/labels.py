"""What makes a row positive. A positive rate never travels without this.

    from claimcheck.labels import LabelRule
    rule = LabelRule(name="top_decile", assay="biophysical_binding",
                     endpoint="pKD",
                     statement="pKD >= 6.0 measured by SPR at 25 C")

## The incident

A positive rate was published with the wrong defining rule attached — a pK_D
threshold set beside an HTS dose-response call, in one column. Two rules, two
assays, one number, and nothing in the artifact said which was which.

`claimcheck` can already say that a column is a float in kcal/mol. It could not
say what made a row a hit, and a hit rate is the most reported and least
qualified number in screening work. A rate carries its rule or it is not a rate.

## Four required fields, and `assay` is the one people leave out

Two rules with different assays are not comparable however similar their
thresholds look — a pIC50 of 6 from a cell viability screen and a pKD of 6 from
SPR are different statements about different things. A rule missing the assay
cannot be checked against another rule, and an uncheckable rule is the state
this prevents.

`thresholdable=False` marks a pre-binarised delivery: the data arrived as calls,
the threshold is not recoverable, and sweeping it is not an available analysis.
Saying so stops someone reporting a sensitivity curve over a threshold that does
not exist.

Ported 2026-08-17 from a predecessor package that has since been retired
and archived. The reasoning is reproduced here rather than cited, because
a citation whose target is inside a zip file is not a citation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class UnusableLabelRule(ValueError):
    """A label rule missing a field that would let anyone recompute it."""


@dataclass(frozen=True)
class LabelRule:
    name: str
    #: How the measurement was made — `biophysical_binding`,
    #: `hts_dose_response`, `cell_viability`.
    assay: str
    #: The quantity thresholded: `pKD`, `pIC50`, `binary_call`.
    endpoint: str
    #: The rule in words, precise enough to recompute from raw values.
    statement: str
    #: Whether a threshold can be moved at all.
    thresholdable: bool = True
    citation: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        missing = [f for f in ("name", "assay", "endpoint", "statement")
                   if not getattr(self, f)]
        if missing:
            raise UnusableLabelRule(
                f"label rule {self.name or '<unnamed>'} is missing {missing}; "
                f"without them it cannot be compared with another rule")

    def describe(self) -> str:
        """One line, for an artifact column or a figure caption."""
        return f"{self.name} ({self.assay}, {self.endpoint})"

    def comparable_to(self, other: "LabelRule") -> tuple[bool, str]:
        """May two positive rates be put side by side?

        The check the incident needed. Same endpoint and same assay, or the two
        rates are measuring different things and the comparison is void however
        similar the thresholds read.
        """
        if self.assay != other.assay:
            return False, (f"{self.name} is {self.assay} and {other.name} is "
                           f"{other.assay}; rates from different assays are not "
                           f"comparable")
        if self.endpoint != other.endpoint:
            return False, (f"{self.name} thresholds {self.endpoint} and "
                           f"{other.name} thresholds {other.endpoint}")
        return True, f"both {self.assay} on {self.endpoint}"

    def as_dict(self) -> dict:
        """For emission beside the rate, so the two cannot be separated."""
        return asdict(self)


def rate(n_positive: int, n_total: int, rule: LabelRule) -> dict:
    """A positive rate that carries its rule. There is no plain-number form.

    Returned as a mapping rather than a float ON PURPOSE: a float can be pulled
    out of context by the next line of code, and every incident behind this
    module is a number that travelled without its definition.
    """
    if n_total <= 0:
        raise ValueError("a rate over zero rows is not a rate")
    if not 0 <= n_positive <= n_total:
        raise ValueError(f"{n_positive} positives out of {n_total} rows")
    return {
        "n_positive": int(n_positive),
        "n_total": int(n_total),
        "rate": n_positive / n_total,
        "label_rule": rule.describe(),
        **{f"label_{k}": v for k, v in rule.as_dict().items()},
    }
