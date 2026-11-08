"""claimcheck: recompute the claim, do not read it."""

from claimcheck.datasets import (
    LEVEL_RANK,
    LEVELS,
    REAL,
    REFERENCE,
    TEST_CASE,
    TOY,
    canonical_level,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "canonical_level",
    "LEVELS",
    "LEVEL_RANK",
    "TOY",
    "TEST_CASE",
    "REFERENCE",
    "REAL",
]
