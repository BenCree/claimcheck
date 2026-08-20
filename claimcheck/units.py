"""Units — an unknown conversion raises. It never returns the input unchanged.

    from claimcheck.units import convert, compatible
    convert(1.0, "kcal/mol", "kJ/mol")     # 4.184
    convert(1.0, "eV", "kcal/mol")         # UnknownUnit

## Why this is here and not left to `pint`

`claimcheck` already asks every declared variable for a unit, and refuses a
claim that touches an undeclared one. That is presence, not agreement: before
this module, a relationship between a column in kcal/mol and a column in kJ/mol
was simply correlated, and correlation is invariant under a scale factor, so the
verdict came out identical and the mistake was invisible.

The failure this prevents is measured, not hypothetical: one interaction energy
recorded in eV in one file and kcal/mol in another, a factor of 23.06 apart,
each internally consistent and neither wrong on its own terms.

A full units library would do more, and `pint` is the obvious candidate. It is
not a dependency here because the whole requirement is a dozen symbols and one
refusal, and `claimcheck`'s rule is that a dependency arrives with a recorded
reason. If the table ever needs dimensional analysis rather than a lookup, that
is the reason, and `pint` should be adopted then.

## Two conventions worth stating

`None` for a unit means **not declared**, never dimensionless. Conflating them is
how a tolerance gets set against the wrong scale; `"1"` is the dimensionless
declaration, written explicitly.

**eV to kcal/mol is deliberately absent.** It needs Avogadro's number, so it is
a change of quantity — per particle to per mole — and not a conversion. Doing it
silently is exactly the confusion above.

Ported 2026-08-17 from a predecessor package that has since been retired
and archived. What survived the port is what could not be got from
Snakemake, `mlcroissant` or `rdflib`; everything else was dropped rather
than carried.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownUnit(KeyError):
    """A conversion this module will not guess at."""


@dataclass(frozen=True)
class Unit:
    symbol: str
    quantity: str
    note: str = ""


#: Deliberately small. A table that grows by guessing is a table that will one
#: day guess wrong; each entry is added when an experiment needs it.
UNITS: dict[str, Unit] = {
    "1": Unit("1", "dimensionless", "declared dimensionless, not undeclared"),
    "g/mol": Unit("g/mol", "molar mass"),
    "Da": Unit("Da", "molar mass", "numerically equal to g/mol"),
    "kcal/mol": Unit("kcal/mol", "molar energy"),
    "kJ/mol": Unit("kJ/mol", "molar energy"),
    "eV": Unit("eV", "energy", "per particle, NOT per mole"),
    "log10(1/M)": Unit("log10(1/M)", "affinity", "higher = stronger binding"),
    # Length, added for trajectory work: RMSD, radius of gyration, contact
    # distances. UDUNITS-2 has no `angstrom` token in its base set, so
    # `0.1 nm` is the portable spelling of the same quantity and both are
    # accepted rather than one being silently rewritten to the other.
    "nm": Unit("nm", "length"),
    "0.1 nm": Unit("0.1 nm", "length", "an angstrom, spelled portably"),
    "ps": Unit("ps", "time"),
    "ns": Unit("ns", "time"),
}

#: Only conversions somebody has checked.
_FACTORS: dict[tuple[str, str], float] = {
    ("g/mol", "Da"): 1.0,
    ("Da", "g/mol"): 1.0,
    ("kcal/mol", "kJ/mol"): 4.184,
    ("kJ/mol", "kcal/mol"): 1.0 / 4.184,
    ("nm", "0.1 nm"): 10.0,
    ("0.1 nm", "nm"): 0.1,
    ("ns", "ps"): 1000.0,
    ("ps", "ns"): 0.001,
}


def convert(value: float, frm: str, to: str) -> float:
    """`value` from one declared unit to another, or raise. Never a no-op."""
    if frm not in UNITS:
        raise UnknownUnit(f"unit {frm!r} is not declared in claimcheck.units")
    if to not in UNITS:
        raise UnknownUnit(f"unit {to!r} is not declared in claimcheck.units")
    if frm == to:
        return value
    if UNITS[frm].quantity != UNITS[to].quantity:
        raise UnknownUnit(
            f"{frm!r} is a {UNITS[frm].quantity} and {to!r} is a "
            f"{UNITS[to].quantity}; that is a change of quantity, not a "
            f"conversion, and this module will not do it silently")
    try:
        return value * _FACTORS[(frm, to)]
    except KeyError:
        raise UnknownUnit(
            f"no checked factor for {frm!r} -> {to!r}. Add one with its source "
            f"rather than letting this return the input unchanged") from None


def compatible(a: str | None, b: str | None) -> tuple[bool, str]:
    """Could a relationship between two columns in these units mean anything?

    Returns `(ok, why)`. Used by `relate` to say something more useful than
    "units present" — two energies in different units are relatable *after* a
    conversion, an energy against a length is not relatable at all, and an
    undeclared unit is neither.
    """
    if a is None or b is None:
        which = "x" if a is None else "y"
        return False, (f"no unit declared for {which} — a relationship between "
                       f"undeclared quantities cannot be compared with anyone "
                       f"else's")
    for u in (a, b):
        if u not in UNITS:
            return False, (f"unit {u!r} is not declared in claimcheck.units; add "
                           f"it there rather than assuming what it means")
    if a == b:
        return True, f"both {a}"
    qa, qb = UNITS[a].quantity, UNITS[b].quantity
    if qa != qb:
        # NOT an error. A correlation between an energy and a length is a
        # perfectly ordinary thing to ask for, and refusing it would be wrong;
        # what matters is that the reader is told the two are different
        # quantities so a "slope" is never read as a conversion factor.
        return True, (f"{a} is a {qa} and {b} is a {qb} — different quantities, "
                      f"so any slope between them carries units and is not a "
                      f"conversion factor")
    if (a, b) not in _FACTORS:
        return False, (f"{a} and {b} are both {qa} but no checked factor "
                       f"relates them; add one to claimcheck.units")
    return True, f"{a} and {b} are both {qa}, factor {_FACTORS[(a, b)]:g}"
