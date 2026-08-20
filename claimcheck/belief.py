"""What else is in doubt, now that this claim has been refuted.

    python -m claimcheck.belief <project-dir> [<out.json>]

`claimcheck.relate` adjudicates each claim **on its own**: `agrees`, `refuted`,
`unverifiable`, one row at a time. That is the right unit for a check and the
wrong unit for a reader, because claims rest on each other. A refutation is
news about more than the row it lands on, and until now the only place that
consequence existed was in somebody's memory.

This module holds the part with no other home: a small graph of what rests on
what, and two questions asked of it.

    from claimcheck.belief import BeliefGraph
    g = BeliefGraph.from_project(".")
    g.caveats_on("mace")        # what belongs beside any number from this
    g.open_questions()          # what is waiting to be specified
    g.stale_if("docked_pose")   # what has to be recomputed if this changes

## Two questions, not one

`open_questions()` and `caveats_on()` look like one query and are not:

* **`open_questions()`** — what should someone go and specify? A `falsified`
  claim raises none: it was tested, the answer is known, and a finished result
  on a to-do list is how a to-do list stops being read.
* **`caveats_on()`** — what warning belongs beside a number? A `falsified`
  claim raises one, loudly.

Both derive from `vocabulary.STATUSES`, so they cannot drift from each other.

## Two *traversals*, not one either

Having refused to let the two questions share one boolean, the implementation
this was ported from then answered the caveat question using the staleness
question's traversal — and `shares_approximation_with`, the one predicate with
epistemic dependence and no staleness dependence, was exactly the edge that
traversal could not cross. A falsified shared approximation reached nobody with
every check green. So the two closures are computed separately, from the two
columns of `vocabulary.PREDICATES`.

## Why this is core and not the `graph` extra

**Stdlib only, and that is a requirement rather than an accident.** Three
reasons, in order of how much they cost when ignored:

1. The question *"what is now in doubt?"* is asked when something has gone
   wrong — often on a machine where the build failed and the optional extras
   are exactly what is missing. A checker that needs a semantic-web stack to
   answer it is a checker that does not answer it.
2. It is a fixed-point computation over about a dozen nodes. Loading an RDF
   engine to breadth-first search twelve nodes is the shape of duplication this
   package exists to refuse.
3. `tests/test_dependencies.py` runs the core behind a meta-path import ban.
   The ban is what keeps (1) true when nobody is watching.

RDF is a **rendering** of what is here, produced at the boundary in
`claimcheck.graph`, and never read back as authority. `claimcheck/graph.py`
also emits EVI's property chain so that a reader with a reasoner derives the
propagation from the same file — see the limits recorded there, which are real:
a reasoner gets a subset of these answers, and the difference is not a bug in
either.

## What no reasoner can do, and what this does instead

Under open-world semantics, absence of a challenge is not a challenge: nothing
entails *"nobody has looked at this, therefore doubt it"*. Two consequences,
both deliberate:

* **`untested` is an asserted status**, not an inference from a missing
  evidence field. Somebody has to write it down. This graph cannot warn you
  about a claim nobody declared, and it does not pretend to.
* **Non-propagation is written in the table**, because silence would not say
  it. `supersedes` and `dominates_error_of` carry doubt nowhere, and that is an
  answer rather than a gap in the axioms.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from claimcheck.vocabulary import (
    BOTH_WAYS,
    CARRIES_CAVEAT,
    DOUBT_KINDS,
    NODE_KINDS,
    OPEN_QUESTION,
    PREDICATES,
    STATUSES,
    TO_OBJECT,
    TO_SUBJECT,
)


class BeliefError(ValueError):
    """A claim the vocabulary forbids. Not a claim with a problem — not a claim."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str = ""


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    object: str
    status: str
    evidence: str = ""
    #: `"x ~ y"`, naming the relationship in `results/claims.json` whose
    #: recomputed verdict set this claim's status. Empty means a person wrote
    #: the status by hand.
    tested_by: str = ""
    #: The sibling project that OWNS this claim, when several projects' results
    #: rest on the same statement. Empty means this project owns it.
    #:
    #: THE PROBLEM IT SOLVES, MEASURED. `measured_affinity --approximates-->
    #: binding_free_energy` — that an assay number stands in for the physical
    #: binding free energy — was declared `assumed` with zero evidence in FOUR
    #: projects independently. Every RMSE against experiment in all four rests
    #: on it. Settle it in one and the other three go stale in silence, which
    #: is the failure a retired method graph had already caused once here.
    #:
    #: `defined_in = "<sibling>"` takes the status, evidence and note from that
    #: project's `context.toml`, exactly as `tested_by` takes them from a
    #: recomputed verdict — declare the authority, then read from it. The nodes
    #: it names must still be declared LOCALLY, so nothing is merged across
    #: projects and no traversal crosses a boundary; only the verdict travels.
    #:
    #: ONE HOP ONLY. A claim reached through `defined_in` may not itself use
    #: `defined_in`. A chain would need cycle detection at every load and would
    #: make "who decides this" a graph walk rather than one name.
    defined_in: str = ""
    #: A nearby fact that is NOT evidence for this claim, and the field exists
    #: because the difference is the easy one to lose.
    #:
    #: The measured case, from the graph this package inherited: `mace_eint
    #: --is_cheaper_surrogate_for--> cascade` is `untested`, and beside it sits
    #: "tier 1 is 1.7 s/compound, the cascade ~390 s, ~350x". That is a
    #: measurement about COST. It is not a measurement that one SUBSTITUTES for
    #: the other, which is what the claim says. Put it in `evidence` and the
    #: claim reads as settled; drop it and the next person re-measures it.
    #:
    #: Four claims in that graph are `untested` or `assumed`, all four carry
    #: exactly this kind of note, and none carries evidence — so folding the
    #: two fields together would have made all four illegal under the rule
    #: that `untested` may not carry evidence, by way of promoting a nearby
    #: number into a warrant. A note never affects a status and never counts
    #: as evidence; it travels with the caveat so a reader sees it.
    note: str = ""

    @property
    def recomputed(self) -> bool:
        return bool(self.tested_by)

    @property
    def open_question(self) -> bool:
        return bool(STATUSES[self.status]["open_question"])

    @property
    def carries_caveat(self) -> bool:
        return bool(STATUSES[self.status]["carries_caveat"])

    @property
    def stale_direction(self) -> str | None:
        d = PREDICATES[self.predicate]["stale"]
        return None if d is None else str(d)

    @property
    def doubt_direction(self) -> str:
        return str(PREDICATES[self.predicate]["doubt"])

    def __str__(self) -> str:
        return f"{self.subject} --{self.predicate}--> {self.object}"


def bearers(c: Claim) -> tuple[str, ...]:
    """The end(s) of a claim whose numbers this caveat belongs beside.

    For a directed predicate that is the **dependent** end and only that end.
    Delivering to both ends handed `[falsified] surrogate
    --approximates--> reference` to the reference — which the refutation
    vindicates — and to everything else citing it: 18 of 50 deliveries on the
    graph that shipped. `NOWHERE` and `BOTH_WAYS` report at both ends, because
    with no dependence declared there is no basis for preferring one.
    """
    d = c.doubt_direction
    if d == TO_SUBJECT:
        return (c.subject,)
    if d == TO_OBJECT:
        return (c.object,)
    return (c.subject, c.object)


def check_claim(c: Claim, nodes: set[str]) -> None:
    """Raise unless the claim is well formed. Six tests.

    Four of these a JSON Schema or SHACL rendering would also enforce, and two
    it cannot: **referential integrity** — no JSON Schema can say "this string
    must be a key of that map", so a claim naming `docked_pse` validates clean
    and creates a phantom node — and the `untested`-with-evidence test, which
    has to see two fields at once.

    The guard on the doubt column is here rather than at import so that a
    vocabulary which grows a predicate breaks on the first claim that uses it,
    naming the file to edit, instead of making the package unimportable for
    everybody.
    """
    if c.predicate not in PREDICATES:
        raise BeliefError(
            f"unknown predicate {c.predicate!r}; the vocabulary has "
            f"{len(PREDICATES)}: {', '.join(sorted(PREDICATES))}")
    if PREDICATES[c.predicate].get("doubt") not in DOUBT_KINDS:
        # NOT the author's fault, and the message says so. Silently defaulting
        # an unclassified predicate would answer the caveat question with a
        # semantics nobody chose.
        raise BeliefError(
            f"predicate {c.predicate!r} is in the vocabulary but its `doubt` "
            f"column is not one of {DOUBT_KINDS}, so there is no answer to "
            f"which end of it a caveat belongs beside. Classify it in "
            f"claimcheck/vocabulary.py, with the reason, and this claim will "
            f"load")
    if c.status not in STATUSES:
        raise BeliefError(
            f"unknown status {c.status!r}; the vocabulary has "
            f"{len(STATUSES)}: {', '.join(sorted(STATUSES))}")
    for role, nid in (("subject", c.subject), ("object", c.object)):
        if nid not in nodes:
            raise BeliefError(
                f"{role} {nid!r} of claim `{c}` is not a declared node. "
                f"Declare it under [[belief.nodes]], or fix the spelling")
    ev = c.evidence.strip()
    if STATUSES[c.status]["evidence_required"] and not ev:
        raise BeliefError(
            f"claim `{c}` has status {c.status!r}, which requires evidence, "
            f"and has none")
    if c.status == "untested" and ev:
        # A claim somebody settled and forgot to restate is the same failure as
        # a stale export, one field earlier.
        raise BeliefError(
            f"claim `{c}` is 'untested' and carries evidence — {ev[:60]!r}. "
            f"If somebody looked, say what they found; 'untested' means nobody "
            f"has")


def _from_owner(raw: dict, where: str, owner: str,
                siblings: Path | None) -> Claim:
    """Read one claim's verdict out of the sibling project that owns it.

    Loads the owner with `from_project`, so a claim settled there BY DATA — a
    `tested_by` recomputed from `results/claims.json` — arrives here already
    settled. That is the whole point: the statement is decided in one place and
    read everywhere a result depends on it.

    **An owning project owns all of its claims.** If the named project uses
    `defined_in` anywhere, this refuses rather than following the chain. That
    single rule gives one-hop resolution, makes cycles impossible without cycle
    detection, and keeps "who decides this" answerable by reading one name.
    """
    if siblings is None:
        raise BeliefError(
            f"{where} is defined_in {owner!r}, but this graph was built from a "
            f"mapping with no directory to resolve it against. Load with "
            f"`BeliefGraph.from_project`, or pass `siblings=`")
    ctx = siblings / owner / "context.toml"
    if not ctx.is_file():
        raise BeliefError(
            f"{where} is defined_in {owner!r}, and {ctx} does not exist. Name a "
            f"sibling project directory, not a file or a path")
    doc = tomllib.loads(ctx.read_text()).get("belief") or {}
    chained = [c for c in (doc.get("claims") or []) if c.get("defined_in")]
    if chained:
        raise BeliefError(
            f"{where} is defined_in {owner!r}, which itself uses `defined_in` "
            f"({len(chained)} claim(s)). An owning project owns all of its "
            f"claims — otherwise 'who decides this' is a graph walk rather "
            f"than one name, and two projects can point at each other")

    key = (str(raw["subject"]), str(raw["predicate"]), str(raw["object"]))
    src = BeliefGraph.from_project(siblings / owner)
    hits = [c for c in src.claims
            if (c.subject, c.predicate, c.object) == key]
    if not hits:
        raise BeliefError(
            f"{where} is defined_in {owner!r}, which asserts no claim "
            f"`{key[0]} --{key[1]}--> {key[2]}`. Importing a claim the owner "
            f"does not make would invent a status nobody wrote")
    if len({(c.status, c.evidence) for c in hits}) > 1:
        # `contradictions()` reports this legitimately WITHIN a project — a
        # claim measured once and later falsified is a history worth keeping.
        # Across projects there is no way to pick, so it is refused here.
        raise BeliefError(
            f"{where} is defined_in {owner!r}, which asserts "
            f"`{key[0]} --{key[1]}--> {key[2]}` under "
            f"{len(hits)} different statuses: "
            f"{sorted({c.status for c in hits})}. Resolve it there first")
    got = hits[0]
    return Claim(subject=key[0], predicate=key[1], object=key[2],
                 status=got.status, evidence=got.evidence,
                 tested_by=got.tested_by, defined_in=owner,
                 # The note stays LOCAL. The owner decides whether the claim
                 # holds; what it means for this project's numbers is this
                 # project's business.
                 note=str(raw.get("note", "") or ""))


@dataclass
class BeliefGraph:
    """Nodes, claims, and the map from an artefact to what it rests on."""

    #: id -> Node.
    nodes: dict[str, Node] = field(default_factory=dict)
    #: The claims, in declaration order.
    claims: list[Claim] = field(default_factory=list)
    #: The name of something you will ask about later — a dataset id, a rule
    #: name, an output path — mapped to the node(s) it rests on.
    #:
    #: **A LIST, because one artefact routinely rests on several.** Written as
    #: a single node this was fine for a dataset table and wrong for a rule: a
    #: workflow rule exercises a representation, an acquisition strategy, a
    #: potential and a threshold at once, and the map it was ported from has
    #: entries naming twelve nodes. Forced to pick one, an artefact inherits
    #: one node's doubt closure and silently none of the rest — and silence
    #: here is indistinguishable from a clean bill of health, which is the same
    #: failure `knows` exists to prevent one level up.
    #:
    #: A bare string in `context.toml` still loads, as a list of one.
    rests_on: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # ----------------------------------------------------------------- build

    @classmethod
    def from_dict(cls, doc: dict,
                  recomputed: dict[str, dict] | None = None,
                  siblings: Path | None = None) -> "BeliefGraph":
        """Build from a plain mapping — the `[belief]` table, as parsed.

        `recomputed` maps `"x ~ y"` to a relationship row out of
        `results/claims.json`. Where a claim names one in `tested_by`, the
        recomputed verdict supplies its status, so validation below sees the
        status a reader will see.
        """
        g = cls()
        for spec in doc.get("nodes") or []:
            nid = str(spec.get("id", "")).strip()
            if not nid:
                raise BeliefError(f"a [[belief.nodes]] block has no `id`: {spec}")
            kind = spec.get("kind")
            if kind not in NODE_KINDS:
                raise BeliefError(
                    f"node {nid!r}: unknown kind {kind!r}; the vocabulary has "
                    f"{len(NODE_KINDS)}: {', '.join(NODE_KINDS)}")
            if nid in g.nodes:
                raise BeliefError(f"node {nid!r} is declared twice")
            g.nodes[nid] = Node(id=nid, kind=str(kind),
                                label=str(spec.get("label", "")))

        # AN EXPLICIT KEY, not dataclass equality. `Claim` is frozen, so
        # `__eq__` covers every field and the duplicate rule would silently
        # loosen each time a field is added. Naming the tuple makes "what
        # counts as the same claim" a decision rather than a side effect of the
        # class definition.
        seen: dict[tuple, int] = {}
        for i, raw in enumerate(doc.get("claims") or []):
            c = cls._claim(raw, i, recomputed, siblings)
            check_claim(c, set(g.nodes))
            key = (c.subject, c.predicate, c.object, c.status, c.evidence)
            if key in seen:
                # A CLAIM ASSERTED TWICE IS NOT ASSERTED TWICE AS HARD, and it
                # double-counts every denominator drawn from `by_status`. Two
                # DIFFERENT statuses on one triple stay legal on purpose and
                # are reported by `contradictions()`.
                raise BeliefError(
                    f"claim `{c}` [{c.status}] is declared twice, at "
                    f"claims[{seen[key]}] and claims[{i}], identical including "
                    f"evidence. Delete one")
            seen[key] = i
            g.claims.append(c)

        for name, spec in (doc.get("rests_on") or {}).items():
            ids = [spec] if isinstance(spec, str) else list(spec)
            if not ids:
                # An empty list reads as "declared, nothing to say" and behaves
                # as "not declared at all" — `knows` would answer True and no
                # caveat could ever arrive. Say which it is.
                raise BeliefError(
                    f"rests_on[{name!r}] is empty. Name the node(s) it rests "
                    f"on, or delete the entry — an artefact declared to rest "
                    f"on nothing reports no caveats for the same reason an "
                    f"undeclared one does")
            for nid in ids:
                if nid not in g.nodes:
                    # Nothing else catches this, and it does not break the
                    # build — it silently stops doubt reaching that artefact,
                    # which is the failure where a file looks better supported
                    # than it is.
                    raise BeliefError(
                        f"rests_on[{name!r}] names {nid!r}, which is not a "
                        f"declared node")
            if len(set(ids)) != len(ids):
                raise BeliefError(
                    f"rests_on[{name!r}] names the same node twice: {ids}")
            g.rests_on[str(name)] = tuple(str(i) for i in ids)
        return g

    @staticmethod
    def _claim(raw: dict, i: int, recomputed: dict[str, dict] | None,
               siblings: Path | None = None) -> Claim:
        """One claim, with a recomputed verdict folded in where it names one."""
        where = f"claims[{i}]"
        for k in ("subject", "predicate", "object"):
            if not str(raw.get(k, "")).strip():
                raise BeliefError(f"{where} has no {k!r}: {raw}")
        owner = str(raw.get("defined_in", "") or "").strip()
        if owner:
            for k in ("status", "evidence", "tested_by"):
                if k in raw:
                    raise BeliefError(
                        f"{where} declares both `defined_in` and {k!r}. The "
                        f"owning project decides this claim; a second answer "
                        f"beside it is one a reader will believe. Delete the "
                        f"`{k}` line, or drop `defined_in` and own it here")
            return _from_owner(raw, where, owner, siblings)
        tested = str(raw.get("tested_by", "") or "").strip()
        if not tested:
            if "status" not in raw:
                raise BeliefError(
                    f"{where} has no `status` and no `tested_by`. Either say "
                    f"what you believe, or name the relationship that settles "
                    f"it")
            return Claim(subject=str(raw["subject"]),
                         predicate=str(raw["predicate"]),
                         object=str(raw["object"]),
                         status=str(raw["status"]),
                         evidence=str(raw.get("evidence", "") or ""),
                         note=str(raw.get("note", "") or ""))
        if "status" in raw:
            # A STATUS THAT IS ABOUT TO BE OVERWRITTEN IS STILL READ. Somebody
            # opens `context.toml`, sees `status = "measured"`, and believes it
            # — while the data says otherwise and the report says otherwise.
            raise BeliefError(
                f"{where} declares both `tested_by` and `status`. The verdict "
                f"recomputed from the data sets the status; a hand-written one "
                f"beside it is a second answer that will be read. Delete the "
                f"`status` line")
        if "evidence" in raw:
            raise BeliefError(
                f"{where} declares both `tested_by` and `evidence`. The "
                f"evidence for a recomputed claim is the recomputation — it is "
                f"filled in from `results/claims.json`")
        # `note` IS allowed here, and the asymmetry is the point. Evidence is
        # what the recomputation supplies and a hand-written second answer
        # beside it would be read; a note is a caveat about the claim that no
        # recomputation produces or contradicts.
        if recomputed is None:
            raise BeliefError(
                f"{where} is tested by {tested!r}, but no recomputed verdicts "
                f"were supplied. Load with `BeliefGraph.from_project`, which "
                f"reads `results/claims.json`, or pass `recomputed=`")
        if f"__ambiguous__{tested}" in recomputed:
            n = recomputed[f"__ambiguous__{tested}"]["__n"]
            raise BeliefError(
                f"{where} is tested by {tested!r}, and {n} relationships in "
                f"`results/claims.json` are over that pair of columns — most "
                f"often the same claim at two resampling units, which is worth "
                f"declaring and cannot be named this way. Give each an `id` in "
                f"`[[relationships]]` and put the id in `tested_by`")
        r = recomputed.get(tested)
        if r is None:
            known = ", ".join(sorted(k for k in recomputed
                                     if not k.startswith("__ambiguous__"))) or "none"
            raise BeliefError(
                f"{where} is tested by {tested!r}, which names no relationship "
                f"in `results/claims.json`. Known: {known}")
        status = str(r["status"])
        if status not in STATUSES:
            raise BeliefError(
                f"{where}: the recomputed verdict carries status {status!r}, "
                f"which is not in the vocabulary")
        # `unverifiable` arrives as `untested`, which must not carry evidence:
        # the checker's reason is a fact about the check, not about the claim.
        # It is reachable through `tested_by` and the row in claims.json.
        evidence = str(r.get("why", "")) if STATUSES[status]["evidence_required"] else ""
        return Claim(subject=str(raw["subject"]), predicate=str(raw["predicate"]),
                     object=str(raw["object"]), status=status,
                     evidence=evidence, tested_by=tested,
                     note=str(raw.get("note", "") or ""))

    @classmethod
    def from_project(cls, root: Path | str) -> "BeliefGraph":
        """`context.toml` for the model, `results/claims.json` for the verdicts.

        The join between the two layers of this package: `relate` recomputes
        each claimed relationship and already reports a vocabulary status per
        row (`measured` / `falsified` / `untested`). Here those statuses become
        the statuses of belief claims, and doubt travels from them.
        """
        root = Path(root)
        ctx = tomllib.loads((root / "context.toml").read_text())
        doc = ctx.get("belief")
        if not doc:
            raise BeliefError(
                f"{root / 'context.toml'} declares no [belief] table, so there "
                f"is nothing to say about what rests on what. An empty answer "
                f"here would read as a clean bill of health")
        # RESOLVED BEFORE TAKING THE PARENT. `Path(".").parent` is `.`, not the
        # directory above — so a project loaded as `claimcheck.belief .` from
        # inside its own directory, which is exactly how every project's own
        # Snakefile invokes it, looked for its siblings inside ITSELF. The
        # in-process load passed and the Snakemake rule failed.
        return cls.from_dict(doc, recomputed=cls._recomputed(root, doc),
                             siblings=root.resolve().parent)

    @staticmethod
    def _recomputed(root: Path, doc: dict) -> dict[str, dict] | None:
        """`"x ~ y"` -> the relationship row, out of `results/claims.json`."""
        # A claim delegating to an owner never reads THIS project's verdicts,
        # so it must not make this project demand a verdict file. Without the
        # `defined_in` guard, a project whose only `tested_by` sits inside an
        # imported claim was refused for a missing file it would never open —
        # and refused with the wrong message, before the clearer refusal for
        # declaring both fields could fire.
        if not any(c.get("tested_by") and not c.get("defined_in")
                   for c in (doc.get("claims") or [])):
            return None
        path = root / "results" / "claims.json"
        if not path.is_file():
            # THE REMEDY SENTENCE HAS TO BE TRUE, and the first version's was
            # not. It said "run `claimcheck.relate` first", which is impossible
            # for the ordinary case that produces this error: a project whose
            # claims are pre-registered and whose tables are still empty.
            # `claimcheck.datasets` refuses a table with a header and no rows,
            # so `relate` cannot run either, and the reader following that
            # advice met a second and less clear error. Found on a real
            # pre-registered project with 4 of 5 tables at zero rows.
            raise BeliefError(
                f"claims declare `tested_by`, so their statuses come from "
                f"{path}, which does not exist — reporting the declared "
                f"statuses instead would publish beliefs nobody has checked. "
                f"If the data is there, run `claimcheck.relate`. If the "
                f"experiment has not produced rows yet, that is the honest "
                f"state and there is nothing to run: say so where a reader "
                f"will see it, and leave the claims pre-registered")
        return BeliefGraph._index(json.loads(path.read_text()))

    @staticmethod
    def _index(doc: dict) -> dict[str, dict]:
        """Verdicts, keyed by relationship id and by `"x ~ y"` where unique."""
        out: dict[str, dict] = {}
        ambiguous: dict[str, int] = {}
        for r in doc["relationships"]:
            rid = str(r.get("id", "") or "")
            if rid:
                if rid in out:
                    raise BeliefError(
                        f"two relationships share the id {rid!r}. "
                        f"An id exists to name one of them")
                out[rid] = r
            # The pair is ALSO a key, so every context written before ids
            # existed keeps working. Where two relationships share a pair the
            # key is withdrawn rather than pointing at one of them: a belief
            # attached to whichever came first is a belief nobody chose.
            key = f"{r['x']} ~ {r['y']}"
            if key in out or key in ambiguous:
                out.pop(key, None)
                ambiguous[key] = ambiguous.get(key, 1) + 1
            else:
                out[key] = r
        for key, n in ambiguous.items():
            # Not raised here. A project may legitimately declare a pair twice
            # and test neither of them, or test them by id — and refusing the
            # whole load for a pair nobody names would stop a correct project
            # building. `_claim` raises if a claim actually reaches for it.
            out[f"__ambiguous__{key}"] = {"__n": n}
        return out

    # --------------------------------------------------------------- closure

    @staticmethod
    def _reach(nid: str, succ: dict[str, set[str]]) -> set[str]:
        """Breadth-first from `nid`, never returning `nid`.

        A node is not stale because of itself, and a cycle back to the start
        must not smuggle it in — self-loops, mutual pairs and mixed 3-cycles
        all terminate here for the same reason: `seen` only grows and `nid` is
        never put into it.
        """
        empty: set[str] = set()
        seen: set[str] = set()
        frontier = {nid}
        while frontier:
            nxt: set[str] = set()
            for f in frontier:
                nxt |= succ.get(f, empty)
            frontier = nxt - seen - {nid}
            seen |= frontier
        return seen

    def _staleness_index(self) -> dict[str, set[str]]:
        """`node -> the nodes one hop of staleness downstream of it`.

        Built fresh rather than cached: `BeliefGraph` is a mutable dataclass
        whose `claims` are appended to in the wild, and a cached closure that no
        longer describes the claims in front of it would be a stale export
        inside the tool that exists to catch stale exports. One pass is O(E),
        and the callers below build it once and reuse it.
        """
        succ: dict[str, set[str]] = {}
        for c in self.claims:
            d = c.stale_direction
            if d == "backward":
                succ.setdefault(c.object, set()).add(c.subject)
            elif d == "forward":
                succ.setdefault(c.subject, set()).add(c.object)
        return succ

    def stale_if(self, nid: str) -> list[str]:
        """Change `nid`, and everything here has to be recomputed."""
        return sorted(self._reach(nid, self._staleness_index()))

    def stale_table(self) -> list[tuple[str, str]]:
        """Every `(changed, goes_stale)` pair. The published closure.

        Built off ONE index. Calling `stale_if` per node rescans every claim
        for every frontier node — O(V².E), measured at 8.0x per doubling on a
        synthetic pipeline: 0.005 s at 50 nodes, 251.8 s at 2000. This is
        O(V.(V+E)), which is output-optimal because the pair list is itself
        Θ(V²). Same sizes, same machine, identical output: 0.0005 s, 1.15 s.
        """
        succ = self._staleness_index()
        return sorted((n, s) for n in self.nodes for s in self._reach(n, succ))

    # -------------------------------------------------------- the two questions

    def open_questions(self) -> list[dict]:
        """Claims with an experiment waiting to be specified.

        `falsified` is deliberately absent. It was tested, the answer is known,
        and there is nothing left to do.

        **`note` is here because this list is a work queue, and not everything
        on it is workable.** The measured case: five claims about a force-field
        line came back `unverifiable` — the correlation was undefined because
        the treatment arm had no rows — and the reason was a MEASURED claim two
        blocks above saying that line cannot type a single ligand in the
        corpus. "Nobody has looked" and "nobody can look here" are the same
        word in a closed seven-term vocabulary, and the difference decides
        whether an entry is a next experiment or a dead end. Without this
        field the queue read as five runnable jobs.
        """
        return [
            {"claim": str(c), "status": c.status,
             "subject": c.subject, "predicate": c.predicate,
             "object": c.object,
             "predicate_means": PREDICATES[c.predicate]["meaning"],
             "why_open": STATUSES[c.status]["meaning"],
             "note": c.note,
             "recomputed_from": c.tested_by,
             "defined_in": c.defined_in}
            for c in self.claims if c.open_question
        ]

    def _doubt_index(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Two maps, because doubt travels two different ways.

        * **rests-on** (`TO_SUBJECT` / `TO_OBJECT`) — transitive. If I rest on
          it and it rests on that, that is mine too.
        * **peers** (`BOTH_WAYS`) — symmetric and deliberately **not**
          transitive, kept in a separate map so the closure can say so. `a` and
          `b` share one assumption, `b` and `c` share another; that gives `a`
          and `c` nothing in common, and closing over the relation would
          silently assert that it did — merging every peer into one equivalence
          class.
        """
        rests: dict[str, set[str]] = {}
        peers: dict[str, set[str]] = {}
        for c in self.claims:
            d = c.doubt_direction
            if d == TO_SUBJECT:
                rests.setdefault(c.subject, set()).add(c.object)
            elif d == TO_OBJECT:
                rests.setdefault(c.object, set()).add(c.subject)
            elif d == BOTH_WAYS:
                peers.setdefault(c.subject, set()).add(c.object)
                peers.setdefault(c.object, set()).add(c.subject)
        return rests, peers

    @classmethod
    def _doubt_reach(cls, nid: str, rests: dict[str, set[str]],
                     peers: dict[str, set[str]]) -> set[str]:
        """Everything whose doubt lands on `nid`.

        Close transitively over rests-on, then cross a shared assumption
        **once**. What a peer shares is one assumption, not a dependency tree,
        so crossing carries doubt about the peer itself and not about the
        peer's whole provenance. The three readings of that edge — peers only,
        peers then continue upward, and full transitive closure — deliver 41,
        53 and 59 caveats on the graph they were measured against. The
        narrowest is the one that reaches the shared approximation without
        burying it in warnings that are only conditionally true.
        """
        empty: set[str] = set()
        base = {nid} | cls._reach(nid, rests)
        return (base | {p for n in base for p in peers.get(n, empty)}) - {nid}

    def supports_edges(self) -> list[tuple[str, str]]:
        """`(a, b)`: a's standing supports b's. ONE HOP of the rests-on
        relation, for renderers that have their own closure.

        **Peers are deliberately absent.** `shares_approximation_with` is
        symmetric and not transitive, so rendering it as an edge of a
        transitive property would assert something this model does not: that
        `a`'s peer's peer is `a`'s problem. `claimcheck.graph` records the
        consequence — a reasoner sees a subset of `caveats_on`, and the missing
        part is exactly this edge.
        """
        rests, _ = self._doubt_index()
        return sorted((a, b) for b, deps in rests.items() for a in deps)

    def in_doubt(self, nid: str) -> list[str]:
        """Every node whose standing `nid` depends on. The doubt closure.

        A DIFFERENT relation from `stale_if` run backwards, and the difference
        is the whole reason the doubt column exists: `shares_approximation_with`
        appears here and in no staleness index anywhere.
        """
        return sorted(self._doubt_reach(nid, *self._doubt_index()))

    def knows(self, name: str) -> bool:
        """Is this artefact joined to the belief graph at all?

        `caveats_on` returns `[]` both for *"not joined"* and for *"joined,
        nothing to say"*, and a caller that cannot tell them apart prints a
        clean bill of health for a name that does not exist.
        """
        return name in self.rests_on

    def caveats_on(self, name: str) -> list[dict]:
        """Warnings that belong beside any number in this artefact.

        Includes `falsified`, which `open_questions()` excludes — that
        difference is why there are two methods. Each claim is delivered to its
        **bearer** (see `bearers`), so a refuted approximation reaches the
        surrogate and not the reference it was compared against.
        """
        return self._caveats_on(name, self._doubt_index())

    def _caveats_on(self, name: str,
                    doubt: tuple[dict[str, set[str]], dict[str, set[str]]]
                    ) -> list[dict]:
        roots = self.rests_on.get(name)
        if roots is None:
            return []
        if isinstance(roots, str):
            # CAUGHT LOUDLY, because it was found silently. `from_dict` always
            # stores a tuple, but the field is a plain dict and code that
            # assigns into it directly — a test, a caller extending the map —
            # can put a bare string there. A string iterates as CHARACTERS, so
            # every membership test fails and this returns "no caveats" for an
            # artefact that has them. Measured while making this change: a
            # summary went from 4 caveat deliveries to 2, and passed.
            raise BeliefError(
                f"rests_on[{name!r}] is the string {roots!r}, not a sequence "
                f"of node ids. A string iterates as characters, so no caveat "
                f"can ever match and this artefact would report clean. Use a "
                f"tuple: ({roots!r},)")
        # Per root, so the caveat can say WHICH declared node carried it here.
        # A union alone answers "does this artefact inherit it" and not "why",
        # and "why" is the first thing a reader asks of a warning.
        reach = {r: {r} | self._doubt_reach(r, *doubt) for r in roots}
        out = []
        for c in self.claims:
            if not c.carries_caveat:
                continue
            via = sorted(r for r, seen in reach.items()
                         if any(b in seen for b in bearers(c)))
            if not via:
                continue
            out.append({
                # A LIST EVEN WHEN IT HOLDS ONE, so a consumer never has to
                # branch on the type of a field to read it.
                "artefact": name, "rests_on": via, "claim": str(c),
                # STRUCTURED, NOT JUST PRINTABLE. The report this replaces
                # reconstructed these by splitting the display string on " --"
                # and "--> ". Every suggestion it made rested on that split, so
                # all of them went wrong, silently, the moment anybody
                # touched `__str__`.
                "subject": c.subject, "predicate": c.predicate,
                "object": c.object, "status": c.status,
                # THE ONLY FIELD THAT SAYS WHY ANYONE BELIEVES IT. A warning a
                # reader cannot follow to its evidence is a warning they will
                # discount.
                "evidence": c.evidence,
                # Separate from evidence, and separately labelled, so that a
                # nearby number cannot be read as the warrant. See `Claim.note`.
                "note": c.note,
                "recomputed_from": c.tested_by,
                "open_question": c.open_question,
                "why": str(STATUSES[c.status]["meaning"]),
            })
        return sorted(out, key=lambda d: (d["status"], d["claim"]))

    def all_caveats(self) -> list[dict]:
        # One index for every artefact, not one per artefact.
        doubt = self._doubt_index()
        return [d for name in sorted(self.rests_on)
                for d in self._caveats_on(name, doubt)]

    def contradictions(self) -> list[tuple[str, str, str, list[str]]]:
        """Triples asserted under more than one status.

        NOT an error, and that was decided rather than overlooked: a claim
        measured once and later falsified is a history worth publishing. Exact
        duplicates are refused at load; this is the other case, counted and
        named so it can be looked at. What it leaves open is which assertion is
        current — `supersedes` relates nodes, not claims.
        """
        by_triple: dict[tuple[str, str, str], set[str]] = {}
        for c in self.claims:
            by_triple.setdefault((c.subject, c.predicate, c.object),
                                 set()).add(c.status)
        return [(s, p, o, sorted(st))
                for (s, p, o), st in sorted(by_triple.items()) if len(st) > 1]

    # --------------------------------------------------------------- summary

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for c in self.claims:
            by_status[c.status] = by_status.get(c.status, 0) + 1
        caveats = self.all_caveats()
        return {
            "nodes": len(self.nodes),
            "claims": len(self.claims),
            "artefacts": len(self.rests_on),
            "recomputed": sum(1 for c in self.claims if c.recomputed),
            "by_status": dict(sorted(by_status.items())),
            "stale_pairs": len(self.stale_table()),
            "open_questions": len(self.open_questions()),
            # DISTINCT CLAIMS, not deliveries. One row per (artefact, claim)
            # pair means four artefacts resting on one node reported `4
            # caveat(s)` two lines under `1 claim(s)`.
            "caveats": len({d["claim"] for d in caveats}),
            "caveat_deliveries": len(caveats),
            # THE UNDELIVERABLE ONES. A graph with claims that carry caveats
            # and nothing declared to rest on them printed `0 caveat(s)` on the
            # same line as the word `falsified`.
            "caveats_undeliverable": (
                sum(1 for c in self.claims if c.carries_caveat)
                if not self.rests_on else 0),
            "contradictory_triples": len(self.contradictions()),
            "vocabulary": {
                "predicates": len(PREDICATES),
                "statuses": len(STATUSES),
                "node_kinds": len(NODE_KINDS),
                "open_question_statuses": list(OPEN_QUESTION),
                "caveat_statuses": list(CARRIES_CAVEAT),
            },
        }

    def report(self) -> dict:
        """Everything a consumer needs, in one JSON-serialisable mapping."""
        return {
            "summary": self.summary(),
            "nodes": [{"id": n.id, "kind": n.kind, "label": n.label}
                      for n in self.nodes.values()],
            "claims": [{"claim": str(c), "subject": c.subject,
                        "predicate": c.predicate, "object": c.object,
                        "status": c.status, "evidence": c.evidence,
                        "note": c.note,
                        "recomputed_from": c.tested_by} for c in self.claims],
            "rests_on": {k: list(v) for k, v in sorted(self.rests_on.items())},
            "open_questions": self.open_questions(),
            "caveats": self.all_caveats(),
            "contradictions": [{"subject": s, "predicate": p, "object": o,
                                "statuses": st}
                               for s, p, o, st in self.contradictions()],
        }


def render(root: Path | str) -> str:
    """The emitter interface, shared with `croissant`, `graph` and `dcat`."""
    g = BeliefGraph.from_project(root)
    return json.dumps(g.report(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else root / "results/belief.json"
    g = BeliefGraph.from_project(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g.report(), indent=2) + "\n")
    for name in sorted(g.rests_on):
        rows = g.caveats_on(name)
        head = f"  {name} (rests on {', '.join(g.rests_on[name])})"
        print(f"{head}: {len(rows)} caveat(s)" if rows
              else f"{head}: nothing in doubt")
        for d in rows:
            print(f"      [{d['status']}] {d['claim']}")
            print(f"                {d['why']}")
            if d["recomputed_from"]:
                print(f"                recomputed from {d['recomputed_from']}")
            # LABELLED, not merged into the line above. A note is the thing
            # that reads like evidence and is not — printing it unlabelled
            # beside the claim is the confusion the field exists to stop.
            if d["note"]:
                print(f"                note (not evidence): {d['note']}")
    for q in g.open_questions():
        print(f"  OPEN  [{q['status']}] {q['claim']}")
    s = g.summary()
    print(f"  {s['claims']} claim(s) over {s['nodes']} node(s), "
          f"{s['caveat_deliveries']} delivery(ies), "
          f"{s['open_questions']} open question(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
