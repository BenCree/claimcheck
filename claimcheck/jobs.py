"""Why a run was launched and what would have counted as success, in PROV-O.

    python -m claimcheck.emit jobs <project-dir> [<out.ttl>]

## What this is for, and why Snakemake does not already do it

Snakemake's metadata store answers *what ran*: the rule, the shell command, the
input digests, the environment, the start and end time. `claimcheck.grading`
reads it and grades every output from that record, and nothing here restates any
of it.

Three things are not in that store and cannot be derived from it:

* **why the run was launched.** A rule name is not a question. "Does an
  acquisition that knows the pocket beat one that does not" is.
* **what would have counted as success**, declared *before* the run. Snakemake
  records the outputs a rule produced. It has no slot for the outputs somebody
  said it must produce, which is the only version that can be wrong.
* **the verdict**, and whether anybody reached one.

And for work submitted to a scheduler rather than executed by Snakemake, the
store holds nothing at all — no rule ever ran locally. That is the ordinary case
for cluster work, so the gap is not a corner.

Measured on the registry this schema was drawn from: of 190 jobs with a
lifecycle, **112 declared nothing to check**, so no check on them could fail,
and **30 failed a check that did run**. A record that cannot distinguish those
two reports 142 problems, and a list where 112 non-events sit beside 30 real
ones is a list that trains a reader to scroll.

## This is NOT a run receipt

`tests/test_dependencies.py` refuses anything that hashes a run's inputs and
writes a record of them, because Snakemake already writes exactly that. Nothing
here hashes anything. The digest of an output is Snakemake's business and
`claimcheck.grading` reads it from Snakemake; this module carries the
*intention* and the *judgement*, which no digest can hold.

## The mapping, in standard terms

| ours | the term, and it is somebody else's |
|---|---|
| the job | `prov:Activity` |
| why it was run | `dct:description` |
| when it ran | `prov:startedAtTime` / `prov:endedAtTime` |
| who ran it | `prov:wasAssociatedWith` -> `prov:SoftwareAgent` |
| the acceptance criterion | `prov:qualifiedAssociation` -> `prov:hadPlan` -> `prov:Plan` |
| what it was to produce | `dct:requires` **on the Plan** |
| what it did produce | `prov:generated` -> `prov:Entity` |
| the verdict | `cc:verdict`, from the three `claimcheck.relate` already uses |
| replaced by a later job | `dct:isReplacedBy` |

**The pair in the middle is the distinction the rest turns on.** An expected output goes on
the *plan*; only a confirmed one becomes `prov:generated`. `prov:generated`
asserts that an entity came into existence, so writing the expected list there
would have every failed job assert the existence of files that are not on disk —
publishing, in the interchange format, the exact claim the check was there to
refuse. Nothing in PROV-O says "planned to generate": `prov:Plan` is "a set of
actions or steps intended by an agent to achieve some goals", and `dct:requires`
is the closest standard term for what those goals name. Both are somebody
else's vocabulary, which is the point; the alternative was minting two terms
of our own.

## Three verdicts, not four, and they are not new here

`claimcheck.relate` already adjudicates in `agrees` / `refuted` /
`unverifiable`, and already maps them onto belief statuses (`measured` /
`falsified` / `untested`). A job's outcome is the same kind of judgement about
the same kind of evidence, so it uses the same three words and inherits the same
mapping. A scheduler-flavoured fourth vocabulary would be a second answer to a
question this package had already answered.

A superseded job is not a fourth outcome either. It is a job with no verdict and
a `dct:isReplacedBy` link, which says more than "N/A" and says it in a term a
stranger's tooling already understands.

## What is deliberately absent

Submitting, polling a scheduler, ssh, retries, and anything that decides whether
to launch. That is execution. This module reads a record somebody else wrote and
says what it means; it never runs a job and never asks a queue about one.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from claimcheck.namespace import BASE, NS
from claimcheck.relate import AGREES, STATUS

#: The verdicts a job record may carry, and the belief status each becomes.
#: Imported rather than restated — if `relate` grows a fourth, this grows one
#: too, and the two cannot drift into disagreeing about the same word.
VERDICTS: dict[str, str] = dict(STATUS)

#: Required of every record, and required for a reason each.
#:   id        — a job nobody can name cannot be found again in a scheduler's
#:               accounting, which is the only place the raw truth survives
#:   question  — the field this whole module exists for. A record without it
#:               documents that something happened, which is what the scheduler
#:               already documents.
_REQUIRED = ("id", "question")


class JobError(ValueError):
    """A job record that cannot be published as it stands."""


@dataclass(frozen=True)
class Job:
    """One submission: the question, the criterion, and the judgement."""

    id: str
    question: str
    #: The outputs the run was declared to produce, BEFORE it ran. An empty
    #: list is the state in which no check can fail.
    expected: tuple[str, ...] = ()
    #: The command that adjudicates them. Empty and `expected` empty together
    #: mean nothing was declared to check.
    check: str = ""
    #: One of `VERDICTS`, or empty for "nobody has adjudicated this yet".
    verdict: str = ""
    #: The outputs confirmed present afterwards. Only these are `prov:generated`.
    generated: tuple[str, ...] = ()
    started: str = ""
    ended: str = ""
    agent: str = ""
    superseded_by: str = ""
    #: Free text about the run that is not a verdict about it.
    note: str = ""

    @property
    def adjudicable(self) -> bool:
        """Could any check on this job have failed?

        The distinction the registry could not draw: a job with nothing
        declared is not a job that failed, and reporting it beside one that did
        is how a list of real defects stops being read.
        """
        return bool(self.expected or self.check)

    @property
    def status(self) -> str:
        """The belief status this verdict carries, or `""` if unadjudicated."""
        return VERDICTS.get(self.verdict, "")

    @property
    def missing(self) -> tuple[str, ...]:
        """Declared outputs not among the confirmed ones."""
        return tuple(p for p in self.expected if p not in set(self.generated))


def check_job(j: Job) -> None:
    """Raise unless the record can be published without misleading somebody.

    Four refusals. Each one is a state that was found in a real registry, and
    each is a state in which the record says something stronger than the
    evidence behind it.
    """
    for k in _REQUIRED:
        if not str(getattr(j, k, "")).strip():
            raise JobError(
                f"job {j.id or '<no id>'!r} has no {k!r}. A run whose purpose "
                f"is unrecorded can be reconstructed from a scheduler's "
                f"accounting as an event and never as a question — 147 of 209 "
                f"entries in the registry this came from were rebuilt that way "
                f"and none of them recovered its question")
    if j.verdict and j.verdict not in VERDICTS:
        raise JobError(
            f"job {j.id!r} carries verdict {j.verdict!r}; the vocabulary has "
            f"{len(VERDICTS)}: {', '.join(sorted(VERDICTS))}. These are "
            f"`claimcheck.relate`'s, deliberately — a scheduler-flavoured "
            f"fourth word would be a second answer to the same question")
    if j.verdict and not j.adjudicable:
        raise JobError(
            f"job {j.id!r} carries verdict {j.verdict!r} and declares neither "
            f"`expected` outputs nor a `check`, so no check on it could have "
            f"failed. Say what would have counted as success, or record no "
            f"verdict — 112 of 190 jobs in the registry this came from were in "
            f"exactly this state and were reported as needing attention")
    if j.verdict == AGREES and j.missing:
        raise JobError(
            f"job {j.id!r} passed, and {len(j.missing)} of its "
            f"{len(j.expected)} declared outputs are not among the confirmed "
            f"ones: {', '.join(j.missing[:3])}. A pass is a claim that the "
            f"criterion was met; an exit status is not that claim. 30 jobs in "
            f"the registry this came from exited 0 with their outputs absent")
    if j.generated and not j.verdict:
        # Not a refusal of the outputs — a refusal of the silence around them.
        raise JobError(
            f"job {j.id!r} confirms {len(j.generated)} output(s) present and "
            f"records no verdict. Something looked; say what it concluded, or "
            f"do not publish the looking as if it were neutral")


@dataclass
class JobLog:
    """The records, and where they were declared."""

    jobs: list[Job] = field(default_factory=list)
    #: The executor named in `[jobs]`, used where a record does not name one.
    agent: str = ""
    source: str = ""

    @classmethod
    def from_project(cls, root: Path | str) -> "JobLog":
        """`[jobs] records = "..."` in `context.toml`, then the JSONL it names."""
        root = Path(root)
        ctx = tomllib.loads((root / "context.toml").read_text())
        spec = ctx.get("jobs")
        if not spec:
            raise JobError(
                f"{root / 'context.toml'} declares no [jobs] table, so there "
                f"is no record of what was submitted or why. An empty answer "
                f"here would read as 'nothing was run'")
        rel = str(spec.get("records", "")).strip()
        if not rel:
            raise JobError("[jobs] declares no `records` file")
        path = root / rel
        if not path.is_file():
            raise JobError(
                f"[jobs] names {rel!r}, which does not exist at {path}. A "
                f"missing log is not an empty one")
        return cls.from_lines(path.read_text().splitlines(),
                             agent=str(spec.get("agent", "")), source=rel)

    @classmethod
    def from_lines(cls, lines: list[str], agent: str = "",
                   source: str = "") -> "JobLog":
        """One JSON object per line; blank lines ignored, bad ones are not.

        A JSONL log is append-only and written by whatever launched the work,
        so a truncated final line is the ordinary shape of an interrupted
        write. Skipping it silently would drop the most recent job — the one
        most likely to be the one somebody is asking about.
        """
        log = cls(agent=agent, source=source)
        seen: dict[str, int] = {}
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise JobError(
                    f"{source or 'the job log'} line {i} is not JSON ({e}). A "
                    f"log written by an interrupted run ends this way, and the "
                    f"last line is the job you are most likely asking about") \
                    from e
            j = Job(
                id=str(raw.get("id", "")).strip(),
                question=str(raw.get("question", "")).strip(),
                expected=tuple(raw.get("expected") or ()),
                check=str(raw.get("check", "") or ""),
                verdict=str(raw.get("verdict", "") or ""),
                generated=tuple(raw.get("generated") or ()),
                started=str(raw.get("started", "") or ""),
                ended=str(raw.get("ended", "") or ""),
                agent=str(raw.get("agent", "") or agent),
                superseded_by=str(raw.get("superseded_by", "") or ""),
                note=str(raw.get("note", "") or ""),
            )
            check_job(j)
            if j.id in seen:
                raise JobError(
                    f"job {j.id!r} appears at line {seen[j.id]} and line {i}. "
                    f"Two records under one id cannot both be published, and "
                    f"picking the later one silently would hide a "
                    f"contradiction between them")
            seen[j.id] = i
            log.jobs.append(j)
        return log

    # ------------------------------------------------------------- questions

    def by_verdict(self) -> dict[str, list[Job]]:
        out: dict[str, list[Job]] = {}
        for j in self.jobs:
            out.setdefault(j.verdict or "unadjudicated", []).append(j)
        return out

    def undeclared(self) -> list[Job]:
        """Jobs with nothing declared to check.

        **A backlog, not a defect.** These ran; nobody said what would have
        counted. Reported separately from failures for that reason.
        """
        return [j for j in self.jobs if not j.adjudicable]

    def summary(self) -> dict:
        by = self.by_verdict()
        return {
            "jobs": len(self.jobs),
            "by_verdict": {k: len(v) for k, v in sorted(by.items())},
            "adjudicable": sum(1 for j in self.jobs if j.adjudicable),
            # THE ONE THAT MUST NOT BE FOLDED INTO A FAILURE COUNT.
            "nothing_declared_to_check": len(self.undeclared()),
            "superseded": sum(1 for j in self.jobs if j.superseded_by),
            "questions_recorded": sum(1 for j in self.jobs if j.question),
            "verdict_statuses": dict(sorted(VERDICTS.items())),
        }

    def report(self) -> dict:
        return {
            "summary": self.summary(),
            "source": self.source,
            "jobs": [{"id": j.id, "question": j.question,
                      "expected": list(j.expected), "check": j.check,
                      "verdict": j.verdict, "status": j.status,
                      "generated": list(j.generated),
                      "missing": list(j.missing),
                      "adjudicable": j.adjudicable,
                      "started": j.started, "ended": j.ended,
                      "agent": j.agent, "superseded_by": j.superseded_by,
                      "note": j.note}
                     for j in self.jobs],
        }


# --------------------------------------------------------------------- PROV

def build(root: Path | str):
    """The job log as PROV-O. Needs the `graph` extra."""
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

    CC = Namespace(NS)
    log = JobLog.from_project(root)

    g = Graph()
    for pfx, ns in (("prov", PROV), ("dct", DCTERMS), ("cc", CC)):
        g.bind(pfx, ns)

    def when(subject, term, value: str) -> None:
        """An instant, typed, or nothing.

        NOT parsed and NOT reformatted. A timestamp this module could not
        interpret is one it must not restate in a shape it invented; it goes
        in as an untyped literal so the string a reader sees is the string the
        record held.
        """
        if not value:
            return
        looks_iso = len(value) >= 10 and value[4] == "-" and value[7] == "-"
        g.add((subject, term,
               Literal(value, datatype=XSD.dateTime) if looks_iso
               else Literal(value)))

    agents: dict[str, object] = {}
    for j in log.jobs:
        act = URIRef(f"{BASE}job/{j.id}")
        g.add((act, RDF.type, PROV.Activity))
        g.add((act, RDFS.label, Literal(j.id)))
        # THE FIELD THIS MODULE EXISTS FOR, in the slot a catalogue reads.
        g.add((act, DCTERMS.description, Literal(j.question)))
        when(act, PROV.startedAtTime, j.started)
        when(act, PROV.endedAtTime, j.ended)
        if j.note:
            g.add((act, RDFS.comment, Literal(j.note)))

        if j.agent:
            ag = agents.get(j.agent)
            if ag is None:
                ag = URIRef(f"{BASE}agent/{j.agent}")
                g.add((ag, RDF.type, PROV.SoftwareAgent))
                g.add((ag, RDFS.label, Literal(j.agent)))
                agents[j.agent] = ag
            g.add((act, PROV.wasAssociatedWith, ag))

        # The acceptance criterion, as a plan the association carries. Emitted
        # whenever anything was declared — including for a job with no verdict,
        # because "this is what it was for" is exactly what a pending job needs
        # to say.
        if j.adjudicable:
            plan = URIRef(f"{BASE}job/{j.id}/criterion")
            g.add((plan, RDF.type, PROV.Plan))
            g.add((plan, RDFS.label,
                   Literal(f"acceptance criterion for {j.id}")))
            if j.check:
                g.add((plan, DCTERMS.description, Literal(j.check)))
            for path in j.expected:
                # dct:requires, NOT prov:generated — see the module docstring.
                # These are the files the run was to produce, which is a claim
                # about intent and never about what is on disk.
                g.add((plan, DCTERMS.requires, Literal(path)))
            assoc = URIRef(f"{BASE}job/{j.id}/association")
            g.add((assoc, RDF.type, PROV.Association))
            g.add((assoc, PROV.hadPlan, plan))
            if j.agent:
                g.add((assoc, PROV.agent, agents[j.agent]))
            g.add((act, PROV.qualifiedAssociation, assoc))

        # …and only what was confirmed becomes an entity that exists.
        for path in j.generated:
            ent = URIRef(f"{BASE}entity/{path}")
            g.add((ent, RDF.type, PROV.Entity))
            g.add((ent, RDFS.label, Literal(path)))
            g.add((ent, PROV.wasGeneratedBy, act))
            g.add((act, PROV.generated, ent))

        if j.verdict:
            g.add((act, CC.verdict, Literal(j.verdict)))
            g.add((act, CC.status, Literal(j.status)))
        if j.superseded_by:
            g.add((act, DCTERMS.isReplacedBy,
                   URIRef(f"{BASE}job/{j.superseded_by}")))
    return g


def render(root: Path | str) -> str:
    """The emitter interface, shared with `croissant`, `graph` and `dcat`."""
    return str(build(root).serialize(format="turtle"))


def main(argv: list[str] | None = None) -> int:
    from claimcheck.emit import main as emit_main
    a = argv if argv is not None else sys.argv[1:]
    return emit_main(["jobs", *a])


if __name__ == "__main__":
    raise SystemExit(main())
