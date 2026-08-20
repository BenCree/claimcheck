"""The closed vocabulary: 7 statuses, 13 predicates, 7 node kinds.

**Hand-written, and imports nothing.** The version this was ported from was
generated from a LinkML schema by a code generator, with `DO NOT EDIT` at the
top and a project check that failed when the committed file stopped
reproducing. That is the right shape for a schema somebody else consumes; it is
the wrong shape here. LinkML is a build-time generator, so carrying it would
mean shipping a `schema/`, a generator script and a regeneration check — three
moving parts, to produce 27 constants that change about once a year. The
package's own `pyproject.toml` already records the decision in the other
direction: *"linkml — generated a vocabulary this package does not have"*. Now
it has one, written out, with the reasons in the file rather than in a schema
nobody reads.

## Two relations, one table, two columns

Every predicate answers two different questions, and they are NOT the same
question:

* **`stale`** — *if I change this, what has to be recomputed?*
* **`doubt`** — *if this is unsettled or refuted, whose numbers must say so?*

The table below agrees with itself on twelve of the thirteen predicates, which
is exactly why they were once collapsed into one relation. The thirteenth is
`shares_approximation_with`: two methods making the same assumption create no
staleness dependence at all — change one and the other need not be recomputed —
and total epistemic dependence, because refuting the shared assumption refutes
it on both sides at once. Answering the doubt question with the staleness
traversal made that the one edge doubt could not cross, so a refuted shared
approximation reached nobody while every check stayed green.

Agreeing twelve times is not being one relation. It is the thirteenth that a
reader needed and did not get.

## Why this is a closed table and not OWL axioms

`stale=None` and `doubt=NOWHERE` are **answers**, not omissions. Under
open-world semantics an ontology can assert that a relation propagates and
cannot assert that one does not: the absence of an axiom is silence, and five
of these thirteen predicates would become silence. A reasoner reading a graph
with no `propertyChainAxiom` for `dominates_error_of` learns nothing about
whether error dominance propagates; this table says it does not.

(The claim that *nobody* can express non-propagation would be too strong, and
was checked: PIMS-II defines `logicallyPrecedes` over chains containing no
`CognitiveRupture`, and the Gene Ontology's `IKR`/`IRD` evidence codes exist to
block propagation down a branch. It can be said in OWL. It is not said by
silence, which is the actual argument for the table.)
"""

from __future__ import annotations

# --------------------------------------------------------------------- doubt

#: The four ways doubt can sit on a claim, used in the `doubt` column below.
#:
#: Each entry decides two things, and both were once wrong:
#:
#: * the **bearer** — the end whose numbers the caveat belongs beside. For a
#:   directed entry that is the *dependent* end and only that end. Delivering
#:   to both ends handed `[falsified] surrogate --approximates--> dft` to
#:   `dft` — the reference that refutation vindicates — and to every other
#:   method that merely cites it. 18 of 50 deliveries on the graph that shipped
#:   were that shape.
#: * the **traversal** — whether doubt crosses the edge at all. `NOWHERE` and
#:   `BOTH_WAYS` differ in exactly this.
TO_SUBJECT = "to_subject"    #: the subject's standing rests on the object's
TO_OBJECT = "to_object"      #: the object's standing rests on the subject's
BOTH_WAYS = "both_ways"      #: the two ends are epistemically entangled
NOWHERE = "nowhere"          #: no epistemic dependence in either direction

#: Every value the `doubt` column may take. `belief.check_claim` refuses a
#: predicate carrying anything else, rather than letting a fourteenth predicate
#: inherit a default nobody chose.
DOUBT_KINDS: tuple[str, ...] = (TO_SUBJECT, TO_OBJECT, BOTH_WAYS, NOWHERE)


# ---------------------------------------------------------------- statuses

#: The statuses a claim may carry.
#:
#:   evidence_required — may the status be asserted with no evidence?
#:   open_question     — is there an experiment waiting to be specified?
#:   carries_caveat    — must a warning travel with any number resting on it?
#:
#: **The last two differ for exactly one status, and that is the point.**
#: `falsified` raises no open question — it was tested, the answer is known,
#: and putting a finished result on a to-do list is how a to-do list stops
#: being read — but it carries the loudest caveat there is, because a number
#: computed on a method that was proven wrong is exactly what a reader must be
#: told about. A repository that answered both questions with one boolean
#: reported an artefact resting on a falsified method as unqualified.
STATUSES: dict[str, dict] = {
    "measured": dict(
        evidence_required=True, open_question=False, carries_caveat=False,
        meaning="we measured it here, and the evidence says where",
    ),
    "published": dict(
        evidence_required=True, open_question=False, carries_caveat=False,
        meaning="someone else measured it; the evidence is a citation",
    ),
    "by_construction": dict(
        evidence_required=False, open_question=False, carries_caveat=False,
        meaning="true by how the thing is built — a definition, NOT evidence. "
                "It requires none, raises no open question and carries no "
                "caveat: counting an analytic statement as support would "
                "inflate every confidence downstream of it.",
    ),
    "assumed": dict(
        evidence_required=False, open_question=True, carries_caveat=True,
        meaning="taken as true, never tested",
    ),
    "contested": dict(
        evidence_required=True, open_question=True, carries_caveat=True,
        meaning="evidence exists on both sides",
    ),
    "falsified": dict(
        evidence_required=True, open_question=False, carries_caveat=True,
        meaning="we tested it and it is false. A RESULT, not a gap — it raises "
                "no open question, and it carries a caveat on every number "
                "resting on it.",
    ),
    "untested": dict(
        evidence_required=False, open_question=True, carries_caveat=True,
        meaning="asserted with nothing behind it. NOBODY HAS LOOKED — and no "
                "reasoner can conclude that from a graph, because absence of a "
                "challenge is not a challenge under open-world semantics. It "
                "has to be said out loud, by whoever wrote the claim.",
    ),
}

#: Statuses meaning 'there is an experiment waiting'. DERIVED, never listed —
#: a second hand-written list is a second thing to keep in step.
OPEN_QUESTION: tuple[str, ...] = tuple(
    k for k, v in STATUSES.items() if v["open_question"])

#: Statuses meaning 'a warning belongs beside any number resting on this'.
#: Includes `falsified`, which `OPEN_QUESTION` does not.
CARRIES_CAVEAT: tuple[str, ...] = tuple(
    k for k, v in STATUSES.items() if v["carries_caveat"])

#: Statuses that may not be asserted with an empty evidence field.
EVIDENCE_REQUIRED: tuple[str, ...] = tuple(
    k for k, v in STATUSES.items() if v["evidence_required"])


# --------------------------------------------------------------- node kinds

#: What a node in the belief graph can be. A closed list so that a typo is
#: refused at load rather than creating a phantom node kind; nothing in the
#: traversal branches on it, and it is not meant to.
NODE_KINDS: tuple[str, ...] = (
    "method",          #: something that computes a number
    "reference",       #: something a method is judged against
    "parameter",       #: a chosen value the numbers depend on
    "condition",       #: a state of the world the numbers assume
    "quantity",        #: a physical quantity, however it is obtained
    "representation",  #: a way of encoding the object of study
    "strategy",        #: a plan for spending a budget
)


# --------------------------------------------------------------- predicates

#: The 13 predicates, each with BOTH directions and its meaning.
#:
#:   stale  — 'backward': the subject goes stale when the object changes.
#:            'forward':  the object goes stale when the subject changes.
#:            None:       this predicate makes nothing stale.
#:   doubt  — one of DOUBT_KINDS, above.
#:
#: The `#:` note on each entry is the reason for its `doubt` value, and is the
#: part that cannot be derived from anything else in this file.
PREDICATES: dict[str, dict] = {
    #: The stand-in is what is in doubt; the physics it stands in for is not.
    #: A refuted approximation is bad news about the subject and, if anything,
    #: a vindication of the object.
    "approximates": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject is a cheaper or rougher stand-in for object's physics",
    ),
    #: Same shape — the surrogate is on trial, not the thing it replaces in
    #: the budget.
    "is_cheaper_surrogate_for": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject is run instead of object to save time",
    ),
    #: The student cannot exceed the teacher, so a doubtful ceiling is doubt
    #: about the student's numbers, not the teacher's.
    "is_distilled_from": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject learns object's labels and cannot exceed it",
    ),
    #: "subject's value changes if object changes" — the dependent end is the
    #: subject, and it is the subject's numbers that inherit the object's doubt.
    "depends_on": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject's value changes if object changes",
    ),
    #: The only FORWARD predicate, and the only place where the bearer is not
    #: the subject: "object is meaningless unless subject holds first", so it
    #: is the object whose meaning is contingent and the object that must say
    #: so.
    "is_precondition_of": dict(
        stale="forward", doubt=TO_OBJECT,
        meaning="object is meaningless unless subject holds first",
    ),
    #: An unsettled validation is unsettled about the subject; the yardstick is
    #: not implicated by what it measured.
    "is_validated_against": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject's correctness is judged by comparing it to object",
    ),
    #: The rescuer is the claimant; the thing being rescued from is not
    #: implicated by the rescue failing.
    "rescues_ranking_errors_of": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject recovers items object ranks wrongly",
    ),
    #: `estimator --preserves_ordering_of--> measured_affinity`, falsified: the
    #: estimator is wrong, and the measurements are the yardstick that showed
    #: it.
    "preserves_ordering_of": dict(
        stale="backward", doubt=TO_SUBJECT,
        meaning="subject reproduces object's ordering",
    ),
    #: **The thirteenth.** Epistemic dependence with no staleness dependence:
    #: changing one does not oblige recomputing the other, but refuting the
    #: shared assumption refutes it on both sides at once. Symmetric because
    #: the assumption has no owner — and deliberately NOT transitive, see
    #: `belief._doubt_reach`.
    "shares_approximation_with": dict(
        stale=None, doubt=BOTH_WAYS,
        meaning="both make the same assumption, so agreement between them is "
                "NOT independent evidence",
    ),
    #: A replacement does not inherit its predecessor's doubt — being wrong is
    #: usually why it was replaced. W3C PROV-CONSTRAINTS defines no inference
    #: propagating invalidation along `wasDerivedFrom`, and "a better method
    #: now exists" is not "the old numbers are wrong". Reported at both ends,
    #: travelling no further.
    "supersedes": dict(
        stale=None, doubt=NOWHERE,
        meaning="subject replaces object. Records that a replacement happened; "
                "it does not by itself make anything out of date.",
    ),
    #: "subject's conclusions are void when object holds" is a CONDITIONAL, and
    #: nothing in the model records whether the condition obtains. Propagating
    #: doubt across it would deliver a warning that is only conditionally true.
    #: Travelling further needs the condition modelled first.
    "is_invalidated_by": dict(
        stale=None, doubt=NOWHERE,
        meaning="subject's conclusions are void when object holds",
    ),
    #: A comparison of error magnitudes. Neither term's standing rests on the
    #: other's — the claim is about which is larger.
    "dominates_error_of": dict(
        stale=None, doubt=NOWHERE,
        meaning="subject contributes more error than object",
    ),
    #: Same shape: a measured comparison between two reorderings, not a
    #: dependence of either on the other.
    "displaces_ranking_more_than": dict(
        stale=None, doubt=NOWHERE,
        meaning="subject reorders the shortlist more than object",
    ),
}

#: The predicates that make something stale, and the ones that do not. Derived,
#: for the same reason `OPEN_QUESTION` is.
PROPAGATES_STALENESS: tuple[str, ...] = tuple(
    k for k, v in PREDICATES.items() if v["stale"] is not None)

def _columns_agree(v: dict) -> bool:
    """Do this predicate's two directions say the same thing?"""
    return ((v["doubt"] == TO_SUBJECT and v["stale"] == "backward")
            or (v["doubt"] == TO_OBJECT and v["stale"] == "forward")
            or (v["doubt"] == NOWHERE and v["stale"] is None))


#: The predicates where the two columns disagree — DERIVED, so that the claim
#: "twelve of thirteen agree" in the docstring above cannot go stale when
#: somebody adds a fourteenth. There is exactly one today, and it is the reason
#: the doubt column exists at all.
DISAGREE: tuple[str, ...] = tuple(
    k for k, v in PREDICATES.items() if not _columns_agree(v))
