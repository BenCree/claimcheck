"""EVI's two axioms, restated so the emitted graph is self-contained.

    python -m claimcheck.emit evi <project-dir> [<out.ttl>]

`claimcheck.graph` emits the belief layer with **EVI's own verbs** —
`evi:supports` and `evi:directlyChallenges` — rather than private predicates,
so that a consumer with a reasoner derives the consequences of a refutation
without being told how to traverse anything. The two axioms that make that
work are:

    evi:supports              a  owl:TransitiveProperty .
    evi:indirectlyChallenges  owl:propertyChainAxiom ( evi:directlyChallenges
                                                       evi:supports ) .

They are **EVI's, not ours.** Micropublications axiomatised the chain in OWL in
2014 and EVI carries it over verbatim. `claimcheck.belief` reimplements it as a
breadth-first search in about twenty lines, and that reimplementation is
defensible — it runs with no reasoner in the broken environment the whole
design is for — but it is a reimplementation, not a gap being filled, and this
file is where that is said out loud.

## Why they are a separate artefact from the data graph

An ontology is not a dataset. `claimcheck.graph` emits statements *about this
project* — every subject an IRI this project minted, no blank nodes, so that
two runs can be diffed. These axioms are statements about somebody else's
vocabulary, and expressing a property chain in RDF needs a list, which needs
blank nodes. Mixing them would have meant loosening the guard that keeps the
data graph diffable, to hold four triples that do not change between runs.

So the data graph carries `owl:imports <https://w3id.org/EVI>` and a reasoner
that follows imports needs nothing from here. This file is for the reasoner
that does not — an air-gapped run, or a triple store with network access
disabled — and for the reader who wants the propagation rule in front of them
without fetching an ontology.

**A restatement is only safe while it is true.** If EVI changes these axioms,
this file is wrong in the specific way this package exists to catch, so it says
which version it was taken from and `tests/test_belief.py` pins the chain to
exactly those two properties in that order.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

#: The ontology, and the version these axioms were read from.
EVI = Namespace("https://w3id.org/EVI#")
EVI_ONTOLOGY = URIRef("https://w3id.org/EVI")
EVI_VERSION = "1.6"

#: The chain, as data, so that the test and the emitter cannot disagree about
#: what was restated. Order matters: `( directlyChallenges supports )` is not
#: the same axiom as `( supports directlyChallenges )`.
CHAIN: tuple[URIRef, ...] = (EVI.directlyChallenges, EVI.supports)

#: The property declared transitive. Without it the chain reaches one hop.
TRANSITIVE: URIRef = EVI.supports


def axioms() -> Graph:
    """The two axioms, plus a comment saying whose they are."""
    g = Graph()
    g.bind("evi", EVI)
    g.bind("owl", OWL)

    g.add((TRANSITIVE, RDF.type, OWL.ObjectProperty))
    g.add((TRANSITIVE, RDF.type, OWL.TransitiveProperty))

    g.add((EVI.indirectlyChallenges, RDF.type, OWL.ObjectProperty))
    lst = BNode()
    Collection(g, lst, list(CHAIN))
    g.add((EVI.indirectlyChallenges, OWL.propertyChainAxiom, lst))

    g.add((EVI_ONTOLOGY, RDFS.comment, Literal(
        f"Two axioms of EVI {EVI_VERSION} restated verbatim so that a graph "
        f"emitted by claimcheck propagates a refutation under a reasoner that "
        f"does not resolve {EVI_ONTOLOGY}. Nothing here is claimcheck's own; "
        f"see claimcheck/evi.py.")))
    return g


def render(root: Path | str) -> str:
    """The emitter interface. The axioms do not depend on the project — that is
    what makes them axioms — so `root` is accepted and ignored."""
    del root
    return axioms().serialize(format="turtle")


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    text = render(Path.cwd())
    if a:
        out = Path(a[0])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"  {out}  —  EVI {EVI_VERSION}, {len(axioms())} triples")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
