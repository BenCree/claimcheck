"""The one place the project's IRI base is written down.

It was `https://w3id.org/provchem/two-layer#` in three modules — the name of a
different project, copied along with the code, appearing in every Croissant and
every graph this emits. Three copies of a constant are three chances for two of
them to disagree, so there is one.

**IT DOES NOT DEREFERENCE, and that is said here rather than discovered.**
`w3id.org` redirects are registered by pull request against a public repository
and nobody has filed one. Until then these IRIs identify things without
resolving to anything — which is legal, common, and the sort of fact that a
reader is entitled to be told instead of finding out with `curl`.

To make them resolve: register `claimcheck` at github.com/perma-id/w3id.org,
then delete this paragraph. Nothing else changes, because nothing else writes
the string.
"""

from __future__ import annotations

#: Terms this project mints: units, verdicts, join reports, origins.
BASE = "https://w3id.org/claimcheck/"
NS = BASE + "ns#"

#: False, and checked by `tests/test_namespace.py` rather than asserted. When
#: the redirect is registered this becomes True and the test starts requiring
#: the IRIs to actually resolve.
RESOLVES = False
