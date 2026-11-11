"""PSDI Guidance, *Sharing Data and FAIR*.

The Physical Sciences Data Infrastructure's guidance on publishing research
data: https://guidance.psdi.ac.uk/docusaurus-pages/docs/category/sharing-data-and-fair/

Not a LiveCoMS article, and the only module here that is not. It earns its place
for the same reason the others do: `GOALS.md` rule 21 promoted *generating
shareable data* to an objective alongside actives-per-wall-clock-hour, and an
objective with no published standard behind it is an opinion. PSDI is the UK
physical-sciences infrastructure, it is the body that operates **BioSimDB** (the
repository our trajectories would go to), and its guidance is explicit enough to
execute. `SHARING.md` is the prose companion; this file is the part that runs.

WHAT THE GUIDANCE ACTUALLY REQUIRES
-----------------------------------
From `.../fair/package/`, a data package must carry provenance information, a
README, **a data dictionary** ("Variable definitions, measurement units, and
codes used"), dataset metadata, chemical annotations ("Use of InChIs and
InChIKeys"), data-quality documentation, citations, and the code that generated
the data. From `.../fair/license/`, licence information must travel *inside* the
package as a separate file, preferably machine-readable, with the **SPDX License
List** named as the reference; upstream obligations must be checked. From
`.../fair/repository/`, persistent identifiers (DOIs). From `.../fair/steps/`,
standard machine-readable formats and comprehensive metadata.

WHY THESE CHECKS AND NOT OTHERS
-------------------------------
Every check below **measures a file in this repo**. None asks the operator to
confirm anything, because a checklist that asks questions is answered by whoever
wants it answered a particular way.

The severities are calibrated against what actually blocks sharing, not against
how much we would like the practice:

* **must** — the artefact is unusable or unredistributable without it. A shared
  structure with no protonation sidecar cannot be interpreted (the PDB format
  cannot express CYM/HIP). A trajectory with no seed cannot be reproduced, and
  we have the measurement: two runs at *identical* seeds diverged to **2.13 A**
  on ligand heavy atoms by the first saved frame because `DeterministicForces`
  was off; with it on the same pair reproduces to **0.000000 A** at every frame.
  A generated dataset whose licence field reads "internal" cannot be given to
  anybody at all.
* **should** — real FAIR gaps that do not stop a competent reuser. Missing
  RO-Crate, missing InChIs, thin column-dictionary coverage.
* **consider** — house-style consistency.

WHAT THIS MODULE DELIBERATELY DOES NOT ENCODE
---------------------------------------------
* **Whether a deposit was accepted, or whether a DOI resolves.** Both need the
  network; checks here are cheap and offline by contract.
* **Whether BioSimDB's UK-affiliation membership rule applies to us.** That is a
  fact about a person, not about a file. `SHARING.md` section 3.1 flags it.
* **Whether a units string is *true*.** `fair-02` can prove a units statement
  exists; it cannot prove the numbers are in those units. That is what
  `GOALS.md` rule 4's known-answer controls are for, and they live elsewhere.
  This distinction matters here, because the sharpest units bug in this repo is
  *not* a missing declaration: `results/interaction_energies.csv` has a column
  named `e_int` in **eV** and `results/score_comparison.csv` has a column named
  `e_int` in **kcal/mol** (factor 23.0605). No metadata check catches two files
  that each describe themselves correctly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bestpractices import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE, NA

PAPER = Paper(
    slug="fair_sharing",
    title="PSDI Guidance: Sharing Data and FAIR",
    authors=("Physical Sciences Data Infrastructure (PSDI); pages by "
             "Cerys Willoughby and Louise Saul"),
    url="https://guidance.psdi.ac.uk/docusaurus-pages/docs/category/sharing-data-and-fair/",
    year="2026",
    scope=("Publishing physical-sciences research data so somebody else can "
           "actually reuse it: the data package and its README, the data "
           "dictionary, licences and upstream obligations, persistent "
           "identifiers, and choosing a repository."),
)

# --------------------------------------------------------------------------
# Paths this module measures.
# --------------------------------------------------------------------------
CROISSANT = "datasets/croissant/*.json"
PROVENANCE = "datasets/provenance/*.md"
CHECKSUMS = "datasets/checksums/*.sha256"
LINEAGE = "datasets/lineage.json"
COLUMN_DICT = "results/phase2_features/column_dictionary.csv"
MD_RUNNER = "md_arms.py"

#: Croissant records describing data **we generate**, and would therefore be the
#: licensor of. The upstream records are somebody else's licence to state, and
#: recording "UNVERIFIED" for one of those is honest bookkeeping rather than a
#: defect -- that case is `fair-10`, at `should`.
OURS = ("al_mace_generated", "qm_scoping_pockets")

#: Directories of generated artefacts that GOALS.md rule 21 commits us to
#: sharing. Kept here rather than inline so adding an artefact updates every
#: check that should see it.
SHAREABLE = (
    "data/protonated",
    "data/protonated_propka",
    "data/protonated_ionpair",
    "results/posebusters",
    "results/crossdock493",
    "results/dock_rescore_full",
)

#: A subset of the SPDX License List (https://spdx.org/licenses/) sufficient for
#: everything this project could plausibly emit. Offline by contract, so the list
#: is embedded; it is a *whitelist of identifiers*, and anything outside it is
#: reported by name rather than silently failed, so a legitimate new licence
#: shows up as a nameable gap instead of a mystery.
SPDX = {
    "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0", "CC-BY-ND-4.0",
    "CC-BY-NC-SA-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "BSD-2-Clause",
    "BSD-3-Clause", "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only",
    "MPL-2.0", "ODbL-1.0", "PDDL-1.0", "Unlicense",
}

#: The canonical licence URLs, mapped back onto their SPDX identifier. A URL is
#: machine-readable and the guidance accepts it, but the SPDX id is what the
#: licence page actually names, so a URL counts as a pass *with a nudge*.
URL_TO_SPDX = {
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "https://spdx.org/licenses/CC-BY-4.0.html": "CC-BY-4.0",
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _croissant(ctx: Context) -> list[tuple[str, dict]]:
    """(stem, parsed record) for every Croissant file, skipping unparseable
    ones so a syntax error surfaces as a named gap rather than a traceback."""
    out = []
    for p in ctx.glob(CROISSANT):
        try:
            out.append((p.stem, json.loads(p.read_text())))
        except Exception:                                    # noqa: BLE001
            out.append((p.stem, {}))
    return out


def _spdx_verdict(value) -> tuple[bool, str]:
    """Is this licence field a usable, machine-readable licence?

    Returns (ok, normalised-or-reason). Deliberately strict about prose: the
    whole point of the SPDX list is that "see the Zenodo record" is not a
    licence, it is a promise to look one up later.
    """
    if not isinstance(value, str) or not value.strip():
        return False, "absent"
    v = value.strip()
    if v in SPDX:
        return True, v
    if v.rstrip("/") + "/" in URL_TO_SPDX or v in URL_TO_SPDX:
        return True, URL_TO_SPDX.get(v, URL_TO_SPDX.get(v.rstrip("/") + "/", v))
    # a bare identifier in the wrong case is still findable; prose is not
    for s in SPDX:
        if v.lower() == s.lower():
            return True, s
    return False, v[:60]


def _receipt_source(ctx: Context) -> str:
    return ctx.read(MD_RUNNER)


def _lineage(ctx: Context) -> dict | None:
    raw = ctx.read(LINEAGE)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:                                        # noqa: BLE001
        return None


# ==========================================================================
# must -- an artefact fails these and it cannot be shared, or cannot be used
# ==========================================================================

def _generated_licence_is_spdx(ctx: Context) -> Result:
    """`.../fair/license/`: the package must carry licence terms, and the page
    names the SPDX License List as the reference for common licences.

    Scoped to data **we** generate. We are the licensor there, so a placeholder
    is not bookkeeping, it is an artefact nobody may legally reuse. FAILS when a
    generated record's licence is prose, 'internal', or absent.
    """
    recs = _croissant(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      "no datasets/croissant/*.json found; testable as soon as "
                      "`python datasets/build_croissant.py` has been run",
                      evidence=CROISSANT)
    seen, bad = [], []
    for stem, d in recs:
        if stem not in OURS:
            continue
        ok, why = _spdx_verdict(d.get("license"))
        seen.append(stem)
        if not ok:
            bad.append(f"{stem}={why!r}")
    if not seen:
        return Result(UNTESTABLE,
                      f"none of {OURS} present among {len(recs)} Croissant "
                      f"records, so no generated dataset was examined",
                      evidence=", ".join(s for s, _ in recs))
    if bad:
        return Result(FAIL,
                      f"{len(bad)}/{len(seen)} generated dataset(s) carry no "
                      f"usable SPDX licence: {'; '.join(bad)}. SHARING.md 3.5 "
                      f"establishes the answer is CC0-1.0 -- OpenBind (Zenodo "
                      f"20026661) is cc-zero and the PDB is CC0-1.0, so no "
                      f"upstream condition flows down.",
                      evidence=f"license field of datasets/croissant/{{{','.join(seen)}}}.json")
    return Result(PASS,
                  f"all {len(seen)} generated dataset(s) carry an SPDX licence",
                  evidence=", ".join(seen))


def _provenance_states_units(ctx: Context) -> Result:
    """`.../fair/package/`: the data dictionary must give "measurement units".

    We extend it to the sign convention, because this project has **shipped** a
    sign inversion -- `benchmark.py` correlated an orientation-corrected score
    against dG instead of pKD and read -0.213 for +0.213 (`FINDINGS` 3.4) -- and
    caught two more. FAILS when a provenance file omits the section.
    """
    files = ctx.glob(PROVENANCE)
    if not files:
        return Result(UNTESTABLE,
                      "no datasets/provenance/*.md; testable once provenance "
                      "records exist", evidence=PROVENANCE)
    pat = re.compile(r"^#{1,3}\s+.*units and sign convention", re.I | re.M)
    missing = [p.name for p in files if not pat.search(p.read_text(errors="replace"))]
    if missing:
        return Result(FAIL,
                      f"{len(missing)}/{len(files)} provenance records have no "
                      f"'Units and sign conventions' section: "
                      f"{', '.join(sorted(missing))}. The house style states "
                      f"them even for datasets we do not hold -- wang_jacs.md "
                      f"and plinder_hiqbind_casf.md use "
                      f"'## Units and sign conventions (expected)'.",
                      evidence="heading scan of " + PROVENANCE)
    return Result(PASS,
                  f"all {len(files)} provenance records state units and sign "
                  f"convention",
                  evidence=", ".join(p.name for p in files))


def _structures_have_sidecars(ctx: Context) -> Result:
    """`.../fair/package/`: "so that users can understand your data".

    Our protonated PDBs carry **standard** residue names on purpose, because
    ApoDock's `get_clean_res_list` silently deletes non-canonical ones. The
    CYM/HIP/HIE/HID states the PDB format cannot express therefore live in a
    JSON sidecar. **A structure shipped without its sidecar is uninterpretable**
    -- you cannot tell a neutral cysteine from a thiolate. FAILS on any orphan.
    """
    checked, orphans = 0, []
    for d in SHAREABLE:
        root = ctx.root / d
        if not root.is_dir():
            continue
        pdbs = sorted(root.glob("*_protonated.pdb"))
        if not pdbs:
            continue
        for p in pdbs:
            checked += 1
            side = p.with_name(p.name.replace("_protonated.pdb",
                                              "_protonation.json"))
            if not side.exists():
                orphans.append(str(side.relative_to(ctx.root)))
    if not checked:
        return Result(UNTESTABLE,
                      "no *_protonated.pdb found under " + ", ".join(SHAREABLE)
                      + "; testable once protonate.py has written an arm",
                      evidence="glob over SHAREABLE")
    if orphans:
        return Result(FAIL,
                      f"{len(orphans)} of {checked} protonated structures have "
                      f"no protonation sidecar, e.g. {orphans[0]}. The PDB "
                      f"cannot express CYM/HIP, so these are uninterpretable.",
                      evidence=f"{len(orphans)} missing *_protonation.json")
    return Result(PASS,
                  f"all {checked} protonated structures are paired with a "
                  f"*_protonation.json sidecar carrying the states the PDB "
                  f"format cannot hold",
                  evidence=f"{checked} PDB/JSON pairs across "
                           f"{sum(1 for d in SHAREABLE if (ctx.root / d).is_dir())} arms")


#: The seven things without which a deposited trajectory is not reusable. The
#: values are the receipt keys `md_arms.py` writes (alternatives, because the
#: emitter changed mid-project); the labels are what a reuser would call them.
#: BioSimDB's own `dsmd` schema has **no slot at all** for seed, integrator or
#: constraints (SHARING.md 3.1), which is why the receipt is deposited as a
#: file rather than transcribed into the repository's metadata form.
TRAJ_FIELDS = {
    "topology": ("system_sha256", "n_atoms"),
    "force field": ("protein_ff", "ligand_ff", "water_model", "ligand_ff_parent"),
    "integrator/thermostat": ("thermostat", "integrator"),
    "barostat/ensemble": ("barostat", "ensemble"),
    "timestep": ("timestep_fs",),
    "seed": ("seed", "replicate_seeds"),
    "determinism flag": ("deterministic_forces",),
}


def _traj_metadata_for(traj: Path, root: Path) -> set[str]:
    """Every metadata key available to one trajectory file.

    The deposit unit is the directory holding the `.dcd`, and the metadata that
    travels with it is the union of its own `receipt.json`, any `receipt.json`
    in an ancestor directory (a replicate inherits its arm's), and the nearest
    `build.json` (which carries the topology and force fields, written once per
    system and shared by both arms). Reading only the leaf receipt would fail a
    trajectory that is in fact fully documented, which is the opposite of what
    this check is for.
    """
    keys: set[str] = set()
    d = traj.parent
    while True:
        for name in ("receipt.json", "build.json"):
            p = d / name
            if p.is_file():
                try:
                    keys |= set(json.loads(p.read_text()))
                except Exception:                            # noqa: BLE001
                    pass
        if d == root or d.parent == d or root not in d.parents:
            break
        d = d.parent
    return keys


def _trajectory_metadata(ctx: Context) -> Result:
    """`.../fair/steps/`: "Provide comprehensive metadata documentation".

    A trajectory without its topology, force field, integrator, thermostat,
    timestep and **random seed** is not reusable, and the cost is measured
    rather than asserted: two runs of the same system at the **same seed** with
    the **same force field** diverged to **2.13 A** on ligand heavy atoms by the
    first saved frame and 3.02 A by 200 ps, because `DeterministicForces` was
    off. With it on the same pair reproduces to **0.000000 A** at every frame.

    Anchored on **trajectory files**, not on receipts, because the question is
    whether a deposited trajectory is reusable -- a receipt with nothing beside
    it is not at risk, and a `.dcd` with no receipt is. Falls back to the
    emitter when no trajectory exists yet, because retrofitting a lost seed onto
    a finished trajectory is impossible, so the emitter has to be right *before*
    the run (`GOALS.md` rule 21).
    """
    trajs = [p for p in ctx.root.glob("results/md_pilot/**/*.dcd")]
    if trajs:
        bad = []
        for t in trajs:
            keys = _traj_metadata_for(t, ctx.root)
            missing = [lab for lab, alts in TRAJ_FIELDS.items()
                       if not (set(alts) & keys)]
            if missing:
                bad.append((t, missing))
        if bad:
            t, missing = max(bad, key=lambda x: len(x[1]))
            fields = sorted({m for _, ms in bad for m in ms})
            return Result(FAIL,
                          f"{len(bad)} of {len(trajs)} trajectories on disk "
                          f"cannot be reproduced from the metadata beside "
                          f"them; missing across all of them: "
                          f"{', '.join(fields)}. Worst is "
                          f"{t.parent.relative_to(ctx.root)} "
                          f"(omits {', '.join(missing)}). These were written "
                          f"by an earlier md_arms.py that did not record the "
                          f"full state point -- re-emit the receipts before "
                          f"depositing.",
                          evidence=str(t.parent.relative_to(ctx.root)
                                       ) + "/receipt.json + build.json")
        return Result(PASS,
                      f"all {len(trajs)} trajectories on disk carry topology, "
                      f"force field, integrator, thermostat, timestep, seed "
                      f"and the determinism flag in receipt.json + build.json",
                      evidence=f"{len(trajs)} *.dcd under results/md_pilot/")

    src = _receipt_source(ctx)
    if not src:
        return Result(UNTESTABLE,
                      f"no trajectory receipts on disk and {MD_RUNNER} absent; "
                      f"testable once either exists",
                      evidence=MD_RUNNER)
    missing = [lab for lab, alts in TRAJ_FIELDS.items()
               if not any(f'"{a}"' in src for a in alts)]
    if missing:
        return Result(FAIL,
                      f"no receipts written yet, and the emitter {MD_RUNNER} "
                      f"does not record {', '.join(missing)}. Capture it at "
                      f"launch: a lost seed cannot be recovered afterwards.",
                      evidence=f"receipt dict in {MD_RUNNER}")
    return Result(PASS,
                  f"no receipts on disk yet, but {MD_RUNNER} records all "
                  f"{len(TRAJ_FIELDS)} reusability fields, so trajectories will "
                  f"be depositable when the run lands. NOTE: BioSimDB's dsmd "
                  f"schema has no slot for seed, integrator or constraints, so "
                  f"the receipt must be deposited as a file (SHARING.md 3.1).",
                  evidence=f"{MD_RUNNER} receipt keys: "
                           + ", ".join(sorted(a for alts in TRAJ_FIELDS.values()
                                              for a in alts if f'\"{a}\"' in src)))


def _checksums_present_and_attributed(ctx: Context) -> Result:
    """`.../fair/package/`: data-quality documentation, and `.../fair/repository/`
    on data integrity (TRUST: Technology).

    Two things at once, because they fail together: does every artefact we have
    committed to sharing have a sha256 manifest, and does every manifest say
    **which host and when**? A manifest with no host is unusable here, because
    `/home` is not shared between paros and thira and the same relative path is
    a different file on each.
    """
    mans = ctx.glob(CHECKSUMS)
    if not mans:
        return Result(FAIL,
                      "no datasets/checksums/*.sha256 at all; nothing shared "
                      "from this repo can be integrity-checked by a reuser",
                      evidence=CHECKSUMS)
    # (a) headers
    headerless = []
    for m in mans:
        head = "\n".join(m.read_text(errors="replace").splitlines()[:8])
        if not (re.search(r"^#.*host=", head, re.M)
                and re.search(r"^#.*taken=", head, re.M)):
            headerless.append(m.name)
    # (b) coverage of the shareable artefacts
    covered = "\n".join(m.read_text(errors="replace") for m in mans)
    uncovered = [d for d in SHAREABLE
                 if (ctx.root / d).is_dir() and Path(d).name not in covered
                 and d not in covered]
    problems = []
    if headerless:
        problems.append(f"{len(headerless)} manifest(s) lack a host/taken "
                        f"header: {', '.join(headerless)}")
    if uncovered:
        problems.append(f"{len(uncovered)} shareable artefact(s) have no "
                        f"manifest entry: {', '.join(uncovered)}")
    if problems:
        return Result(FAIL, "; ".join(problems) + ". The protonated structures "
                            "are the most valuable thing here and a reuser "
                            "cannot verify what they downloaded.",
                      evidence=f"{len(mans)} manifests, "
                               f"{sum(1 for d in SHAREABLE if (ctx.root / d).is_dir())} "
                               f"shareable dirs on disk")
    return Result(PASS,
                  f"{len(mans)} checksum manifests, all carrying host and "
                  f"timestamp headers, covering every shareable artefact",
                  evidence=", ".join(m.name for m in mans))


# ==========================================================================
# should -- genuine FAIR gaps that do not stop a competent reuser
# ==========================================================================

def _column_dictionary_declares_units(ctx: Context) -> Result:
    """`.../fair/package/`: "Data dictionary: Variable definitions, measurement
    units, and codes used."

    Two failure modes, and this project has both: a dictionary that omits units,
    and a dictionary that covers a fraction of the CSVs. Reported together
    because a reuser hitting either one is equally stuck.
    """
    raw = ctx.read(COLUMN_DICT)
    csvs = list(ctx.results.rglob("*.csv")) if ctx.results.is_dir() else []
    if not raw:
        return Result(FAIL,
                      f"no column dictionary at {COLUMN_DICT}, against "
                      f"{len(csvs)} result CSVs. Every shared CSV needs "
                      f"variable definitions and units.",
                      evidence=COLUMN_DICT)
    header = raw.splitlines()[0].split(",") if raw.splitlines() else []
    n_rows = max(0, len(raw.splitlines()) - 1)
    has_units = "units" in [h.strip() for h in header]
    if not has_units:
        return Result(FAIL,
                      f"{COLUMN_DICT} documents {n_rows} columns but has no "
                      f"'units' column; header is {header[:8]}",
                      evidence=COLUMN_DICT)
    # coverage: one dictionary against the whole results tree
    if len(csvs) > 3 * n_rows or len(csvs) > 50:
        return Result(FAIL,
                      f"{COLUMN_DICT} declares units for {n_rows} columns of "
                      f"pose_matrix/compound_matrix, but it is the ONLY column "
                      f"dictionary in the repo and there are {len(csvs)} CSVs "
                      f"under results/. Everything outside those two matrices "
                      f"ships undocumented.",
                      evidence=f"1 dictionary ({n_rows} columns) vs "
                               f"{len(csvs)} CSVs")
    return Result(PASS,
                  f"column dictionary documents {n_rows} columns with units "
                  f"against {len(csvs)} result CSVs",
                  evidence=COLUMN_DICT)


def _repo_carries_a_licence_file(ctx: Context) -> Result:
    """`.../fair/license/`: the guidance "recommends including a separate licence
    file within data packages to ensure accessibility after download".

    A licence recorded only in a metadata field does not survive someone
    downloading a tarball.
    """
    names = ("LICENSE", "LICENCE", "LICENSE.md", "LICENCE.md", "LICENSE.txt",
             "COPYING")
    found = [n for n in names if ctx.exists(n)]
    if found:
        return Result(PASS, f"repository root carries {found[0]}",
                      evidence=found[0])
    return Result(FAIL,
                  "no LICENSE/LICENCE/COPYING file at the repository root, so "
                  "there is nothing to copy into a data package. SHARING.md 3.5 "
                  "establishes CC0-1.0 for everything derived from OpenBind "
                  "(cc-zero) and the PDB (CC0-1.0); the PL-REX-derived tables "
                  "are CC-BY-4.0 and need their own notice.",
                  evidence="ls LICENSE LICENCE COPYING -> none")


def _chemical_annotation(ctx: Context) -> Result:
    """`.../fair/package/`: "Use of InChIs and InChIKeys to describe the
    chemical entities in your research", and `.../fair/steps/`: "Employ InChI &
    SMILES for chemical structures".

    We ship SMILES. SMILES is not canonical across toolkits; the InChIKey is
    what makes a compound joinable to PubChem, ChEMBL and PChProp (277,117
    molecules, keyed on InChIKey).
    """
    hits = []
    for pat in (CROISSANT, PROVENANCE):
        for p in ctx.glob(pat):
            if re.search(r"inchi", p.read_text(errors="replace"), re.I):
                hits.append(p.name)
    if hits:
        return Result(PASS,
                      f"InChI/InChIKey referenced in {len(hits)} metadata "
                      f"record(s)", evidence=", ".join(sorted(hits)[:6]))
    return Result(FAIL,
                  "no InChI or InChIKey anywhere in datasets/croissant/ or "
                  "datasets/provenance/. We key compounds on SMILES and an "
                  "OpenBind compound_group, neither of which joins to an "
                  "external database. Adding an InChIKey column to "
                  "benchmark_metadata.csv is a one-off RDKit pass.",
                  evidence="case-insensitive 'inchi' scan of "
                           f"{CROISSANT} and {PROVENANCE}")


def _ro_crate_manifest(ctx: Context) -> Result:
    """`.../fair/package/`: "An RO-Crate is a Research Object (or RO) formed of
    a collection of data (a crate) and a special `ro-crate-metadata.json` file."

    Measured, not assumed: **all six** records in PSDI's own *AI Ready Datasets*
    community ship `ro-crate-metadata.json` *and* a Croissant file *and* a
    README. That is the house pattern we would be depositing into.
    """
    found = ctx.glob("**/ro-crate-metadata.json")
    if found:
        return Result(PASS,
                      f"{len(found)} RO-Crate manifest(s) present",
                      evidence=", ".join(str(p.relative_to(ctx.root))
                                         for p in found[:5]))
    n_croissant = len(ctx.glob(CROISSANT))
    return Result(FAIL,
                  f"{n_croissant} Croissant records and no ro-crate-metadata.json. "
                  f"PSDI's AI-Ready deposits carry README.md + "
                  f"ro-crate-metadata.json + croissant*.json together (6/6 "
                  f"records, measured 2026-08-02); we have one of the three.",
                  evidence="glob **/ro-crate-metadata.json -> 0")


def _upstream_licences_verified(ctx: Context) -> Result:
    """`.../fair/license/`: "Compatibility with upstream licence obligations".

    Redistributing a derivative of someone else's data requires knowing their
    terms. Distinct from `fair-01`: this is about datasets we **consume**, where
    an honest "UNVERIFIED" is correct bookkeeping but still blocks redistribution.
    """
    recs = _croissant(ctx)
    if not recs:
        return Result(UNTESTABLE, "no Croissant records to inspect",
                      evidence=CROISSANT)
    unverified = []
    for stem, d in recs:
        if stem in OURS:
            continue
        lic = str(d.get("license", ""))
        if not lic:
            unverified.append(f"{stem}=<absent>")
        elif re.search(r"unverified|not read|not downloaded|confirm before",
                       lic, re.I):
            unverified.append(stem)
    if unverified:
        return Result(FAIL,
                      f"{len(unverified)} upstream dataset(s) have an "
                      f"unverified licence: {', '.join(unverified)}. Two are "
                      f"now settled and the records should be updated -- "
                      f"OpenBind Zenodo 20026661 is 'cc-zero' (CC0-1.0), read "
                      f"from the Zenodo API 2026-08-02. The real blocker is "
                      f"elsewhere: /home/ben/code/csbrt-gcmc-fep carries NO "
                      f"LICENSE file, and absence of a licence is not "
                      f"permission (SHARING.md 3.5.3).",
                      evidence="license fields of the non-generated Croissant "
                               "records")
    return Result(PASS,
                  f"all {len(recs) - len(OURS)} upstream licences verified",
                  evidence=CROISSANT)


def _lineage_records_derivation(ctx: Context) -> Result:
    """`.../fair/package/`: "Provenance information: Creator details, dates, and
    project context", and "Processed data: Include analysed materials with
    explanations of derivation from original sources".

    The question a reuser asks first is *where did this come from and can I
    regenerate it*. Croissant cannot express that -- it describes one dataset's
    files and columns and has no `wasDerivedFrom` (verified: `isBasedOn` appears
    in none of our nine records). `datasets/lineage.json` carries it as PROV-O.

    FAILS when a generated entity has no parent, no generating activity, or an
    activity with neither a seed nor a stated reason it has none.
    """
    doc = _lineage(ctx)
    if doc is None:
        return Result(FAIL,
                      f"no parseable {LINEAGE}. Nothing records what our "
                      f"generated datasets were derived from, by which script, "
                      f"at which seed -- so none of them can be regenerated.",
                      evidence=LINEAGE)
    graph = doc.get("@graph", [])
    ents = [n for n in graph if "prov:Entity" in (n.get("@type") or [])
            or n.get("@type") == "prov:Entity"]
    acts = [n for n in graph if n.get("@type") == "prov:Activity"]
    if not ents or not acts:
        return Result(FAIL,
                      f"{LINEAGE} holds {len(ents)} entities and {len(acts)} "
                      f"activities; a lineage graph needs both",
                      evidence=LINEAGE)
    generated = [e for e in ents if e.get("wasGeneratedBy")
                 or e.get("wasDerivedFrom")]
    orphans = [e["@id"] for e in generated
               if not (e.get("wasDerivedFrom") and e.get("wasGeneratedBy"))
               and e.get("wasDerivedFrom")]
    seedless = [a["@id"] for a in acts
                if "seed" not in a and "almace:seed_reason" not in a]
    # an activity may legitimately have no seed, but must say so
    nulls = [a["@id"] for a in acts
             if a.get("seed") is None and "almace:seed_reason" not in a
             and "seed" in a]
    problems = []
    if orphans:
        problems.append(f"{len(orphans)} entity(ies) derived from a parent but "
                        f"with no generating activity: {', '.join(orphans[:3])}")
    if seedless or nulls:
        bad = sorted(set(seedless) | set(nulls))
        problems.append(f"{len(bad)} activity(ies) record neither a seed nor a "
                        f"reason there is none: {', '.join(bad[:3])}")
    if problems:
        return Result(FAIL, "; ".join(problems), evidence=LINEAGE)
    vocab = doc.get("almace:vocabulary", {})
    typed = sum(1 for e in ents if e.get("datatype"))
    untyped = [e["@id"] for e in ents if not e.get("datatype")]
    if untyped:
        return Result(FAIL,
                      f"{len(untyped)} entity(ies) carry no datatype from the "
                      f"controlled vocabulary: {', '.join(untyped[:3])}",
                      evidence=LINEAGE)
    return Result(PASS,
                  f"{len(ents)} entities ({typed} typed against a "
                  f"{max(0, len(vocab) - 1)}-term vocabulary) and {len(acts)} "
                  f"activities; every generated entity names its parents and "
                  f"its script, and every activity states a seed or why it has "
                  f"none",
                  evidence=LINEAGE)


def _croissant_states_conventions(ctx: Context) -> Result:
    """`.../fair/package/`: the data dictionary, in machine-readable form.

    Our Croissant dialect carries a `units_and_conventions` RecordSet giving
    quantity, units and direction for every column a reuser could get backwards.
    """
    recs = _croissant(ctx)
    if not recs:
        return Result(UNTESTABLE, "no Croissant records to inspect",
                      evidence=CROISSANT)
    missing = [stem for stem, d in recs
               if "units_and_conventions" not in
               [r.get("@id") for r in d.get("recordSet", [])]]
    if missing:
        return Result(FAIL,
                      f"{len(missing)}/{len(recs)} Croissant records have no "
                      f"units_and_conventions RecordSet: {', '.join(missing)}",
                      evidence=CROISSANT)
    return Result(PASS,
                  f"all {len(recs)} Croissant records carry a machine-readable "
                  f"units_and_conventions RecordSet giving quantity, units and "
                  f"direction",
                  evidence=CROISSANT)


# ==========================================================================
# consider -- house style, and one that genuinely does not apply
# ==========================================================================

def _croissant_filename_convention(ctx: Context) -> Result:
    """`.../fair/package/`: "Use a consistent file naming convention".

    PSDI's own AI-Ready deposits are inconsistent here -- `croissant.json`,
    `croissant_metadata.json`, `croissant-metadata.json` and
    `MLTaskTemplate_croissantMetadata.json` all appear across six records -- so
    this is house style, not a standard. `croissant.json` matches their PChProp
    export, which is the closest analogue to what we would deposit.
    """
    recs = ctx.glob(CROISSANT)
    if not recs:
        return Result(UNTESTABLE, "no Croissant records", evidence=CROISSANT)
    return Result(PASS,
                  f"{len(recs)} records under datasets/croissant/ named "
                  f"<dataset>.json, one per dataset. On deposit each becomes "
                  f"croissant.json inside its own package, matching PSDI's "
                  f"PChProp export.",
                  evidence=", ".join(p.name for p in recs[:4]) + ", ...")


def _pchprop_deposit(ctx: Context) -> Result:
    """`.../fair/repository/`: "Discipline-specific repositories" first.

    PChProp is the PSDI discipline repository for molecular property data, and
    it is the one Ben asked about by name. Recorded as N/A with the measurement
    that settles it, rather than omitted -- a repository we examined and ruled
    out is a finding, and leaving it out would let someone re-propose it.
    """
    return Result(NA,
                  "PChProp accepts one row per MOLECULE keyed on InChIKey, with "
                  "a closed six-property vocabulary (boiling_point, "
                  "melting_point, solubility, henrys_law_constant, "
                  "critical_micelle_concentration, miscibility) plus 13 RDKit "
                  "2-D descriptors. Every crystallographic field is null across "
                  "its 277,117 entries. It has no protein, no receptor axis and "
                  "no conformer axis, so our MACE descriptors (per pose), E_int "
                  "(protein-ligand) and strain (per conformer) have nowhere to "
                  "go. Measured from its OPTIMADE endpoint 2026-08-02.",
                  evidence="https://index.psdi.ac.uk/optimade/pchprop/v1/structures"
                           " -- _psdi_properties has 6 keys, all bulk-phase")


CHECKS = [
    # --- must: without these the artefact is unusable or unshareable ------
    Check(id="fair-01", severity="must",
          section="guidance/fair/license/ -- 'Selecting a Licence'; SPDX License List",
          question="Does every dataset we GENERATE carry a machine-readable "
                   "SPDX licence identifier?",
          measures="license field of datasets/croissant/{al_mace_generated,"
                   "qm_scoping_pockets}.json against the SPDX list",
          fn=_generated_licence_is_spdx),
    Check(id="fair-02", severity="must",
          section="guidance/fair/package/ -- 'Data dictionary: Variable "
                  "definitions, measurement units, and codes used'",
          question="Does every provenance record state units AND the sign "
                   "convention?",
          measures="heading scan of datasets/provenance/*.md",
          fn=_provenance_states_units),
    Check(id="fair-03", severity="must",
          section="guidance/fair/package/ -- 'Documentation and Metadata'",
          question="Is every shared structure file paired with the metadata "
                   "needed to interpret it?",
          measures="*_protonated.pdb vs *_protonation.json across "
                   "data/protonated{,_propka,_ionpair}/",
          fn=_structures_have_sidecars),
    Check(id="fair-04", severity="must",
          section="guidance/fair/steps/ -- 'Provide comprehensive metadata "
                  "documentation'",
          question="Do trajectories record topology, force field, integrator, "
                   "thermostat, timestep, seed and the determinism flag?",
          measures="results/md_pilot/**/receipt.json, else the receipt dict in "
                   "md_arms.py",
          fn=_trajectory_metadata),
    Check(id="fair-05", severity="must",
          section="guidance/fair/repository/ -- TRUST: Technology; "
                  "guidance/fair/package/ -- data-quality documentation",
          question="Are checksums present for every shareable artefact, and "
                   "does every manifest say which host and when?",
          measures="datasets/checksums/*.sha256 headers and coverage of the "
                   "SHAREABLE directories",
          fn=_checksums_present_and_attributed),

    # --- should: real gaps, not blockers ---------------------------------
    Check(id="fair-06", severity="should",
          section="guidance/fair/package/ -- 'Data dictionary'",
          question="Do our result CSVs have a column dictionary that declares "
                   "units, and does it cover them?",
          measures="results/phase2_features/column_dictionary.csv vs the CSV "
                   "count under results/",
          fn=_column_dictionary_declares_units),
    Check(id="fair-07", severity="should",
          section="guidance/fair/license/ -- 'include a separate licence file "
                  "within data packages'",
          question="Is there a licence file that would travel inside a data "
                   "package?",
          measures="LICENSE/LICENCE/COPYING at the repository root",
          fn=_repo_carries_a_licence_file),
    Check(id="fair-08", severity="should",
          section="guidance/fair/package/ -- 'Use of InChIs and InChIKeys'; "
                  "guidance/fair/steps/ -- 'Employ InChI & SMILES'",
          question="Are chemical entities annotated with InChI/InChIKey, not "
                   "only SMILES?",
          measures="case-insensitive 'inchi' scan of datasets/croissant/*.json "
                   "and datasets/provenance/*.md",
          fn=_chemical_annotation),
    Check(id="fair-09", severity="should",
          section="guidance/fair/package/ -- 'Research Objects' / RO-Crate",
          question="Does a data package manifest exist in the form PSDI's own "
                   "repositories use?",
          measures="glob **/ro-crate-metadata.json",
          fn=_ro_crate_manifest),
    Check(id="fair-10", severity="should",
          section="guidance/fair/license/ -- 'Compatibility with upstream "
                  "licence obligations'",
          question="Is the upstream licence verified for every dataset we "
                   "consume and might redistribute a derivative of?",
          measures="license field of the non-generated Croissant records",
          fn=_upstream_licences_verified),
    Check(id="fair-11", severity="should",
          section="guidance/fair/package/ -- 'Provenance information'; "
                  "'Processed data ... explanations of derivation'",
          question="Does every generated dataset record what it was derived "
                   "from, by which script, and at what seed?",
          measures="datasets/lineage.json (PROV-O: wasDerivedFrom, "
                   "wasGeneratedBy, seed)",
          fn=_lineage_records_derivation),
    Check(id="fair-12", severity="should",
          section="guidance/fair/package/ -- machine-readable data dictionary",
          question="Does every Croissant record carry a machine-readable "
                   "units-and-conventions table?",
          measures="units_and_conventions RecordSet in datasets/croissant/*.json",
          fn=_croissant_states_conventions),

    # --- consider ---------------------------------------------------------
    Check(id="fair-13", severity="consider",
          section="guidance/fair/package/ -- 'Use a consistent file naming "
                  "convention'",
          question="Do our Croissant files follow one naming convention?",
          measures="filenames under datasets/croissant/",
          fn=_croissant_filename_convention),
    Check(id="fair-14", severity="consider",
          section="guidance/fair/repository/ -- 'Discipline-specific repositories'",
          question="Should any of our molecular data be deposited in PChProp?",
          measures="PChProp's OPTIMADE property vocabulary against our data "
                   "shapes",
          fn=_pchprop_deposit),
]
