"""Mey et al., *Best Practices for Alchemical Free Energy Calculations*.

LiveCoMS **2**(1):18378 (2020), doi:10.33011/livecoms.2.1.18378. The explicit
checklist is the paper's **Sec. 12**, split into seven boxes: *Know what you
want to simulate*, *Preparing your simulations*, *Running absolute
simulations*, *Running relative simulations*, *How do I know which simulations
are unreliable?*, *Why are they not reliable?* and *Data analysis*. Section
references in `Check.section` point at the prose that defines the threshold,
with `§12` added when the item also appears verbatim in the checklist.

**Why this paper, and why now.** We have not yet run a single free-energy
calculation. `PLAN.md` Phase 4 commits to SOMD2 on a 32-ligand / 52-edge
network and records the protocol as *"11-window / 5 ns / no-replica-exchange
… labelled unvalidated for convergence"*. Every check below that FAILs is
therefore a design decision we can still make correctly rather than a result we
have to retract. That is the whole value of encoding this one before the run,
not after.

**What each outcome means here.**

* `FAIL` — the practice applies to work already on disk (a protocol record, an
  analysis script, a plot) and the artefact contradicts the paper.
* `UNTESTABLE` — the practice governs an artefact we do not yet produce. Each
  such check names the **exact file that would make it testable**, and the
  threshold is already coded, so the check starts working the day the file
  lands. `UNTESTABLE` is never a pass.
* `PASS` — measured, with the number in `evidence`.

Deliberately **not** encoded, because they cannot fail on a file and belong in
the report as principles: single-vs-dual topology choice (§7.1.1 — no
efficiency study exists to prefer either), annihilation vs decoupling (§7.2.3 —
"no consensus on which is better"), the reversible-binding cross-check (§8.5 —
10–100 µs, far outside budget), and selection bias in retrospective look-backs
(§5.4).
"""

from __future__ import annotations

import csv
import io
import re
from itertools import combinations

from bestpractices import (FAIL, NA, PASS, UNTESTABLE, Check, Context, Paper,
                           Result)

PAPER = Paper(
    slug="alchemical",
    title="Best Practices for Alchemical Free Energy Calculations "
          "[Article v1.0]",
    authors="Mey, Allen, Bruce Macdonald, Chodera, Hahn, Kuhn, Michel, "
            "Mobley, Naden, Prasad, Rizzi, Scheen, Shirts, Tresadern, Xu",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/"
        "v2i1e18378",
    doi="10.33011/livecoms.2.1.18378",
    year="2020",
    scope="Setup, protocol, convergence diagnostics and reporting for "
          "relative and absolute alchemical binding free energy "
          "calculations — governs all of PLAN.md Phase 4 (SOMD2 RBFE on the "
          "32-ligand / 52-edge network) and the planned ABFE toolkit.",
)

# --------------------------------------------------------------------------
# helpers — all read-only, all cheap
# --------------------------------------------------------------------------

#: Vendored upstream trees. Grepping them tells us nothing about our practice
#: and `mace/` alone contains an MAE loss and a dozen scatter plots that would
#: make several checks pass vacuously.
_VENDORED = ("mace/", "mace-off/", "EV-A71_2A_benchmark/",
             "traversing_chem_space/", "albench/", "datasets/croissant/")

#: Where a free-energy campaign would write. Checked explicitly rather than by
#: recursive glob so the checks stay cheap, and so the UNTESTABLE branch can
#: name the exact directory that would switch it on.
#:
#: **2026-08-19: the sixth entry, and why its absence was a defect rather than
#: a gap.** The first five have never existed. This project's only alchemical
#: corpus -- 471 completed SOMD2 edges, 964 legs, 30 arms -- landed in
#: `results/paros/rowanfep/`, so alch-13, alch-15, alch-16 and alch-17 reported
#: UNTESTABLE *while the data they were written for was on disk*. That is the
#: same class of failure as a checker that cannot fail: the check was
#: structurally unable to see the thing it was written for, and it would have
#: gone on saying "no free-energy output exists" indefinitely.
#:
#: The entry added is a PURPOSE-BUILT SUBDIRECTORY, not the results directory
#: itself, and that is deliberate. `results/paros/rowanfep/overlap.csv` is a
#: per-arm SUMMARY (min, median, fraction below 0.03); the `overlap*.csv` glob
#: in `_overlap_matrix` would have picked it up and fed a table with a header
#: row to a float matrix parser. `.../bestpractice/` holds only files written
#: to the schema each UNTESTABLE branch documents, so what these checks read is
#: what they asked for. Written by `rule rowanfep_bestpractice_export`.
#:
#: NOT ONE THRESHOLD IN THIS MODULE WAS CHANGED to accommodate the corpus.
_FEP_DIRS = ("results/fep", "results/rbfe", "results/abfe",
             "results/somd2", "results/alchemical",
             "results/paros/rowanfep/bestpractice")

#: Documents that record protocol decisions. `PLAN.md` is the spec, `RUNS.md`
#: the run register, `GOALS.md` the standing rules.
_PROTOCOL_DOCS = ("PLAN.md", "RUNS.md", "GOALS.md", "FINDINGS.md")

_META_CSV = "data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv"
_SCORES_CSV = "results/score_comparison.csv"


def _sources(ctx: Context):
    """Our own Python, as (path, text). Vendored trees excluded."""
    out = []
    for p in list(ctx.glob("*.py")) + list(ctx.glob("cluster/*.py")):
        rel = str(p.relative_to(ctx.root))
        if any(rel.startswith(v) for v in _VENDORED):
            continue
        out.append((rel, p.read_text(errors="replace")))
    return out


def _docs(ctx: Context):
    return [(d, ctx.read(d)) for d in _PROTOCOL_DOCS if ctx.exists(d)]


def _fep_dirs_present(ctx: Context):
    return [d for d in _FEP_DIRS if ctx.exists(d)]


#: A line that names alchemical run parameters.
_DESCRIPTOR = re.compile(r"(?i)\d+\s*[-‑ ]?windows?\b|"
                         r"ns\s*/\s*(?:leg|window)|\bns/leg\b|"
                         r"\d+\s*ns\s*(?:per|/)\s*(?:leg|window|state)")

#: Someone else's published protocol. PLAN.md:129 records Khoualdi et al.'s
#: "0.1 ns/window, 5 independently-equilibrated replicas" — a citation, not a
#: decision of ours. Counting it would let another group's replicate count
#: satisfy our checklist, which is precisely the vacuous pass this package
#: exists to prevent.
_FOREIGN = re.compile(r"(?i)khoualdi|they also tested|\bet al\b|rowan|fep\+|"
                      r"schr[oö]dinger|published|literature")


def _protocol_records(ctx: Context):
    """Windows of the protocol documents that declare *our* alchemical run.

    A record is a descriptor line (window count, ns/leg) whose paragraph calls
    itself a protocol, with lines describing another group's work excluded by
    name. Returns (doc, lineno, line, context) — context is +/-3 lines, since
    our own record is split across PLAN.md:645-646.

    Failure direction is deliberate: a protocol recorded in wording this does
    not recognise yields *no* records, and the callers then report UNTESTABLE
    ("no protocol recorded") rather than a pass.
    """
    out = []
    for name, text in _docs(ctx):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not _DESCRIPTOR.search(line) or _FOREIGN.search(line):
                continue
            lo, hi = max(0, i - 3), min(len(lines), i + 4)
            ctxt = "\n".join(lines[lo:hi])
            if not re.search(r"(?i)protocol|per leg|per window", ctxt):
                continue
            out.append((name, i + 1, line.strip(), ctxt))
    return out


#: Where `ROWAN_32` has lived. `benchmark.py` first, because that is where the
#: docstring below promised to look; `albench/config.py` because that is where
#: the literal moved to (benchmark.py:73 now says
#: `from albench.config import ROWAN_32`).
_ROWAN32_SOURCES = ("benchmark.py", "albench/config.py")


def _rowan32(ctx: Context):
    """The 32 ligand IDs, parsed from source rather than hardcoded.

    **2026-08-19: this returned [] and had been doing so silently.** The list
    was refactored out of `benchmark.py` into `albench/config.py`, leaving this
    regex matching nothing -- and alch-09, alch-14 and alch-19 reporting
    UNTESTABLE with the message "metadata.csv not readable", which was not the
    reason. A stale locator, reported as a missing input.

    That is the same defect as `_FEP_DIRS` above and the second instance of it
    in this module: a check structurally unable to see data that is present.
    The fix follows the literal to where it now lives rather than hardcoding
    it here, which is what the original docstring asked for. **No threshold
    changes** -- the checks these feed still FAIL on a >5 heavy-atom
    transformation and on an uncorrected charge change exactly as before.
    """
    for name in _ROWAN32_SOURCES:
        src = ctx.read(name)
        m = re.search(r"ROWAN_32\s*=\s*\[(.*?)\]", src, re.S)
        if m:
            ids = re.findall(r'"([^"]+)"', m.group(1))
            if ids:
                return ids
    return []


def _heavy_from_smiles(smiles: str) -> int:
    """Heavy-atom count straight from SMILES — no RDKit, no import cost.

    Bracket atoms collapse to one token first so `[C@H]` counts once and its
    explicit H is not double counted. Validated below against RDKit.
    """
    flat = re.sub(r"\[[^\]]*\]", "X", smiles)
    return len(re.findall(r"Cl|Br|[BCNOPSFIbcnops]|X", flat))


def _ligand_table(ctx: Context):
    """(rows_by_id, agree, compared) for the 32.

    `agree/compared` is a **known-answer test** of `_heavy_from_smiles`
    against the RDKit-derived `n_heavy` column in `results/score_comparison.
    csv` — GOALS standing rule 4. A check that depends on the counter must
    refuse to report if the counter fails its own test.
    """
    meta_txt = ctx.read(_META_CSV)
    if not meta_txt:
        return {}, 0, 0
    rows = list(csv.DictReader(io.StringIO(meta_txt)))

    agree = compared = 0
    scores_txt = ctx.read(_SCORES_CSV)
    if scores_txt:
        by_group = {r["compound_group"]: r for r in rows}
        for r in csv.DictReader(io.StringIO(scores_txt)):
            ref, g = r.get("n_heavy"), r.get("compound_group")
            if ref and g in by_group:
                compared += 1
                agree += int(float(ref)) == _heavy_from_smiles(
                    by_group[g]["smiles"])

    # The 32 are keyed by the suffix of complex_name; seven of them sit in a
    # compound_group named after a different representative, so matching on
    # compound_group silently loses those seven (25/32).
    wanted, out = _rowan32(ctx), {}
    for lig in wanted:
        for r in rows:
            if r.get("complex_name", "").endswith("-" + lig):
                out[lig] = r
                break
    return out, agree, compared


def _formal_charge(smiles: str) -> int:
    q = 0
    for tok in re.findall(r"\[[^\]]*\]", smiles):
        m = re.search(r"([+-])(\d*)", tok)
        if m:
            q += (1 if m.group(1) == "+" else -1) * int(m.group(2) or 1)
    return q


# --------------------------------------------------------------------------
# 1. Data analysis — the statistics the paper insists you report
# --------------------------------------------------------------------------

def _reporting_statistics(ctx: Context) -> Result:
    """§12 *Data analysis*: "Compute both correlation and ranking coefficients
    and ranking statistics (e.g. r, ρ, MUE and τ)"."""
    src = ctx.read("benchmark.py")
    if not src:
        return Result(UNTESTABLE, "benchmark.py absent — nothing reports our "
                                  "predictions against experiment",
                      evidence="benchmark.py")
    blob = src + "\n".join(t for _, t in _sources(ctx))
    have = {
        "Pearson r": bool(re.search(r"(?i)pearsonr?\b", blob)),
        "Spearman rho": bool(re.search(r"(?i)spearmanr?\b", blob)),
        "Kendall tau": bool(re.search(r"(?i)kendall", blob)),
        "MUE/MAE": bool(re.search(
            r"(?i)\bmue\b|mean_absolute_error|mean unsigned", blob)),
    }
    missing = [k for k, v in have.items() if not v]
    ev = ("benchmark.py + all top-level/cluster .py (vendored trees "
          f"excluded): present={[k for k, v in have.items() if v]}")
    if missing:
        return Result(FAIL, f"missing {', '.join(missing)}. benchmark.py "
                            "reports Pearson/Spearman/fitted-RMSE only; the "
                            "paper additionally requires a rank correlation "
                            "(tau) and MUE, which is 'less sensitive to "
                            "dataset size' and so the statistic that lets our "
                            "n=32 be compared with anyone else's set",
                      evidence=ev)
    return Result(PASS, "correlation, ranking and MUE statistics all "
                        "computed", evidence=ev)


def _replicates_declared(ctx: Context) -> Result:
    """§12 *Data analysis*: "Make sure you have run enough replicates to
    ensure statistical reliability (>3)"; §8.7 "triplicates … at minimum,
    with starting points that are expected to be uncorrelated"."""
    recs = _protocol_records(ctx)
    if not recs:
        return Result(UNTESTABLE, "no alchemical protocol is recorded in "
                                  f"{', '.join(_PROTOCOL_DOCS)}, so there is "
                                  "no replicate count to check",
                      evidence=str(_PROTOCOL_DOCS))
    pat = re.compile(r"(?i)(\d+)\s*(?:x\s*)?(?:independent(?:ly)?[- ]\w*\s*)?"
                     r"(?:replicates?|repeats?|replicas\b)")
    counts = []
    for name, i, line, ctxt in recs:
        for chunk in ctxt.splitlines():
            if _FOREIGN.search(chunk):
                continue
            m = pat.search(chunk)
            if m:
                counts.append((name, i, int(m.group(1))))
    ev = (f"{len(recs)} protocol record(s): "
          + "; ".join(f"{n}:{i} \"{ln[:70]}\"" for n, i, ln, _ in recs[:2]))
    if not counts:
        n, i, ln, _ = recs[0]
        return Result(FAIL, "no replicate count is attached to the alchemical "
                      f"protocol — the record at {n}:{i} fixes windows and "
                      "ns/leg but not the number of independent repeats. The "
                      "paper's floor is 3, from starting points 'expected to "
                      "be uncorrelated', because MBAR's own error bar is "
                      "explicitly 'likely an underestimated error'. Khoualdi "
                      "et al. ran 5 and still found their inter-replicate SE "
                      "was 1.5 kcal/mol; one replica would have hidden that",
                      evidence=ev)
    worst = min(c for *_, c in counts)
    if worst < 3:
        return Result(FAIL, f"declared replicate count {worst} is below the "
                            "paper's floor of 3", evidence=ev)
    return Result(PASS, f"{worst} replicates declared", evidence=ev)


def _estimator_and_decorrelation(ctx: Context) -> Result:
    """§8.3 "We recommend MBAR if all energy differences are available";
    §8.1/§8.2 automated equilibration detection and subsampling at the
    statistical inefficiency g; §12 *Running absolute*: "Make sure you
    subsample the data in your free energy estimation protocol"."""
    blob = "\n".join(t for _, t in _sources(ctx))
    blob += "\n".join(t for _, t in _docs(ctx))
    est = re.search(r"(?i)pymbar|alchemlyb|\bMBAR\b|bennett acceptance|"
                    r"thermodynamic integration", blob)
    # Deliberately narrow: a bare "subsample" appears in UMAP and sweep code
    # and has nothing to do with decorrelating alchemical samples at g.
    dec = re.search(r"(?i)statistical inefficiency|detect_?[eE]quilibration|"
                    r"subsample_correlated|integrated autocorrelation", blob)
    ev = (f"estimator match={bool(est)}, decorrelation match={bool(dec)} over "
          f"{len(_sources(ctx))} project .py files + "
          f"{len(_PROTOCOL_DOCS)} protocol docs")
    missing = []
    if not est:
        missing.append("no free-energy estimator named (MBAR/BAR/TI)")
    if not dec:
        missing.append("no equilibration-detection or decorrelation step "
                       "named")
    if missing:
        return Result(FAIL, "; ".join(missing) + ". The paper's recommended "
                      "stack is pymbar/alchemlyb: automated equilibration "
                      "detection (§8.1), subsampling at the statistical "
                      "inefficiency g (§8.2), then MBAR (§8.3). Note the gap "
                      "is *ours*, not the tooling's — pymbar 4.2.0 and "
                      "openmmtools 0.25.3 are already installed in the "
                      "`openfe` env; no code here imports them and no "
                      "document names which estimator our numbers will come "
                      "from", evidence=ev)
    return Result(PASS, "estimator and decorrelation step both named",
                  evidence=ev)


def _error_bars_on_plots(ctx: Context) -> Result:
    """§12 *Data analysis*: "Include error bars in all your visual analyses";
    §8.7 *Presenting your data*: equal axis ranges, 1:1 aspect, 1- and
    2-kcal/mol confidence bands, experimental error bars."""
    rel = "plot_interaction_energy.py"
    src = ctx.read(rel)
    if not src:
        return Result(UNTESTABLE, f"{rel} absent — no computed-vs-experimental"
                                  " plot to inspect", evidence=rel)
    bars = bool(re.search(r"errorbar|yerr\s*=|xerr\s*=|fill_between", src))
    bands = bool(re.search(r"(?i)kcal.{0,20}band|1 kcal|2 kcal|axline|"
                           r"set_aspect", src))
    ev = f"{rel}: error-bar calls={bars}, band/aspect calls={bands}"
    gaps = []
    if not bars:
        gaps.append("no error bars")
    if not bands:
        gaps.append("no 1/2 kcal-mol confidence bands and no 1:1 aspect")
    if gaps:
        return Result(FAIL, f"{rel} plots MACE-OFF E_int against experimental "
                            f"pKD with {' and '.join(gaps)}. benchmark.py "
                            "already computes bootstrap CIs — they simply are "
                            "not drawn", evidence=ev)
    return Result(PASS, "computed-vs-experimental plot carries uncertainty",
                  evidence=ev)


# --------------------------------------------------------------------------
# 2. Protocol — the choices that must be made before launching
# --------------------------------------------------------------------------

def _sampling_scheme(ctx: Context) -> Result:
    """§7.2.4: "Currently, we recommend using Hamiltonian replica exchange
    type sampling schemes … If these are not available in the code of choice,
    running independent simulations at different λ values can be acceptable,
    especially when configurational sampling is fast"."""
    recs = _protocol_records(ctx)
    if not recs:
        return Result(UNTESTABLE, "no alchemical protocol recorded",
                      evidence=str(_PROTOCOL_DOCS))
    neg = re.compile(r"(?i)no[- ]replica[- ]exchange|without replica[- ]"
                     r"exchange|no hrex")
    pos = re.compile(r"(?i)hamiltonian replica[- ]exchange|replica[- ]exchange|"
                     r"\bhrex\b|lambda[- ]exchange")
    negated = [(n, i, ln) for n, i, ln, c in recs if neg.search(c)]
    affirmed = [(n, i, ln) for n, i, ln, c in recs
                if pos.search(c) and not neg.search(c)]
    ev = ("; ".join(f"{n}:{i} \"{ln[:70]}\"" for n, i, ln, _ in recs[:2])
          + f" | negations={len(negated)} affirmations={len(affirmed)}")
    if negated:
        # A negation in the protocol record dominates a mention elsewhere: a
        # Phase 6 list of knobs to tune is not a declaration of what we ran.
        n, i, ln = negated[0]
        ctxt = next(c for nn, ii, _, c in recs if (nn, ii) == (n, i))
        if not re.search(r"(?i)because|justif|sampling is fast|"
                         r"not (?:available|supported)", ctxt):
            return Result(FAIL, f"{n}:{i} declares an independent-window "
                          "protocol ('no-replica-exchange') with no recorded "
                          "justification. The paper's recommendation is "
                          "Hamiltonian replica exchange; independent windows "
                          "are 'acceptable' only 'when configurational "
                          "sampling is fast', a claim that has to be made and "
                          "defended, not left implicit — and the same record "
                          "already calls itself 'unvalidated for convergence'",
                          evidence=ev)
        return Result(PASS, "independent-window sampling, with a recorded "
                            "justification", evidence=ev)
    if affirmed:
        return Result(PASS, "Hamiltonian replica exchange declared",
                      evidence=ev)
    return Result(FAIL, "the protocol record names no lambda-sampling scheme; "
                        "the paper requires the choice to be deliberate",
                  evidence=ev)


def _softcore_and_lambda_spacing(ctx: Context) -> Result:
    """§7.2.3: "transformations involving insertions or deletions of atoms
    should employ a soft-core potential path"; "we heavily recommend against
    using the linear approach". Spacing: states "spaced so that the
    statistical uncertainties between neighboring states be approximately
    equal … within 30–50%"."""
    blob = "\n".join(t for _, t in _sources(ctx))
    blob += "\n".join(t for _, t in _docs(ctx))
    soft = bool(re.search(r"(?i)soft[- ]?core", blob))
    sched = bool(re.search(r"(?i)lambda[ _-]?(schedule|spacing|values)|"
                           r"λ[- ](schedule|spacing)", blob))
    win = _protocol_records(ctx)
    ev = (f"soft-core mentioned={soft}, lambda-schedule mentioned={sched}; "
          f"protocol records={len(win)}")
    gaps = []
    if not soft:
        gaps.append("no soft-core potential recorded")
    if not sched:
        gaps.append("no lambda schedule or spacing rule recorded")
    if gaps and win:
        n, i, _, _ = win[0]
        return Result(FAIL, "; ".join(gaps) + f". {n}:{i} fixes a window "
                      "*count* but not the alchemical path those windows sit "
                      "on. A count without a soft-core form and a spacing "
                      "rule is not a protocol — 11 evenly spaced windows on a "
                      "linear path is the failure mode §7.2.3 spends three "
                      "pages warning about", evidence=ev)
    if gaps:
        return Result(UNTESTABLE, "; ".join(gaps) + " — and no window count "
                      "is declared either, so there is no protocol to check",
                      evidence=ev)
    return Result(PASS, "soft-core path and lambda spacing both recorded",
                  evidence=ev)


def _constraints_in_single_topology(ctx: Context) -> Result:
    """§7.1.1 *Constraints and relative free energy calculations*: "the most
    general solution to this problem is simply to avoid the use of constraints
    (and thus use a smaller timestep if necessary, usually of around 1 fs) in
    any relative free energy calculation involving a transformation of a
    constrained bond." The paper names BioSimSpace/Sire — our SOMD2 route — as
    **single topology**, so this applies to us directly."""
    rel = "md_arms.py"
    src = ctx.read(rel)
    if not src:
        return Result(UNTESTABLE, f"{rel} absent — no MD template to inspect",
                      evidence=rel)
    constrained = re.search(r"constraints\s*=\s*HBonds", src)
    ts = re.search(r'"--timestep".{0,200}?default=([\d.]+)', src, re.S)
    hmr = re.search(r'"--hydrogen-mass".{0,200}?default=([\d.]+)', src, re.S)
    blob = "\n".join(t for _, t in _docs(ctx)) + src
    caveat = re.search(r"(?i)constrained bond|constraint.{0,40}perturb|"
                       r"unconstrain.{0,40}(rbfe|alchem)", blob)
    ev = (f"{rel}: constraints=HBonds match={bool(constrained)}, "
          f"timestep default={ts.group(1) if ts else '?'} fs, "
          f"HMR default={hmr.group(1) if hmr else '?'} amu; "
          f"constrained-bond caveat recorded={bool(caveat)}")
    if constrained and not caveat:
        return Result(FAIL, "the only MD template in the project constrains "
                            "all bonds to hydrogen (HBonds + hydrogen mass "
                            "repartitioning at a 4 fs timestep) and nothing "
                            "records that this must not be inherited by a "
                            "single-topology RBFE. In a benzene->benzyl "
                            "alcohol-style mutation a constrained H-X bond "
                            "changes length or ceases to be constrained; "
                            "engines that do not add that term to dH/dlambda "
                            "return a wrong free energy silently",
                      evidence=ev)
    if constrained:
        return Result(PASS, "H-bond constraints are used and the RBFE caveat "
                            "is recorded", evidence=ev)
    return Result(PASS, "MD template does not constrain bonds to hydrogen",
                  evidence=ev)


def _net_charge_change(ctx: Context) -> Result:
    """§7.2.2: "If the net charge of the system changes as the alchemical
    variable changes … this can pose major challenges", finite-size artifacts
    that PME "do[es] not handle … effectively"."""
    ligs, agree, compared = _ligand_table(ctx)
    if not ligs:
        return Result(UNTESTABLE, f"{_META_CSV} or benchmark.py's ROWAN_32 "
                                  "list not readable", evidence=_META_CSV)
    charges = {k: _formal_charge(v["smiles"]) for k, v in ligs.items()}
    nonzero = {k: q for k, q in charges.items() if q}
    ev = (f"{len(ligs)}/32 ligands read from {_META_CSV}; formal charges "
          f"{sorted(set(charges.values()))}")
    if nonzero:
        blob = "\n".join(t for _, t in _docs(ctx))
        corrected = re.search(r"(?i)co-?alchemical ion|alchemical ion|"
                              r"charge correction|Rocklin", blob)
        if not corrected:
            return Result(FAIL, f"{len(nonzero)} of {len(ligs)} ligands carry "
                          f"a formal charge {sorted(set(nonzero.values()))} "
                          "and no charge-correction or co-alchemical-ion "
                          "scheme is recorded; any edge joining a charged to "
                          "a neutral ligand carries a finite-size artifact",
                          evidence=ev)
        return Result(PASS, "charged ligands present but a correction scheme "
                            "is recorded", evidence=ev)
    return Result(PASS, f"all {len(ligs)} ligands in the network are neutral, "
                        "so no edge can change net charge and the hardest "
                        "open problem in §7.2.2 does not arise for us",
                  evidence=ev)


# --------------------------------------------------------------------------
# 3. The perturbation network
# --------------------------------------------------------------------------

def _network_redundancy_claim(ctx: Context) -> Result:
    """§7.1.1 *Perturbation maps*: "adding redundancy into the network means
    that a better error analysis can be carried out by looking at cycle
    closure errors". For a connected graph the number of independent cycles is
    E - V + 1; a claimed cycle count that does not satisfy it is wrong."""
    # All three numbers must come from **one** sentence. Read separately they
    # cross-contaminate: PLAN.md:129 says "27 ligands" about Khoualdi et al.'s
    # ABFE set, and pairing that with our 52 edges invents a network nobody
    # designed.
    sources = ["datasets/provenance/rowan_xtal_32.md"] + list(_PROTOCOL_DOCS)
    pat = re.compile(r"(\d+)\s+edges\b[^.\n]{0,60}?(\d+)\s+ligands\b"
                     r"[^.\n]{0,60}?(\d+)\s+independent cycles")
    hit = src = None
    for name in sources:
        flat = re.sub(r"[*`]", "", ctx.read(name))
        flat = re.sub(r"\s+", " ", flat)
        m = pat.search(flat)
        if m:
            hit, src = m, name
            break
    if not hit:
        return Result(UNTESTABLE, "no single statement describes the network "
                      "as (edges, ligands, independent cycles), so there is "
                      "no redundancy claim to verify. Reading the three "
                      "numbers from separate sentences is unsafe: PLAN.md "
                      "records another group's 27-ligand ABFE set",
                      evidence=f"searched {sources}")
    E, V, C = (int(hit.group(i)) for i in (1, 2, 3))
    rank = E - V + 1
    ev = (f"{src}: E={E}, V={V}, claimed independent cycles={C}; "
          f"E-V+1={rank}")
    if rank != C:
        return Result(FAIL, f"the documented network is internally "
                      f"inconsistent: {E} edges over {V} ligands gives "
                      f"{rank} independent cycles if connected, not {C}",
                      evidence=ev)
    if rank <= 0:
        return Result(FAIL, "the documented network is a tree or a forest — "
                            "no cycles, so cycle-closure consistency checking "
                            "is impossible", evidence=ev)
    return Result(PASS, f"the documented {E}-edge / {V}-ligand network is "
                        f"arithmetically consistent with its claimed {C} "
                        "independent cycles (this verifies the claim, not the "
                        "graph — see alch-11 for the graph itself)",
                  evidence=ev)


def _network_edge_list_present(ctx: Context) -> Result:
    """§12 *Running relative*: "Generate a perturbation network by your method
    of choice; check whether you have enough cycle closures to check
    consistency in the results"; §8.5 *Cycle closure error*."""
    # The four original globs reach the repo root, `results/` one level deep,
    # and `data/`/`datasets/` recursively -- but NOT into a free-energy output
    # directory, which is the one place a network that actually ran would write
    # its edge list. The declared `_FEP_DIRS` are added rather than a blanket
    # `results/**`, so the reach follows the same declaration the other checks
    # use. This widens where the file may be found; it does not change what
    # counts as finding it.
    found = [str(p.relative_to(ctx.root))
             for pat in ("*edges*.tsv", "results/*edges*.tsv",
                         "data/**/*edges*.tsv", "datasets/**/*edges*.tsv",
                         *(f"{d}/**/*edges*.tsv" for d in _FEP_DIRS))
             for p in ctx.glob(pat)]
    if not found:
        return Result(UNTESTABLE, "the perturbation-network edge list is not "
                      "in this repo. datasets/provenance/rowan_xtal_32.md "
                      "places it on **paros** at "
                      "`/home/ben/code/csbrt-gcmc-fep/loch_fep_pipeline/"
                      "fep_edges/rowan_xtal_edges_full.tsv` (<10 KB). Copying "
                      "that file in makes this check and alch-14 executable, "
                      "and nothing about the network — cycle "
                      "count, per-edge transformation size, charge changes — "
                      "is verifiable until it is here",
                      evidence="globs: *edges*.tsv, results/, data/, datasets/")
    return Result(PASS, f"edge list present: {found[0]}",
                  evidence=", ".join(found[:3]))


def _transformation_size(ctx: Context) -> Result:
    """§12 *Running relative*: "Make sure your molecular transformations are
    realistic (1–5 heavy atoms for reliable computations)"."""
    ligs, agree, compared = _ligand_table(ctx)
    if not ligs:
        return Result(UNTESTABLE, f"{_META_CSV} not readable",
                      evidence=_META_CSV)
    if compared and agree != compared:
        return Result(UNTESTABLE, "the SMILES heavy-atom counter failed its "
                      f"known-answer test ({agree}/{compared} agree with the "
                      "RDKit-derived n_heavy column) — refusing to report a "
                      "number from an unvalidated metric (GOALS rule 4)",
                      evidence=f"{agree}/{compared} vs {_SCORES_CSV}")
    heavy = {k: _heavy_from_smiles(v["smiles"]) for k, v in ligs.items()}
    lo, hi = min(heavy.values()), max(heavy.values())
    over = [(a, b) for a, b in combinations(sorted(heavy), 2)
            if abs(heavy[a] - heavy[b]) > 5]
    ev = (f"heavy-atom counts {lo}-{hi} over {len(heavy)} ligands "
          f"(counter validated {agree}/{compared} against "
          f"{_SCORES_CSV}); {len(over)} of "
          f"{len(heavy) * (len(heavy) - 1) // 2} pairs exceed 5")
    # Same reach as alch-13's locator, and extended for the same reason: this
    # pair of globs stops at `results/` one level deep, so a network that
    # actually ran and wrote its edge list into a free-energy output directory
    # was invisible here while alch-13 could already see it. Two locators for
    # one file, disagreeing about where to look. The 1-5 heavy-atom threshold
    # below is untouched.
    edges = (list(ctx.glob("*edges*.tsv"))
             + list(ctx.glob("results/*edges*.tsv"))
             + [q for dd in _FEP_DIRS for q in ctx.glob(f"{dd}/**/*edges*.tsv")])
    if not edges:
        if hi - lo <= 5:
            return Result(PASS, f"the whole series spans {hi - lo} heavy "
                          "atoms, so no possible edge can exceed the "
                          "1-5 heavy-atom guideline", evidence=ev)
        return Result(UNTESTABLE, f"the series spans {hi - lo} heavy atoms "
                      f"({lo}-{hi}) and {len(over)} ligand pairs differ by "
                      "more than 5, so a bad network *could* violate the "
                      "guideline — but the edge list is not in this repo, so "
                      "which pairs are actually simulated is unknown. See "
                      "alch-13 for the file that fixes this",
                      evidence=ev)
    rows = [ln.split("\t") for ln in
            edges[0].read_text(errors="replace").splitlines() if ln.strip()]
    bad = []
    for row in rows:
        ids = [c.strip() for c in row if c.strip() in heavy]
        if len(ids) == 2 and abs(heavy[ids[0]] - heavy[ids[1]]) > 5:
            bad.append((ids[0], ids[1],
                        abs(heavy[ids[0]] - heavy[ids[1]])))
    ev += f"; {len(rows)} rows read from {edges[0].name}"
    if bad:
        return Result(FAIL, f"{len(bad)} edges transform more than 5 heavy "
                      f"atoms, worst {max(b[2] for b in bad)} "
                      f"({bad[0][0]}<->{bad[0][1]})", evidence=ev)
    return Result(PASS, "every edge stays within the 1-5 heavy-atom "
                        "guideline", evidence=ev)


# --------------------------------------------------------------------------
# 4. Convergence diagnostics — coded now, executable the day we have output
# --------------------------------------------------------------------------

def _overlap_matrix(ctx: Context) -> Result:
    """§8.5 *Overlap matrix*: "the values of the first off-diagonals … should
    at least be 0.03 to obtain a reliable free energy estimate"; "the matrix
    should be at least tridiagonal". §12 *Why are they not reliable?*:
    "Inspect the overlap matrices for lack of overlap"."""
    dirs = _fep_dirs_present(ctx)
    if not dirs:
        return Result(UNTESTABLE, "no free-energy output exists. This check "
                      "reads a per-edge overlap matrix from "
                      f"{_FEP_DIRS[0]}/<edge>/overlap.csv (a K x K matrix of "
                      "floats, K = number of lambda windows) and FAILs any "
                      "edge whose first off-diagonal drops below 0.03. Have "
                      "the SOMD2 analysis write that file per edge and the "
                      "threshold is enforced automatically",
                      evidence=f"none of {_FEP_DIRS} exists")
    found = [p for d in dirs for p in ctx.glob(f"{d}/**/overlap*.csv")]
    if not found:
        return Result(UNTESTABLE, f"{dirs[0]} exists but contains no "
                      "overlap*.csv; MBAR's overlap matrix is not being "
                      "written out", evidence=f"searched {dirs}")
    worst, worst_f = 1.0, ""
    for p in found:
        rows = [[float(x) for x in ln.split(",") if x.strip()]
                for ln in p.read_text().splitlines() if ln.strip()]
        for i in range(len(rows) - 1):
            for j in (i + 1,):
                if j < len(rows[i]) and rows[i][j] < worst:
                    worst, worst_f = rows[i][j], p.name
    ev = f"{len(found)} overlap matrices; min first off-diagonal {worst:.4f}"
    if worst < 0.03:
        return Result(FAIL, f"first off-diagonal falls to {worst:.4f} in "
                      f"{worst_f}, below the 0.03 reliability threshold — add "
                      "lambda windows or respace them rather than running "
                      "longer", evidence=ev)
    return Result(PASS, f"all first off-diagonals >= 0.03 (min {worst:.4f})",
                  evidence=ev)


def _per_edge_reliability(ctx: Context) -> Result:
    """§12 *How do I know which simulations are unreliable?*: "Standard error
    (σ) should not be >1 kcal·mol−1"; relative "Energy differences >~15
    kcal·mol−1 are likely unreliable"; absolute "Energies <~−15 kcal·mol−1 are
    likely unreliable"."""
    dirs = _fep_dirs_present(ctx)
    if not dirs:
        return Result(UNTESTABLE, "no free-energy output exists. This check "
                      f"reads {_FEP_DIRS[0]}/edges.csv with columns "
                      "(edge, ddg, sigma) and FAILs any edge with sigma > 1.0 "
                      "kcal/mol or |ddg| > 15 kcal/mol. Worth noting before "
                      "we run: Khoualdi et al.'s ABFE campaign, our published "
                      "adversarial control, reported an inter-replicate SE of "
                      "1.5 kcal/mol — already over this threshold — and an "
                      "inter-complex spread of 14.7, at the 15 kcal/mol "
                      "unreliability line. By this checklist those numbers "
                      "were flagged before the ranking analysis began",
                      evidence=f"none of {_FEP_DIRS} exists")
    found = [p for d in dirs for p in ctx.glob(f"{d}/**/*edges*.csv")]
    if not found:
        return Result(UNTESTABLE, f"{dirs[0]} exists but holds no per-edge "
                      "results table", evidence=f"searched {dirs}")
    bad_s, bad_d, n = [], [], 0
    for p in found:
        for r in csv.DictReader(io.StringIO(p.read_text(errors="replace"))):
            n += 1
            s = next((r[k] for k in r if re.search(r"(?i)sigma|std|err", k)),
                     None)
            d = next((r[k] for k in r if re.search(r"(?i)^dd?g|ddg", k)), None)
            try:
                if s is not None and float(s) > 1.0:
                    bad_s.append((r, float(s)))
                if d is not None and abs(float(d)) > 15.0:
                    bad_d.append((r, float(d)))
            except ValueError:
                continue
    ev = (f"{n} edges over {len(found)} files; sigma>1: {len(bad_s)}, "
          f"|ddG|>15: {len(bad_d)}")
    if bad_s or bad_d:
        return Result(FAIL, f"{len(bad_s)} edges exceed the 1 kcal/mol "
                      f"standard-error limit and {len(bad_d)} exceed the "
                      "15 kcal/mol magnitude limit", evidence=ev)
    return Result(PASS, f"all {n} edges within sigma<=1 and |ddG|<=15 "
                        "kcal/mol", evidence=ev)


def _cycle_closure_computed(ctx: Context) -> Result:
    """§8.5 *Cycle closure error*: "the sum of free energy changes along edges
    of a closed cycle should be zero … Significant deviations … typically
    indicate insufficient configurational sampling"."""
    dirs = _fep_dirs_present(ctx)
    if not dirs:
        return Result(UNTESTABLE, "no free-energy output exists, so the 21 "
                      "independent cycles PLAN.md relies on as the validation "
                      "gate have closed nothing. This check reads "
                      f"{_FEP_DIRS[0]}/cycle_closure.csv (columns cycle, "
                      "sum_ddg, sigma) and FAILs when a cycle's |sum| exceeds "
                      "its propagated uncertainty. It is the single most "
                      "valuable diagnostic we can run without any "
                      "experimental data, and PLAN.md already commits to it "
                      "before comparison to experiment",
                      evidence=f"none of {_FEP_DIRS} exists")
    found = [p for d in dirs for p in ctx.glob(f"{d}/**/cycle*closure*.csv")]
    if not found:
        return Result(UNTESTABLE, f"{dirs[0]} exists but no cycle-closure "
                      "table has been written", evidence=f"searched {dirs}")
    bad, n = [], 0
    for p in found:
        for r in csv.DictReader(io.StringIO(p.read_text(errors="replace"))):
            n += 1
            s = next((r[k] for k in r if re.search(r"(?i)sum", k)), None)
            e = next((r[k] for k in r if re.search(r"(?i)sigma|err", k)), None)
            try:
                if s is not None and abs(float(s)) > max(
                        1.0, 2.0 * float(e or 0.0)):
                    bad.append((r, float(s)))
            except ValueError:
                continue
    ev = f"{n} cycles; {len(bad)} closing outside 2 sigma or 1 kcal/mol"
    if bad:
        return Result(FAIL, f"{len(bad)} of {n} cycles do not close",
                      evidence=ev)
    return Result(PASS, f"all {n} cycles close within tolerance", evidence=ev)


# --------------------------------------------------------------------------
# 5. Setup and system preparation
# --------------------------------------------------------------------------

def _md_stability_across_size_range(ctx: Context) -> Result:
    """§5.3: "before launching large numbers of alchemical free energy
    calculations it is always recommended to test the system using classical
    MD simulations … Metrics such as ligand and protein RMSD and RMSF can be
    inspected"; §5.2: the single receptor "needs to be capable of
    accommodating the smallest through to largest ligands"."""
    ligs, agree, compared = _ligand_table(ctx)
    if not ligs:
        return Result(UNTESTABLE, f"{_META_CSV} not readable",
                      evidence=_META_CSV)
    built = {p.parent.name for p in ctx.glob("results/md_pilot/*/build.json")}
    read = {p.parent.parent.name
            for p in ctx.glob("results/md_pilot/*/*/readouts.json")}
    heavy = {k: _heavy_from_smiles(v["smiles"]) for k, v in ligs.items()}
    smallest = min(heavy, key=heavy.get)
    largest = max(heavy, key=heavy.get)
    ev = (f"{len(built)} systems built, {len(read)} with RMSD/RMSF readouts "
          f"({', '.join(sorted(read)) or 'none'}); smallest ligand "
          f"{smallest} ({heavy[smallest]} heavy), largest {largest} "
          f"({heavy[largest]} heavy)")
    if not built:
        return Result(UNTESTABLE, "no MD systems have been built for the "
                      "series", evidence=ev)
    missing = [x for x in (smallest, largest) if x not in read]
    if missing:
        return Result(FAIL, f"plain-MD stability read-outs exist for "
                      f"{len(read)}/{len(ligs)} ligands and cover neither "
                      f"end of the size range ({', '.join(missing)} "
                      "untested). One receptor conformation has to hold every "
                      "ligand in the series; the extremes are exactly where "
                      "it will not, and md_readouts.py already computes "
                      "ligand RMSD, ligand RMSF and protein RMSF",
                      evidence=ev)
    return Result(PASS, f"stability read-outs cover both size extremes "
                        f"({smallest}, {largest})", evidence=ev)


def _assay_conditions_recorded(ctx: Context) -> Result:
    """§5.1: "computational protein and ligand preparation protocols should
    reflect experimental pH"; §6.1: "Care should be taken when preparing
    ligands and proteins to match the pH of the experimental assay, if
    known"."""
    prov = [(str(p.relative_to(ctx.root)), p.read_text(errors="replace"))
            for p in ctx.glob("datasets/provenance/*.md")]
    if not prov:
        return Result(UNTESTABLE, "no provenance records to inspect",
                      evidence="datasets/provenance/*.md")
    pat = re.compile(r"(?i)assay pH|pH of the assay|experimental pH|"
                     r"pH\s*=\s*[\d.]+|buffer pH")
    hits = [n for n, t in prov if pat.search(t)]
    ph7 = bool(re.search(r"(?i)pH 7", ctx.read("protonate.py")))
    ev = (f"{len(prov)} provenance files searched, {len(hits)} record an "
          f"assay pH; protonate.py assumes pH 7 = {ph7}")
    if not hits:
        return Result(FAIL, "no provenance record states the pH (or "
                      "temperature) of the assay that produced our "
                      "experimental pKD values, yet protonate.py fixes every "
                      "protonation state at pH 7 and FINDINGS records a "
                      "His21 pKa of 9.37 driving HIP in 825/925 structures. "
                      "If the assay is not at pH 7 that whole decision is "
                      "conditioned on an unrecorded assumption",
                      evidence=ev)
    return Result(PASS, f"assay pH recorded in {', '.join(hits[:3])}",
                  evidence=ev)


def _cost_counted_before_launch(ctx: Context) -> Result:
    """§4.5 *Can I afford the calculation?* and §12 *Know what you want to
    simulate*: "Are the projected computational expense and runtime realistic
    for my scientific goals?"."""
    txt = ctx.read("results/cost_curve.csv")
    if not txt:
        return Result(UNTESTABLE, "results/cost_curve.csv absent — no cost "
                      "estimate for any tier", evidence="results/cost_curve.csv")
    rows = list(csv.DictReader(io.StringIO(txt)))
    fe = [r for r in rows
          if re.search(r"(?i)rbfe|abfe|fep|alchem", r.get("method", ""))]
    ev = f"{len(rows)} tiers costed; free-energy rows: " + \
         "; ".join(f"{r['method']}={r['sec']}s/cpd k={r['k']} "
                   f"feasible={r.get('feasible')}" for r in fe)
    if not fe:
        return Result(FAIL, "the cost curve costs every other tier but has no "
                      "row for the planned RBFE/ABFE campaign, so its expense "
                      "has never been projected", evidence=ev)
    return Result(PASS, "the alchemical tier is costed before launch, and the "
                        "verdict is recorded rather than assumed",
                  evidence=ev)


def _potency_range_ceiling(ctx: Context) -> Result:
    """§8.7: "Brown et al. have provided a useful analysis in terms of upper
    bounds of expected possible correlations between experiment and
    computation with a given potency range … for potency ranges of 2 log units
    it would be impossible to get a higher correlation in R than 0.8"."""
    ligs, _, _ = _ligand_table(ctx)
    if not ligs:
        return Result(UNTESTABLE, f"{_META_CSV} not readable",
                      evidence=_META_CSV)
    pk = []
    for v in ligs.values():
        try:
            pk.append(float(v["experimental_pKD"]))
        except (KeyError, ValueError):
            continue
    if len(pk) < 2:
        return Result(UNTESTABLE, "no experimental pKD values on the series",
                      evidence=f"{len(pk)} values")
    span = max(pk) - min(pk)
    blob = "\n".join(t for _, t in _docs(ctx))
    recorded = re.search(r"(?i)brown et al|correlation ceiling|"
                         r"attainable correlation|upper bound.{0,30}"
                         r"correlation", blob)
    ev = (f"{len(pk)} pKD values, {min(pk):.3f}-{max(pk):.3f}, span "
          f"{span:.3f} log units (~{span * 1.364:.2f} kcal/mol); ceiling "
          f"recorded in project docs={bool(recorded)}")
    if not recorded:
        return Result(FAIL, f"the benchmark series spans only {span:.2f} log "
                      "units and nothing in our documents records the "
                      "resulting ceiling on attainable correlation. At ~2 log "
                      "units the paper puts the maximum achievable R near "
                      "0.8; Rowan's published Pearson 0.426 on exactly these "
                      "32 has to be read against that ceiling, not against "
                      "1.0, or we will over-read both their result and ours",
                      evidence=ev)
    return Result(PASS, f"potency span {span:.2f} log units and the "
                        "correlation ceiling are both recorded", evidence=ev)


def _data_open_sourceable(ctx: Context) -> Result:
    """§12 *Know what you want to simulate*: "Can I open-source my data?" —
    and §4.2/§9 on cross-package reproducibility, which needs the inputs."""
    crois = ctx.glob("datasets/croissant/*.json")
    sums = ctx.glob("datasets/checksums/*.sha256")
    prov = ctx.glob("datasets/provenance/*.md")
    ev = (f"{len(crois)} Croissant records, {len(sums)} checksum manifests, "
          f"{len(prov)} provenance notes")
    if not (crois and sums and prov):
        return Result(FAIL, "the inputs are not packaged for release "
                            "(machine-readable metadata, checksums and "
                            "provenance are not all present), so a third "
                            "party cannot reproduce a free energy we publish",
                      evidence=ev)
    return Result(PASS, "inputs carry provenance, checksums and "
                        "machine-readable metadata", evidence=ev)


def _absolute_specific(ctx: Context) -> Result:
    """§7.1.2: absolute calculations "must handle the standard state and use
    restraints" — the correction is
    dG_restr = -kT ln(c* V_L) - kT ln(xi_L / 8 pi^2). §12 *Running absolute*:
    "Set up your restraints correctly"; "Apply the appropriate correction
    terms"."""
    blob = "\n".join(t for _, t in _sources(ctx))
    blob += "\n".join(t for _, t in _docs(ctx))
    # Only *our* intent counts. PLAN.md discusses Khoualdi et al.'s failed
    # ABFE at length; that is a citation, not a plan of ours.
    goal = None
    for name, text in _docs(ctx):
        for i, line in enumerate(text.splitlines(), 1):
            if _FOREIGN.search(line):
                continue
            if re.search(r"(?i)\bABFE\b|absolute binding free energ", line) \
                    and re.search(r"(?i)priority|goal|phase|toolkit|we will|"
                                  r"planned", line):
                goal = (name, i, line.strip())
                break
        if goal:
            break
    if not goal:
        return Result(NA, "no absolute binding free energy calculation of our "
                          "own is planned — every ABFE mention in the "
                          "protocol documents cites another group's campaign "
                          "— so the standard-state and restraint corrections "
                          "do not yet apply", evidence="no own-ABFE plan line")
    restr = re.search(r"(?i)boresch|flat[- ]bottom|harmonic restraint|"
                      r"restraint correction|RMSD restraint", blob)
    # "non-standard state" appears in the protonation discussion; excluded.
    std = re.search(r"(?i)(?<!non-)standard[- ]state|1660\s*(?:Å|A)\^?3", blob)
    ev = (f"our ABFE intent at {goal[0]}:{goal[1]} \"{goal[2][:70]}\"; "
          f"restraint scheme named={bool(restr)}, "
          f"standard state named={bool(std)}")
    gaps = []
    if not restr:
        gaps.append("no restraint scheme named (Boresch / flat-bottom / "
                    "harmonic)")
    if not std:
        gaps.append("no standard-state correction named")
    if gaps:
        return Result(FAIL, "ABFE is planned but " + "; and ".join(gaps) +
                      ". The restraint choice is not cosmetic: Boresch "
                      "restraints permit exactly one binding mode, "
                      "flat-bottom restraints permit several sites, and the "
                      "standard-state term differs between them. Khoualdi et "
                      "al. did not restrain to the intended pose and reported "
                      "substantial ligand shifts",
                      evidence=ev)
    return Result(PASS, "restraint scheme and standard-state correction both "
                        "recorded", evidence=ev)


# --------------------------------------------------------------------------

CHECKS = [
    # --- Data analysis and reporting (§8, §12 "Data analysis") ---
    Check(id="alch-01", severity="must", section="§12 Data analysis; §8.7",
          question="Do we compute a correlation coefficient, a ranking "
                   "coefficient and MUE for predictions against experiment?",
          measures="benchmark.py and every project .py, for pearson / "
                   "spearman / kendall / MUE",
          fn=_reporting_statistics),
    Check(id="alch-02", severity="must", section="§12 Data analysis; §8.7, §4.2",
          question="Does the recorded free-energy protocol declare at least 3 "
                   "replicates from uncorrelated starting points?",
          measures="PLAN.md / RUNS.md / GOALS.md lines mentioning "
                   "RBFE/ABFE/SOMD2/FEP",
          fn=_replicates_declared),
    Check(id="alch-03", severity="should", section="§7.2.4",
          question="Is the lambda-sampling scheme declared, and if it is not "
                   "Hamiltonian replica exchange, is the choice justified?",
          measures="the protocol lines in PLAN.md / RUNS.md",
          fn=_sampling_scheme),
    Check(id="alch-04", severity="should", section="§7.2.3; §8.4",
          question="Does the protocol record a soft-core path and a lambda "
                   "spacing rule, not just a window count?",
          measures="project .py + protocol docs for soft-core and lambda "
                   "schedule",
          fn=_softcore_and_lambda_spacing),
    Check(id="alch-05", severity="must", section="§8.1, §8.2, §8.3; §12",
          question="Are a free-energy estimator and a decorrelation / "
                   "equilibration-detection step named anywhere?",
          measures="project .py + protocol docs for pymbar / alchemlyb / "
                   "MBAR / statistical inefficiency",
          fn=_estimator_and_decorrelation),
    Check(id="alch-06", severity="should", section="§12 Data analysis; §8.7",
          question="Does our computed-vs-experimental plot carry error bars "
                   "and the 1 and 2 kcal/mol confidence bands?",
          measures="plot_interaction_energy.py",
          fn=_error_bars_on_plots),

    # --- Setup (§5, §6, §7.1) ---
    Check(id="alch-07", severity="should", section="§5.3, §5.2",
          question="Has plain MD stability been checked across the size range "
                   "of the series before launching alchemical runs?",
          measures="results/md_pilot/*/build.json and */*/readouts.json vs "
                   "heavy-atom counts of the 32",
          fn=_md_stability_across_size_range),
    Check(id="alch-08", severity="should", section="§5.1, §6.1",
          question="Is the pH of the experimental assay recorded, so the "
                   "protonation protocol can be shown to match it?",
          measures="datasets/provenance/*.md and protonate.py",
          fn=_assay_conditions_recorded),
    Check(id="alch-09", severity="should", section="§7.2.2",
          question="Can any perturbation in the network change net charge?",
          measures="formal charge from SMILES for all 32 ligands in "
                   "data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv",
          fn=_net_charge_change),
    Check(id="alch-10", severity="should", section="§7.1.1 Constraints",
          question="Is constrained-bond handling recorded for the planned "
                   "single-topology RBFE?",
          measures="md_arms.py constraints/timestep/HMR defaults and the "
                   "protocol docs",
          fn=_constraints_in_single_topology),
    Check(id="alch-11", severity="should", section="§4.5; §12 Know what you "
                                                   "want to simulate",
          question="Was the cost of the alchemical campaign projected before "
                   "launching it?",
          measures="results/cost_curve.csv",
          fn=_cost_counted_before_launch),

    # --- The network (§7.1.1, §8.5, §12 "Running relative") ---
    Check(id="alch-12", severity="should", section="§7.1.1 Perturbation maps",
          question="Is the documented network redundancy arithmetically "
                   "consistent (E - V + 1 independent cycles)?",
          measures="edge / ligand / cycle counts in PLAN.md and "
                   "datasets/provenance/rowan_xtal_32.md",
          fn=_network_redundancy_claim),
    Check(id="alch-13", severity="must", section="§12 Running relative; §8.5",
          question="Is the perturbation-network edge list actually present, "
                   "so cycle closure can be checked?",
          measures="*edges*.tsv across the repo",
          fn=_network_edge_list_present),
    Check(id="alch-14", severity="should", section="§12 Running relative",
          question="Does every transformation stay within the 1-5 heavy-atom "
                   "guideline?",
          measures="heavy-atom counts from SMILES (known-answer tested "
                   "against n_heavy) crossed with the edge list",
          fn=_transformation_size),

    # --- Convergence diagnostics (§8.5, §12 "unreliable") ---
    Check(id="alch-15", severity="must", section="§8.5 Overlap matrix; §12",
          question="Do the first off-diagonals of every overlap matrix reach "
                   "0.03?",
          measures="results/fep*/**/overlap*.csv",
          fn=_overlap_matrix),
    Check(id="alch-16", severity="must",
          section="§12 How do I know which simulations are unreliable?",
          question="Is every edge within sigma <= 1 kcal/mol and |ddG| <= 15 "
                   "kcal/mol?",
          measures="results/fep*/**/*edges*.csv",
          fn=_per_edge_reliability),
    Check(id="alch-17", severity="must", section="§8.5 Cycle closure error",
          question="Have the network's independent cycles been closed and the "
                   "closure errors reported?",
          measures="results/fep*/**/cycle*closure*.csv",
          fn=_cycle_closure_computed),

    # --- Absolute-specific (§7.1.2, §12 "Running absolute") ---
    Check(id="alch-18", severity="must", section="§7.1.2; §12 Running absolute",
          question="If ABFE is planned, are the restraint scheme and the "
                   "standard-state correction both recorded?",
          measures="project .py + protocol docs for Boresch / flat-bottom / "
                   "standard state",
          fn=_absolute_specific),

    # --- Framing (§8.7, §12 "Know what you want to simulate") ---
    Check(id="alch-19", severity="consider", section="§8.7",
          question="Is the potency range of the benchmark series recorded, so "
                   "the ceiling on attainable correlation is known?",
          measures="experimental pKD span of the 32 vs the project docs",
          fn=_potency_range_ceiling),
    Check(id="alch-20", severity="consider",
          section="§12 Know what you want to simulate; §4.2",
          question="Can the inputs behind a published free energy be "
                   "open-sourced and reproduced?",
          measures="datasets/croissant, datasets/checksums, "
                   "datasets/provenance",
          fn=_data_open_sourceable),
]
