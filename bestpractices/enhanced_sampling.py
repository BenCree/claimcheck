"""Hénin, Lelièvre, Shirts, Valsson & Delemotte — *Enhanced Sampling Methods for
Molecular Dynamics Simulations* [Article v1.0], LiveCoMS **4**(1):1583 (2022).

**What this paper is, and is not.** It is a LiveCoMS *Perpetual Review*, not a
*Best Practices* article, so it ships no numbered checklist box. What it does
ship — and what is encoded here — are (a) two explicitly numeric convergence
criteria for multi-state methods (§8.1.1, §8.2.4), (b) a hard precondition on
the BAR/MBAR/WHAM/exponential-averaging family (§3.4), (c) a statement of what
makes a method comparison legitimate (§1.3), and (d) several sentences of the
form "one must ..." / "it is a good practice to ...". Everything below traces to
one of those.

**Why it applies to us at all.** We run no enhanced sampling. That is precisely
the point: this paper is the published statement of what unbiased MD cannot
reach, and of the diagnostics that distinguish "the simulation ran" from "the
simulation sampled". Our MD is 5 replicates x 500 ps of unbiased dynamics from a
single minimised structure, and the objective it feeds is a *ranking*, so the
question the paper forces is not "is the free energy converged" but the weaker
and still unanswered one: **did anything move**.

The sharpest sentence in the paper for this project is §8.2.4:

    "Note that equal visitation of all states does not of course guarantee
    sampling in the coordinate space. [...] One must thus explicitly check the
    slow degrees of freedom of the system to see if enhanced sampling actually
    occurred."

That is the general form of the failure mode this repo has already shipped three
times — a diagnostic reporting success while the thing it stands proxy for never
happened (`FINDINGS` §3.6, §3.7, §1.11).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE, NA

PAPER = Paper(
    slug="enhanced_sampling",
    title="Enhanced Sampling Methods for Molecular Dynamics Simulations "
          "[Article v1.0]",
    authors="Jérôme Hénin, Tony Lelièvre, Michael R. Shirts, Omar Valsson, "
            "Lucie Delemotte",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v4i1e1583",
    doi="10.33011/livecoms.4.1.1583",
    year="2022",
    scope="What unbiased MD can and cannot sample, and the diagnostics that "
          "separate a simulation that ran from one that sampled: multi-basin "
          "evidence, aggregate sampling time, biased-vs-unbiased ensembles, "
          "state overlap as a precondition for BAR/MBAR, and what makes a "
          "method comparison legitimate.",
)

# --------------------------------------------------------------------------
# shared readers.  Everything here is read-only and touches only small files.
# --------------------------------------------------------------------------

MD_ROOT = "results/md_pilot"


def _jsons(ctx: Context, pattern: str):
    """(path, parsed) for every JSON matching `pattern`, skipping unreadable
    ones rather than raising — a check must report, not crash."""
    out = []
    for p in ctx.glob(pattern):
        try:
            out.append((p, json.loads(p.read_text())))
        except (ValueError, OSError):
            continue
    return out


def _receipts(ctx: Context):
    return _jsons(ctx, f"{MD_ROOT}/*/*/receipt.json")


def _readouts(ctx: Context):
    return _jsons(ctx, f"{MD_ROOT}/*/*/readouts.json")


def _rel(ctx: Context, p: Path) -> str:
    try:
        return str(p.relative_to(ctx.root))
    except ValueError:
        return str(p)


def _source_files(ctx: Context):
    """Our own analysis code: repo-root and cluster modules. Deliberately not
    recursive — vendored trees are not ours to judge."""
    return ctx.glob("*.py") + ctx.glob("cluster/*.py")


def _aggregate_ns(receipts) -> float:
    """Total production time actually run, in ns. `replicates` is absent from
    older receipts; those are counted as 1, which is generous."""
    ps = 0.0
    for _, r in receipts:
        n = r.get("replicates") or 1
        ps += float(r.get("prod_ps") or 0.0) * n
    return ps / 1000.0


# --------------------------------------------------------------------------
# es-01  Did anything move?
# --------------------------------------------------------------------------

#: Any statistic that could show a slow degree of freedom changed state.
_BASIN_TOKENS = (
    "transition", "basin", "metastable", "round.?trip", "state.?index",
    "dwell", "markov", r"\bmsm\b", "n_states", "cluster", "hopping",
    "torsion.?flip", "rotamer",
)


def _multi_basin_evidence(ctx: Context) -> Result:
    src = ctx.read("md_readouts.py")
    if not src:
        return Result(UNTESTABLE,
                      "md_readouts.py is absent, so there is no read-out layer "
                      "to inspect; testable as soon as one exists",
                      evidence="expected at repo root")
    recs = _receipts(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      f"no MD receipts under {MD_ROOT}/*/*/receipt.json — the "
                      "criterion becomes testable at the first production "
                      "trajectory",
                      evidence=f"{MD_ROOT} empty or absent")

    found = sorted({t for t in _BASIN_TOKENS if re.search(t, src, re.I)})
    ns = _aggregate_ns(recs)
    longest = max((float(r.get("prod_ps") or 0.0) for _, r in recs), default=0.0)
    if found:
        return Result(PASS,
                      "the read-out layer computes at least one state-change "
                      f"statistic ({', '.join(found)}), so a reported mean can "
                      "be shown to average over more than one basin",
                      evidence=f"md_readouts.py matches {found}")
    return Result(
        FAIL,
        f"{len(recs)} trajectories, {ns:.2f} ns aggregate, longest single run "
        f"{longest:.0f} ps, and md_readouts.py computes no transition, dwell, "
        "basin or clustering statistic of any kind. Every channel (lig_rmsd, "
        "lig_rmsf, ca_rmsf, rgyr, contact_persistence) is a within-basin "
        "fluctuation measure, so a small replicate SD is equally consistent "
        "with 'converged' and with 'all five replicates stayed in the basin "
        "they were minimised into'. The paper's requirement is to check the "
        "slow degrees of freedom explicitly, not to infer them from a "
        "well-behaved fast one.",
        evidence=f"md_readouts.py: 0/{len(_BASIN_TOKENS)} state-change tokens; "
                 f"channels are RMSD/RMSF/Rgyr/contact-persistence only")


# --------------------------------------------------------------------------
# es-02  Aggregate time, not run length, is the barrier-crossing currency
# --------------------------------------------------------------------------

def _aggregate_time_recorded(ctx: Context) -> Result:
    recs = _receipts(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      f"no receipts under {MD_ROOT}/*/*/receipt.json",
                      evidence=f"{MD_ROOT} empty or absent")
    ro = {p.parent: d for p, d in _readouts(ctx)}
    bad = []
    for p, r in recs:
        has_len = r.get("prod_ps") is not None
        n = r.get("replicates")
        if n is None:
            n = (ro.get(p.parent) or {}).get("n_replicates")
        if not (has_len and n is not None):
            bad.append(_rel(ctx, p))
    if not bad:
        return Result(PASS,
                      f"all {len(recs)} receipts carry both prod_ps and a "
                      "replicate count, so aggregate sampling time is "
                      "recoverable from the artefact alone",
                      evidence=f"aggregate {_aggregate_ns(recs):.2f} ns over "
                               f"{len(recs)} runs")
    return Result(
        FAIL,
        f"{len(bad)} of {len(recs)} MD receipts do not record how many "
        "replicates they represent, so the quantity the paper says governs "
        "barrier crossing — *aggregate* simulation time — cannot be read off "
        "the artefact. Run length alone does not answer it: §9.1 states the "
        "crossing probability 'depends on the total number of attempts', not "
        "on the length of any one run.",
        evidence=f"missing replicate count: {', '.join(bad[:4])}"
                 + (" ..." if len(bad) > 4 else ""))


# --------------------------------------------------------------------------
# es-03  A restrained simulation samples a different distribution
# --------------------------------------------------------------------------

#: Receipt keys that would state the production ensemble's bias status.
_PROD_BIAS_KEYS = ("production_restrained", "prod_restrained",
                   "restraints_released", "backbone_k_production",
                   "production_biased")


def _production_bias_disclosed(ctx: Context) -> Result:
    recs = _receipts(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      f"no receipts under {MD_ROOT}/*/*/receipt.json",
                      evidence=f"{MD_ROOT} empty or absent")
    biased_looking = [(p, r) for p, r in recs
                      if (r.get("backbone_restraints") or 0) > 0
                      and (r.get("backbone_k") or 0) > 0]
    if not biased_looking:
        return Result(PASS,
                      "no receipt records a nonzero restraint, so every "
                      "trajectory on disk is an unbiased sample of the target "
                      "distribution",
                      evidence=f"{len(recs)} receipts, all backbone_k = 0")
    undisclosed = [_rel(ctx, p) for p, r in biased_looking
                   if not any(k in r for k in _PROD_BIAS_KEYS)]
    if not undisclosed:
        return Result(PASS,
                      "every restrained run records whether the restraint was "
                      "active during production",
                      evidence=f"{len(biased_looking)} restrained runs, all "
                               f"carry one of {_PROD_BIAS_KEYS}")
    src = ctx.read("md_arms.py")
    released = bool(re.search(r'setParameter\(\s*"k"\s*,\s*0(\.0)?\s*\)', src))
    ks = sorted({r.get("backbone_k") for _, r in biased_looking})
    ns = sorted({r.get("backbone_restraints") for _, r in biased_looking})
    return Result(
        FAIL,
        f"{len(undisclosed)} of {len(recs)} receipts record a localizing "
        f"potential ({ns} restrained atoms at k={ks}) but no field saying "
        "whether it was on during production. "
        + ("md_arms.py does release it (setParameter k=0 before the production "
           "step), so the physics is right — but the receipt is the artefact "
           "that travels, and from the receipt alone one cannot tell whether "
           "the reported per-residue RMSF is from the target distribution or "
           "from one convolved with a localizing function (§5). "
           if released else
           "No release call is visible in md_arms.py, so on the artefact's own "
           "evidence these read-outs are from a biased ensemble and require "
           "reweighting (§6). ")
        + "Fix is one boolean in the receipt.",
        evidence=f"receipt keys present: backbone_restraints, backbone_k; "
                 f"absent: {'/'.join(_PROD_BIAS_KEYS)}; "
                 f"md_arms.py release call found={released}")


# --------------------------------------------------------------------------
# es-04  Two independent routes to the same quantity must be compared
# --------------------------------------------------------------------------

def _independent_cross_check(ctx: Context) -> Result:
    """§7.2.1's metadynamics advice generalised: estimate the quantity a second,
    independent way and treat disagreement as evidence of non-convergence or of
    a bad coordinate."""
    script = "validate_gcmc_waters.py"
    artefact = "results/gcmc_vs_crystal_waters.csv"
    have_script, have_artefact = ctx.exists(script), ctx.exists(artefact)
    if not have_script and not have_artefact:
        return Result(
            FAIL,
            "no sampling-derived quantity in this repo is compared against an "
            "independent estimate of the same thing. The GCMC hydration-site "
            "occupancies are consumed as given.",
            evidence=f"absent: {script}, {artefact}")
    if not have_artefact:
        return Result(UNTESTABLE,
                      f"{script} exists but has produced no output, so the "
                      "agreement between the two routes cannot be read; "
                      f"testable once {artefact} is written",
                      evidence=f"{script} present, {artefact} absent")
    rows = ctx.read(artefact).strip().splitlines()
    return Result(
        PASS,
        "GCMC-predicted hydration-site occupancies are cross-checked against "
        "an entirely independent estimate of the same quantity — waters "
        "actually modelled by crystallographers in the deposited complexes. "
        "This is the generalised form of the paper's advice, and it is the one "
        "place we do it.",
        evidence=f"{artefact}: {max(0, len(rows) - 1)} rows; produced by {script}")


# --------------------------------------------------------------------------
# es-05  Exponential averaging needs samples from the distribution it names
# --------------------------------------------------------------------------

_EXPAVG_PAT = re.compile(r"logsumexp|exp\(\s*-\s*\w+\s*/\s*k_?[bt]", re.I)
#: Phrases that would tell a reader the weighted set is not an equilibrium
#: sample, so the result is not being passed off as a thermodynamic free energy.
_DISCLAIM_PAT = re.compile(
    r"smoothing parameter|rather than a physical|not a physical|"
    r"not an equilibrium|not Boltzmann|no statistical basis|underflow", re.I)


def _boltzmann_weighting_disclosed(ctx: Context) -> Result:
    users, silent = [], []
    for p in _source_files(ctx):
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        if p.name == Path(__file__).name or not _EXPAVG_PAT.search(src):
            continue
        users.append(_rel(ctx, p))
        if not _DISCLAIM_PAT.search(src):
            silent.append(_rel(ctx, p))
    if not users:
        return Result(NA,
                      "no module applies a Boltzmann/exponential weight to a "
                      "set of configurations, so the precondition on eq. 58 "
                      "cannot be violated here",
                      evidence=f"scanned {len(_source_files(ctx))} modules, "
                               "no logsumexp or exp(-E/kT) aggregation")
    if silent:
        return Result(
            FAIL,
            f"{len(silent)} module(s) collapse a set of energies with a "
            "Boltzmann weight without stating what distribution the set is "
            "drawn from. Eq. 58 is an estimator only when 'the Ni samples are "
            "from the ith state'; docked or optimised geometries are optimiser "
            "output, not Boltzmann samples, so the weighted sum estimates "
            "nothing — and §3.4 warns the result is dominated by the tail "
            "regardless.",
            evidence=f"undisclosed: {', '.join(silent)}")
    return Result(
        PASS,
        f"{len(users)} module(s) use a Boltzmann weight and each states that "
        "the temperature is a smoothing parameter rather than a physical one, "
        "i.e. that the result is not being offered as a free energy",
        evidence=f"disclosed: {', '.join(users)}")


# --------------------------------------------------------------------------
# es-06  Overlap is a precondition, not a diagnostic
# --------------------------------------------------------------------------

_WINDOW_PAT = re.compile(r"\b(\d{1,3})[\s-]*windows?\b", re.I)
_ALCHEM_PAT = re.compile(r"\bRBFE\b|\bABFE\b|\bFEP\b|alchemical", re.I)
_OVERLAP_IMPORT = re.compile(r"^\s*(from|import)\s+(pymbar|alchemlyb)", re.M)


def _lambda_overlap_demonstrated(ctx: Context) -> Result:
    docs = [d for d in ("PLAN.md", "RUNS.md", "GOALS.md") if ctx.exists(d)]
    declared = []
    for d in docs:
        text = ctx.read(d)
        if not _ALCHEM_PAT.search(text):
            continue
        for line in text.splitlines():
            m = _WINDOW_PAT.search(line)
            if m:
                declared.append(f"{d}: {line.strip()[:90]}")
    if not declared:
        return Result(UNTESTABLE,
                      "no lambda-window protocol is declared anywhere in the "
                      "planning documents, so there is no protocol whose "
                      "overlap could be demonstrated; testable the moment one "
                      "is written down",
                      evidence=f"scanned {docs} for '<N> window' + FEP/RBFE/ABFE")

    have = []
    for p in _source_files(ctx):
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        if _OVERLAP_IMPORT.search(src):
            have.append(_rel(ctx, p))
    artefacts = [_rel(ctx, p) for p in ctx.glob("results/*")
                 if re.search(r"overlap|mbar|fep|rbfe|abfe", p.name, re.I)]
    if have or artefacts:
        return Result(PASS,
                      "an alchemical protocol is declared and the repo carries "
                      "the machinery/artefacts that report inter-state overlap",
                      evidence=f"estimators: {have}; artefacts: {artefacts}")
    return Result(
        FAIL,
        f"a lambda-window protocol is declared ({len(declared)} mention(s)) but "
        "nothing in the repo computes or stores an overlap between neighbouring "
        "states. §3.4 is unambiguous that BAR/MBAR/WHAM/exponential averaging "
        "'rely on the overlap between the thermodynamic states i and j' and "
        "'cannot be used when the states do not have overlap' — overlap is the "
        "estimator's precondition, so a window count chosen without it is a "
        "guess. §8.1.1 gives the working target: neighbours want 'moderate "
        "overlap (perhaps 5-20%, depending on the method)'.",
        evidence=declared[0] + (f" (+{len(declared)-1} more)"
                                if len(declared) > 1 else ""))


# --------------------------------------------------------------------------
# es-07  The explicit numeric diagnostics for multi-state methods
# --------------------------------------------------------------------------

_MULTISTATE_CODE = re.compile(
    r"ReplicaExchange|MultiStateSampler|openmmtools\.multistate|"
    r"expanded[_ ]ensemble|\bSAMS\b|state_index|replica_exchange", re.I)


def _multistate_diagnostics(ctx: Context) -> Result:
    """§8.1.1 and §8.2.4 are the paper's only explicitly numeric criteria:

      * each replica transitions to another state >= ~30% of proposed exchanges;
      * ~20-40 traversals of the full state range, else the weights/free
        energies across the range are not well estimated;
      * the state graph must be connected — no bottleneck splitting it in two;
      * all states visited, and none an order of magnitude less than the rest.
    """
    impls = []
    for p in _source_files(ctx):
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        if _MULTISTATE_CODE.search(src):
            impls.append(_rel(ctx, p))
    if not impls:
        return Result(
            UNTESTABLE,
            "no replica-exchange, expanded-ensemble or lambda-hopping "
            "simulation exists in this repo, so its state-index trajectory, "
            "round-trip count and transition matrix cannot be measured. This "
            "is NOT a pass: the planned RBFE protocol is declared *without* "
            "replica exchange, which removes the mechanism these diagnostics "
            "measure without removing the need for overlap (see es-06). "
            "Testable at the first run that writes a per-replica state index; "
            "the thresholds to apply are >=30% transitions, 20-40 traversals "
            "of the full range, and a connected state graph.",
            evidence="scanned repo-root and cluster/ modules for "
                     "ReplicaExchange/MultiStateSampler/state_index: 0 hits")
    return Result(
        UNTESTABLE,
        f"multi-state sampling code is present ({', '.join(impls)}) but this "
        "check does not yet parse its state-index output. Encode the parse "
        "before quoting any free energy from it; the thresholds are >=30% "
        "transitions per proposed exchange, 20-40 traversals of the full "
        "state range, and no bottleneck in the transition matrix.",
        evidence=f"implementations found: {impls}")


# --------------------------------------------------------------------------
# es-08  "Fairly" — a comparison at matched parameters
# --------------------------------------------------------------------------

def _comparison_is_fair(ctx: Context) -> Result:
    """§1.3: comparison requires a benchmark 'known to a high accuracy and
    precision', studied 'fairly' — matched parameters, one variable."""
    bench = ctx.read("benchmark.py")
    controls = [c for c in ("oracle", "inverted", "random")
                if re.search(rf'"{c}"', bench)]
    arms = ctx.read("md_arms.py")
    matched = {
        "shared system": bool(re.search(r"system\.pdb", arms)),
        "shared charges": bool(re.search(r"charge_fingerprint", arms)),
        "paired seeds": bool(re.search(r"seed", arms)),
    }
    ro = ctx.read("md_readouts.py")
    asserts_match = bool(re.search(
        r"(assert|refus|mismatch|!=).{0,120}(seed|charge_fingerprint|system_sha)",
        ro + arms, re.I | re.S))

    missing = [k for k, v in matched.items() if not v]
    if len(controls) == 3 and not missing and asserts_match:
        return Result(
            PASS,
            "the A-vs-B design meets §1.3's 'fairly': a known-answer benchmark "
            f"with all three controls ({', '.join(controls)}), one shared "
            "system built once, shared charges, seeds paired across arms, and "
            "the comparison refuses to run when those do not match",
            evidence=f"benchmark.py controls={controls}; md_arms.py matched="
                     f"{sorted(matched)}; equality guard found={asserts_match}")
    gaps = []
    if len(controls) < 3:
        gaps.append(f"benchmark controls present: {controls or 'none'}")
    if missing:
        gaps.append(f"unmatched between arms: {missing}")
    if not asserts_match:
        gaps.append("nothing refuses the comparison when arms differ")
    return Result(
        FAIL,
        "the comparison does not meet §1.3's requirement that each method be "
        "studied 'fairly' on a benchmark whose answer is known: " + "; ".join(gaps),
        evidence="; ".join(gaps))


# --------------------------------------------------------------------------
# es-09  One easy model system does not license transfer
# --------------------------------------------------------------------------

def _no_transfer_from_one_system(ctx: Context) -> Result:
    """§1.3: 'the ground truth is seldom known beyond simple systems such as
    alanine dipeptide or a few fast-folding miniproteins, and there is no
    guarantee that methods that work well for those systems will work well on
    more complex systems.'"""
    text = ctx.read("DATASETS.md")
    if not text:
        return Result(UNTESTABLE,
                      "DATASETS.md is absent, so the target coverage of our "
                      "ground truth cannot be counted",
                      evidence="expected DATASETS.md at repo root")
    try:
        tier0 = text.split("## TIER 0", 1)[1].split("## TIER 1", 1)[0]
    except IndexError:
        return Result(UNTESTABLE,
                      "DATASETS.md has no parseable 'TIER 0' block, so held "
                      "datasets cannot be enumerated; testable once the table "
                      "structure is restored",
                      evidence="no '## TIER 0' ... '## TIER 1' span")

    held_affinity = []
    for line in tier0.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        name, targets, truth, status = cells[0], cells[2], cells[3], cells[-1]
        if "not held" in status.lower() or "---" in name:
            continue
        if not re.search(r"pkd|affinit", truth, re.I):
            continue
        m = re.search(r"\d+", targets)
        if m:
            held_affinity.append((re.sub(r"[*`]", "", name), int(m.group())))
    if not held_affinity:
        return Result(UNTESTABLE,
                      "no TIER 0 row parsed as a held dataset with affinity "
                      "ground truth; the table format has probably changed",
                      evidence="parsed 0 rows from the TIER 0 table")

    best = max(n for _, n in held_affinity)
    if best > 1:
        return Result(PASS,
                      f"hit-finding ground truth spans {best} targets, so a "
                      "protocol validated here is not being generalised from a "
                      "single system",
                      evidence=f"held with affinities: {held_affinity}")
    return Result(
        FAIL,
        "every dataset we hold with affinity ground truth covers exactly one "
        f"protein ({', '.join(n for n, _ in held_affinity)}), so any statement "
        "that a sampling protocol 'works' is a statement about one target. "
        "§1.3's warning is exactly this: results are known accurately only on a "
        "small set of systems, and 'there is no guarantee that methods that "
        "work well for those systems will work well on more complex systems'. "
        "Compounding it, FINDINGS §2.11 records that our richest-ground-truth "
        "subset is itself too easy to separate methods.",
        evidence=f"held with affinities: {held_affinity}; "
                 "DATASETS.md TIER 0 summary states we have never measured "
                 "hit-finding across more than one protein")


# --------------------------------------------------------------------------

CHECKS = [
    Check(id="es-01", severity="must", section="§8.2.4, §1",
          question="Is there any evidence that MD sampling left the basin it "
                   "started in?",
          measures="md_readouts.py channels + results/md_pilot/*/*/receipt.json",
          fn=_multi_basin_evidence),
    Check(id="es-02", severity="should", section="§9.1",
          question="Can aggregate sampling time — the quantity that governs "
                   "barrier crossing — be recovered from the artefact alone?",
          measures="results/md_pilot/*/*/receipt.json, readouts.json",
          fn=_aggregate_time_recorded),
    Check(id="es-03", severity="should", section="§5, §6",
          question="Does each trajectory record whether its production ensemble "
                   "was biased by a localizing potential?",
          measures="results/md_pilot/*/*/receipt.json + md_arms.py",
          fn=_production_bias_disclosed),
    Check(id="es-04", severity="should", section="§7.2.1",
          question="Is any sampling-derived quantity estimated a second, "
                   "independent way and the two compared?",
          measures="validate_gcmc_waters.py + results/gcmc_vs_crystal_waters.csv",
          fn=_independent_cross_check),
    Check(id="es-05", severity="should", section="§3.4",
          question="Where energies are collapsed with a Boltzmann weight, is it "
                   "stated that the set is not an equilibrium sample?",
          measures="*.py, cluster/*.py source",
          fn=_boltzmann_weighting_disclosed),
    Check(id="es-06", severity="should", section="§3.4, §8.1.1",
          question="For any declared lambda-window protocol, is inter-state "
                   "overlap computed rather than assumed?",
          measures="PLAN.md/RUNS.md/GOALS.md window declarations vs pymbar/"
                   "alchemlyb use and results/ artefacts",
          fn=_lambda_overlap_demonstrated),
    Check(id="es-07", severity="consider", section="§8.1.1, §8.2.4",
          question="For multi-state sampling: >=30% transitions, 20-40 "
                   "traversals of the state range, connected state graph?",
          measures="*.py, cluster/*.py for replica-exchange / expanded-ensemble "
                   "implementations",
          fn=_multistate_diagnostics),
    Check(id="es-08", severity="should", section="§1.3",
          question="Is the A-vs-B comparison run 'fairly' — known-answer "
                   "benchmark, matched parameters, one variable?",
          measures="benchmark.py controls + md_arms.py / md_readouts.py guards",
          fn=_comparison_is_fair),
    Check(id="es-09", severity="should", section="§1.3",
          question="Does the ground truth span more than one system, or is a "
                   "protocol's success being generalised from one?",
          measures="DATASETS.md TIER 0 table",
          fn=_no_transfer_from_one_system),
]
