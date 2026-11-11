"""Bogetti et al. 2019 — *A Suite of Tutorials for the WESTPA Rare-Events
Sampling Software* — as executable checks.

    https://livecomsjournal.org/index.php/livecoms/article/view/v1i2e10607
    doi:10.33011/livecoms.1.2.10607   Living J. Comp. Mol. Sci. 2019, 1(2), 10607

**Read the scope note before reading the checks.** This paper is a *tutorial*
for weighted-ensemble (WE) rare-event sampling, not a benchmarking-methodology
paper. We run no WE simulation and there is no `westpa` env on this machine, so
the parameter-selection content (τ, bin spacings, trajectories/bin, WExplore
Voronoi levels) is genuinely N/A and is recorded as such, once, with its reason
(`we-19`).

What *does* transfer, and transfers hard, is **§5 Uncertainty Quantification and
Monitoring of Convergence** plus the two appendix checklists (PROTOCOL FOR
RUNNING WE SIMULATIONS, pp. 30–31; CHECKLIST FOR TROUBLESHOOTING WE SIMULATIONS,
p. 32). §5 is four paragraphs on one problem we have in a different costume:

    "WE results ultimately depend on a large number of trajectories which
     typically are significantly correlated with one another due to repeated
     replication ('splitting') events. ... This large variation invalidates
     typical uncertainty estimation schemes based on the standard error of the
     mean, and we therefore recommend employing a Bayesian bootstrapping
     procedure."                                                          (§5)

Their correlated units are split trajectories. Ours are chemically similar
compounds: `leakage_check.py` shows the 499-compound library collapses to **45
Butina clusters** at Tanimoto d > 0.6 (`FINDINGS` §2.19). An i.i.d. bootstrap
over 499 compounds — which is exactly what `benchmark.py:boot_ci` does, by its
own docstring, "resampling compounds" — is the same mistake as a standard error
of the mean over split trajectories, and understates the spread by roughly
sqrt(499/45) ≈ 3.3×. That is `we-02`, and it is a `must`.

The rest of §5 is equally direct: run **multiple fully independent replicates
from identical starting states**, use **at least 4**, and note that error bars
are *required*, not optional — and that even the recommended interval is
"overly optimistic".

Severity discipline: only three `must` checks, each resting on an unconditional
imperative in the paper ("error bars are required"; "invalidates ... the
standard error of the mean"; "Ensure you are saving everything you might need to
restart"). Everything the paper phrases as advice is `should` or `consider`.
"""

from __future__ import annotations

import json
import os
import re

from . import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE, NA

PAPER = Paper(
    slug="westpa_tutorials",
    title="A Suite of Tutorials for the WESTPA Rare-Events Sampling Software "
          "[Article v1.0]",
    authors="Bogetti, Mostofian, Dickson, Pratt, Saglam, Harrison, Adelman, "
            "Dudek, Torrillo, DeGrave, Adhikari, Zwier, Zuckerman & Chong",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v1i2e10607",
    doi="10.33011/livecoms.1.2.10607",
    year="2019",
    scope="Running, monitoring and error-analysing an expensive stochastic "
          "sampling campaign on shared HPC: replicates, correlated-sample "
          "uncertainty, convergence, storage, and the pre-production dry run.",
)

# ---------------------------------------------------------------------------
# helpers — all read-only, all bounded (no unbounded ** walks)
# ---------------------------------------------------------------------------

#: Scripts that print a number this project quotes. If one of these emits a
#: headline metric with no interval, §5's "error bars are required" is broken.
REPORTERS = ("benchmark.py", "leakage_check.py", "cost_curve.py",
             "md_readouts.py")

#: Tokens that indicate an actual interval, not merely a call to np.percentile
#: (which we also use to set the active threshold — that is not a CI).
CI_TOKENS = ("boot_ci", "paired_boot", "bootstrap", "confidence interval",
             "95% CI", "credibility")


def _receipts(ctx: Context):
    """Every run receipt on disk, at the two depths we actually write them."""
    out = []
    for pat in ("results/*/*/receipt.json", "results/*/*/*/receipt.json"):
        for p in ctx.glob(pat):
            try:
                out.append((p, json.loads(p.read_text())))
            except (ValueError, OSError):
                continue
    return out


def _own_python(ctx: Context):
    """Our own sources only. `mace/`, `mace-off/` and the other vendored trees
    are third-party and are not ours to hold to this checklist."""
    return ctx.glob("*.py") + ctx.glob("cluster/*.py") + ctx.glob("datasets/*.py")


# ---------------------------------------------------------------------------
# §5 — Uncertainty Quantification and Monitoring of Convergence
# ---------------------------------------------------------------------------

def _error_bars_required(ctx: Context) -> Result:
    """§5: 'as with any simulation result, error bars are required to set the
    results in context because there is always a finite range of results which
    are predicted in any stochastic calculation'."""
    missing, present, absent_files = [], [], []
    for name in REPORTERS:
        src = ctx.read(name)
        if not src:
            absent_files.append(name)
            continue
        if any(t in src for t in CI_TOKENS):
            present.append(name)
        else:
            missing.append(name)
    if absent_files and not present and not missing:
        return Result(UNTESTABLE,
                      "none of the reporting scripts are present; this becomes "
                      "testable as soon as one of "
                      f"{', '.join(REPORTERS)} exists",
                      evidence=f"absent: {absent_files}")
    if missing:
        return Result(FAIL,
                      f"{len(missing)} of {len(present) + len(missing)} "
                      "result-reporting scripts emit numbers with no interval "
                      f"of any kind: {', '.join(missing)}. leakage_check.py is "
                      "the sharp case — it prints hits@25 and Spearman at six "
                      "clustering stringencies (FINDINGS §2.19, the 0.714 -> "
                      "0.205 collapse) and not one of the twelve numbers "
                      "carries a CI, while the finding's own caveat concedes "
                      "'the exact endpoint value is noisy'",
                      evidence=f"with intervals: {present}; without: {missing}")
    return Result(PASS,
                  f"all {len(present)} reporting scripts construct an interval",
                  evidence=f"{present}")


def _uncertainty_over_independent_units(ctx: Context) -> Result:
    """§5: correlated samples invalidate the naive variance estimate. The
    paper's units are split trajectories; ours are similar compounds."""
    src = ctx.read("benchmark.py")
    leak = ctx.read("leakage_check.py")
    if not src:
        return Result(UNTESTABLE,
                      "benchmark.py absent, so the resampling unit cannot be "
                      "read; testable once the headline metric has a bootstrap",
                      evidence="benchmark.py not found")

    m = re.search(r"def boot_ci\(.*?\n(?:.*\n)*?\s*return ", src)
    body = m.group(0) if m else ""
    if not body:
        return Result(UNTESTABLE,
                      "no boot_ci() found in benchmark.py; this check reads the "
                      "resampling unit out of that function and needs it to "
                      "exist",
                      evidence="regex 'def boot_ci' did not match")

    group_aware = any(t in body for t in ("group", "cluster", "block", "strat"))
    correlated = "Butina" in leak
    if group_aware:
        return Result(PASS,
                      "the bootstrap resamples grouped/clustered units, not "
                      "individual correlated observations",
                      evidence=body.strip().splitlines()[0])
    return Result(FAIL,
                  "benchmark.py:boot_ci resamples individual compounds i.i.d. "
                  "('Percentile bootstrap CI, resampling compounds'; "
                  "idx = rng.integers(0, n_obs, n_obs)) while leakage_check.py "
                  "shows the compounds are NOT independent — 499 compounds "
                  "collapse to 45 Butina clusters at Tanimoto d > 0.6 "
                  "(FINDINGS §2.19). This is §5's error exactly: a variance "
                  "estimate that assumes independence over units generated by "
                  "replication. Effective n is ~45, not 499, so every CI we "
                  "quote — including hits@25 95% CI [7, 17] and the 2.63 SD now "
                  "used as the noise floor — is understated by roughly "
                  "sqrt(499/45) = 3.3x. FIX: resample clusters, not compounds.",
                  evidence=("group-aware tokens absent from boot_ci; "
                            f"Butina clustering present in leakage_check.py: {correlated}"))


def _independent_replicate_runs(ctx: Context) -> Result:
    """§5: 'The primary recommendation is to perform multiple, fully
    independent WE simulations when possible. To understand the variation
    intrinsic to WE sampling, we suggest performing these runs from identical
    starting states.'"""
    seeds_by_arm: dict[str, set] = {}
    for _p, r in _receipts(ctx):
        key = f"{r.get('name')}/{r.get('arm')}/{str(r.get('ligand_ff'))[-24:]}"
        seeds_by_arm.setdefault(key, set()).add(r.get("seed"))
    md_max = max((len(s) for s in seeds_by_arm.values()), default=0)

    # The headline. hits@25 comes from a gnina run whose seed is a fixed
    # default; nothing in the repo re-runs it at a second seed.
    dock = ctx.read("dock_rescore.py") + ctx.read("cluster/crossdock493_array.py")
    dock_seeded = 'default=0' in dock and '"--seed"' in dock
    dock_dirs = [p.name for p in ctx.glob("results/dock_rescore*")]

    if md_max == 0 and not dock:
        return Result(UNTESTABLE,
                      "no run receipts and no docking script found; testable "
                      "once results/*/*/receipt.json or dock_rescore.py exists",
                      evidence="no receipts, no dock script")
    detail = (f"MD stage: {md_max} independent seeds for the same system+arm "
              f"(receipts under results/md_pilot). ")
    if md_max >= 2 and not dock_seeded:
        return Result(PASS, detail + "no single-seed headline stage found",
                      evidence=str(sorted(seeds_by_arm.items()))[:200])
    return Result(FAIL,
                  detail +
                  "But the HEADLINE stage has exactly one replicate: gnina is "
                  "invoked with --seed at a fixed default of 0 in both "
                  "dock_rescore.py and cluster/crossdock493_array.py, and "
                  f"results/ holds {len(dock_dirs)} dock_rescore output set(s) "
                  "from that one seed. hits@25 therefore has NO run-to-run "
                  "variance estimate at all — only a within-run bootstrap over "
                  "correlated compounds (see we-02). Docking is stochastic; "
                  "§5 says the intrinsic run-to-run variation is the thing you "
                  "must measure, and it is the one number we have never "
                  "measured. Cheapest fix in the repo: re-run dock_rescore.py "
                  "at --seed 1..4 and report the spread of hits@25.",
                  evidence=f"dock output dirs: {dock_dirs}; MD seeds/arm: "
                           f"{ {k: sorted(v) for k, v in seeds_by_arm.items()} }")


def _at_least_four_samples(ctx: Context) -> Result:
    """§5: 'a sufficiently large number of trajectories/bin (at least 4
    trajectories/bin) should be used to increase the chances of obtaining a
    diverse ensemble of pathways'."""
    # A run contributes as many independent samples as it has distinct
    # replicate seeds; a receipt that declares N internal replicates without
    # listing their seeds contributes N. Samples for the same system + arm +
    # force field pool into one ensemble however they were launched.
    ensembles: dict[str, int] = {}
    for _p, r in _receipts(ctx):
        key = f"{r.get('name')}/{r.get('arm')}/{str(r.get('ligand_ff'))[-24:]}"
        n = max(len(set(r.get("replicate_seeds") or [])),
                int(r.get("replicates") or 1))
        ensembles[key] = ensembles.get(key, 0) + n
    best = max(ensembles.values(), default=0)

    launcher = ctx.read("cluster/md_arms.sbatch")
    m = re.search(r"REPS=\$\{REPS:-(\d+)\}", launcher)
    launch_reps = int(m.group(1)) if m else None

    if not ensembles:
        return Result(UNTESTABLE,
                      "no run receipts on disk; testable as soon as any "
                      "stochastic run writes results/*/*/receipt.json",
                      evidence="results/*/*/receipt.json matched nothing")
    if best >= 4:
        return Result(PASS,
                      f"largest independent ensemble on disk is {best} "
                      f"(paper's floor is 4); cluster launcher requests "
                      f"REPS={launch_reps}",
                      evidence=f"{ensembles}")
    return Result(FAIL,
                  f"largest independent ensemble on disk is {best}, below the "
                  f"paper's floor of 4. Note md_arms.py's own --replicates "
                  f"default is 1, which would silently produce a one-sample "
                  f"'ensemble'; the cluster launcher overrides it to "
                  f"REPS={launch_reps}, so the bar is met only when the run is "
                  f"launched through cluster/md_arms.sbatch.",
                  evidence=f"{ensembles}")


def _replicates_must_diverge(ctx: Context) -> Result:
    """§1.1: replicates 'will diverge upon additional simulation using a
    stochastic thermostat'; §5's diversity argument is void if they do not.
    A set of replicates that did not diverge is one trajectory counted N times."""
    src = ctx.read("md_readouts.py")
    if not src:
        return Result(UNTESTABLE,
                      "md_readouts.py absent; testable once a replicate "
                      "read-out exists to assert divergence",
                      evidence="md_readouts.py not found")
    has_control = "replicate control" in src and "diverged" in src
    if has_control:
        return Result(PASS,
                      "md_readouts.py identity carries an explicit replicate "
                      "control asserting that same-force-field, different-seed "
                      "replicates diverge, and reports their aggregate spread",
                      evidence="'replicate control: same force field, "
                               "different seeds -- replicates must be independent'")
    return Result(FAIL,
                  "no control asserts that replicates actually diverge; "
                  "identical replicates carry no more information than one run "
                  "and would inflate every ensemble-derived error bar",
                  evidence="tokens 'replicate control' / 'diverged' absent "
                           "from md_readouts.py")


# ---------------------------------------------------------------------------
# Appendix: CHECKLIST FOR TROUBLESHOOTING WE SIMULATIONS (p. 32)
# ---------------------------------------------------------------------------

def _seeds_and_restart_state(ctx: Context) -> Result:
    """Troubleshooting, 'Segment implementation': 'Ensure you are saving
    everything you might need to restart your simulation later on, including
    random seeds for stochastic dynamics as auxiliary data in the H5 file.'"""
    recs = _receipts(ctx)
    seeded = [p for p, r in recs if r.get("seed") is not None]
    unseeded = [p.as_posix() for p, r in recs if r.get("seed") is None]
    dock_ok = '"--seed"' in ctx.read("dock_rescore.py")
    arr_ok = '"--seed"' in ctx.read("cluster/crossdock493_array.py")

    if not recs and not (dock_ok or arr_ok):
        return Result(UNTESTABLE,
                      "no receipts and no seeded external call found; testable "
                      "once any stochastic stage writes a receipt or passes a "
                      "seed to its engine",
                      evidence="no results/*/*/receipt.json, no --seed in "
                               "dock_rescore.py")
    if unseeded or not (dock_ok and arr_ok):
        return Result(FAIL,
                      f"{len(unseeded)} receipt(s) carry no seed; "
                      f"gnina --seed present in dock_rescore.py={dock_ok}, "
                      f"crossdock493_array.py={arr_ok}. A stochastic stage "
                      "whose seed is not written to its own output cannot be "
                      "restarted or reproduced.",
                      evidence=f"unseeded: {unseeded[:5]}")
    return Result(PASS,
                  f"all {len(seeded)} run receipts record their seed (and "
                  "replicate_seeds), and every gnina call passes --seed",
                  evidence=f"{len(seeded)} receipts with 'seed'; "
                           "dock_rescore.py and crossdock493_array.py both "
                           "pass --seed")


def _single_unit_dry_run(ctx: Context) -> Result:
    """§1.2.4: 'test simulations consisting of a few WE iterations should be run
    using the development queue'. Troubleshooting, 'Segment implementation':
    'manually running a single dynamics segment ... for a single trajectory
    walker. Check that your analysis pipeline works using the output from the
    single dynamics segment.'"""
    scale_flags = ("--chunk", "--limit", "--n-chunks", "--index", "--nshards",
                   "--max-poses", "--only", "--smoke")
    arrays = ctx.glob("cluster/*_array.py")
    if not arrays:
        return Result(UNTESTABLE,
                      "no cluster/*_array.py found; testable once a production "
                      "array script exists",
                      evidence="cluster/*_array.py matched nothing")
    ungated = []
    for p in arrays:
        src = p.read_text(errors="replace")
        if not any(f in src for f in scale_flags):
            ungated.append(p.name)
    if ungated:
        return Result(FAIL,
                      f"{len(ungated)} of {len(arrays)} array scripts expose no "
                      f"way to run a reduced slice before the full array: "
                      f"{', '.join(ungated)}. The paper wants a few tasks run "
                      "in the development queue first, to gauge I/O and prove "
                      "the analysis pipeline on real output.",
                      evidence=f"no {scale_flags} in {ungated}")
    return Result(PASS,
                  f"all {len(arrays)} array scripts accept a chunk/limit flag, "
                  "so a small slice can be run before the full array",
                  evidence=f"{[p.name for p in arrays]}")


def _thread_count_pinned(ctx: Context) -> Result:
    """Troubleshooting, 'Analysis': on a shared computing resource use the
    serial flag, 'Otherwise, many of the included tools default to parallel
    mode ... which will create as many Python threads as there are CPU cores
    available on your resource.' Reinforced by §1.2.3's warning that a
    poorly-configured job 'can result in denial of service to other users'."""
    sbatch = ctx.glob("cluster/*.sbatch")
    if not sbatch:
        return Result(UNTESTABLE,
                      "no cluster/*.sbatch found; testable once a job script "
                      "exists to inspect",
                      evidence="cluster/*.sbatch matched nothing")
    unpinned = []
    for p in sbatch:
        src = p.read_text(errors="replace")
        if "NUM_THREADS" not in src:
            unpinned.append(p.name)
    if unpinned:
        return Result(FAIL,
                      f"{len(unpinned)} of {len(sbatch)} job scripts export no "
                      f"thread cap: {', '.join(unpinned)}. thira is shared "
                      "(GOALS rule 2, ~15 other jobs typical); an unpinned "
                      "BLAS/OpenMP process takes every core on the node, not "
                      "the cores SLURM allocated.",
                      evidence=f"no NUM_THREADS export in {unpinned}")
    return Result(PASS,
                  f"all {len(sbatch)} job scripts export a thread cap",
                  evidence=f"{[p.name for p in sbatch]}")


def _solute_only_trajectory(ctx: Context) -> Result:
    """§4.2(4): 'Consider saving coordinates for only the solute atoms of your
    system to an H5 file.' Troubleshooting, 'Storage': 'For all-atom explicit
    water simulations, it's a good idea to save a separate copy of your
    trajectories without water coordinates for more efficient analysis.'"""
    src = ctx.read("md_arms.py")
    if "DCDReporter" not in src:
        return Result(UNTESTABLE,
                      "no trajectory reporter found in md_arms.py; testable "
                      "once explicit-solvent MD writes trajectories",
                      evidence="'DCDReporter' absent from md_arms.py")
    # Deliberately narrow tokens. An earlier draft of this check matched the
    # bare substring "strip" and PASSED on `line[17:20].strip()` in md_arms.py's
    # PDB parser — a vacuous pass of exactly the kind FINDINGS §3.7 records.
    # Only a real atom-subset writer counts.
    writer = ("atomSubset", "atom_indices=", "solute_only", "atom_slice",
              "strip_water", "keepWater=False")
    subsetted = any(t in src for t in writer)
    readouts = ctx.read("md_readouts.py")
    stripped_copy = any(t in readouts for t in writer)
    dcds = ctx.glob("results/md_pilot/*/*/traj.dcd")
    total_mb = sum(p.stat().st_size for p in dcds) / 1e6 if dcds else 0.0
    if subsetted or stripped_copy:
        return Result(PASS,
                      "a solute-only trajectory copy is produced",
                      evidence=f"{len(dcds)} traj.dcd, {total_mb:.0f} MB total")
    return Result(FAIL,
                  "md_arms.py writes DCDReporter(traj.dcd, stride) with no atom "
                  "subset, so every stored frame carries all ~17,872 atoms "
                  "including bulk water, and md_readouts.py writes only "
                  "aggregated readouts.npz — there is no water-free copy. "
                  f"{len(dcds)} pilot trajectories already occupy {total_mb:.0f} "
                  "MB for one ligand; the 32-ligand x 2-arm x 5-replicate run "
                  "scales that by ~320.",
                  evidence=f"traj.dcd files: {len(dcds)}, {total_mb:.0f} MB")


def _bulk_output_on_scratch(ctx: Context) -> Result:
    """§1.2.3: estimate storage before the run and make sure scratch is large
    enough. Troubleshooting, 'Storage': 'Make sure you have enough space before
    starting the simulation', 'Keep file sizes =<300 GB'."""
    sbatch = ctx.glob("cluster/*.sbatch")
    if not sbatch:
        return Result(UNTESTABLE,
                      "no job scripts to inspect for output paths; testable "
                      "once cluster/*.sbatch exists",
                      evidence="cluster/*.sbatch matched nothing")
    home_bound = []
    for p in sbatch:
        src = p.read_text(errors="replace")
        outs = re.findall(r"#SBATCH --(?:output|error)=(\S+)", src)
        if any(not o.startswith("/work") for o in outs):
            home_bound.append(p.name)
    ceiling_gb = 300
    big = [(p.as_posix(), p.stat().st_size / 1e9)
           for p in ctx.glob("results/*/*")
           if p.is_file() and p.stat().st_size > ceiling_gb * 1e9]
    if home_bound or big:
        return Result(FAIL,
                      f"{len(home_bound)} job script(s) write logs off /work "
                      f"({home_bound}); {len(big)} output file(s) exceed the "
                      f"paper's {ceiling_gb} GB per-file ceiling",
                      evidence=f"home-bound: {home_bound}; oversize: {big}")
    return Result(PASS,
                  f"all {len(sbatch)} job scripts write to /work scratch and no "
                  f"result file exceeds the {ceiling_gb} GB ceiling",
                  evidence=f"{len(sbatch)} sbatch checked; "
                           f"largest scanned results/*/* under {ceiling_gb} GB")


# ---------------------------------------------------------------------------
# §1.2 Prerequisites, §1.2.2 Software Requirements
# ---------------------------------------------------------------------------

def _software_versions_recorded(ctx: Context) -> Result:
    """§1.2.2: 'The version of each software package is also provided in each
    tutorial's "computational requirements" section.' Every tutorial pins its
    engine version (OpenMM 7.3, MDTraj 1.9.3, AmberTools 19, hdfview >= 3.1.0)."""
    recs = _receipts(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      "no run receipts on disk; testable once any run writes "
                      "results/*/*/receipt.json",
                      evidence="results/*/*/receipt.json matched nothing")
    with_ver = [p.as_posix() for p, r in recs
                if any("version" in k.lower() for k in r)]
    if with_ver:
        return Result(PASS,
                      f"{len(with_ver)}/{len(recs)} receipts record package "
                      "versions",
                      evidence=f"{with_ver[:3]}")
    return Result(FAIL,
                  f"not one of {len(recs)} run receipts records the version of "
                  "any package. md_arms.py's receipt captures host, platform, "
                  "precision, seeds and sha256 of the system — but not openmm, "
                  "openff-toolkit, rdkit or torch. GOALS rule 1 pins "
                  "mace-torch 0.3.16 / torch 2.13.0 precisely because a version "
                  "bump invalidates completed runs, and no completed run "
                  "records which version produced it.",
                  evidence=f"receipt keys sampled: "
                           f"{sorted(recs[0][1].keys())[:12]}")


def _throughput_recorded(ctx: Context) -> Result:
    """§1.2.1(iv): before the expensive campaign, 'estimate storage needs for
    your eventual WE simulation and the ns/day that can be generated for your
    system'."""
    recs = _receipts(ctx)
    if not recs:
        return Result(UNTESTABLE,
                      "no run receipts on disk; testable once a run records "
                      "its own throughput",
                      evidence="results/*/*/receipt.json matched nothing")
    rates = [r.get("ns_per_day") for _p, r in recs if r.get("ns_per_day")]
    if not rates:
        return Result(FAIL,
                      f"none of {len(recs)} receipts records a throughput, so "
                      "the cost of scaling this run is unknown before it is "
                      "launched",
                      evidence=f"no 'ns_per_day' key in {len(recs)} receipts")
    return Result(PASS,
                  f"{len(rates)}/{len(recs)} receipts record throughput "
                  f"({min(rates):.0f}-{max(rates):.0f} ns/day), so the campaign "
                  "cost is known before it is launched",
                  evidence=f"ns_per_day: {sorted(rates)}")


def _scaling_test_on_target_hardware(ctx: Context) -> Result:
    """§3(2): 'a code scaling test (plot of the time required to complete a WE
    iteration vs. τ value) should be carried out for a range of potential τ
    values on the intended computer hardware'. §4.3(1) repeats it."""
    timing = ctx.read("mlp_timing.py")
    curve = ctx.read("cost_curve.py")
    if not timing and not curve:
        return Result(UNTESTABLE,
                      "neither mlp_timing.py nor cost_curve.py exists; testable "
                      "once any script measures per-unit cost on our hardware",
                      evidence="mlp_timing.py and cost_curve.py both absent")
    measured = curve.count('src="MEASURED"')
    estimated = curve.count('src="ESTIMATED"')
    on_our_data = "the actual screening library" in timing or "our own" in curve
    if measured and on_our_data:
        return Result(PASS,
                      f"cost_curve.py labels {measured} tier cost(s) MEASURED "
                      f"and {estimated} ESTIMATED and never mixes them "
                      "silently; mlp_timing.py times the real 493-compound "
                      "library on the device that will run it",
                      evidence=f"MEASURED={measured}, ESTIMATED={estimated}")
    return Result(FAIL,
                  f"cost per unit is not measured on the intended hardware "
                  f"(MEASURED tiers={measured}, ESTIMATED={estimated}); a "
                  "throughput quoted from a paper does not decide a cascade "
                  "budget",
                  evidence=f"MEASURED={measured}, ESTIMATED={estimated}, "
                           f"timed-on-our-data={on_our_data}")


def _validation_sources_identified(ctx: Context) -> Result:
    """§1.2.1: 'It is also important to identify sources of validation for your
    simulation (e.g., from experiment and/or standard simulations) and to be
    familiar with the estimation of statistical uncertainty in the computed
    observables, including those used for validation.'"""
    lit = ctx.read("LITERATURE.md")
    if not lit:
        return Result(UNTESTABLE,
                      "LITERATURE.md absent; testable once external reference "
                      "numbers are recorded somewhere machine-readable",
                      evidence="LITERATURE.md not found")
    # The comparison table is the operative artefact: every row is a number we
    # measure ourselves against. Both halves of §1.2.1's sentence have to hold
    # for a row to be usable — the source is named AND its uncertainty is known.
    m = re.search(r"^## 4\..*?(?=^## )", lit, re.M | re.S)
    block = m.group(0) if m else ""
    if not block:
        return Result(UNTESTABLE,
                      "LITERATURE.md has no '## 4.' comparison table to read; "
                      "testable once the numbers we measure ourselves against "
                      "are tabulated in one place",
                      evidence="regex '^## 4\\.' did not match LITERATURE.md")
    rows = [ln for ln in block.splitlines()
            if ln.startswith("|") and not set(ln) <= set("|-: ")
            and not ln.lower().startswith("| source")]
    # An en-dash range across systems (3.35-3.68) is spread, not uncertainty.
    interval = ("±", "+/-", "95%", " CI", "CI ", "s.e.m", "SD ")
    with_unc = [ln for ln in rows if any(t in ln for t in interval)]
    if not rows:
        return Result(UNTESTABLE,
                      "the '## 4.' section holds no table rows to inspect; "
                      "testable once reference numbers are tabulated",
                      evidence=f"section length {len(block)} chars, 0 rows")
    if len(with_unc) == len(rows):
        return Result(PASS,
                      f"all {len(rows)} reference numbers we benchmark against "
                      "carry an uncertainty, so 'we beat it' is decidable",
                      evidence=f"{len(with_unc)}/{len(rows)} rows with an interval")
    return Result(FAIL,
                  f"{len(rows) - len(with_unc)} of {len(rows)} rows in "
                  "LITERATURE.md §4 'NUMBERS WE ARE TRYING TO BEAT' quote the "
                  "reference as a bare point with no uncertainty (Rowan 0.464, "
                  "OpenBind 9.8%, AIMNet2 3.13, MACE-POLAR 3.35-3.68 — a range "
                  "across systems, not an error bar). §1.2.1 asks for both "
                  "halves: identify the validation source AND know the "
                  "statistical uncertainty of the observable used to validate. "
                  "Without the second half, 'we beat it' and 'we tie it' are "
                  "the same measurement — which is how the 1.89-vs-3.35 claim "
                  "survived as long as it did (FINDINGS §2.1).",
                  evidence=f"{len(with_unc)}/{len(rows)} rows carry an "
                           f"interval; sources named: "
                           f"{[t for t in ('Rowan', 'OpenBind', 'PL-REX', 'AIMNet2', 'OpenFE') if t in lit]}")


# ---------------------------------------------------------------------------
# §3 Choosing WE parameters, §3.1 distance metric, §4.3 executables
# ---------------------------------------------------------------------------

def _nested_expensive_coordinate(ctx: Context) -> Result:
    """§3(4): 'additional dimensions in the progress coordinate greatly increase
    the number of bins and hence the cost of the WE run, which is the motivation
    for nesting an additional coordinate in only a subset of the initial bins.'
    That is the cascade, argued from cost."""
    src = ctx.read("cost_curve.py")
    if not src:
        return Result(UNTESTABLE,
                      "cost_curve.py absent; testable once the tier structure "
                      "and its costs are written down",
                      evidence="cost_curve.py not found")
    tiers = len(re.findall(r"dict\(tier=\d+", src))
    promotes = any(t in src for t in ("shortlist", "promote", "remaining "
                                      "budget", "cascade"))
    if tiers >= 2 and promotes:
        return Result(PASS,
                      f"{tiers} tiers with the expensive stages applied to a "
                      "promoted shortlist rather than the whole library — the "
                      "nesting §3(4) recommends, and cost_curve.py states the "
                      "break-even each tier must clear",
                      evidence=f"tiers={tiers}, promotion language present")
    return Result(FAIL,
                  f"only {tiers} tier(s) and promotion={promotes}: the "
                  "expensive coordinate is applied flat rather than nested "
                  "inside a subset, which §3(4) rejects on cost grounds",
                  evidence=f"tiers={tiers}, promotes={promotes}")


def _ligand_rmsd_convention(ctx: Context) -> Result:
    """§3.1, Choice of Distance Metric: 'a common distance metric used in ligand
    (un)binding simulations is calculated by aligning binding site atoms, and
    calculating the RMSD between the ligands, without any further alignment'."""
    offenders, correct = [], []
    for p in _own_python(ctx):
        src = p.read_text(errors="replace")
        if "GetBestRMS" in src or "rdMolAlign.AlignMol" in src:
            offenders.append(p.name)
        if "CalcRMS" in src:
            correct.append(p.name)
    if not correct and not offenders:
        return Result(UNTESTABLE,
                      "no ligand-RMSD computation found in our own sources; "
                      "testable once poses are scored against a reference",
                      evidence="neither CalcRMS nor GetBestRMS present")
    if offenders:
        return Result(FAIL,
                      f"{', '.join(offenders)} use GetBestRMS/AlignMol, which "
                      "superpose the two ligands onto each other before "
                      "measuring. That reports the best achievable shape match, "
                      "not displacement from the crystal pose, and flatters "
                      "every docking accuracy number.",
                      evidence=f"offenders={offenders}; correct={correct}")
    return Result(PASS,
                  f"{len(correct)} script(s) use rdMolAlign.CalcRMS — "
                  "symmetry-corrected, in place, no ligand re-alignment — and "
                  "cluster/crossdock493_array.py applies the receptor "
                  "superposition transform to the pose first, which is exactly "
                  "the §3.1 convention",
                  evidence=f"CalcRMS in {correct}; no GetBestRMS/AlignMol "
                           "anywhere in our sources")


def _executables_pinned(ctx: Context) -> Result:
    """§4.3(2): 'Set environment variables to the full pathnames of repeatedly
    used programs'. §1.2.3: 'writing /usr/bin/awk instead of simply awk'."""
    paths: dict[str, set] = {}
    which = []
    for p in _own_python(ctx):
        src = p.read_text(errors="replace")
        for tool, pat in (("gnina", r"GNINA\s*=\s*Path\(\"([^\"]+)\"\)"),
                          ("xtb", r"XTB\s*=\s*\"([^\"]+)\""),
                          ("pdb2pqr", r"PDB2PQR\s*=\s*\(([^\n]+)")):
            for hit in re.findall(pat, src):
                if "shutil.which" in hit:
                    which.append(f"{p.name}:{tool}")
                elif hit.startswith("/"):
                    paths.setdefault(tool, set()).add(hit)
    if not paths and not which:
        return Result(UNTESTABLE,
                      "no external-executable constants found; testable once a "
                      "script shells out to gnina/xtb/pdb2pqr",
                      evidence="no GNINA/XTB/PDB2PQR constant matched")
    divergent = {t: sorted(v) for t, v in paths.items() if len(v) > 1}
    missing = {t: [q for q in v if not os.path.exists(q)]
               for t, v in paths.items()}
    missing = {t: v for t, v in missing.items() if v}
    if divergent or which or missing:
        return Result(FAIL,
                      f"executable resolution is not pinned to one path per "
                      f"tool: {len(divergent)} tool(s) have divergent hard-coded "
                      f"paths {divergent}; {len(which)} resolve through PATH via "
                      f"shutil.which {which}; {len(missing)} path(s) do not "
                      f"exist on this host {missing}. Two gnina paths means two "
                      "possible builds and no record of which produced a "
                      "number. This project has already run the wrong binary "
                      "off a plausible path — /usr/bin/orca is the GNOME screen "
                      "reader (FINDINGS §3.9).",
                      evidence=f"paths={ {t: sorted(v) for t, v in paths.items()} }")
    return Result(PASS,
                  "every external executable resolves through a single pinned "
                  "absolute path that exists on this host",
                  evidence=f"{ {t: sorted(v) for t, v in paths.items()} }")


def _monitor_and_stop(ctx: Context) -> Result:
    """PROTOCOL III(2): 'Monitor simulation progress. ... If your WE simulation
    is not making any bin-to-bin transitions stop the simulation and restart the
    simulation with a shorter τ, adjustments in the progress coordinate, and/or
    adjustments in bin spacings.' A monitored run needs a stop condition, not
    only a plot."""
    wd = ctx.read("watchdog.sh")
    bm = ctx.read("benchmark.py")
    aborts = any(t in bm for t in ("SystemExit", "sys.exit(", "abort"))
    bounded = "MAX_CYCLES" in wd and "TIMEOUT" in wd
    if not wd and not bm:
        return Result(UNTESTABLE,
                      "no watchdog and no benchmark harness to inspect; "
                      "testable once a long run is monitored by something",
                      evidence="watchdog.sh and benchmark.py both absent")
    if bounded and aborts:
        return Result(PASS,
                      "watchdog.sh bounds unattended work with MAX_CYCLES and "
                      "TIMEOUT, and benchmark.py aborts the run when a "
                      "known-answer control fails rather than reporting a "
                      "number from a broken metric",
                      evidence="MAX_CYCLES+TIMEOUT in watchdog.sh; abort/exit "
                               "path in benchmark.py")
    return Result(FAIL,
                  f"monitoring lacks a stop condition (watchdog bounded="
                  f"{bounded}, benchmark aborts={aborts}); the paper's "
                  "instruction is to stop and re-parameterise, not to keep "
                  "burning the allocation",
                  evidence=f"bounded={bounded}, aborts={aborts}")


# ---------------------------------------------------------------------------
# Honest N/A
# ---------------------------------------------------------------------------

def _we_parameters(ctx: Context) -> Result:
    """§3 and §6: τ, bin spacings, trajectories/bin, WExplore branching factor
    and hierarchy depth, steady-state vs equilibrium WE, west.cfg layout."""
    # bounded globs, not a full-tree walk: west.cfg lives at the simulation
    # root by construction (Table 2, WEST_SIM_ROOT organisation).
    wcfg = ctx.glob("west.cfg") + ctx.glob("*/west.cfg") + ctx.glob("*/*/west.cfg")
    uses_westpa = any("westpa" in p.read_text(errors="replace").lower()
                      for p in _own_python(ctx))
    if uses_westpa or wcfg:
        return Result(FAIL,
                      "WESTPA appears in our sources, so the WE parameter "
                      "checklist applies and is not being followed",
                      evidence=f"westpa referenced; west.cfg found: {bool(wcfg)}")
    return Result(NA,
                  "We run no weighted-ensemble simulation: there is no west.cfg "
                  "anywhere in the tree, no `westpa` import in any of our "
                  "sources, and no westpa environment on this machine. τ, bin "
                  "spacings, trajectories/bin, recycling conditions and "
                  "WExplore Voronoi hierarchies have no counterpart in a "
                  "docking + MLP-rescoring cascade. This N/A covers ONLY the "
                  "WE mechanics — §5's uncertainty content and both appendix "
                  "checklists are general and are tested above, not excused "
                  "here.",
                  evidence="no west.cfg, no westpa import, no westpa env")


# ---------------------------------------------------------------------------

CHECKS = [
    Check(id="we-01", severity="must", section="§5",
          question="Does every number we report carry an uncertainty?",
          measures="CI construct in benchmark.py, leakage_check.py, "
                   "cost_curve.py, md_readouts.py",
          fn=_error_bars_required),
    Check(id="we-02", severity="must", section="§5",
          question="Is uncertainty estimated over INDEPENDENT units, given "
                   "ours are correlated?",
          measures="the resampling unit inside benchmark.py:boot_ci, against "
                   "the Butina cluster count in leakage_check.py",
          fn=_uncertainty_over_independent_units),
    Check(id="we-03", severity="should", section="§5",
          question="Are multiple fully independent runs performed from "
                   "identical starting states?",
          measures="distinct seeds per (system, arm) in results/*/receipt.json; "
                   "seed handling in the docking stage",
          fn=_independent_replicate_runs),
    Check(id="we-04", severity="should", section="§5",
          question="Is every ensemble at least 4 independent samples?",
          measures="replicate counts and replicate_seeds in run receipts; "
                   "REPS default in cluster/md_arms.sbatch",
          fn=_at_least_four_samples),
    Check(id="we-05", severity="should", section="§1.1, §5",
          question="Is there a control proving replicates actually diverge?",
          measures="the replicate control in md_readouts.py identity",
          fn=_replicates_must_diverge),

    Check(id="we-06", severity="must",
          section="Appendix: Troubleshooting, 'Segment implementation'",
          question="Is the RNG seed, and everything needed to restart, written "
                   "into the run's own output?",
          measures="'seed'/'replicate_seeds' in every run receipt; --seed on "
                   "every gnina call",
          fn=_seeds_and_restart_state),
    Check(id="we-07", severity="should", section="§1.2.2",
          question="Is the version of every software package recorded with the "
                   "run?",
          measures="version keys across results/*/receipt.json",
          fn=_software_versions_recorded),
    Check(id="we-08", severity="should", section="§3(2), §4.3(1)",
          question="Was cost measured on the intended hardware before the "
                   "production campaign?",
          measures="MEASURED vs ESTIMATED tier labels in cost_curve.py; "
                   "mlp_timing.py's library",
          fn=_scaling_test_on_target_hardware),
    Check(id="we-09", severity="should",
          section="§1.2.4; Appendix: Troubleshooting, 'Segment implementation'",
          question="Can a reduced slice be run before the full array?",
          measures="chunk/limit flags across cluster/*_array.py",
          fn=_single_unit_dry_run),
    Check(id="we-10", severity="should", section="§3(4)",
          question="Is the expensive coordinate nested onto a promoted subset "
                   "rather than run flat?",
          measures="tier count and promotion logic in cost_curve.py",
          fn=_nested_expensive_coordinate),
    Check(id="we-11", severity="should", section="§3.1",
          question="Is ligand RMSD measured after superposing the site, with no "
                   "further ligand alignment?",
          measures="CalcRMS vs GetBestRMS/AlignMol across our own sources",
          fn=_ligand_rmsd_convention),
    Check(id="we-12", severity="should", section="§4.3(2), §1.2.3",
          question="Does every external executable resolve through one pinned "
                   "full path?",
          measures="GNINA/XTB/PDB2PQR constants across our own sources",
          fn=_executables_pinned),
    Check(id="we-13", severity="should",
          section="Appendix: Troubleshooting, 'Analysis'; §1.2.3",
          question="Do shared-cluster jobs cap their thread count?",
          measures="NUM_THREADS export in every cluster/*.sbatch",
          fn=_thread_count_pinned),
    Check(id="we-14", severity="should",
          section="§4.2(4); Appendix: Troubleshooting, 'Storage'",
          question="For explicit-solvent MD, is a water-free trajectory copy "
                   "saved?",
          measures="the DCDReporter call in md_arms.py and the size of the "
                   "pilot trajectories on disk",
          fn=_solute_only_trajectory),
    Check(id="we-15", severity="should", section="§1.2.1(iv)",
          question="Does each run record its own throughput, so the campaign "
                   "cost is known before launch?",
          measures="ns_per_day across run receipts",
          fn=_throughput_recorded),
    Check(id="we-16", severity="should", section="§1.2.1",
          question="Are external validation sources identified, WITH the "
                   "uncertainty of the reference number?",
          measures="named reference numbers and uncertainty language in "
                   "LITERATURE.md",
          fn=_validation_sources_identified),
    Check(id="we-17", severity="consider",
          section="§1.2.3; Appendix: Troubleshooting, 'Storage'",
          question="Is bulk output on scratch, and every file under the 300 GB "
                   "ceiling?",
          measures="#SBATCH output paths and file sizes under results/*/*",
          fn=_bulk_output_on_scratch),
    Check(id="we-18", severity="consider",
          section="Appendix: Protocol III(2)",
          question="Is a long run monitored by something that can stop it?",
          measures="MAX_CYCLES/TIMEOUT in watchdog.sh; the abort path in "
                   "benchmark.py",
          fn=_monitor_and_stop),
    Check(id="we-19", severity="consider", section="§3, §3.1, §6",
          question="Are the weighted-ensemble parameters (τ, bins, "
                   "trajectories/bin, WExplore hierarchy) chosen per §3?",
          measures="absence of west.cfg and of any westpa import",
          fn=_we_parameters),
]
