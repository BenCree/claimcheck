"""Hatch et al., *Best Practices for Developing Monte Carlo Methodologies in
Molecular Simulations [Article v1.0]*, LiveCoMS 6(1):3289 (2025),
doi:10.33011/livecoms.6.1.3289.

**Why this paper governs us.** We run grand canonical Monte Carlo water
sampling: `grand.samplers.StandardGCMCSphereSampler` inside an 8 A sphere,
across 1,096 receptors in five cohorts, and the occupancies it produces are the
input to the water thesis. Every one of those numbers is a mu-V-T ensemble
average whose validity rests on the acceptance probability of an insertion and a
deletion trial. This is the paper that says how such a trial is derived, and how
you show that the derivation was implemented correctly rather than merely
written down.

**What the paper actually contains.** A boxed **BEST PRACTICES CHECKLIST FOR
DERIVING MC TRIAL ACCEPTANCE** on page 5, reproduced in `_CHECKLIST` below so
that a reader of this file can see which items are encoded and which are not.
Most of its items govern a derivation on paper and cannot be measured against a
run. Four can: the random number generator must be seeded for reproducibility;
the ensemble of the trial must be determined and its microstate probabilities
used; the implementation must be tested against a tractable expectation; and the
derivation or implementation must be documented for later reference. The
remaining checks here come from Section 2 (the acceptance probability and
detailed balance), Section 7.2 (Eqs. 28 and 29, the mu-V-T insertion and
deletion acceptances this sampler implements), Section 12.1 (energy tests) and
Section 12.2.1 (block averages, with equilibration discarded).

**The situation this module exists to catch.** On all 489 runs of job 81166 the
sampler attempted 35,000 exchange moves and accepted a median of 13. A water
count that is flat because almost nothing is being exchanged is indistinguishable
from one that is flat because it has equilibrated, and an occupancy computed
under near-zero exchange measures how long one water happened to sit in a place
under molecular dynamics rather than the probability that the site is wet.
`mc-05` is the check that separates the two, and it needs no threshold: if the
number of accepted moves over the whole run is smaller than the total variation
of the water count over production alone, then most of the recorded fluctuation
cannot be exchange, whatever the acceptance rate was.

**What is measured, and what is read.** Every check reads a file the campaign
wrote, or the source of the sampler that was actually installed, never a script's
statement of what it meant to do. Three checks parse
`grand/samplers.py` from the interpreter the cluster job itself names, so that a
`grand` upgrade that changed Eq. 28 would be caught here rather than in a result.

**Inputs are discovered, not hardcoded.** A GCMC run is any directory holding a
`timing.json` that records `gcmc_moves_attempted`, so a cohort added later is
checked with no edit to this file. Set `ALMACE_GCMC_RUNS` to a colon-separated
list of directories or globs to point the module at a different set. Verdicts are
computed per cohort and the worst cohort decides, because a mean over cohorts
would report which cohorts happened to be on disk.

Section numbers are those of the published article:

    Sec. 2 acceptance and detailed balance . Sec. 3 the boxed checklist .
    Sec. 7 unbiased grand canonical trials . Sec. 12.1 energy tests .
    Sec. 12.2.1 block averages . Sec. 12.3 workflow for testing a new trial

Two of the checks below PASS on the current data and one is N/A with a measured
reason, which is deliberate: a checklist that can only complain is one nobody
runs, and one that cannot complain has not checked anything.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from io import StringIO
from pathlib import Path

from . import NA, PASS, FAIL, UNTESTABLE, Check, Context, Paper, Result

PAPER = Paper(
    slug="monte_carlo",
    title="Best Practices for Developing Monte Carlo Methodologies in "
          "Molecular Simulations [Article v1.0]",
    authors="Harold W. Hatch, David S. Corti, David A. Kofke, Vincent K. Shen",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v6i1e3289",
    doi="10.33011/livecoms.6.1.3289",
    year="2025",
    scope="How a Monte Carlo trial's acceptance probability is derived, "
          "implemented and shown to be correct: the ensemble fixes the ratio "
          "of microstate probabilities, the trial's own steps fix the ratio of "
          "transition probabilities, the random number generator is seeded, "
          "and the result is tested against a tractable expectation before it "
          "is used on anything real.",
)

#: The boxed checklist on page 5, printed here so a reader can see which items
#: this module encodes. Items marked `-> mc-NN` are executable against our
#: files; the rest govern a derivation on paper and are not measurable from a
#: run directory. Listing them rather than dropping them is the point: an
#: unencoded item is work not done, not an item the paper does not have.
_CHECKLIST = """
  Write the MC trial algorithm as an ordered list of all steps required to
    fully describe both the forward and reverse trials.          (on paper)
  Check detailed balance.                                        -> mc-05, mc-09
  Determine if each step is deterministic, stochastic or needs to be broken
    into multiple steps.
      Stochastic steps often use a random number generator
        (seeded for reproducibility).                            -> mc-02
  Determine the probability for each step.                       (on paper)
  Determine the ensemble of the trial.                           -> mc-01, mc-12
  Use Eq. 4 to obtain the trial acceptance probability.          -> mc-08
  Test the implementation of the trial acceptance probability.
      Verify that the acceptance probability leads to simulation results that
        agree with theory and trusted sources.                   -> mc-07, mc-04
      Start with the simplest possible simulations that are fast, such as
        ideal gases with tractable theoretical expressions.      -> mc-07
  Document your derivation or software implementation for later
    reference.                                                   -> mc-11
"""

# --------------------------------------------------------------------------
# constants, every one of them from the paper or from the sampler's own
# documented defaults. None is tuned to our data.
# --------------------------------------------------------------------------

#: Sec. 12.2.1, Code Block 10: `np.array_split(values[equil:], block_size)`
#: with `block_size=100`, so the paper's own mu-V-T example forms 100 blocks
#: after discarding equilibration. A block containing no accepted exchange holds
#: one value of N repeated, which is not a sample of N. So the number of
#: accepted exchanges is the ceiling on how many blocks the paper's own
#: procedure could form, and 100 is the number it forms.
BLOCKS_IN_THE_PAPERS_EXAMPLE = 100

#: Sec. 12.2.1, the same code block: the first 1,000 trials are discarded as
#: equilibration before any average is formed.
EQUILIBRATION_IS_DISCARDED = True

#: `grand.samplers.GCMCSphereSampler.__init__` defaults, and the temperature
#: `gcmc_apo.py` passes to `BAOABIntegrator`. These are NOT recorded in
#: `timing.json`, which is what mc-12 reports; mc-01 recomputes the Adams value
#: from them and says out loud that it had to assume them.
EXCESS_MU_KCAL_PER_MOL = -6.09
STANDARD_VOLUME_A3 = 30.345
TEMPERATURE_K = 300.0
R_KCAL_PER_MOL_K = 0.001987204259

#: The Adams value is a logarithm, so a relative tolerance is the wrong shape.
#: grand computes B in float64 from a radius we record to one decimal place;
#: anything beyond this is a different radius or a hand-supplied `adams`.
ADAMS_ABS_TOL = 1e-6

#: TIP3P at 300 K and 1 bar, 0.0332 molecules per cubic angstrom. Used only by
#: mc-07, and only if a water-only reference run exists to compare against.
BULK_WATER_DENSITY_PER_A3 = 0.0332
#: Sec. 12.3 asks for agreement with a trusted source, not for exactness. A
#: water-only sphere at the equilibrium Adams value should hold the bulk number
#: within this fraction; wider than this and the acceptance is not reproducing
#: the density it was constructed to reproduce.
BULK_TOLERANCE = 0.10

#: One-sided tail below which a run's accepted count is declared inconsistent
#: with the bench that justified the sampler choice (mc-04).
BENCH_ALPHA = 1e-3

_CACHE: dict = {}


# --------------------------------------------------------------------------
# discovery and measurement helpers
# --------------------------------------------------------------------------

_RUN_PATTERNS = (
    "datasets/*/*/data/*/*/timing.json",
    "datasets/*/*/data/*/timing.json",
    "datasets/*/data/*/*/timing.json",
    "results/*/*/timing.json",
    "results/*/*/*/timing.json",
)


def _gcmc_runs(ctx: Context) -> list[dict]:
    """Every GCMC run on disk, found by what its `timing.json` records.

    A run qualifies on content, `gcmc_moves_attempted` being present, not on
    where it sits, so a cohort written next month is checked with no edit here.
    `ALMACE_GCMC_RUNS` overrides the search with a colon-separated list of
    directories or globs.
    """
    key = ("runs", str(ctx.root), os.environ.get("ALMACE_GCMC_RUNS", ""))
    if key in _CACHE:
        return _CACHE[key]

    paths: list[Path] = []
    override = os.environ.get("ALMACE_GCMC_RUNS", "").strip()
    if override:
        for piece in override.split(":"):
            p = Path(piece)
            if p.is_dir():
                paths.extend(sorted(p.glob("*/timing.json")))
                if (p / "timing.json").exists():
                    paths.append(p / "timing.json")
            else:
                base = Path(piece)
                root = base.anchor or "."
                paths.extend(sorted(Path(root).glob(str(base).lstrip("/"))))
    else:
        for pattern in _RUN_PATTERNS:
            paths.extend(ctx.glob(pattern))

    runs, seen = [], set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            meta = json.loads(p.read_text())
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(meta, dict) or "gcmc_moves_attempted" not in meta:
            continue
        d = p.parent
        runs.append({
            "dir": d,
            "cohort": d.parent,
            "name": d.name,
            "meta": meta,
            "trace": d / "trace.csv",
            "sites": d / "sites.csv",
        })
    runs.sort(key=lambda r: str(r["dir"]))
    _CACHE[key] = runs
    return runs


def _cohorts(ctx: Context) -> dict[Path, list[dict]]:
    """Runs grouped by the directory that holds them, which is the cohort."""
    out: dict[Path, list[dict]] = {}
    for r in _gcmc_runs(ctx):
        out.setdefault(r["cohort"], []).append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def _rel(ctx: Context, p: Path) -> str:
    try:
        return str(p.relative_to(ctx.root))
    except ValueError:
        return str(p)


def _trace(run: dict) -> list[tuple[str, int]]:
    """`(phase, n_waters_in_sphere)` per recorded cycle, or [] if absent."""
    p = run["trace"]
    if not p.exists():
        return []
    key = ("trace", str(p))
    if key in _CACHE:
        return _CACHE[key]
    rows = []
    try:
        for row in csv.DictReader(StringIO(p.read_text(errors="replace"))):
            try:
                rows.append((row["phase"], int(float(row["n_waters_in_sphere"]))))
            except (KeyError, TypeError, ValueError):
                continue
    except Exception:                                            # noqa: BLE001
        rows = []
    _CACHE[key] = rows
    return rows


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _log_poisson_cdf(k: int, lam: float) -> float:
    """ln P(X <= k) for X ~ Poisson(lam), in log space so a large lam does not
    underflow the way `exp(-lam)` does."""
    if lam <= 0:
        return 0.0
    terms = [i * math.log(lam) - math.lgamma(i + 1) for i in range(0, k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms)) - lam


def _poisson_lower_limit(k: int, alpha: float = 0.025) -> float:
    """The rate lam for which P(X >= k) = alpha, the exact lower confidence
    limit on a Poisson count. For k = 8 this returns 3.45, which is the number
    that keeps mc-04 from treating a bench of eight accepted moves as though it
    were known exactly."""
    if k <= 0:
        return 0.0
    lo, hi = 1e-9, max(10.0, 4.0 * k)
    while math.exp(_log_poisson_cdf(k - 1, hi)) > alpha:
        hi *= 2
        if hi > 1e9:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # P(X >= k) = 1 - P(X <= k-1)
        if 1.0 - math.exp(_log_poisson_cdf(k - 1, mid)) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _launchers(ctx: Context) -> list[tuple[str, str]]:
    """Our own scripts that drive a `grand` sampler, as `(path, source)`.

    Found by what they import rather than by name, so a second launcher is
    covered automatically.
    """
    key = ("launchers", str(ctx.root))
    if key in _CACHE:
        return _CACHE[key]
    out = []
    for pattern in ("datasets/*/*/*.py", "datasets/*/*/cluster/*.py",
                    "cluster/*.py", "*.py"):
        for p in ctx.glob(pattern):
            try:
                text = p.read_text(errors="replace")
            except Exception:                                    # noqa: BLE001
                continue
            if "grand.samplers" in text or "GCMCSphereSampler" in text:
                out.append((_rel(ctx, p), text))
    _CACHE[key] = out
    return out


def _sampler_source(ctx: Context) -> tuple[str, str] | None:
    """`grand/samplers.py` as `(path, source)`, from the interpreter the cluster
    job itself names.

    The environment is read out of the submission scripts rather than assumed,
    because the checklist's point is to measure the code that ran. Falls back to
    whatever `grand` this interpreter can see, and returns None rather than
    raising so a check can report UNTESTABLE.
    """
    key = ("sampler", str(ctx.root))
    if key in _CACHE:
        return _CACHE[key]

    candidates: list[Path] = []
    rx = re.compile(r"([\w$/.~-]*(?:/envs/[\w.-]+|miniforge3?[\w/.-]*)"
                    r"/bin/python[\d.]*)")
    for pattern in ("cluster/*.sbatch", "datasets/*/*/cluster/*.sbatch",
                    "datasets/*/*/*.sh", "cluster/*.sh"):
        for p in ctx.glob(pattern):
            try:
                text = p.read_text(errors="replace")
            except Exception:                                    # noqa: BLE001
                continue
            if "gcmc" not in text.lower() and "gcmc" not in p.name.lower():
                continue
            for m in rx.finditer(text):
                interp = m.group(1).replace("$HOME", os.path.expanduser("~"))
                interp = os.path.expanduser(interp)
                prefix = Path(interp).parent.parent
                candidates.extend(sorted(
                    prefix.glob("lib/python*/site-packages/grand/samplers.py")))

    try:
        import importlib.util
        spec = importlib.util.find_spec("grand.samplers")
        if spec and spec.origin:
            candidates.append(Path(spec.origin))
    except Exception:                                            # noqa: BLE001
        pass
    # Last resort: sibling environments of the interpreter running this file.
    candidates.extend(sorted(Path(sys.prefix).parent.glob(
        "*/lib/python*/site-packages/grand/samplers.py")))

    for c in candidates:
        try:
            src = c.read_text(errors="replace")
        except Exception:                                        # noqa: BLE001
            continue
        _CACHE[key] = (str(c), src)
        return _CACHE[key]
    _CACHE[key] = None
    return None


def _sphere_sampler_block(src: str) -> str:
    """The `StandardGCMCSphereSampler` class body, so an acceptance expression
    from the box sampler cannot stand in for the sphere sampler's."""
    m = re.search(r"class\s+StandardGCMCSphereSampler\b.*?(?=\nclass\s|\Z)",
                  src, re.S)
    return m.group(0) if m else ""


def _no_runs(ctx: Context) -> Result:
    return Result(UNTESTABLE,
                  "no GCMC run found: no timing.json recording "
                  "gcmc_moves_attempted under this tree",
                  evidence="searched " + ", ".join(_RUN_PATTERNS)
                           + f" below {ctx.root}; set ALMACE_GCMC_RUNS to "
                             "point elsewhere")


# --------------------------------------------------------------------------
# Sec. 3 checklist, "Determine the ensemble of the trial"
# --------------------------------------------------------------------------

def _adams_matches_the_sphere(ctx: Context) -> Result:
    """The mu-V-T ensemble parameter recomputed from each run's own geometry.

    The checklist requires the ensemble to be determined and its microstate
    probabilities used; for mu-V-T that is B = beta*mu_ex + ln(V/V0), and for a
    sphere sampler V is the sphere, not the box. Recomputing B from the radius
    each run recorded catches a run whose B and whose geometry disagree, which
    is what happens the first time somebody changes the radius and passes
    `adams=` by hand.

    FAILS when any run's recorded `adams_B` differs from the value its own
    recorded `sphere_radius_A` implies by more than 1e-6, or when a run records
    no `adams_B` at all.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    beta_mu = EXCESS_MU_KCAL_PER_MOL / (R_KCAL_PER_MOL_K * TEMPERATURE_K)
    bad, missing, ok, worst = [], [], 0, 0.0
    for r in runs:
        B = r["meta"].get("adams_B")
        rad = r["meta"].get("sphere_radius_A")
        if B is None or rad is None:
            missing.append(_rel(ctx, r["dir"]))
            continue
        v = 4.0 * math.pi * float(rad) ** 3 / 3.0
        want = beta_mu + math.log(v / STANDARD_VOLUME_A3)
        d = abs(float(B) - want)
        worst = max(worst, d)
        if d > ADAMS_ABS_TOL:
            bad.append(f"{_rel(ctx, r['dir'])}: B={float(B):.6f} but a "
                       f"{float(rad):.1f} A sphere implies {want:.6f}")
        else:
            ok += 1
    if missing:
        return Result(FAIL,
                      f"{len(missing)} of {len(runs)} runs record no Adams "
                      f"value or no sphere radius, so the ensemble they "
                      f"sampled cannot be recovered from the run",
                      evidence="; ".join(missing[:4]))
    if bad:
        return Result(FAIL,
                      f"{len(bad)} of {len(runs)} runs carry an Adams value "
                      f"that does not match the sphere they recorded, so the "
                      f"insertion acceptance used a volume the run did not use",
                      evidence="; ".join(bad[:4]))
    return Result(PASS,
                  f"all {ok} runs carry B = beta*mu_ex + ln(V_sphere/V0) for "
                  f"their own recorded radius, to {worst:.2e}",
                  evidence=f"recomputed with mu_ex = "
                           f"{EXCESS_MU_KCAL_PER_MOL} kcal/mol, V0 = "
                           f"{STANDARD_VOLUME_A3} A^3, T = {TEMPERATURE_K} K, "
                           f"which are grand's defaults and the launcher's "
                           f"integrator temperature, NOT values the run files "
                           f"record (see mc-12)")


def _ensemble_parameters_are_recorded(ctx: Context) -> Result:
    """Sec. 3, "Determine the ensemble of the trial", and Sec. 7: the mu-V-T
    microstate probability depends on mu, V and T, so a run file that records
    only B cannot be checked without reading the launcher.

    FAILS when a run records `adams_B` but not the temperature, the excess
    chemical potential and the standard volume that produced it.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    wanted = ("temperature_K", "excess_chemical_potential_kcal_per_mol",
              "standard_volume_A3")
    have = {w: sum(1 for r in runs if w in r["meta"]) for w in wanted}
    absent = [w for w in wanted if have[w] == 0]
    if absent:
        return Result(FAIL,
                      f"none of {len(runs)} runs records "
                      f"{', '.join(absent)}, so B can only be verified against "
                      f"the launcher's hardcoded defaults rather than against "
                      f"the run's own declaration of its ensemble",
                      evidence="what would have to be written into timing.json: "
                               + ", ".join(absent)
                               + "; recorded today: "
                               + ", ".join(k for k in ("adams_B",
                                                       "sphere_radius_A")
                                           if k in runs[0]["meta"]))
    return Result(PASS,
                  f"all {len(runs)} runs record the temperature, the excess "
                  f"chemical potential and the standard volume beside B",
                  evidence=", ".join(f"{k}={have[k]}/{len(runs)}"
                                     for k in wanted))


# --------------------------------------------------------------------------
# Sec. 3 checklist, "Stochastic steps often use a random number generator
# (seeded for reproducibility)"
# --------------------------------------------------------------------------

def _rng_is_seeded_and_recorded(ctx: Context) -> Result:
    """Two things, because either one alone is worthless: the generator the
    sampler draws from is seeded, and the seed travels with the run.

    `grand` decides every trial from the process-global `numpy.random` state.
    A launcher that seeds only OpenMM's velocity generator has seeded a
    different generator, and the run is not reproducible however carefully the
    seed is passed.

    FAILS when no run records a seed, or when the launcher never seeds the
    generator the installed sampler actually draws from.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    recorded = sum(1 for r in runs
                   if any(k in r["meta"] for k in ("seed", "rng_seed",
                                                   "numpy_seed")))
    src = _sampler_source(ctx)
    launchers = _launchers(ctx)
    draws_from_numpy = bool(src) and "np.random." in _sphere_sampler_block(src[1])
    seeds_numpy = [p for p, t in launchers
                   if re.search(r"np(?:umpy)?\.random\.seed\(|"
                                r"np(?:umpy)?\.random\.set_state\(", t)]
    takes_a_seed = [p for p, t in launchers if re.search(r"--seed\b", t)]

    problems = []
    if recorded == 0:
        problems.append(f"0 of {len(runs)} runs record a seed in timing.json")
    if draws_from_numpy and not seeds_numpy:
        problems.append("the installed sampler decides every trial from the "
                        "global numpy.random state and no launcher seeds it")
    if problems:
        return Result(FAIL,
                      "; ".join(problems)
                      + ". The Monte Carlo chain is therefore not reproducible, "
                        "and a cohort labelled as a second seed cannot be "
                        "distinguished from a plain rerun",
                      evidence=(f"launchers taking --seed: "
                                f"{', '.join(takes_a_seed[:3]) or 'none'}; "
                                f"launchers seeding numpy.random: "
                                f"{', '.join(seeds_numpy[:3]) or 'none'}; "
                                f"sampler source: "
                                f"{src[0] if src else 'not found'}. "
                                f"What would have to be written: a "
                                f"numpy.random.seed call before sampler.move, "
                                f"and a 'seed' key in timing.json"))
    return Result(PASS,
                  f"{recorded} of {len(runs)} runs record their seed and the "
                  f"launcher seeds the generator the sampler draws from",
                  evidence=f"seeded in {', '.join(seeds_numpy[:3])}")


# --------------------------------------------------------------------------
# Sec. 2 and Sec. 12.2.1: was there enough accepted exchange to average over?
# --------------------------------------------------------------------------

def _enough_accepted_to_block_average(ctx: Context) -> Result:
    """Sec. 12.2.1, Code Block 10, applied to the quantity we report.

    The paper's own mu-V-T example forms 100 blocks of the particle number
    after discarding equilibration. In a mu-V-T chain N changes only on an
    accepted insertion or deletion, so a run that accepted fewer than 100
    exchanges cannot fill 100 blocks with distinct values of N: most blocks hold
    one number repeated. Section 2 makes the same point from the other side,
    that chi should be close to 1 so that trials are likely to be accepted.

    FAILS when the median run of a cohort accepted fewer than 100 exchange
    moves across the whole run.
    """
    cohorts = _cohorts(ctx)
    if not cohorts:
        return _no_runs(ctx)
    rows, failing = [], []
    for c, runs in cohorts.items():
        acc = [int(r["meta"].get("gcmc_moves_accepted") or 0) for r in runs]
        att = [int(r["meta"].get("gcmc_moves_attempted") or 0) for r in runs]
        med, lo, hi = _median(acc), min(acc), max(acc)
        rate = _median([a / t if t else float("nan")
                        for a, t in zip(acc, att)])
        rows.append(f"{_rel(ctx, c)} n={len(runs)}: accepted median {med:.0f} "
                    f"[{lo}, {hi}] of {_median(att):.0f} attempted, "
                    f"rate {rate:.5f}")
        if med < BLOCKS_IN_THE_PAPERS_EXAMPLE:
            failing.append((c, med, hi, rate, len(runs)))
    if failing:
        c, med, hi, rate, n = min(failing, key=lambda f: f[1])
        return Result(FAIL,
                      f"{len(failing)} of {len(cohorts)} cohorts accepted too "
                      f"few exchanges to form the "
                      f"{BLOCKS_IN_THE_PAPERS_EXAMPLE} blocks the paper's own "
                      f"mu-V-T example forms. Worst: {_rel(ctx, c)}, "
                      f"{n} runs, median {med:.0f} accepted (best run {hi}), "
                      f"acceptance rate {rate:.5f}",
                      evidence=" | ".join(rows))
    return Result(PASS,
                  f"every cohort's median run accepted at least "
                  f"{BLOCKS_IN_THE_PAPERS_EXAMPLE} exchanges",
                  evidence=" | ".join(rows))


def _acceptance_agrees_with_our_own_bench(ctx: Context) -> Result:
    """Sec. 12.3: start from a result obtained with a well-tested set of trials
    and known to high confidence, and investigate any difference.

    We have such a reference and it is written into the launcher that chose the
    sampler. The check reads the bench out of that docstring rather than holding
    a copy, carries the bench's own Poisson lower limit so a reference of eight
    accepted moves is not treated as exact, and asks whether each production run
    is consistent with it.

    FAILS when the production campaign accepts at a rate below the bench's lower
    confidence limit with a tail below 1e-3. Reports UNTESTABLE, never PASS, if
    the launcher no longer records a bench.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    words = {"none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12}
    rx = re.compile(r"accepted\s+([a-z]+|\d+),\s*([\d.]+)\s*%", re.I)
    bench = None
    for path, text in _launchers(ctx):
        for m in rx.finditer(text):
            token, pct = m.group(1).lower(), float(m.group(2))
            k = words.get(token, int(token) if token.isdigit() else None)
            if k is None or pct <= 0:
                continue
            bench = {"path": path, "k": k, "rate": pct / 100.0,
                     "n": round(k / (pct / 100.0)) if k else None}
            break
        if bench:
            break
    if bench is None or not bench["k"]:
        return Result(UNTESTABLE,
                      "no launcher records the bench that justified the "
                      "sampler choice, so the production rate has nothing "
                      "trusted to be compared against",
                      evidence="looked for a recorded accepted count and "
                               "percentage in "
                               + ", ".join(p for p, _ in _launchers(ctx))[:200])

    lo_rate = _poisson_lower_limit(bench["k"]) / bench["n"]
    inconsistent, rates = [], []
    per_cohort: dict[Path, list[int]] = {}
    for r in runs:
        att = int(r["meta"].get("gcmc_moves_attempted") or 0)
        acc = int(r["meta"].get("gcmc_moves_accepted") or 0)
        if att <= 0:
            continue
        rates.append(acc / att)
        lam = lo_rate * att
        bad = lam > 0 and _log_poisson_cdf(acc, lam) < math.log(BENCH_ALPHA)
        per_cohort.setdefault(r["cohort"], [0, 0])
        per_cohort[r["cohort"]][1] += 1
        if bad:
            per_cohort[r["cohort"]][0] += 1
            inconsistent.append(r)
    rows = [f"{_rel(ctx, c)}: {n_bad} of {n} inconsistent"
            for c, (n_bad, n) in sorted(per_cohort.items(),
                                        key=lambda kv: str(kv[0]))]
    if not rates:
        return Result(UNTESTABLE, "no run records an attempted-move count")
    med = _median(rates)
    if inconsistent:
        worst = min(inconsistent,
                    key=lambda r: (r["meta"]["gcmc_moves_accepted"]
                                   / r["meta"]["gcmc_moves_attempted"]))
        return Result(FAIL,
                      f"{len(inconsistent)} of {len(rates)} runs accept far "
                      f"below the bench that justified this sampler: campaign "
                      f"median {med:.5f} against a bench of {bench['k']} of "
                      f"{bench['n']} ({bench['rate']:.4f}), whose own 95% "
                      f"lower limit is {lo_rate:.5f}. The sampler was chosen "
                      f"because it accepts more per unit wall time, and it "
                      f"does not accept at the rate that argument assumed",
                      evidence=f"bench read from {bench['path']}; worst run "
                               f"{_rel(ctx, worst['dir'])} accepted "
                               f"{worst['meta']['gcmc_moves_accepted']} of "
                               f"{worst['meta']['gcmc_moves_attempted']}, "
                               f"expected at least "
                               f"{lo_rate * worst['meta']['gcmc_moves_attempted']:.0f} "
                               f"at the bench's lower limit. Per cohort: "
                               + " | ".join(rows)
                               + ". A cohort with a smaller move budget can be "
                                 "below the bench rate and still not reach "
                                 f"p < {BENCH_ALPHA}, so a low count here is "
                                 "less power, not agreement")
    return Result(PASS,
                  f"every one of {len(rates)} runs accepts at a rate "
                  f"consistent with the bench ({bench['k']} of {bench['n']}); "
                  f"campaign median {med:.5f}",
                  evidence=f"bench read from {bench['path']}, lower limit "
                           f"{lo_rate:.5f}. Per cohort: " + " | ".join(rows))


# --------------------------------------------------------------------------
# Sec. 7.2 and Sec. 3 "Check detailed balance": is the reported fluctuation
# exchange at all?
# --------------------------------------------------------------------------

def _fluctuation_is_exchange_not_diffusion(ctx: Context) -> Result:
    """The check with no threshold in it.

    The recorded water count in the sphere changes in two ways: an accepted
    insertion or deletion, which is the grand canonical exchange the ensemble
    average is built on, and a water crossing the sphere boundary under the
    molecular dynamics between batches, which is not. Summing the absolute
    change between consecutive production samples gives the total variation,
    and every unit of it came from one of the two. The number of accepted moves
    over the WHOLE run, equilibration included, is therefore an upper bound on
    the part exchange can explain.

    FAILS when accepted moves over the whole run are fewer than the production
    total variation, because exchange then cannot account for the fluctuation
    being reported as grand canonical even if every accepted move happened
    during production.
    """
    cohorts = _cohorts(ctx)
    if not cohorts:
        return _no_runs(ctx)
    rows, failing, tested = [], [], 0
    no_trace = []
    for c, runs in cohorts.items():
        fracs, tvs, accs = [], [], []
        for r in runs:
            tr = [n for phase, n in _trace(r) if phase.startswith("prod")]
            if len(tr) < 2:
                continue
            tv = sum(abs(tr[i + 1] - tr[i]) for i in range(len(tr) - 1))
            acc = int(r["meta"].get("gcmc_moves_accepted") or 0)
            tvs.append(tv)
            accs.append(acc)
            fracs.append(min(1.0, acc / tv) if tv else 1.0)
        if not fracs:
            no_trace.append(f"{_rel(ctx, c)} ({len(runs)} runs)")
            continue
        tested += len(fracs)
        f = _median(fracs)
        rows.append(f"{_rel(ctx, c)} n={len(fracs)}: production total "
                    f"variation median {_median(tvs):.0f} waters against "
                    f"{_median(accs):.0f} accepted moves, so at most "
                    f"{100 * f:.1f}% of it is exchange")
        if _median(accs) < _median(tvs):
            failing.append((c, f, _median(tvs), _median(accs), len(fracs)))
    if not rows:
        return Result(UNTESTABLE,
                      "no run writes a per-cycle trace, so the water count's "
                      "fluctuation cannot be attributed",
                      evidence="what would have to be written: trace.csv with "
                               "phase and n_waters_in_sphere per cycle, as "
                               "gcmc_apo.py --trace already emits. Cohorts "
                               "without it: " + "; ".join(no_trace[:6]))
    if failing:
        c, f, tv, acc, n = min(failing, key=lambda x: x[1])
        detail = (f"{len(failing)} of {len(rows)} cohorts with a trace report a "
                  f"water-count fluctuation that accepted exchanges cannot "
                  f"explain. Worst: {_rel(ctx, c)}, {n} runs, production total "
                  f"variation {tv:.0f} waters against {acc:.0f} accepted moves "
                  f"over the whole run, so at least {100 * (1 - f):.1f}% of the "
                  f"recorded fluctuation is water crossing the sphere boundary "
                  f"under molecular dynamics. An occupancy from these samples "
                  f"measures residence time, not the probability that a site "
                  f"is wet")
        if no_trace:
            detail += (f". A further {len(no_trace)} cohorts write no trace and "
                       f"could not be tested")
        return Result(FAIL, detail, evidence=" | ".join(rows))
    return Result(PASS,
                  f"in every cohort the accepted exchanges can account for the "
                  f"production fluctuation in the water count ({tested} runs)",
                  evidence=" | ".join(rows))


def _both_directions_are_counted(ctx: Context) -> Result:
    """Sec. 7.2: insertion and deletion are the reverse of each other and are
    derived together. A chain that accepted only insertions has never
    demonstrated the reverse trial, and the combined counter cannot say.

    FAILS when no run records accepted insertions and accepted deletions
    separately.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    keys = ("gcmc_insertions_accepted", "gcmc_deletions_accepted",
            "n_inserted", "n_deleted")
    have = sum(1 for r in runs if any(k in r["meta"] for k in keys))
    if have == 0:
        return Result(FAIL,
                      f"none of {len(runs)} runs separates accepted insertions "
                      f"from accepted deletions, so a chain that only filled "
                      f"and a chain that exchanged both ways record the same "
                      f"number",
                      evidence="what would have to be written into "
                               "timing.json: gcmc_insertions_accepted and "
                               "gcmc_deletions_accepted, which grand does not "
                               "count separately either, so the counters have "
                               "to be added around insertionMove and "
                               "deletionMove or read back from the sampler log")
    if have < len(runs):
        return Result(FAIL,
                      f"only {have} of {len(runs)} runs record the two "
                      f"directions separately",
                      evidence="; ".join(_rel(ctx, r["dir"]) for r in runs
                                         if not any(k in r["meta"]
                                                    for k in keys))[:300])
    return Result(PASS,
                  f"all {len(runs)} runs count accepted insertions and "
                  f"accepted deletions separately")


# --------------------------------------------------------------------------
# Sec. 12.2.1: equilibration is discarded before the average
# --------------------------------------------------------------------------

def _equilibration_is_discarded(ctx: Context) -> Result:
    """Code Block 10 discards the first 1,000 trials before averaging.

    Measured rather than assumed: the reported `mean_waters_in_sphere` is
    compared with the mean over the trace's production rows and with the mean
    over every row. The two differ, so the check distinguishes them rather than
    passing whatever it is given.

    FAILS when the reported mean matches the all-rows mean better than the
    production-only mean.
    """
    cohorts = _cohorts(ctx)
    if not cohorts:
        return _no_runs(ctx)
    rows, failing, tested, distinguishing = [], [], 0, 0
    for c, runs in cohorts.items():
        bad, ok, n = [], 0, 0
        for r in runs:
            tr = _trace(r)
            prod = [n_ for phase, n_ in tr if phase.startswith("prod")]
            allv = [n_ for _, n_ in tr]
            rep = r["meta"].get("mean_waters_in_sphere")
            if not prod or not allv or rep is None:
                continue
            n += 1
            m_prod = sum(prod) / len(prod)
            m_all = sum(allv) / len(allv)
            if abs(m_prod - m_all) > 1e-9:
                distinguishing += 1
            if abs(rep - m_prod) > abs(rep - m_all) + 1e-9:
                bad.append(f"{_rel(ctx, r['dir'])}: reported {rep:.3f}, "
                           f"production-only {m_prod:.3f}, all rows {m_all:.3f}")
            else:
                ok += 1
        if not n:
            continue
        tested += n
        rows.append(f"{_rel(ctx, c)} n={n}: {ok} match the production-only "
                    f"mean, {len(bad)} match the all-rows mean")
        if bad:
            failing.append((c, bad))
    if not tested:
        return Result(UNTESTABLE,
                      "no run writes a per-cycle trace, so it cannot be "
                      "checked whether equilibration entered the average",
                      evidence="what would have to be written: trace.csv with "
                               "a phase column, as gcmc_apo.py --trace emits")
    if failing:
        c, bad = failing[0]
        return Result(FAIL,
                      f"{sum(len(b) for _, b in failing)} of {tested} runs "
                      f"report a mean that includes equilibration",
                      evidence="; ".join(bad[:3]))
    if distinguishing == 0:
        return Result(UNTESTABLE,
                      f"the production-only and all-rows means are identical "
                      f"in all {tested} runs, so this check cannot tell them "
                      f"apart and its PASS would mean nothing",
                      evidence=" | ".join(rows))
    return Result(PASS,
                  f"all {tested} runs report the production-only mean; the two "
                  f"candidates differ in {distinguishing} of them, so the "
                  f"check distinguished rather than accepted",
                  evidence=" | ".join(rows))


def _production_accounting_is_separate(ctx: Context) -> Result:
    """The counter that governs the reported statistic is the one for
    production, and it is not recorded.

    `grand` accumulates `n_accepted` from the first equilibration move onward
    and the launcher never calls `sampler.reset()`, so the number in every
    timing.json spans both phases. The statistic being reported comes from
    production alone, so the count that supports it is unknown.

    FAILS when no run records a production-only accepted count.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    keys = ("gcmc_moves_accepted_production", "gcmc_moves_accepted_prod",
            "gcmc_moves_accepted_equil")
    have = sum(1 for r in runs if any(k in r["meta"] for k in keys))
    resets = [p for p, t in _launchers(ctx) if re.search(r"\.reset\(\)", t)]
    if have == 0:
        return Result(FAIL,
                      f"none of {len(runs)} runs records how many exchanges "
                      f"were accepted during production, and no launcher "
                      f"resets the sampler between the phases, so the single "
                      f"recorded count spans equilibration and production "
                      f"together",
                      evidence=f"launchers calling sampler.reset(): "
                               f"{', '.join(resets) or 'none'}. What would "
                               f"have to be written into timing.json: "
                               f"gcmc_moves_accepted_equil and "
                               f"gcmc_moves_accepted_production")
    if have < len(runs):
        return Result(FAIL,
                      f"only {have} of {len(runs)} runs separate the accepted "
                      f"count by phase")
    return Result(PASS,
                  f"all {len(runs)} runs record a production-only accepted "
                  f"count")


# --------------------------------------------------------------------------
# Sec. 3 "Test the implementation", Sec. 12.3 workflow
# --------------------------------------------------------------------------

def _tested_against_a_tractable_expectation(ctx: Context) -> Result:
    """The checklist's penultimate box: verify the acceptance probability
    against theory and trusted sources, starting with the simplest system that
    has a tractable expression.

    The tractable case for a GCMC sphere sampler is the sphere in bulk water
    with no protein: at the equilibrium Adams value the mean occupancy must be
    the bulk number density times the sphere volume, which is a number known to
    three figures and needs no new theory. The check looks for such a run among
    the runs on disk, identified by its own record of zero protein atoms, and
    compares what it found.

    FAILS when no such reference run exists, and equally when one exists and
    disagrees with the bulk expectation by more than 10%.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    refs = [r for r in runs
            if (r["meta"].get("n_protein_atoms") == 0
                and r["meta"].get("mean_waters_in_sphere") is not None)]
    if not refs:
        return Result(FAIL,
                      f"none of {len(runs)} GCMC runs is a reference run: "
                      f"every one places water in a protein pocket, where the "
                      f"expected occupancy is not known independently, so the "
                      f"acceptance implementation has never been compared "
                      f"against a tractable expectation",
                      evidence="what would have to be run: the same sampler on "
                               "the same sphere with no protein, whose mean "
                               "occupancy must equal "
                               f"{BULK_WATER_DENSITY_PER_A3} waters/A^3 times "
                               "the sphere volume (71.2 waters in an 8 A "
                               "sphere). One run, minutes on one GPU, written "
                               "to a cohort directory this check would then "
                               "find by its recorded n_protein_atoms = 0")
    bad = []
    for r in refs:
        rad = float(r["meta"].get("sphere_radius_A") or 0.0)
        v = 4.0 * math.pi * rad ** 3 / 3.0
        want = BULK_WATER_DENSITY_PER_A3 * v
        got = float(r["meta"]["mean_waters_in_sphere"])
        if want <= 0 or abs(got - want) / want > BULK_TOLERANCE:
            bad.append(f"{_rel(ctx, r['dir'])}: {got:.1f} waters against "
                       f"{want:.1f} expected in a {rad:.1f} A sphere")
    if bad:
        return Result(FAIL,
                      f"{len(bad)} of {len(refs)} water-only reference runs "
                      f"miss the bulk expectation by more than "
                      f"{100 * BULK_TOLERANCE:.0f}%, so the acceptance "
                      f"probability does not reproduce the density it was "
                      f"constructed from",
                      evidence="; ".join(bad[:4]))
    return Result(PASS,
                  f"{len(refs)} water-only reference runs reproduce the bulk "
                  f"number density within {100 * BULK_TOLERANCE:.0f}%",
                  evidence="; ".join(
                      f"{_rel(ctx, r['dir'])}: "
                      f"{float(r['meta']['mean_waters_in_sphere']):.1f} waters"
                      for r in refs[:4]))


# --------------------------------------------------------------------------
# Sec. 7.2 Eqs. 28 and 29, read out of the installed sampler
# --------------------------------------------------------------------------

def _acceptance_expression_matches_the_paper(ctx: Context) -> Result:
    """Eq. 28 for insertion and Eq. 29 for deletion, checked against the source
    of the sampler that ran rather than against the launcher's description of
    it.

    Eq. 28: chi = V z exp(-beta dU) / (P_w (N+1)), which with exp(B) = V z and
    a randomly oriented rigid water is exp(B) exp(-beta dU) / (N+1).
    Eq. 29: chi = P_w N exp(-beta dU) / (V z), which is N exp(-B) exp(-beta dU).

    The two are each other's reverse only if the volume in B is the volume the
    insertion samples from and the deletion selects within, which for a sphere
    sampler is the sphere.

    FAILS when the installed sampler's insertion acceptance loses its 1/(N+1),
    when the deletion acceptance loses its factor of N, when either loses its
    exp(-beta dU), or when B is built from a volume other than the sphere the
    insertion draws into.
    """
    src = _sampler_source(ctx)
    if not src:
        return Result(UNTESTABLE,
                      "grand/samplers.py could not be located, so the "
                      "acceptance expression that ran cannot be read",
                      evidence="looked for the interpreter named in the GCMC "
                               "submission scripts, then for an importable "
                               "grand, then for sibling environments of "
                               f"{sys.prefix}")
    path, text = src
    block = _sphere_sampler_block(text)
    if not block:
        return Result(UNTESTABLE,
                      "no StandardGCMCSphereSampler class in the installed "
                      "grand, so the sphere sampler's acceptance could not be "
                      "isolated",
                      evidence=path)

    ins = re.search(r"acc_prob\s*=\s*(.+)", block[block.find("def insertionMove"):])
    dele = re.search(r"acc_prob\s*=\s*(.+)", block[block.find("def deletionMove"):])
    problems = []
    if not ins or not dele:
        return Result(UNTESTABLE,
                      "the sphere sampler defines no acceptance expression "
                      "this check recognises",
                      evidence=path)
    ins_e, del_e = ins.group(1).strip(), dele.group(1).strip()
    if "exp(self.B)" not in ins_e.replace(" ", "").replace("math.", ""):
        problems.append(f"insertion carries no exp(B): {ins_e}")
    if "/(self.N+1)" not in ins_e.replace(" ", ""):
        problems.append(f"insertion does not divide by N+1 as Eq. 28 requires: "
                        f"{ins_e}")
    if "self.kT" not in ins_e or "self.energy" not in ins_e:
        problems.append(f"insertion carries no exp(-beta dU): {ins_e}")
    if "exp(-self.B)" not in del_e.replace(" ", "").replace("math.", ""):
        problems.append(f"deletion carries no exp(-B): {del_e}")
    if not re.match(r"self\.N\s*\*", del_e):
        problems.append(f"deletion does not multiply by N as Eq. 29 requires: "
                        f"{del_e}")
    if "self.kT" not in del_e or "self.energy" not in del_e:
        problems.append(f"deletion carries no exp(-beta dU): {del_e}")

    vol = re.search(r"volume\s*=\s*\(4\s*\*\s*np\.pi\s*\*\s*sphereRadius\s*\*\*"
                    r"\s*3\)\s*/\s*3", text)
    badams = re.search(r"self\.B\s*=\s*excessChemicalPotential\s*/\s*self\.kT"
                       r"\s*\+\s*math\.log\(volume\s*/\s*standardVolume\)", text)
    if not (vol and badams):
        problems.append("B is not built as beta*mu_ex + ln(V_sphere/V0) from "
                        "the sphere volume, so the insertion's volume and the "
                        "Adams value refer to different regions")

    if problems:
        return Result(FAIL,
                      f"the installed sampler's acceptance does not match "
                      f"Eqs. 28 and 29: " + "; ".join(problems[:3]),
                      evidence=path)
    return Result(PASS,
                  "the installed sphere sampler implements Eq. 28 and Eq. 29 "
                  "with B built from the sphere volume the insertion draws into",
                  evidence=f"{path}: insertion `{ins_e[:70]}`; deletion "
                           f"`{del_e[:70]}`")


def _reverse_trial_returns_to_the_same_region(ctx: Context) -> Result:
    """The checklist's detailed-balance box: every step of the reverse trial
    must be able to return to the exact state the forward trial began in.

    For a sphere sampler that means the insertion draws its position inside the
    sphere and the deletion selects only among the waters inside the same
    sphere. An insertion that could place a water the deletion can never pick,
    or a deletion that could remove a water the insertion could never have
    placed, breaks detailed balance silently: nothing raises and no number
    looks wrong.

    FAILS when the installed insertion routine does not bound its draw by the
    sphere radius, or when the deletion routine selects from a set not built
    from the waters inside the sphere.
    """
    src = _sampler_source(ctx)
    if not src:
        return Result(UNTESTABLE,
                      "grand/samplers.py could not be located",
                      evidence=f"searched the GCMC submission scripts' "
                               f"interpreter and siblings of {sys.prefix}")
    path, text = src
    m_ins = re.search(r"def insertRandomWater\(self.*?(?=\n    def )", text, re.S)
    m_del = re.search(r"def deleteRandomWater\(self.*?(?=\n    def )", text, re.S)
    if not (m_ins and m_del):
        return Result(UNTESTABLE,
                      "the installed grand defines no insertRandomWater or "
                      "deleteRandomWater to read",
                      evidence=path)
    ins, dele = m_ins.group(0), m_del.group(0)
    problems = []
    if "self.sphere_radius" not in ins:
        problems.append("the insertion does not bound its random position by "
                        "the sphere radius")
    if "gcmc_resids" not in dele and "gcmc_wats" not in dele:
        problems.append("the deletion does not select from the waters recorded "
                        "as inside the sphere")
    if problems:
        return Result(FAIL,
                      "insertion and deletion do not address the same region, "
                      "so they are not each other's reverse: "
                      + "; ".join(problems),
                      evidence=path)
    return Result(PASS,
                  "the insertion draws inside the sphere radius and the "
                  "deletion selects only among the waters inside the same "
                  "sphere, so the two trials are each other's reverse",
                  evidence=f"{path}: insertRandomWater uses "
                           f"self.sphere_radius, deleteRandomWater selects "
                           f"from the recorded in-sphere waters")


# --------------------------------------------------------------------------
# Sec. 12.1 energy tests
# --------------------------------------------------------------------------

def _running_energy_is_checked(ctx: Context) -> Result:
    """Sec. 12.1: a running total energy accumulated from per-trial increments
    drifts from the true total, so it must be recomputed and compared
    infrequently.

    Whether this applies is a property of the implementation, so it is read out
    of the implementation. If the sampler accumulates increments, the check
    asks for the periodic recomputation. If it recomputes the full potential
    energy at every trial, there is no running total to drift and the practice
    is genuinely not applicable.

    FAILS when the sampler accumulates energy increments and nothing recomputes
    the total.
    """
    src = _sampler_source(ctx)
    if not src:
        return Result(UNTESTABLE,
                      "grand/samplers.py could not be located, so it is not "
                      "known whether a running energy is accumulated",
                      evidence=f"searched siblings of {sys.prefix}")
    path, text = src
    block = _sphere_sampler_block(text)
    accumulates = bool(re.search(r"self\.energy\s*\+=|"
                                 r"self\.energy\s*=\s*self\.energy\s*\+", block))
    recomputes = bool(re.search(
        r"final_energy\s*=\s*self\.context\.getState\(getEnergy=True\)"
        r"\.getPotentialEnergy\(\)", block))
    if accumulates:
        periodic = bool(re.search(r"getPotentialEnergy\(\)", block))
        if not periodic:
            return Result(FAIL,
                          "the sampler accumulates energy increments and never "
                          "recomputes the total, so round-off grows with the "
                          "number of trials unchecked",
                          evidence=path)
        return Result(PASS,
                      "the sampler accumulates increments and also recomputes "
                      "the total energy",
                      evidence=path)
    if recomputes:
        return Result(NA,
                      "the practice does not apply: the installed sampler "
                      "recomputes the full potential energy from the context "
                      "at every trial and at the start of every batch rather "
                      "than accumulating increments, so there is no running "
                      "total for round-off to corrupt",
                      evidence=f"{path}: final_energy comes from "
                               f"context.getState(getEnergy=True) inside both "
                               f"insertionMove and deletionMove, and "
                               f"self.energy is reset from a full state at the "
                               f"top of move()")
    return Result(UNTESTABLE,
                  "the installed sampler neither accumulates nor visibly "
                  "recomputes an energy this check recognises",
                  evidence=path)


# --------------------------------------------------------------------------
# Sec. 3 "Document your derivation or software implementation"
# --------------------------------------------------------------------------

def _exchange_accounting_travels_with_the_data(ctx: Context) -> Result:
    """The checklist's last box, in the only form that survives a file being
    copied away from its directory: the acceptance accounting is on every row
    of the table a reader will use, not only in a log.

    FAILS when a cohort's site table omits the accepted or attempted counts, or
    when the counts on its rows disagree with the run's own timing.json.
    """
    runs = _gcmc_runs(ctx)
    if not runs:
        return _no_runs(ctx)
    missing, mismatched, ok = [], [], 0
    for r in runs:
        p = r["sites"]
        if not p.exists():
            continue
        try:
            rows = list(csv.DictReader(StringIO(p.read_text(errors="replace"))))
        except Exception:                                        # noqa: BLE001
            continue
        if not rows:
            continue
        head = rows[0]
        if ("gcmc_moves_accepted" not in head
                or "gcmc_moves_attempted" not in head):
            missing.append(_rel(ctx, p))
            continue
        try:
            acc = {int(row["gcmc_moves_accepted"]) for row in rows}
            att = {int(row["gcmc_moves_attempted"]) for row in rows}
        except (TypeError, ValueError):
            missing.append(_rel(ctx, p))
            continue
        # `or -1` would be wrong here: a run that accepted zero moves is
        # exactly the case this project has, and zero is falsy.
        m_acc = r["meta"].get("gcmc_moves_accepted")
        m_att = r["meta"].get("gcmc_moves_attempted")
        if (m_acc is None or m_att is None
                or acc != {int(m_acc)} or att != {int(m_att)}):
            mismatched.append(f"{_rel(ctx, p)}: rows say {sorted(acc)}/"
                              f"{sorted(att)}, timing.json says "
                              f"{r['meta'].get('gcmc_moves_accepted')}/"
                              f"{r['meta'].get('gcmc_moves_attempted')}")
        else:
            ok += 1
    if not (ok or missing or mismatched):
        return Result(UNTESTABLE, "no run writes a site table to check",
                      evidence=f"{len(runs)} runs found, none with sites.csv")
    if missing or mismatched:
        return Result(FAIL,
                      f"of {ok + len(missing) + len(mismatched)} site tables, "
                      f"{len(missing)} carry no exchange accounting and "
                      f"{len(mismatched)} disagree with the run's own "
                      f"timing.json",
                      evidence="; ".join((missing + mismatched)[:4]))
    return Result(PASS,
                  f"all {ok} site tables carry the accepted and attempted "
                  f"counts on every row, agreeing with their run's timing.json",
                  evidence="so a reader of the occupancies can ask how much "
                           "exchange produced them without finding the log")


# --------------------------------------------------------------------------

CHECKS = [
    # --- Sec. 3, the boxed checklist on page 5 ------------------------------
    Check(id="mc-01", severity="must",
          section="Sec. 3 checklist, 'Determine the ensemble of the trial'; "
                  "Sec. 7.2 Eq. 28",
          question="Does each run's Adams value match the sphere it recorded?",
          measures="adams_B against beta*mu_ex + ln(V_sphere/V0) recomputed "
                   "from each run's own sphere_radius_A",
          fn=_adams_matches_the_sphere),
    Check(id="mc-02", severity="must",
          section="Sec. 3 checklist, 'Stochastic steps often use a random "
                  "number generator (seeded for reproducibility)'",
          question="Is the generator the trials are decided from seeded, and "
                   "the seed recorded with the run?",
          measures="a seed key in timing.json, and a numpy.random seeding call "
                   "in the launcher against the generator the installed "
                   "sampler draws from",
          fn=_rng_is_seeded_and_recorded),
    Check(id="mc-03", severity="must",
          section="Sec. 12.2.1 Code Block 10; Sec. 2",
          question="Were enough exchanges accepted to block-average the "
                   "particle number, as the paper's own example does?",
          measures="gcmc_moves_accepted per run against the 100 blocks of "
                   "Code Block 10, per cohort",
          fn=_enough_accepted_to_block_average),
    Check(id="mc-04", severity="must",
          section="Sec. 12.3; Sec. 3 checklist, 'Verify that the acceptance "
                  "probability leads to simulation results that agree with "
                  "theory and trusted sources'",
          question="Does the campaign accept at the rate the bench that chose "
                   "this sampler measured?",
          measures="per-run accepted counts against the bench recorded in the "
                   "launcher's own docstring, with the bench's Poisson lower "
                   "limit carried",
          fn=_acceptance_agrees_with_our_own_bench),
    Check(id="mc-05", severity="must",
          section="Sec. 7.2; Sec. 3 checklist, 'Check detailed balance'",
          question="Is the reported fluctuation in the water count exchange, "
                   "or is it water diffusing across the sphere boundary?",
          measures="production total variation of n_waters_in_sphere in "
                   "trace.csv against gcmc_moves_accepted for the whole run",
          fn=_fluctuation_is_exchange_not_diffusion),
    Check(id="mc-06", severity="must",
          section="Sec. 12.2.1 Code Block 10, block(values, equil=1000)",
          question="Is equilibration excluded from the reported average?",
          measures="mean_waters_in_sphere against the production-only mean and "
                   "the all-rows mean of trace.csv",
          fn=_equilibration_is_discarded),
    Check(id="mc-07", severity="must",
          section="Sec. 3 checklist, 'Test the implementation of the trial "
                  "acceptance probability'; Sec. 12.3",
          question="Has the acceptance implementation been tested against a "
                   "tractable expectation?",
          measures="a water-only reference run, found by its recorded "
                   "n_protein_atoms = 0, against bulk density times sphere "
                   "volume",
          fn=_tested_against_a_tractable_expectation),
    Check(id="mc-08", severity="must",
          section="Sec. 7.2 Eqs. 28 and 29; Sec. 3 checklist, 'Use Eq. 4 to "
                  "obtain the trial acceptance probability'",
          question="Does the installed sampler's acceptance expression carry "
                   "the factors Eqs. 28 and 29 require?",
          measures="the acceptance lines of StandardGCMCSphereSampler in the "
                   "grand installed in the interpreter the cluster job names",
          fn=_acceptance_expression_matches_the_paper),
    Check(id="mc-09", severity="must",
          section="Sec. 3 checklist, 'Ensure all steps conducted in the "
                  "reverse trial are capable of returning ... to the exact "
                  "same old state'",
          question="Do the insertion and the deletion address the same region, "
                   "so that each is the other's reverse?",
          measures="insertRandomWater and deleteRandomWater in the installed "
                   "sampler",
          fn=_reverse_trial_returns_to_the_same_region),

    # --- Sec. 12, implementation issues -------------------------------------
    Check(id="mc-10", severity="should",
          section="Sec. 12.1",
          question="Is a running total energy recomputed and compared, or is "
                   "none accumulated?",
          measures="whether the installed sampler accumulates increments or "
                   "recomputes the full potential energy per trial",
          fn=_running_energy_is_checked),
    Check(id="mc-11", severity="should",
          section="Sec. 3 checklist, 'Document your derivation or software "
                  "implementation for later reference'",
          question="Does the exchange accounting travel on every row of the "
                   "table a reader will use?",
          measures="gcmc_moves_accepted and gcmc_moves_attempted per row of "
                   "sites.csv against the run's timing.json",
          fn=_exchange_accounting_travels_with_the_data),
    Check(id="mc-12", severity="should",
          section="Sec. 3 checklist, 'Determine the ensemble of the trial'",
          question="Does each run record the temperature, the excess chemical "
                   "potential and the standard volume that define its "
                   "ensemble?",
          measures="keys present in timing.json across every cohort",
          fn=_ensemble_parameters_are_recorded),
    Check(id="mc-13", severity="should",
          section="Sec. 7.2",
          question="Are accepted insertions and accepted deletions counted "
                   "separately, and is the production count separate from the "
                   "equilibration count?",
          measures="per-direction keys in timing.json, and sampler.reset() "
                   "calls in the launcher",
          fn=_both_directions_are_counted),
    Check(id="mc-14", severity="should",
          section="Sec. 12.2.1, block(values, equil=1000)",
          question="Is the accepted-move count that supports the reported "
                   "statistic recorded for production alone?",
          measures="production-phase accepted counts in timing.json, and "
                   "sampler.reset() between the phases",
          fn=_production_accounting_is_separate),
]
