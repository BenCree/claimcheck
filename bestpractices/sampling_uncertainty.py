"""Grossfield et al., *Best Practices for Quantification of Uncertainty and
Sampling Quality in Molecular Simulations*, LiveCoMS 1(1):5067 (2018).

The paper carries an eight-item **Best Practices Checklist** (§2) plus a set of
harder requirements buried in §7. Its central claim, and the reason it belongs
in this package, is stated in the glossary (§1.2.1) and is the single sentence
that indicts most of what we currently report:

    "Standard Deviation: The positive square root of the variance ... This is a
    measure of the width of the distribution of x, and is, in itself, *not* a
    measure of the statistical uncertainty."

and the recommendation that follows from it (§7.1):

    "(b) otherwise, such as for biomolecular simulations, *authors should report
    (and graph) their best estimates of 95 % confidence intervals*."

Everything below either measures a file we actually wrote or is not here. Three
checks execute the paper's own statistics on our own data — the split-half
stationarity test of §4.1 on `state.csv`, and Eq. 5 of §7.3.1 (the effective
sample size) on `readouts.npz` — so that a FAIL arrives with the number the
paper says we should have computed and did not.

Section numbers are those of the published article:

    §1.2.1 glossary · §2 checklist · §3 pre-simulation sanity · §4 semi-
    quantitative checks · §5 equilibration/burn-in · §6 global sampling ·
    §7 uncertainty in specific observables · §8 enhanced sampling

The §2 checklist is a bulleted list, not a numbered one. "§2 checklist item N"
below is our ordinal into that list in the order it is printed — 1 planning,
2 cherry-picking, 3 semiquantitative checks, 4 equilibration removal, 5 global
sampling, 6 confidence intervals, 7 enhanced sampling, 8 reporting — and is
said here so nobody looks for a number the paper does not print.

Two of the six `must` checks currently PASS. That is deliberate: a checklist
that can only complain is one nobody runs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path

from . import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE, NA

PAPER = Paper(
    slug="sampling_uncertainty",
    title="Best Practices for Quantification of Uncertainty and Sampling "
          "Quality in Molecular Simulations [Article v1.0]",
    authors="Alan Grossfield, Paul N. Patrone, Daniel R. Roe, Andrew J. "
            "Schultz, Daniel Siderius, Daniel M. Zuckerman",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v1i1e5067",
    doi="10.33011/livecoms.1.1.5067",
    year="2018",
    scope="How to establish that a simulation sampled enough, and what error "
          "bar to put on a number derived from it. Covers conventional MD/MC "
          "and, in §8, enhanced sampling. Explicitly excludes systematic error "
          "from the force field or the model (§1.1) — verification, not "
          "validation.",
)

# --------------------------------------------------------------------------
# where we look
# --------------------------------------------------------------------------

LEADERBOARD = "results/leaderboard.csv"
SWEEP_SUMMARY = "results/sweep/sweep_representation_summary.csv"
SWEEP_MANIFEST = "results/sweep/manifest.csv"
STATE_GLOB = "results/md_pilot/*/*/state.csv"
NPZ_GLOB = "results/md_pilot/*/*/readouts.npz"
RECEIPT_GLOB = "results/md_pilot/*/*/receipt.json"
READOUT_GLOB = "results/md_pilot/*/*/readouts.json"

#: Columns in the leaderboard that are reported figures of merit, i.e. numbers
#: a reader would compare between methods. Each one needs an interval.
FOM_RE = re.compile(r"^(hits@\d+|EF@\d+|rowan32_(spearman|pearson|rmse_fitted))$")

#: The thermodynamic channels §4.1 names explicitly: "potential energy, system
#: size ..., temperature ..., and/or density".
ZEROTH_ORDER = ("Potential Energy (kJ/mole)", "Box Volume (nm^3)",
                "Density (g/mL)", "Temperature (K)")

#: OUR threshold, not the paper's. §4.1 mandates the test ("Is each time series
#: fluctuating about an average value or drifting overall?") but gives no
#: number. We operationalise it as: the two halves' means must not differ by
#: more than one standard deviation of the series itself, i.e. by more than a
#: typical fluctuation. Deliberately generous — it fires only on visible drift.
DRIFT_SD = 1.0


# --------------------------------------------------------------------------
# small helpers — stdlib only, so the suite runs anywhere
# --------------------------------------------------------------------------

def _rows(ctx: Context, rel: str):
    """(header, rows) from a repo CSV; ([], []) if absent or empty."""
    txt = ctx.read(rel)
    if not txt.strip():
        return [], []
    rdr = csv.DictReader(io.StringIO(txt))
    return list(rdr.fieldnames or []), list(rdr)


def _floats(text: str, key: str):
    """Every float under `key` in a StateDataReporter CSV, in file order."""
    rdr = csv.DictReader(io.StringIO(text))
    fields = [f for f in (rdr.fieldnames or [])]
    # OpenMM writes the header as #"Step","Time (ps)",...
    clean = {f: f.lstrip('#').strip('"') for f in fields}
    src = next((f for f, c in clean.items() if c == key), None)
    if src is None:
        return []
    out = []
    for row in rdr:
        try:
            out.append(float(row[src]))
        except (TypeError, ValueError):
            pass
    return out


def _mean_sd(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd


def _split_half_drift(v):
    """|mean(second half) - mean(first half)| / SD(series).

    §2 checklist item 3: "An individual trajectory can be divided into two
    parts and analyzed as if two simulations had been run." This is that test,
    expressed in units of the series' own fluctuation.
    """
    if len(v) < 8:
        return None
    h = len(v) // 2
    m1, _ = _mean_sd(v[:h])
    m2, _ = _mean_sd(v[h:])
    _, sd = _mean_sd(v)
    if sd <= 0:
        return float("inf")
    return abs(m2 - m1) / sd


def _n_independent(v):
    """Eq. 5 of §7.3.1: N_ind = N / (1 + 2 * sum_j C_j).

    C_j is the stationary autocorrelation of Eq. 2. The sum is truncated at the
    first non-positive C_j — §7.3.1 permits any N_max greater than tau and
    warns explicitly that "care must be exercised to avoid integrating pure
    noise over too large of an interval, since this can generate Brownian
    motion". The initial-positive-sequence truncation is the standard guard.
    """
    n = len(v)
    if n < 8:
        return None, 0
    m, sd = _mean_sd(v)
    if sd <= 0:
        return None, 0
    s2 = sd * sd
    total, lag = 0.0, 0
    for j in range(1, n - 1):
        c = sum((v[k] - m) * (v[k + j] - m) for k in range(n - j)) / (n - j) / s2
        if c <= 0:
            break
        total += c
        lag = j
    return n / (1.0 + 2.0 * total), lag


def _py_sources(ctx: Context):
    """Our own analysis scripts — repo root and cluster/, never vendored trees.

    mace/, mace-off/ and the other vendored packages are somebody else's code
    and holding them to our checklist would be both unfair and noisy.
    """
    out = [p for p in ctx.glob("*.py")]
    out += [p for p in ctx.glob("cluster/*.py")]
    return sorted(out)


def _code_lines(text: str):
    """Lines with docstrings and comments blanked, so a check greps *code*.

    Written after this check's first run flagged `md_readouts.py:48` — a
    docstring line whose whole point is to say the +-1.5 noise floor is NOT
    used. Matching prose and calling it a violation is how a checklist loses
    its authority.
    """
    out, in_doc, delim = [], False, ""
    for ln in text.splitlines():
        if in_doc:
            out.append("")
            if delim in ln:
                in_doc = False
            continue
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            out.append("")
            continue
        for d in ('"""', "'''"):
            if ln.count(d) % 2 == 1:            # opens and does not close
                in_doc, delim = True, d
                ln = ln.split(d)[0]
                break
        out.append(ln.split("#")[0] if "#" in ln else ln)
    return out


#: Terms that mean a script is looking at a trajectory. §4.2's all-to-all RMSD
#: map and §6's decorrelation analyses are trajectory analyses; the same words
#: applied to a docking matrix are a different thing entirely, and treating
#: them as equivalent is exactly the vacuous PASS this package exists to stop.
TRAJ_CONTEXT = re.compile(
    r"MDAnalysis|mdtraj|\bUniverse\(|\.dcd\b|\.xtc\b|trajectory|traj\.", re.I)


def _rel(ctx: Context, p: Path) -> str:
    try:
        return str(p.relative_to(ctx.root))
    except ValueError:
        return str(p)


# ==========================================================================
# §7 — uncertainty on the numbers we actually report
# ==========================================================================

def _ci_on_every_fom(ctx: Context) -> Result:
    hdr, rows = _rows(ctx, LEADERBOARD)
    if not hdr:
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} is absent or empty, so there is nothing "
                      f"to audit. Running `benchmark.py` once makes this "
                      f"testable.", evidence=LEADERBOARD)
    foms = [c for c in hdr if FOM_RE.match(c)]
    if not foms:
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} exists but carries no recognised figure "
                      f"of merit; FOM_RE may need updating for new columns.",
                      evidence=f"columns: {', '.join(hdr)}")
    bare = [c for c in foms
            if not ({f"{c}_lo", f"{c}_hi"} <= set(hdr)
                    or {f"{c}_ci_lo", f"{c}_ci_hi"} <= set(hdr))]
    if not bare:
        return Result(PASS,
                      f"all {len(foms)} figures of merit carry a lo/hi pair",
                      evidence=f"{LEADERBOARD}: {', '.join(foms)}")
    return Result(FAIL,
                  f"{len(bare)} of {len(foms)} reported figures of merit carry "
                  f"no interval at all: {', '.join(bare)}. hits@25 is the "
                  f"project objective and it is the barest of them — "
                  f"`benchmark.py` computes its bootstrap CI at line ~221 and "
                  f"prints it, then writes only the point estimate to the "
                  f"leaderboard.",
                  evidence=f"{LEADERBOARD} header has {len(rows)} data row(s); "
                           f"with-interval: "
                           f"{', '.join(sorted(set(foms) - set(bare))) or 'none'}")


def _sd_is_not_an_uncertainty(ctx: Context) -> Result:
    hdr, rows = _rows(ctx, SWEEP_SUMMARY)
    if not hdr:
        return Result(UNTESTABLE,
                      f"{SWEEP_SUMMARY} is absent; the AL sweep summary is "
                      f"where means-over-seeds are reported. Re-running the "
                      f"sweep aggregation makes this testable.",
                      evidence=SWEEP_SUMMARY)
    means = [c for c in hdr if c.startswith("mean_")]
    sds = [c for c in hdr if c.startswith("std_") or c.startswith("sd_")]
    has_n = any(c in hdr for c in ("n", "n_seeds", "n_runs", "n_replicates"))
    has_ci = any(c.endswith(("_lo", "_hi", "_ci_lo", "_ci_hi")) for c in hdr)
    if has_ci:
        return Result(PASS,
                      f"{len(means)} mean column(s) accompanied by interval "
                      f"columns", evidence=f"{SWEEP_SUMMARY}: {', '.join(hdr)}")
    if not means:
        return Result(UNTESTABLE,
                      f"{SWEEP_SUMMARY} has no `mean_*` column, so no reported "
                      f"mean can be audited here.",
                      evidence=f"columns: {', '.join(hdr)}")
    # How wrong is it? n lives in the manifest, not on the row.
    n_seeds = None
    _, man = _rows(ctx, SWEEP_MANIFEST)
    seeds = {r.get("seeds") for r in man if r.get("seeds")}
    if len(seeds) == 1:
        try:
            n_seeds = int(next(iter(seeds)))
        except ValueError:
            n_seeds = None
    factor = f", overstating the error bar by sqrt({n_seeds}) = " \
             f"{math.sqrt(n_seeds):.0f}x" if n_seeds else ""
    return Result(FAIL,
                  f"{len(rows)} rows report {len(means)} mean(s) with "
                  f"{len(sds)} standard deviation(s) and no confidence "
                  f"interval. §1.2.1 is explicit that a standard deviation "
                  f"\"is, in itself, *not* a measure of the statistical "
                  f"uncertainty\"{factor}. The row does not even carry n "
                  f"({'n is ' + str(n_seeds) + ', and it lives in ' + SWEEP_MANIFEST if n_seeds else 'n is recorded nowhere on the row'}), "
                  f"so a reader cannot form the interval themselves.",
                  evidence=f"{SWEEP_SUMMARY}: means={means} sds={sds} "
                           f"n_column={has_n}")


def _effective_sample_size(ctx: Context) -> Result:
    """§7.2: "one requires an estimate for the independent number of
    observations in a given simulation"."""
    npz = ctx.glob(NPZ_GLOB)
    jsons = ctx.glob(READOUT_GLOB)
    if not npz and not jsons:
        return Result(UNTESTABLE,
                      "no MD read-outs exist yet, so there is no "
                      "trajectory-derived mean to audit. Running "
                      "`md_readouts.py extract` on one arm makes this "
                      "testable.", evidence=NPZ_GLOB)

    # Is the number recorded anywhere in the artefact we ship?
    keys = set()
    for j in jsons:
        try:
            keys |= set(json.loads(j.read_text()).keys())
        except (ValueError, OSError):
            pass
    recorded = {k for k in keys
                if re.search(r"n_ind|n_eff|neff|tau|corr_time|stat_ineff", k, re.I)}
    if recorded:
        return Result(PASS,
                      f"read-outs record an effective-sample-size field: "
                      f"{', '.join(sorted(recorded))}",
                      evidence=f"{len(jsons)} readouts.json")

    # Not recorded. Compute it ourselves so the FAIL carries the number.
    try:
        import numpy as np
    except ImportError:
        return Result(FAIL,
                      f"no readouts.json among {len(jsons)} carries an "
                      f"effective sample size, correlation time or statistical "
                      f"inefficiency, so every mean over frames is implicitly "
                      f"treated as n = n_frames. (numpy is not importable "
                      f"here, so the size of the error could not be measured.)",
                      evidence=f"keys present: {', '.join(sorted(keys))[:200]}")

    worst = None
    for p in npz:
        try:
            z = np.load(p)
        except (OSError, ValueError):
            continue
        for ch in ("rmsd", "drift", "rgyr"):
            if ch not in z:
                continue
            v = [float(x) for x in np.asarray(z[ch], dtype=float)]
            ni, _ = _n_independent(v)
            if ni is None or ni <= 0:
                continue
            ratio = len(v) / ni
            if worst is None or ratio > worst[0]:
                worst = (ratio, _rel(ctx, p), ch, len(v), ni)
    if worst is None:
        return Result(UNTESTABLE,
                      f"{len(npz)} readouts.npz found but none carried a "
                      f"usable time series (keys rmsd/drift/rgyr), so Eq. 5 "
                      f"could not be evaluated.", evidence=NPZ_GLOB)
    ratio, path, ch, n, ni = worst
    return Result(FAIL,
                  f"no readouts.json carries an effective sample size, "
                  f"correlation time or statistical inefficiency, so every "
                  f"mean over frames is implicitly treated as n = n_frames. "
                  f"Evaluating Eq. 5 (§7.3.1) on our own saved series says "
                  f"that is wrong by up to {ratio:.1f}x in n, i.e. "
                  f"{math.sqrt(ratio):.1f}x in any standard uncertainty formed "
                  f"as SD/sqrt(n_frames).",
                  evidence=f"worst: {path}:{ch} — {n} frames are worth "
                           f"N_ind = {ni:.1f} independent samples "
                           f"(over {len(npz)} runs)")


def _bootstrap_does_not_drop_silently(ctx: Context) -> Result:
    """§2 item 8: "detailed enough to permit reproduction of reported
    findings". A resampling loop that skips replicates changes the number of
    bootstrap samples to a value recorded nowhere."""
    sources = _py_sources(ctx)
    if not sources:
        return Result(UNTESTABLE,
                      "no analysis scripts were found to scan, so 'no "
                      "offending loop' would mean 'nothing was looked at'. "
                      "That is a vacuous pass, not a pass.",
                      evidence="glob('*.py') and glob('cluster/*.py') empty")
    offenders = {}
    for p in sources:
        try:
            lines = _code_lines(p.read_text(errors="replace"))
        except OSError:
            continue
        # Does the file account for what it kept?
        accounted = any(re.search(r"n_kept|n_drop|n_fail|kept\b|n_used|n_valid",
                                  ln) for ln in lines)
        if accounted:
            continue
        for i, ln in enumerate(lines):
            # Any rng object: rng.integers(...), idxr.integers(...), ...
            if not re.search(r"\.integers\(|\bresample\(", ln):
                continue
            # The draw must be re-executed per replicate, i.e. sit *inside* a
            # loop. `eint_dyad_arms.py` draws the whole index matrix up front
            # and its later `continue` skips the reference arm, not a
            # replicate — flagging that would be wrong.
            indent = len(ln) - len(ln.lstrip())
            inside = any(
                re.match(r"\s*(for|while)\b", lines[j])
                and (len(lines[j]) - len(lines[j].lstrip())) < indent
                for j in range(max(0, i - 4), i))
            if not inside:
                continue
            window = lines[i:i + 14]
            for k, w in enumerate(window):
                if re.match(r"\s*continue\s*$", w):
                    # keyed on (file, line) so two resampling calls sharing one
                    # `continue` are reported once, not twice
                    offenders[(_rel(ctx, p), i + 1 + k)] = \
                        window[max(0, k - 1)].strip()[:60]
                    break
    offenders = [(f, n, prev) for (f, n), prev in sorted(offenders.items())]
    if not offenders:
        return Result(PASS,
                      "no resampling loop skips a replicate without the file "
                      "accounting for how many survived",
                      evidence=f"scanned {len(_py_sources(ctx))} scripts")
    detail = "; ".join(f"{f}:{n} ({prev!r} -> continue)"
                       for f, n, prev in offenders[:4])
    return Result(FAIL,
                  f"{len(offenders)} bootstrap loop(s) discard resamples and "
                  f"no file records how many survived, so the denominator of "
                  f"the reported percentile interval is unknown and the "
                  f"interval is not reproducible: {detail}",
                  evidence=", ".join(f"{f}:{n}" for f, n, _ in offenders))


def _degenerate_bootstrap_guard(ctx: Context) -> Result:
    """§7.6: "bootstrapping confidence intervals are not quantitatively
    reliable in certain cases such as with small-sample sizes or distributions
    that are skewed or heavy-tailed."

    Our own instance of exactly this (FINDINGS §3.9c): a bootstrap CI over a
    degenerate 32/32 arm had zero width, excluded zero, and was quoted as
    SEPARATES against Fisher p = 1.000.
    """
    src = ctx.read("benchmark.py")
    if not src:
        return Result(UNTESTABLE,
                      "benchmark.py is absent, so the CI estimator this "
                      "project quotes cannot be inspected.",
                      evidence="benchmark.py")
    m = re.search(r"def boot_ci\(.*?\n(?=\ndef |\nclass |\Z)", src, re.S)
    if not m:
        return Result(UNTESTABLE,
                      "benchmark.py no longer defines `boot_ci`; the check "
                      "needs updating to point at the current estimator.",
                      evidence="benchmark.py: def boot_ci not found")
    body = m.group(0)
    guarded = bool(re.search(
        r"ptp\(|max\(vals\)\s*[-=]|len\(set\(|nunique|degener|zero.width|"
        r"np\.allclose\(|np\.std\(vals\)", body))
    if guarded:
        return Result(PASS,
                      "boot_ci inspects the spread of its own resample "
                      "distribution before returning an interval",
                      evidence="benchmark.py: def boot_ci")
    return Result(FAIL,
                  "boot_ci returns percentiles of the resample distribution "
                  "with no degeneracy guard: if every resample gives the same "
                  "value it returns a zero-width interval that necessarily "
                  "excludes zero. Its only guard is `if not vals: return "
                  "(nan, nan)`, which fires when *all* replicates fail, not "
                  "when they all agree.",
                  evidence=f"benchmark.py: boot_ci body, {len(body.splitlines())} "
                           f"lines, no spread test")


def _one_sigma_is_not_a_decision_rule(ctx: Context) -> Result:
    """§7.1: "1 s(x) uncertainties only correspond to a 68 % level of
    confidence, which may be too low" — and the recommendation is a 95 %
    interval, i.e. an expanded uncertainty k*s with k from Table 1."""
    sources = _py_sources(ctx)
    if not sources:
        return Result(UNTESTABLE,
                      "no analysis scripts were found to scan, so 'no 1-SD "
                      "gate' would mean 'nothing was looked at'.",
                      evidence="glob('*.py') and glob('cluster/*.py') empty")
    hits = []
    for p in sources:
        try:
            lines = _code_lines(p.read_text(errors="replace"))
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            # A noise-floor constant *used in a comparison* — i.e. it decides
            # something. Merely printing a null distribution's SD, correctly
            # labelled, is legitimate and is deliberately not flagged.
            if re.search(r"\bNOISE\w*\b", ln) and re.search(r"[<>]=?", ln):
                hits.append((_rel(ctx, p), i + 1,
                             "a 1-SD constant decides a verdict",
                             ln.strip()[:90]))
            # The retracted +/-1.5 — the SD of a *random* ranking (FINDINGS
            # §3.9/PLAN:57) — still bound to a live constant or printed as ours.
            if re.search(r"\bNOISE\w*\s*=\s*1\.5\b", ln) or \
               re.search(r"noise floor[^\n]{0,12}1\.5", ln, re.I):
                hits.append((_rel(ctx, p), i + 1,
                             "the retracted +/-1.5 (SD of a RANDOM ranking) is "
                             "still live", ln.strip()[:90]))
    if not hits:
        return Result(PASS,
                      "no analysis script gates a conclusion on a one-standard-"
                      "deviation quantity",
                      evidence=f"scanned {len(_py_sources(ctx))} scripts")
    detail = "; ".join(f"{f}:{n} {why}" for f, n, why, _ in hits[:4])
    return Result(FAIL,
                  f"{len(hits)} site(s) decide feasibility or significance "
                  f"against a one-standard-deviation number, which is a 68 % "
                  f"level of confidence where the paper asks for 95 %: "
                  f"{detail}. At k = 1 the threshold is roughly half what a "
                  f"95 % interval requires.",
                  evidence="; ".join(f"{f}:{n}: {ln}" for f, n, _, ln in hits[:6]))


def _significant_figures(ctx: Context) -> Result:
    """§7.1: "if the mean of a quantity is calculated to be 1.23456 with
    uncertainty +-0.1 based on a 95 % confidence interval, then only two
    significant figures should be reported for the mean (1.2)."."""
    txt = ctx.read(LEADERBOARD)
    if not txt.strip():
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} is absent, so no reported precision can "
                      f"be compared against its interval.", evidence=LEADERBOARD)
    hdr, rows = _rows(ctx, LEADERBOARD)
    bad = []
    for col in hdr:
        lo, hi = f"{col}_lo", f"{col}_hi"
        if lo not in hdr or hi not in hdr:
            continue
        for r in rows:
            try:
                u = (float(r[hi]) - float(r[lo])) / 2.0
                raw = (r[col] or "").strip()
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if u <= 0:
                continue
            # decimals the 95 % half-width supports, plus one guard digit
            allowed = max(0, -math.floor(math.log10(u))) + 1
            written = len(raw.split(".")[1]) if "." in raw else 0
            if written > allowed:
                bad.append((r.get("name", "?"), col, raw, u, allowed, written))
    if not bad:
        return Result(PASS,
                      "every value with an interval is quoted to a precision "
                      "that interval supports",
                      evidence=f"{LEADERBOARD}, {len(rows)} row(s)")
    n, col, raw, u, allowed, written = bad[0]
    return Result(FAIL,
                  f"{len(bad)} value(s) are quoted to more decimals than their "
                  f"own 95 % interval supports — e.g. {col} = {raw} for "
                  f"'{n}' with a half-width of {u:.2f}, which supports "
                  f"{allowed} decimal(s) (one guard digit included) but is "
                  f"written to {written}.",
                  evidence="; ".join(f"{a}/{b}={c} +-{d:.2f}"
                                     for a, b, c, d, _, _ in bad[:5]))


def _uq_procedure_is_recorded(ctx: Context) -> Result:
    """§2 item 8: "Describe the meaning and basis of uncertainties given in
    figures or tables in the captions for those items, e.g., 'Error bars
    represent 95 % confidence intervals based on bootstrapping results from the
    independent simulations.'" """
    hdr, rows = _rows(ctx, LEADERBOARD)
    if not hdr:
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} is absent, so there is no reported "
                      f"interval whose basis could be recorded.",
                      evidence=LEADERBOARD)
    has_interval = any(c.endswith(("_lo", "_hi")) for c in hdr)
    meta = [c for c in hdr
            if re.search(r"ci_level|conf_level|ci_method|coverage|n_boot|"
                         r"resampl|estimator", c, re.I)]
    if not has_interval:
        return Result(FAIL,
                      f"{LEADERBOARD} reports no interval at all, so there is "
                      f"nothing whose meaning could be described. §2 item 8 "
                      f"presupposes an uncertainty exists.",
                      evidence=f"columns: {', '.join(hdr)}")
    if meta:
        return Result(PASS,
                      f"the interval's basis is machine-readable: "
                      f"{', '.join(meta)}", evidence=f"{LEADERBOARD}")
    return Result(FAIL,
                  f"{LEADERBOARD} carries lo/hi columns but records neither "
                  f"the confidence level, the estimator, nor the number of "
                  f"resamples. A reader of the file cannot tell a 95 % "
                  f"percentile bootstrap from a 68 % normal interval, and "
                  f"§2 item 8 requires exactly that to be stated.",
                  evidence=f"{len(rows)} row(s); interval columns "
                           f"{[c for c in hdr if c.endswith(('_lo', '_hi'))]}")


def _raw_data_still_present(ctx: Context) -> Result:
    """§2 item 8: "We strongly urge publication of unprocessed simulation data
    (measurements/observations) and post-processing scripts ... so that readers
    can exactly reproduce the processed results and uncertainty estimates."."""
    hdr, rows = _rows(ctx, LEADERBOARD)
    if not rows:
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} has no rows, so no headline number "
                      f"points at a source file yet.", evidence=LEADERBOARD)
    if "source" not in hdr:
        return Result(FAIL,
                      f"{LEADERBOARD} does not record where each number came "
                      f"from; a `source` column is the minimum that makes a "
                      f"result re-derivable.", evidence=f"columns: {', '.join(hdr)}")
    missing = sorted({r["source"] for r in rows
                      if r.get("source") and not ctx.exists(r["source"])})
    if missing:
        return Result(FAIL,
                      f"{len(missing)} leaderboard row(s) cite a source file "
                      f"that no longer exists, so the number cannot be "
                      f"re-derived: {', '.join(missing)}",
                      evidence=f"{len(rows)} row(s) checked")
    return Result(PASS,
                  f"all {len(rows)} leaderboard rows cite a source file that "
                  f"is still present",
                  evidence=", ".join(sorted({r['source'] for r in rows})))


# ==========================================================================
# §4 / §5 — did the sampling happen at all
# ==========================================================================

def _zeroth_order_stationarity(ctx: Context) -> Result:
    """§4.1: "one should inspect the time series for a number of simple scalar
    values, such as potential energy, system size ..., temperature ... and/or
    density. ... If *any* value appears to be systematically changing, then the
    system may not be equilibrated"."""
    states = ctx.glob(STATE_GLOB)
    if not states:
        return Result(UNTESTABLE,
                      f"no {STATE_GLOB} exists, so the zeroth-order time "
                      f"series cannot be inspected. Any `md_arms.py run` "
                      f"writes one and makes this testable.",
                      evidence=STATE_GLOB)
    worst, checked, skipped = None, 0, 0
    for p in states:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for ch in ZEROTH_ORDER:
            v = _floats(txt, ch)
            d = _split_half_drift(v)
            if d is None:
                skipped += 1
                continue
            checked += 1
            if worst is None or d > worst[0]:
                worst = (d, _rel(ctx, p), ch, len(v))
    if worst is None:
        return Result(UNTESTABLE,
                      f"{len(states)} state.csv found but none carried >= 8 "
                      f"points of any of {ZEROTH_ORDER}, so §4.1's test could "
                      f"not be run. This is exactly the vacuous-PASS case the "
                      f"package exists to prevent.",
                      evidence=f"{skipped} channel(s) too short")
    d, path, ch, n = worst
    where = f"worst channel: {path}:{ch}, {n} points, |dmean|/SD = {d:.2f}"
    if d > DRIFT_SD:
        return Result(FAIL,
                      f"a production time series is still drifting: the two "
                      f"halves' means differ by {d:.2f} SD of the series "
                      f"itself. §4.1 says the system may not be equilibrated "
                      f"and further investigation is warranted; §7.2 says a "
                      f"reliable uncertainty on such an observable cannot be "
                      f"obtained at all.",
                      evidence=f"{where}; {checked} channel(s) across "
                               f"{len(states)} run(s), threshold {DRIFT_SD} SD")
    return Result(PASS,
                  f"all {checked} zeroth-order series across {len(states)} run(s) "
                  f"fluctuate about a mean; the largest split-half shift is "
                  f"{d:.2f} SD (threshold {DRIFT_SD})",
                  evidence=where)


def _observable_of_interest_stationarity(ctx: Context) -> Result:
    """§2 item 3: "Critically examine the time series of a number of
    observables, both those of interest *and others*." The system-wide channels
    are the "others"; ligand RMSD, drift and Rg are the ones we score on."""
    npz = ctx.glob(NPZ_GLOB)
    if not npz:
        return Result(UNTESTABLE,
                      f"no {NPZ_GLOB} exists, so the observables we actually "
                      f"score on have no saved time series to inspect. "
                      f"`md_readouts.py extract` writes them.",
                      evidence=NPZ_GLOB)
    try:
        import numpy as np
    except ImportError:
        return Result(UNTESTABLE,
                      "numpy is not importable in this interpreter, so the "
                      ".npz time series cannot be read. Run the suite in the "
                      "project environment.", evidence="import numpy")
    worst, checked = None, 0
    for p in npz:
        try:
            z = np.load(p)
        except (OSError, ValueError):
            continue
        for ch in ("rmsd", "drift", "rgyr"):
            if ch not in z:
                continue
            v = [float(x) for x in np.asarray(z[ch], dtype=float)]
            d = _split_half_drift(v)
            if d is None:
                continue
            checked += 1
            if worst is None or d > worst[0]:
                worst = (d, _rel(ctx, p), ch, len(v))
    if worst is None:
        return Result(UNTESTABLE,
                      f"{len(npz)} readouts.npz found but none carried a "
                      f"usable rmsd/drift/rgyr series of >= 8 points.",
                      evidence=NPZ_GLOB)
    d, path, ch, n = worst
    where = f"worst: {path}:{ch}, {n} frames, |dmean|/SD = {d:.2f}"
    if d > DRIFT_SD:
        return Result(FAIL,
                      f"a scored observable is drifting rather than "
                      f"fluctuating: split-half shift {d:.2f} SD. Its mean is "
                      f"a transient, not an ensemble average.",
                      evidence=f"{where}; {checked} series over {len(npz)} runs")
    return Result(PASS,
                  f"all {checked} scored series over {len(npz)} runs fluctuate "
                  f"about a mean; largest split-half shift {d:.2f} SD "
                  f"(threshold {DRIFT_SD})", evidence=where)


def _burn_in_is_discarded(ctx: Context) -> Result:
    """§2 item 4: "Remove an 'equilibration' (a.k.a. 'burn-in', or transient)
    portion of a single MD or MC trajectory and perform analyses only on the
    remaining 'production' portion"."""
    receipts = ctx.glob(RECEIPT_GLOB)
    if not receipts:
        return Result(UNTESTABLE,
                      f"no {RECEIPT_GLOB} exists, so no run records an "
                      f"equilibration time. `md_arms.py run` writes one.",
                      evidence=RECEIPT_GLOB)
    bad, ok = [], 0
    for r in receipts:
        try:
            d = json.loads(r.read_text())
        except (ValueError, OSError):
            continue
        eq = float(d.get("equil_ps") or 0.0)
        run = _rel(ctx, r.parent)
        if eq <= 0:
            bad.append(f"{run}: equil_ps = {eq}")
            continue
        # And was the *saved* data actually restricted to production?
        state = r.parent / "state.csv"
        if state.exists():
            t = _floats(state.read_text(errors="replace"), "Time (ps)")
            if t and t[0] < eq:
                bad.append(f"{run}: first saved frame at {t[0]:.1f} ps < "
                           f"equil_ps {eq:.1f}")
                continue
        ok += 1
    if bad:
        return Result(FAIL,
                      f"{len(bad)} of {len(receipts)} run(s) analyse data from "
                      f"before the end of equilibration: {'; '.join(bad[:3])}",
                      evidence=f"{ok} run(s) clean")
    return Result(PASS,
                  f"all {ok} run(s) discard a burn-in and the first saved "
                  f"frame lies after it",
                  evidence=f"{RECEIPT_GLOB}: equil_ps > 0 and "
                           f"state.csv starts after it in every run")


def _tequil_sensitivity_tested(ctx: Context) -> Result:
    """§5: "if values of observables estimated from the production phase depend
    sensitively on the choice of t_equil, it is likely that further sampling is
    required" — and Fig. 3, "select t_equil in a systematic way"."""
    receipts = ctx.glob(RECEIPT_GLOB)
    if not receipts:
        return Result(UNTESTABLE,
                      f"no {RECEIPT_GLOB} exists, so the choice of t_equil "
                      f"cannot be examined.", evidence=RECEIPT_GLOB)
    groups = {}
    for r in receipts:
        try:
            d = json.loads(r.read_text())
        except (ValueError, OSError):
            continue
        groups.setdefault(float(d.get("prod_ps") or 0.0), set()).add(
            float(d.get("equil_ps") or 0.0))
    varied = {k: v for k, v in groups.items() if len(v) > 1}
    src = " ".join(ctx.read(f) for f in ("md_arms.py", "md_readouts.py"))
    scanned = bool(re.search(r"equil[_-]?scan|t_?equil.*(scan|sweep|sensitiv)|"
                             r"reverse.cumulative|detectEquilibration", src, re.I))
    if varied or scanned:
        return Result(PASS,
                      "t_equil has been varied within a protocol, or a "
                      "systematic equilibration-detection routine exists",
                      evidence=f"prod_ps groups with >1 equil_ps: {varied}; "
                               f"scan routine in source: {scanned}")
    summary = ", ".join(f"prod {k:.0f} ps -> equil {sorted(v)}"
                        for k, v in sorted(groups.items()))
    return Result(FAIL,
                  f"t_equil is a hard-coded constant per protocol and its "
                  f"effect on the reported observables has never been "
                  f"measured. §5 asks for a systematic choice and warns that "
                  f"sensitivity to it means more sampling is needed; we would "
                  f"not currently detect that sensitivity if it existed.",
                  evidence=f"{len(receipts)} receipt(s): {summary}")


# ==========================================================================
# §3 / §4.4 / §6 — sampling quality
# ==========================================================================

def _multiple_independent_runs(ctx: Context) -> Result:
    """§2 item 1: "Consider multiple runs instead of a single simulation.
    Diverse starting structures enable a check on sampling for equilibrium
    ensembles, which should not depend on the starting structure." §7.2 adds
    that if the starting configuration is shared you must "recognize and report
    that the uncertainty refers to simulations started from a particular
    configuration"."""
    outs = ctx.glob(READOUT_GLOB)
    if not outs:
        return Result(UNTESTABLE,
                      f"no {READOUT_GLOB} exists, so replicate structure "
                      f"cannot be counted.", evidence=READOUT_GLOB)
    reps, starts = {}, set()
    for p in outs:
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        reps[_rel(ctx, p.parent)] = d.get("n_replicates")
    for p in ctx.glob(RECEIPT_GLOB):
        try:
            starts.add(json.loads(p.read_text()).get("system_sha256"))
        except (ValueError, OSError):
            pass
    multi = {k: v for k, v in reps.items() if isinstance(v, int) and v > 1}
    if multi:
        return Result(PASS,
                      f"{len(multi)} arm(s) aggregate more than one replicate: "
                      f"{multi}",
                      evidence=f"{len(starts)} distinct starting system hash(es)")
    counted = {k: v for k, v in reps.items() if v is not None}
    n_runs = len(ctx.glob(RECEIPT_GLOB))
    shared = (f"and all {n_runs} run(s) share a single starting configuration "
              f"(one system_sha256), so any spread they show is over velocity "
              f"seeds alone" if len(starts) == 1 else
              f"across {len(starts)} distinct starting configurations")
    return Result(FAIL,
                  f"every arm with a recorded replicate count has "
                  f"n_replicates = 1, so no run-to-run variance exists to form "
                  f"an uncertainty from; {shared}. §7.2 requires that "
                  f"limitation to be reported with the number, and the receipt "
                  f"does not carry it.",
                  evidence=f"n_replicates by arm: {counted or 'not recorded'}; "
                           f"{len(reps) - len(counted)} arm(s) do not record it "
                           f"at all; {len(starts)} distinct system hash(es) "
                           f"over {n_runs} run(s)")


def _global_sampling_measure(ctx: Context) -> Result:
    """§2 item 5: "Consider computing a quantitative measure of global
    sampling, i.e., attempt to estimate the number of statistically independent
    samples in a trajectory." §6 names decorrelation analysis (LOOS), block
    covariance overlap (BCOM), and combined clustering (§4.4)."""
    found, traj_scripts = [], []
    for p in _py_sources(ctx):
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        # The measure only counts if it is applied to a trajectory. An
        # "all-to-all RMSD" over a 32x32 cross-docking matrix is a different
        # object entirely and passing on it would be vacuous — which is what
        # the first version of this check did.
        if not TRAJ_CONTEXT.search(txt):
            continue
        traj_scripts.append(_rel(ctx, p))
        for pat, name in ((r"all[_ -]to[_ -]all", "all-to-all RMSD map"),
                          (r"combined[_ ]clust", "combined clustering"),
                          (r"\bBCOM\b|covariance[_ ]overlap", "BCOM"),
                          (r"\bLOOS\b|pyloos", "LOOS"),
                          (r"block[_ ]?averag", "block averaging"),
                          (r"decorrelation|statistical[_ ]inefficiency|"
                           r"effective sample size", "decorrelation analysis")):
            if re.search(pat, txt, re.I):
                found.append(f"{name} in {_rel(ctx, p)}")
    if not traj_scripts:
        return Result(UNTESTABLE,
                      "no script in this repo analyses a trajectory, so there "
                      "is nothing for a global sampling measure to be applied "
                      "to. Writing any MDAnalysis-based trajectory analysis "
                      "makes this testable.",
                      evidence=f"scanned {len(_py_sources(ctx))} scripts for "
                               f"{TRAJ_CONTEXT.pattern}")
    if found:
        return Result(PASS,
                      f"a global sampling measure is applied to trajectory "
                      f"data: {'; '.join(sorted(set(found))[:4])}",
                      evidence=f"{len(traj_scripts)} trajectory script(s) of "
                               f"{len(_py_sources(ctx))} scanned")
    return Result(FAIL,
                  f"none of the {len(traj_scripts)} script(s) that touch a "
                  f"trajectory computes a global sampling measure — no "
                  f"all-to-all RMSD map, no combined clustering, no block "
                  f"averaging, no decorrelation/BCOM analysis. Our only "
                  f"sampling evidence is RMSD to a single reference, which "
                  f"§4.2 says \"should really be considered as another "
                  f"equilibration test\" and is degenerate: \"you cannot tell "
                  f"if the simulation is exploring new states that are "
                  f"equidistant from the chosen reference\".",
                  evidence=f"trajectory scripts: {', '.join(traj_scripts)}")


def _all_points_for_small_n(ctx: Context) -> Result:
    """§7.1(c): "when feasible and especially for a small number of independent
    measurements (n < 10), authors should consider plotting all of the points
    instead of an average with error bars"."""
    outs = ctx.glob(READOUT_GLOB)
    if not outs:
        return Result(UNTESTABLE,
                      f"no {READOUT_GLOB} exists, so it cannot be told whether "
                      f"per-run values are retained.", evidence=READOUT_GLOB)
    per_run = len(outs)
    if per_run < 2:
        return Result(UNTESTABLE,
                      f"only {per_run} run has read-outs, so there is no "
                      f"small-n aggregate to check. Extracting a second "
                      f"replicate makes this testable.",
                      evidence=READOUT_GLOB)
    return Result(PASS,
                  f"all {per_run} runs keep their own readouts.json, so every "
                  f"individual point survives alongside any aggregate — which "
                  f"is what §7.1(c) asks for at n < 10",
                  evidence="; ".join(_rel(ctx, p.parent) for p in outs[:6]))


def _uniform_compound_set(ctx: Context) -> Result:
    """§2 item 2: "Use all of the available data unless there is an objective
    and compelling reason not to ... When used, sampling metrics should be
    applied uniformly to *all* simulations to further avoid bias."."""
    hdr, rows = _rows(ctx, LEADERBOARD)
    if not rows:
        return Result(UNTESTABLE,
                      f"{LEADERBOARD} has no rows to compare.",
                      evidence=LEADERBOARD)
    if "n_all" not in hdr:
        return Result(FAIL,
                      f"{LEADERBOARD} does not record how many compounds each "
                      f"method was scored on, so two hits@25 values cannot be "
                      f"known to be comparable. `benchmark.py` drops rows with "
                      f"a missing score, so the set can differ silently.",
                      evidence=f"columns: {', '.join(hdr)}")
    counts = {}
    for r in rows:
        counts.setdefault(r.get("n_all"), []).append(r.get("name"))
    if len(counts) > 1:
        return Result(FAIL,
                      f"methods on the same leaderboard were scored on "
                      f"different compound sets, so their hits@N are not "
                      f"comparable: "
                      f"{ {k: v for k, v in counts.items()} }",
                      evidence=f"{len(rows)} row(s), n_all values "
                               f"{sorted(counts)}")
    n = next(iter(counts))
    return Result(PASS,
                  f"all {len(rows)} methods were scored on the same "
                  f"{n} compounds",
                  evidence=f"{LEADERBOARD}: n_all == {n} for every row")


def _known_answer_controls(ctx: Context) -> Result:
    """§2 item 1: "Check and validate your code/method via a simple benchmark
    system." The paper points at shirtsgroup/software-physical-validation for
    the physics; the analysis pipeline needs the same treatment."""
    src = ctx.read("benchmark.py")
    if not src:
        return Result(UNTESTABLE,
                      "benchmark.py is absent, so the scoring pipeline's "
                      "controls cannot be inspected.", evidence="benchmark.py")
    has = {name: bool(re.search(pat, src))
           for name, pat in (("oracle", r'"oracle"'),
                             ("inverted", r'"inverted"'),
                             ("random", r'"random"'))}
    default_on = bool(re.search(r"if not args\.skip_self_test", src))
    ci_contains_zero = bool(re.search(r"lo <= 0 <= hi", src))
    missing = [k for k, v in has.items() if not v]
    if missing or not default_on:
        return Result(FAIL,
                      f"the scoring pipeline's known-answer controls are "
                      f"incomplete or not run by default: missing "
                      f"{missing or 'none'}; runs by default = {default_on}.",
                      evidence="benchmark.py: self_test")
    return Result(PASS,
                  f"oracle / inverted / random controls run by default, and "
                  f"the random arm additionally requires its bootstrap CI to "
                  f"contain zero ({ci_contains_zero})",
                  evidence="benchmark.py: self_test, gated on "
                           "`if not args.skip_self_test`")


# ==========================================================================
# §8 — enhanced sampling
# ==========================================================================

def _enhanced_sampling_diagnostics(ctx: Context) -> Result:
    """§8.1 lists four required REMD diagnostics — exchange acceptance, replica
    round trips, replica residence time, and distributions from coordinate
    trajectories — and §8.2 the weighted-ensemble analogue."""
    hits = []
    for p in _py_sources(ctx):
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if re.search(r"\bREMD\b|replica[_ ]exchange|ReplicaExchange|"
                     r"metadynamic|\bWESTPA\b|weighted[_ ]ensemble|"
                     r"HREX|hamiltonian[_ ]replica", txt, re.I):
            hits.append(_rel(ctx, p))
    if not hits:
        return Result(NA,
                      "no replica-exchange, metadynamics or weighted-ensemble "
                      "simulation exists in this repo — production sampling is "
                      "conventional Langevin MD (md_arms.py), so §8's "
                      "diagnostics (exchange acceptance, round trips, "
                      "residence time) have no input to run on. This flips to "
                      "a live requirement the moment RBFE/ABFE lands, since "
                      "those are normally run with lambda-Hamiltonian replica "
                      "exchange.",
                      evidence=f"scanned {len(_py_sources(ctx))} scripts; no "
                               f"REMD/metadynamics/WE reference")
    diag = [n for n, pat in (("exchange acceptance", r"acceptance"),
                             ("round trips", r"round[_ ]?trip"),
                             ("residence time", r"residence"))
            if any(re.search(pat, Path(f).read_text(errors="replace"), re.I)
                   for f in [str(ctx.root / h) for h in hits])]
    if len(diag) >= 2:
        return Result(PASS,
                      f"enhanced sampling present and diagnosed: "
                      f"{', '.join(diag)}", evidence=", ".join(hits))
    return Result(FAIL,
                  f"enhanced sampling appears in {', '.join(hits)} but only "
                  f"{len(diag)} of §8.1's four diagnostics are computed "
                  f"({', '.join(diag) or 'none'}). Uncertainty from correlated "
                  f"replicas cannot be assessed without them.",
                  evidence=", ".join(hits))


# --------------------------------------------------------------------------

CHECKS = [
    # --- §7: what error bar goes on the number -----------------------------
    Check(id="unc-01", severity="must", section="§7.1(b), §2 checklist item 6",
          question="Does every reported figure of merit carry a 95 % "
                   "confidence interval?",
          measures=f"columns of {LEADERBOARD}", fn=_ci_on_every_fom),
    Check(id="unc-02", severity="must", section="§7.2, §7.3.1 Eq. 5",
          question="Is the number of independent samples behind a "
                   "trajectory-derived mean estimated and recorded?",
          measures=f"{READOUT_GLOB} keys; Eq. 5 evaluated on {NPZ_GLOB}",
          fn=_effective_sample_size),
    Check(id="unc-03", severity="must", section="§7.6, §2 checklist item 8",
          question="Do our bootstraps report how many resamples they actually "
                   "used?",
          measures="resampling loops in *.py and cluster/*.py",
          fn=_bootstrap_does_not_drop_silently),
    Check(id="unc-04", severity="must", section="§7.1",
          question="Are decisions gated on a 95 % interval rather than a "
                   "one-standard-deviation number?",
          measures="threshold constants and comparisons in *.py",
          fn=_one_sigma_is_not_a_decision_rule),
    Check(id="unc-05", severity="should", section="§1.2.1 glossary, §7.1(b)",
          question="Where an uncertainty is given, is it a confidence interval "
                   "rather than a bare standard deviation?",
          measures=f"columns of {SWEEP_SUMMARY}", fn=_sd_is_not_an_uncertainty),
    Check(id="unc-06", severity="should", section="§7.6",
          question="Does the bootstrap guard against a degenerate resample "
                   "distribution?",
          measures="benchmark.py: boot_ci", fn=_degenerate_bootstrap_guard),
    Check(id="unc-07", severity="should", section="§7.1",
          question="Are values quoted to more significant figures than their "
                   "own uncertainty supports?",
          measures=f"value/lo/hi triples in {LEADERBOARD}",
          fn=_significant_figures),
    Check(id="unc-08", severity="should", section="§2 checklist item 8",
          question="Is the meaning and basis of each reported uncertainty "
                   "recorded with it?",
          measures=f"columns of {LEADERBOARD}", fn=_uq_procedure_is_recorded),
    Check(id="unc-09", severity="should", section="§2 checklist item 8",
          question="Is the unprocessed data behind every headline number still "
                   "present?",
          measures=f"`source` column of {LEADERBOARD}",
          fn=_raw_data_still_present),

    # --- §4/§5: did the sampling happen ------------------------------------
    Check(id="unc-10", severity="must", section="§4.1, §2 checklist item 3",
          question="Do the zeroth-order system-wide time series fluctuate "
                   "about a mean rather than drift?",
          measures=f"split-half means of {ZEROTH_ORDER[0]} etc. in {STATE_GLOB}",
          fn=_zeroth_order_stationarity),
    Check(id="unc-11", severity="must", section="§5, §2 checklist item 4",
          question="Is an equilibration/burn-in portion discarded before "
                   "analysis?",
          measures=f"equil_ps in {RECEIPT_GLOB} vs first frame in state.csv",
          fn=_burn_in_is_discarded),
    Check(id="unc-12", severity="should", section="§4.1, §2 checklist item 3",
          question="Are the observables we actually score on checked for drift, "
                   "not just the thermodynamic ones?",
          measures=f"split-half means of rmsd/drift/rgyr in {NPZ_GLOB}",
          fn=_observable_of_interest_stationarity),
    Check(id="unc-13", severity="should", section="§6, §4.2, §2 checklist item 5",
          question="Is there any quantitative measure of global sampling?",
          measures="all-to-all RMSD / combined clustering / BCOM / block "
                   "averaging in *.py",
          fn=_global_sampling_measure),
    Check(id="unc-14", severity="consider", section="§5",
          question="Has the sensitivity of production observables to the choice "
                   "of t_equil been tested?",
          measures=f"equil_ps spread across {RECEIPT_GLOB}",
          fn=_tequil_sensitivity_tested),

    # --- §2/§3: study design ------------------------------------------------
    Check(id="unc-15", severity="should", section="§3, §7.2, §2 checklist item 1",
          question="Are there multiple independent runs, and is a shared "
                   "starting configuration reported as a limitation?",
          measures=f"n_replicates in {READOUT_GLOB}; system_sha256 in receipts",
          fn=_multiple_independent_runs),
    Check(id="unc-16", severity="should", section="§2 checklist item 2",
          question="Was every method on the leaderboard scored on the same "
                   "compound set?",
          measures=f"`n_all` column of {LEADERBOARD}", fn=_uniform_compound_set),
    Check(id="unc-17", severity="should", section="§3, §2 checklist item 1",
          question="Is the analysis pipeline validated against known answers "
                   "by default?",
          measures="benchmark.py: self_test", fn=_known_answer_controls),
    Check(id="unc-18", severity="consider", section="§7.1(c)",
          question="For n < 10 runs, are the individual points retained rather "
                   "than only an aggregate?",
          measures=f"count of {READOUT_GLOB}", fn=_all_points_for_small_n),

    # --- §8 ------------------------------------------------------------------
    Check(id="unc-19", severity="should", section="§8.1, §8.2",
          question="If enhanced sampling is used, are its convergence "
                   "diagnostics computed?",
          measures="REMD/metadynamics/WE references in *.py",
          fn=_enhanced_sampling_diagnostics),
]
