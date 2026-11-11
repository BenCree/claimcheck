"""Braun et al., *Best Practices for Foundations in Molecular Simulations*.

LiveCoMS 1(1):5957 (2018), doi:10.33011/livecoms.1.1.5957. The article carries a
one-page checklist (p. 23) under six headings, reproduced here as executable
checks against **our own MD stack** (`md_arms.py` -> `results/md_pilot/`) and
against the artefact the objective metric is computed from
(`results/score_comparison.csv`).

Scope note, stated once so no check has to repeat it. The paper governs
**classical condensed-phase MD**: "for the rest of this document we will restrict
ourselves to classical MD" (§1). Our MACE interaction energies are a
single-point MLP evaluation on a truncated pocket, not MD, so ensemble/thermostat
/barostat rules do not reach them — but the **electrostatics** rules do, because
those are stated for the model and not for the propagator, and a truncated pocket
is precisely the "finite system" case the checklist's cutoff box addresses.

What this module deliberately does not encode:

* the *reading list* (§3.1-3.5 name textbooks, not practices) — not checkable;
* "determine reference states you are trying to emulate" — we have `LITERATURE.md`
  and a stated objective, and any check would be a tautology over a file we
  always keep;
* "ensure you have sufficient storage, memory and computer time" — subsumed by
  `found-13`, which measures the cost model rather than asking about disk.
"""

from __future__ import annotations

import csv
import json
import re

from bestpractices import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE

PAPER = Paper(
    slug="foundations",
    title="Best Practices for Foundations in Molecular Simulations [Article v1.0]",
    authors=("Efrem Braun, Justin Gilmer, Heather B. Mayes, David L. Mobley, "
             "Jacob I. Monroe, Samarjeet Prasad, Daniel M. Zuckerman"),
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v1i1e5957",
    doi="10.33011/livecoms.1.1.5957",
    year="2018",
    scope=("Setting up and running classical condensed-phase MD correctly: "
           "ensemble and state point, periodicity and cutoffs, thermostat and "
           "barostat, timestep, and the minimise/equilibrate/produce protocol."),
)

MD_RUNNER = "md_arms.py"
READOUTS = "md_readouts.py"
RECEIPTS = "results/md_pilot/*/*/receipt.json"
SYSTEMS = "results/md_pilot/*/system.pdb"


# --------------------------------------------------------------------------
# small shared parsers
# --------------------------------------------------------------------------

def _cutoff_nm(src: str) -> float | None:
    """The nonbonded cutoff `md_arms.py` passes to `createSystem`, in nm."""
    m = re.search(r"nonbondedCutoff\s*=\s*([\d.]+)\s*\*\s*unit\.nanometer", src)
    return float(m.group(1)) if m else None


def _cryst1(text: str) -> tuple[float, float, float] | None:
    """(a, b, c) in Angstrom from the first CRYST1 record."""
    for line in text.splitlines():
        if line.startswith("CRYST1"):
            try:
                return (float(line[6:15]), float(line[15:24]), float(line[24:33]))
            except ValueError:
                return None
    return None


def _state_series(path) -> tuple[list[float], dict[str, list[float]]] | None:
    """(time_ps, {column: values}) from an OpenMM StateDataReporter CSV."""
    try:
        rows = list(csv.reader(path.open()))
    except OSError:
        return None
    if len(rows) < 12:
        return None
    hdr = [h.lstrip('#').strip('"') for h in rows[0]]
    body = [r for r in rows[1:] if len(r) == len(hdr)]
    if len(body) < 12 or "Time (ps)" not in hdr:
        return None
    cols: dict[str, list[float]] = {}
    for i, name in enumerate(hdr):
        try:
            cols[name] = [float(r[i]) for r in body]
        except ValueError:
            continue
    if "Time (ps)" not in cols:
        return None
    return cols["Time (ps)"], cols


def _trend_over_sd(t: list[float], y: list[float]) -> float:
    """|OLS trend across the whole window| / SD of the series.

    An operationalisation of Fig. 4: a property whose straight-line change over
    the production window is larger than its own scatter is still moving
    systematically. Unitless, so it compares across observables.
    """
    n = len(t)
    mt, my = sum(t) / n, sum(y) / n
    sxx = sum((a - mt) ** 2 for a in t)
    if sxx == 0:
        return 0.0
    slope = sum((a - mt) * (b - my) for a, b in zip(t, y)) / sxx
    sd = (sum((b - my) ** 2 for b in y) / n) ** 0.5
    if sd == 0:
        return 0.0
    return abs(slope * (t[-1] - t[0])) / sd


def _receipts(ctx: Context) -> list[tuple[str, dict]]:
    out = []
    for p in ctx.glob(RECEIPTS):
        try:
            out.append((p.parent.name, json.loads(p.read_text())))
        except (OSError, ValueError):
            continue
    return out


def _repo_python(ctx: Context) -> list:
    return ctx.glob("*.py") + ctx.glob("cluster/*.py") + ctx.glob("albench/*.py")


# --------------------------------------------------------------------------
# TAKE STOCK OF YOUR PLANS  (checklist box 1)
# --------------------------------------------------------------------------

def _cost_counted(ctx: Context) -> Result:
    """"Count the cost ... determine whether it is tractable to simulate this
    given the size of your system, your computational resources, and the expense
    of the simulation." (checklist box 1; §5 "Should you run MD?")"""
    text = ctx.read("results/cost_curve.csv")
    if not text:
        return Result(UNTESTABLE, "no results/cost_curve.csv — nothing records "
                                  "what a stage costs before we commit to it",
                      evidence="expected results/cost_curve.csv")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or "method" not in rows[0]:
        return Result(UNTESTABLE, "cost_curve.csv has no `method` column",
                      evidence=str(list(rows[0]) if rows else []))
    dyn = [r for r in rows
           if re.search(r"\bMD\b|FEP|RBFE|ABFE|dynamic", r["method"], re.I)]
    if not dyn:
        return Result(FAIL, "the cost model prices no dynamics stage at all, so "
                            "no MD/FE stage has had its cost counted",
                      evidence="; ".join(r["method"] for r in rows))
    ruled_out = [r["method"] for r in dyn if str(r.get("feasible")).lower() == "false"]
    return Result(PASS,
                  f"{len(dyn)} dynamics stage(s) priced before launch"
                  + (f"; {ruled_out[0]} ruled infeasible on cost" if ruled_out else ""),
                  evidence="; ".join(
                      f"{r['method']} = {r['sec']}s/lig ({r['src']}, "
                      f"feasible={r.get('feasible')})" for r in dyn))


def _state_point_recorded(ctx: Context) -> Result:
    """"Pick the desired ensemble (NVT, NPT, NVE, muVT)" and "What temperature,
    pressure, etc. are you interested in?" (checklist box 1)"""
    rec = _receipts(ctx)
    if not rec:
        return Result(UNTESTABLE, "no run receipts under results/md_pilot/",
                      evidence=RECEIPTS)
    missing = {"ensemble": [], "pressure": []}
    for tag, r in rec:
        keys = {k.lower() for k in r}
        if not any("ensemble" in k for k in keys):
            missing["ensemble"].append(tag)
        if not any("pressure" in k or k.startswith("p_") or k.endswith("_bar")
                   for k in keys):
            missing["pressure"].append(tag)
    if not missing["ensemble"] and not missing["pressure"]:
        return Result(PASS, f"all {len(rec)} receipts name the ensemble and both "
                            "halves of the state point")
    return Result(FAIL,
                  f"{len(missing['ensemble'])}/{len(rec)} receipts name no ensemble "
                  f"and {len(missing['pressure'])}/{len(rec)} record no target "
                  "pressure; temperature alone does not pin an NPT state point, "
                  "and the 1 bar setpoint exists only as a literal in md_arms.py",
                  evidence="receipt keys: " + ", ".join(sorted(rec[0][1])[:12]) + " ...")


# --------------------------------------------------------------------------
# PREPARE TO IMPLEMENT YOUR PLANS  (checklist box 2) / §4.1 force fields
# --------------------------------------------------------------------------

def _forcefield_recorded(ctx: Context) -> Result:
    """"it is important to pay particular attention (and make note of) exactly
    what version you are using and how you obtained it so you will be able to
    accurately detail this in any subsequent publications" (§4.1)"""
    rec = _receipts(ctx)
    if not rec:
        return Result(UNTESTABLE, "no run receipts under results/md_pilot/",
                      evidence=RECEIPTS)
    want = {
        "ligand force field": lambda k: "ligand_ff" in k or "lig_ff" in k,
        "protein force field": lambda k: "protein_ff" in k or "prot_ff" in k,
        "water model": lambda k: "water" in k,
        "engine version": lambda k: "openmm" in k or "version" in k,
    }
    keys = {k.lower() for _, r in rec for k in r}
    absent = [name for name, pred in want.items() if not any(pred(k) for k in keys)]
    if not absent:
        return Result(PASS, "receipts carry the whole model, not just the ligand half")
    return Result(FAIL,
                  f"receipt.json records no {' / no '.join(absent)}. The run is "
                  "reproducible only from md_arms.py's module-level literals "
                  "(PROTEIN_FF = amber14-all.xml + amber14/tip3p.xml), which the "
                  f"artefact does not carry and nothing versions. {len(rec)} "
                  "receipts affected",
                  evidence="keys present: " + ", ".join(sorted(keys)))


def _cutoff_matches_forcefield(ctx: Context) -> Result:
    """"The choice of how to apply a cutoff, such as through direct truncation,
    shifting of the potential energy function, or through the use of switching
    functions, should be maintained if identical matches to prior work computing
    the properties of interest are desired." (§4.1)"""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    cut = _cutoff_nm(src)
    has_switch = "switchDistance" in src or "switch_width" in src
    if cut is None:
        return Result(UNTESTABLE, "could not parse nonbondedCutoff from md_arms.py",
                      evidence="regex nonbondedCutoff=<x>*unit.nanometer")
    if has_switch:
        return Result(PASS, "a switching distance is set explicitly, so the LJ "
                            "truncation can be matched to the force field spec")
    return Result(FAIL,
                  f"createSystem sets nonbondedCutoff={cut} nm and no "
                  "switchDistance, so OpenMM truncates Lennard-Jones abruptly. "
                  "openff_unconstrained-2.2.1.offxml declares "
                  'cutoff="9.0 * angstrom" switch_width="1.0 * angstrom" — the '
                  "cutoff matches, the switching function does not, so the ligand "
                  "is not being simulated with the nonbonded treatment Sage was "
                  "fitted under",
                  evidence="md_arms.py `_make_system`; offxml <vdW ... "
                           'switch_width="1.0 * angstrom ** 1">')


def _forcefield_reproduces_reference(ctx: Context) -> Result:
    """"it is good practice to ensure that you properly implemented the force
    field by replicating energies, forces, or other observables from prior
    publications" (checklist box 2)

    The distinction this check turns on: an *internal* known-answer control
    (feed the tool an identity and require zero) proves the tool self-consistent.
    It cannot detect a force field that is applied consistently and wrongly.
    Only an external number can do that.
    """
    external = re.compile(
        r"(replicat|reproduc)\w*\s+(a\s+|the\s+)?(published|literature|reference)|"
        r"published\s+(energ|force|dipole)|literature\s+(energ|force)", re.I)
    internal = re.compile(r"known[- _]answer|identity diff", re.I)
    ext, inte = [], []
    if not _repo_python(ctx):
        return Result(UNTESTABLE, "no python sources to inspect",
                      evidence="*.py, cluster/*.py, albench/*.py")
    for p in _repo_python(ctx):
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        if external.search(t):
            ext.append(p.name)
        if internal.search(t):
            inte.append(p.name)
    if ext:
        return Result(PASS,
                      "a control replicates an energy, force or observable from "
                      "prior work with our own force field",
                      evidence=", ".join(sorted(set(ext))[:6]))
    if inte:
        return Result(FAIL,
                      f"{len(set(inte))} scripts carry known-answer controls, but "
                      "every one of them validates a *metric* (hits@N, a rank "
                      "correlation, a parameter diff) rather than the force field. "
                      "Nothing in the repo evaluates an energy or force with our "
                      "amber14 + Sage system and compares it to a published value. "
                      "FINDINGS §2.1 is what that gap costs: an offset-corrected "
                      "1.89 kcal/mol quoted against raw literature values, where "
                      "the raw number was 8.47",
                      evidence="internal-only controls: "
                               + ", ".join(sorted(set(inte))[:6]))
    return Result(FAIL, "no known-answer or reference-value control found in any "
                        "top-level or cluster script",
                  evidence=f"searched {len(_repo_python(ctx))} python files")


# --------------------------------------------------------------------------
# DETERMINE HANDLING OF CUTOFFS  (checklist box 3) / §3.4, §4.2, §4.7
# --------------------------------------------------------------------------

def _lattice_sum(ctx: Context) -> Result:
    """"electrostatics are long-range enough that either the cutoff needs to be
    larger than the system size (for finite systems) or periodicity is needed
    along with full treatment of long-range electrostatics" (checklist box 3);
    "severe errors often result from neglecting electrostatic interactions beyond
    some cutoff distance" (§3.4)"""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    if re.search(r"nonbondedMethod\s*=\s*(PME|LJPME|Ewald)", src):
        return Result(PASS, "the periodic MD system uses a lattice sum (PME), "
                            "not a truncated Coulomb potential",
                      evidence="md_arms.py: createSystem(nonbondedMethod=PME, ...)")
    bad = re.search(r"nonbondedMethod\s*=\s*(CutoffPeriodic|CutoffNonPeriodic|NoCutoff)",
                    src)
    return Result(FAIL,
                  f"electrostatics truncated ({bad.group(1)}) in a condensed-phase "
                  "periodic system" if bad else
                  "no nonbondedMethod found in md_arms.py — cannot show that "
                  "long-range electrostatics are treated at all",
                  evidence=bad.group(0) if bad else "regex nonbondedMethod=<x>")


def _minimum_image(ctx: Context) -> Result:
    """"a cut-off of many non-bonded interactions should be chosen that is less
    than half the length of the simulation box in any dimension" (§4.2)"""
    src = ctx.read(MD_RUNNER)
    cut = _cutoff_nm(src) if src else None
    if cut is None:
        return Result(UNTESTABLE, "cannot read the cutoff from md_arms.py",
                      evidence=MD_RUNNER)
    boxes = []
    for p in ctx.glob(SYSTEMS):
        try:
            cell = _cryst1(p.read_text(errors="replace"))
        except OSError:
            continue
        if cell:
            boxes.append((p.parent.name, min(cell) / 10.0))     # A -> nm
    if not boxes:
        return Result(UNTESTABLE,
                      "no built system carries a CRYST1 record, so the box is "
                      "unknown and the minimum-image condition cannot be checked",
                      evidence=SYSTEMS)
    worst_name, worst = min(boxes, key=lambda b: b[1])
    bad = [n for n, L in boxes if L / 2.0 <= cut]
    if bad:
        return Result(FAIL,
                      f"{len(bad)}/{len(boxes)} systems have a box shorter than "
                      f"2x the {cut} nm cutoff, so atoms interact with their own "
                      f"periodic images (worst {worst_name}, {worst:.2f} nm)",
                      evidence=", ".join(bad[:6]))
    return Result(PASS,
                  f"{len(boxes)} systems checked; smallest box {worst:.2f} nm "
                  f"({worst_name}), half-box {worst/2:.2f} nm vs {cut} nm cutoff",
                  evidence=f"CRYST1 across {SYSTEMS}")


def _lj_cutoff_range(ctx: Context) -> Result:
    """"Nonpolar interactions can often be safely treated with cutoffs of 1-1.5 nm
    as long as the system size is at least twice that" (checklist box 3)"""
    src = ctx.read(MD_RUNNER)
    cut = _cutoff_nm(src) if src else None
    if cut is None:
        return Result(UNTESTABLE, "cannot read the cutoff from md_arms.py",
                      evidence=MD_RUNNER)
    if 1.0 <= cut <= 1.5:
        return Result(PASS, f"LJ cutoff {cut} nm is inside the paper's 1-1.5 nm band")
    if cut > 1.5:
        return Result(PASS, f"LJ cutoff {cut} nm exceeds the paper's band — "
                            "conservative, costs speed, no artefact")
    return Result(FAIL,
                  f"LJ cutoff {cut} nm is below the 1-1.5 nm band the paper calls "
                  "safe. It is defensible on §4.1 grounds (Sage and amber14 were "
                  "both parameterised at 9 A) but the two rules conflict here and "
                  "we have never measured which way the error goes",
                  evidence="md_arms.py: nonbondedCutoff=0.9 * unit.nanometer")


def _score_artefact_records_its_cutoff(ctx: Context) -> Result:
    """Finite-system arm of checklist box 3, plus §4.1's "make note of exactly
    what you used". The MLP interaction energy is computed on a *truncated*
    pocket: everything beyond the radius contributes zero, which is a hard
    electrostatic cutoff by construction. The artefact must at minimum say what
    radius it used."""
    head = ctx.read("results/score_comparison.csv").split("\n", 1)[0]
    if not head:
        return Result(UNTESTABLE, "results/score_comparison.csv absent",
                      evidence="results/score_comparison.csv")
    cols = [c.strip() for c in head.split(",")]
    if any(re.search(r"radius|cutoff|n_pocket", c, re.I) for c in cols):
        return Result(PASS, "the headline scoring artefact records the pocket "
                            "radius it was computed at",
                      evidence=head)
    src_cols = [c.strip() for c in
                ctx.read("results/interaction_energies.csv").split("\n", 1)[0].split(",")]
    return Result(FAIL,
                  "results/score_comparison.csv — the file every hits@25 number is "
                  "computed from — carries no column naming the truncation radius, "
                  "so the electrostatic cutoff of our headline number is not "
                  "recoverable from the artefact. FINDINGS §3.16 records the "
                  "consequence: the 493 headline is a 10 A score labelled '@6 A', "
                  "and the radius had to be back-inferred from n_pocket_atoms in a "
                  "different file",
                  evidence=f"score_comparison.csv: {', '.join(cols)} | "
                           f"interaction_energies.csv: {', '.join(src_cols)}")


# --------------------------------------------------------------------------
# SETTINGS FOR THE DESIRED ENSEMBLE  (checklist box 4) / §4.4, §4.5
# --------------------------------------------------------------------------

def _thermostat_samples_an_ensemble(ctx: Context) -> Result:
    """"Pick a thermostat that gives the correct distribution of temperatures,
    not just the correct average temperature" (checklist box 4). Berendsen and
    simple velocity rescaling sample neither the canonical nor the isokinetic
    distribution: "Its usage can lead to simulation artifacts, so it is not
    recommended" (§4.4.3, Table 1)."""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    banned = [n for n in ("AndersenThermostat", "Berendsen") if n in src]
    good = [n for n in ("LangevinMiddleIntegrator", "LangevinIntegrator",
                        "NoseHoover", "NoseHooverIntegrator") if n in src]
    if "Berendsen" in src:
        return Result(FAIL, "Berendsen thermostat in use; Table 1 marks its "
                            "ensemble 'Undefined' and both structural and "
                            "dynamical properties incorrect",
                      evidence="md_arms.py contains 'Berendsen'")
    if not good:
        return Result(FAIL, "no canonical thermostat found — the run does not "
                            "demonstrably sample any defined ensemble",
                      evidence="searched md_arms.py for Langevin/Nose-Hoover")
    return Result(PASS,
                  f"{good[0]} — Table 1: canonical ensemble, correct structural "
                  "properties. Stochastic and local, so it does not conserve "
                  "momentum and gives incorrect dynamical properties (see "
                  "found-14)",
                  evidence=f"md_arms.py: {good[0]}"
                           + (f"; also present: {banned}" if banned else ""))


def _barostat_has_a_thermostat(ctx: Context) -> Result:
    """"Barostat algorithms control pressure alone, not temperature, so if the
    target ensemble is isothermal-isobaric, they must be applied with a
    thermostat. If a barostat is applied without a thermostat, only ... NPH"
    (§4.5.2)"""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    baro = re.search(r"(MonteCarloBarostat|MonteCarloAnisotropicBarostat|"
                     r"MonteCarloMembraneBarostat)", src)
    if not baro:
        return Result(UNTESTABLE,
                      "no barostat found in md_arms.py — this check exists to "
                      "catch NPH-by-accident and has nothing to inspect",
                      evidence=MD_RUNNER)
    thermo = re.search(r"(LangevinMiddleIntegrator|LangevinIntegrator|"
                       r"NoseHoover|AndersenThermostat)", src)
    if thermo:
        return Result(PASS,
                      f"{baro.group(1)} paired with {thermo.group(1)}, so the "
                      "target ensemble is NPT and not NPH",
                      evidence=f"md_arms.py: {baro.group(1)} + {thermo.group(1)}")
    return Result(FAIL, f"{baro.group(1)} applied with no thermostat — this "
                        "samples NPH, not NPT",
                  evidence=f"md_arms.py: {baro.group(1)}")


def _no_dynamical_observables(ctx: Context) -> Result:
    """"it is preferable that a thermostat not be used if it is desired to
    calculate dynamical properties" (§4.4.2); Table 1 marks Langevin's dynamical
    properties incorrect; MC barostats mean "long-time dynamics are not
    accurately reproduced in any sense" (§4.5.3). So no read-out may be a
    dynamical quantity."""
    src = ctx.read(READOUTS)
    if not src:
        return Result(UNTESTABLE, f"{READOUTS} absent", evidence=READOUTS)
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    bad = sorted({m.group(0).lower() for m in re.finditer(
        r"diffusi\w*|residence[ _]time|lifetime|survival[ _]probab\w*|"
        r"relaxation[ _]time|rate[ _]constant|velocity[ _]autocorrel\w*", body)})
    if bad:
        return Result(FAIL,
                      "read-outs include dynamical quantities, which a Langevin "
                      "thermostat plus a Monte Carlo barostat cannot deliver: "
                      + ", ".join(bad),
                      evidence=READOUTS)
    return Result(PASS,
                  "every channel is configurational (RMSD to start, RMSF, Rg, "
                  "fraction-of-frames contact persistence), which the canonical "
                  "ensemble delivers correctly under any of the Table 1 "
                  "thermostats",
                  evidence="md_readouts.py: no diffusion/residence/lifetime/rate "
                           "channel")


# --------------------------------------------------------------------------
# TIMESTEP  (checklist box 5) / §4.6.4
# --------------------------------------------------------------------------

def _timestep_validated_in_nve(ctx: Context) -> Result:
    """"Test this choice by running a simulation in the microcanonical ensemble,
    and ensure that energy is conserved." (checklist box 5; §4.6.4 "After
    choosing a timestep, a test simulation should be run in the microcanonical
    ensemble")"""
    rec = _receipts(ctx)
    dts = sorted({r.get("timestep_fs") for _, r in rec if r.get("timestep_fs")})
    if not dts:
        return Result(UNTESTABLE,
                      "no run receipt names a timestep, so there is no choice to "
                      "validate; testable once md_arms.py has been run",
                      evidence=RECEIPTS)
    hits = []
    for p in _repo_python(ctx) + ctx.glob("*.sh") + ctx.glob("cluster/*.sh"):
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        if re.search(r"microcanonical|\bNVE\b", t) and re.search(
                r"energy[ _]?(drift|conserv)", t, re.I):
            hits.append(p.name)
    if hits:
        return Result(PASS, "an NVE energy-conservation test exists for the "
                            "production timestep",
                      evidence=", ".join(sorted(set(hits))))
    return Result(FAIL,
                  f"no NVE energy-conservation test anywhere in the repo, yet "
                  f"production runs at {dts or ['?']} fs — the largest timestep in "
                  "the paper's own worked examples is 2 fs, and ours is only "
                  "defensible via HBonds constraints plus 3.0 amu hydrogen-mass "
                  "repartitioning, neither of which has been validated here. "
                  "Testable the moment one run writes a thermostat-free, "
                  "barostat-free state.csv and we quote the drift in kJ/mol/ns/atom",
                  evidence=f"searched {len(_repo_python(ctx))} py + shell scripts "
                           "for microcanonical|NVE with an energy-drift check")


# --------------------------------------------------------------------------
# RUN PROTOCOL  (checklist box 6) / §4.3.3-4.3.5
# --------------------------------------------------------------------------

def _production_follows_equilibration(ctx: Context) -> Result:
    """"Production must always be preceded by equilibration appropriate for the
    target production ensemble, and production data should never be collected
    immediately after a change in conditions (such as rescaling a box size,
    energy minimizing, or suddenly changing the temperature or pressure) except
    in very specific applications where this is the goal." (§4.3.5)"""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    rel = re.search(r"setParameter\(\s*[\"']k[\"']\s*,\s*0(\.0)?\s*\)", src)
    if not rel:
        return Result(UNTESTABLE,
                      "no restraint release found in md_arms.py; this check "
                      "exists to catch data collection that begins in the same "
                      "breath as a change in conditions and has nothing to inspect",
                      evidence=MD_RUNNER)
    tail = src[rel.end():]
    rep = re.search(r"(DCDReporter|StateDataReporter)\(", tail)
    if not rep:
        return Result(UNTESTABLE, "restraint released but no reporter attached "
                                  "afterwards — protocol unrecognised",
                      evidence=MD_RUNNER)
    between = tail[:rep.start()]
    if re.search(r"\.step\(", between):
        return Result(PASS, "a relaxation step separates the restraint release "
                            "from the start of data collection",
                      evidence="md_arms.py: sim.step(...) between "
                               "setParameter('k', 0.0) and the reporters")
    line = src[:rel.start()].count("\n") + 1
    return Result(FAIL,
                  "the backbone positional restraints (k = 100 kJ/mol/nm^2 on ~556 "
                  "atoms) are released and production data collection begins with "
                  "no propagation in between, so the first frames are post-release "
                  "relaxation recorded as production. The paper names exactly this "
                  "pattern of change-then-collect",
                  evidence=f"md_arms.py:{line} setParameter('k', 0.0) is followed "
                           f"directly by sim.reporters.append(...) then "
                           f"sim.step(prod_steps)")


def _equilibration_is_recorded(ctx: Context) -> Result:
    """"test that your equilibration protocol actually allows you to reach
    equilibrium in the target ensemble" (checklist box 6); "This phase of
    equilibration can be monitored by assessing the temperature and pressure of
    the system, as well as the kinetic and potential energy, to ensure these
    reach a steady state on average." (§4.3.4)"""
    rec = {tag: r for tag, r in _receipts(ctx)}
    if not rec:
        return Result(UNTESTABLE, "no run receipts under results/md_pilot/",
                      evidence=RECEIPTS)
    unmonitored, checked = [], 0
    for p in ctx.glob("results/md_pilot/*/*/state.csv"):
        s = _state_series(p)
        r = rec.get(p.parent.name)
        if not s or r is None:
            continue
        checked += 1
        t0, eq = s[0][0], float(r.get("equil_ps") or 0.0)
        if t0 > eq:
            unmonitored.append((p.parent.name, eq, t0))
    if not checked:
        return Result(UNTESTABLE, "no readable state.csv beside a receipt",
                      evidence="results/md_pilot/*/*/state.csv")
    if not unmonitored:
        return Result(PASS, f"{checked} runs record energies/temperature from "
                            "t=0, so equilibration can be shown rather than assumed")
    tag, eq, t0 = unmonitored[0]
    return Result(FAIL,
                  f"{len(unmonitored)}/{checked} runs log nothing during "
                  f"equilibration — reporters are attached only after "
                  f"sim.step(eq_steps), so the first sample is at t = {t0:.0f} ps "
                  f"for {eq:.0f} ps of equilibration ({tag}). We assert the system "
                  "equilibrated; we cannot show it",
                  evidence="; ".join(f"{n}: equil {e:.0f} ps, first sample "
                                     f"{s:.0f} ps" for n, e, s in unmonitored[:5]))


def _no_systematic_drift_in_production(ctx: Context) -> Result:
    """"if any observed properties still exhibit a systematic trend with respect
    to simulation time (e.g. Figure 4) this should be taken as a sign that
    equilibration is not yet complete" (§4.3.4)"""
    rec = {tag: r for tag, r in _receipts(ctx)}
    worst, checked = [], 0
    for p in ctx.glob("results/md_pilot/*/*/state.csv"):
        s = _state_series(p)
        if not s:
            continue
        t, cols = s
        checked += 1
        for name in ("Potential Energy (kJ/mole)", "Density (g/mL)"):
            if name in cols:
                ratio = _trend_over_sd(t, cols[name])
                if ratio > 1.0:
                    worst.append((p.parent.name, name.split(" (")[0], ratio))
    if not checked:
        return Result(UNTESTABLE, "no readable production state.csv",
                      evidence="results/md_pilot/*/*/state.csv")
    if not worst:
        return Result(PASS,
                      f"{checked} runs: no production observable moves by more "
                      "than its own SD across the window",
                      evidence="threshold |OLS trend across window| / SD > 1.0")
    worst.sort(key=lambda w: -w[2])
    runs = sorted({w[0] for w in worst})
    eq = {n: (rec.get(n) or {}).get("equil_ps") for n in runs}
    clean_eq = sorted({(rec.get(t2) or {}).get("equil_ps")
                       for t2 in rec if t2 not in runs} - {None})
    return Result(FAIL,
                  f"{len(runs)}/{checked} runs are still drifting in production: "
                  + ", ".join(f"{n} {c} at {r:.2f}x its own SD"
                              for n, c, r in worst[:3])
                  + f". Equilibration for the drifting runs was "
                    f"{sorted(set(v for v in eq.values() if v))} ps against "
                    f"{clean_eq} ps for the rest — the paper's Fig. 4 bottom "
                    "panel, and the drifting pair is the determinism probe whose "
                    "agreement is quoted as RUNS.md gate 4",
                  evidence="threshold |OLS trend across window| / SD > 1.0; "
                           + "; ".join(f"{n}/{c}={r:.2f}" for n, c, r in worst))


def _replicates_differ_in_configuration(ctx: Context) -> Result:
    """"even very small differences in initial configuration ... lead to
    exponential divergence ... An even better way to generate independent
    realizations is to begin with different starting configurations, such as
    different conformations of the molecule(s) being simulated, as this leads to
    behavior which is immediately different." (§4.3.3)"""
    src = ctx.read(MD_RUNNER)
    if not src:
        return Result(UNTESTABLE, f"{MD_RUNNER} absent", evidence=MD_RUNNER)
    m = re.search(r"for r in range\(args\.replicates\):(.*?)\n    total_ns",
                  src, re.S)
    if not m:
        return Result(UNTESTABLE, "replicate loop not found in md_arms.py",
                      evidence=MD_RUNNER)
    loop = m.group(1)
    reseeds_velocities = "setVelocitiesToTemperature" in loop
    same_positions = re.search(r"setPositions\(\s*minimised\s*\)", loop) is not None
    varies_config = bool(re.search(r"conformer|starting[ _]structure|"
                                   r"different[ _]pose|frame\[", loop))
    if varies_config:
        return Result(PASS, "replicates start from different configurations",
                      evidence="md_arms.py replicate loop")
    if same_positions and reseeds_velocities:
        return Result(FAIL,
                      "every replicate restores the same minimised coordinates and "
                      "differs only in its velocity seed. That is the weaker of the "
                      "two options the paper offers, and it matters here because "
                      "the within-arm replicate spread (RUNS.md gate 5: SD of mean "
                      "ligand RMSD = 0.075 A) is our noise floor — a floor built "
                      "from one starting structure understates the spread of "
                      "genuinely independent realisations",
                      evidence="md_arms.py: sim.context.setPositions(minimised) + "
                               "setVelocitiesToTemperature(T, seed_r) inside the loop")
    return Result(UNTESTABLE, "replicate loop does not match either pattern",
                  evidence="md_arms.py replicate loop")


def _save_interval_justified(ctx: Context) -> Result:
    """"storing data more frequently than intervals of the autocorrelation time
    results in storage of essentially redundant data ... storing data more
    frequently than intervals of the autocorrelation time is generally
    unnecessary" (§4.3.5)"""
    rec = _receipts(ctx)
    if not rec:
        return Result(UNTESTABLE, "no run receipts under results/md_pilot/",
                      evidence=RECEIPTS)
    saves = sorted({r.get("save_ps") for _, r in rec if r.get("save_ps")})
    hits = []
    for p in _repo_python(ctx):
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        if re.search(r"autocorrelat|statistical_inefficiency|detectEquilibration|"
                     r"subsample_correlated", t, re.I):
            hits.append(p.name)
    if hits:
        return Result(PASS, f"save interval {saves} ps set against a measured "
                            "correlation time",
                      evidence=", ".join(sorted(set(hits))))
    return Result(FAIL,
                  f"frames are stored every {saves} ps with no correlation time "
                  "measured anywhere in the repo, so we cannot say whether 100 "
                  "frames per 200 ps are 100 samples or 3. This is the same "
                  "quantity that would tell us how many effective samples the "
                  "within-arm noise floor rests on. pymbar.timeseries is already "
                  "installed in the grand and fegrow envs",
                  evidence=f"searched {len(_repo_python(ctx))} python files for "
                           "autocorrelation / statistical inefficiency")


CHECKS = [
    # --- must: stated by the paper as requirements ------------------------
    Check(id="found-01", severity="must", section="checklist box 3; §3.4, §4.7",
          question="Does the condensed-phase MD use lattice-sum electrostatics "
                   "rather than a truncated Coulomb potential?",
          measures="md_arms.py createSystem call", fn=_lattice_sum),
    Check(id="found-02", severity="must", section="§4.2",
          question="Is the nonbonded cutoff below half the shortest box vector "
                   "of every built system?",
          measures="CRYST1 of results/md_pilot/*/system.pdb vs md_arms.py cutoff",
          fn=_minimum_image),
    Check(id="found-03", severity="must", section="§4.5.2",
          question="Is the barostat applied together with a thermostat, so the "
                   "target ensemble is NPT and not NPH?",
          measures="md_arms.py force/integrator construction",
          fn=_barostat_has_a_thermostat),
    Check(id="found-04", severity="must", section="§4.3.5",
          question="Is production data collection separated from the last change "
                   "in conditions?",
          measures="md_arms.py restraint release vs reporter attachment",
          fn=_production_follows_equilibration),
    Check(id="found-05", severity="must", section="checklist box 5; §4.6.4",
          question="Has the production timestep been validated by an NVE "
                   "energy-conservation test?",
          measures="repo-wide search for a microcanonical drift test",
          fn=_timestep_validated_in_nve),

    # --- should ------------------------------------------------------------
    Check(id="found-06", severity="should", section="checklist box 6; §4.3.4",
          question="Is the equilibration phase instrumented, so that reaching "
                   "equilibrium can be shown rather than assumed?",
          measures="first timestamp in each state.csv vs equil_ps in its receipt",
          fn=_equilibration_is_recorded),
    Check(id="found-07", severity="should", section="§4.3.4 (Fig. 4)",
          question="Are production observables free of a systematic trend with "
                   "simulation time?",
          measures="OLS trend / SD of potential energy and density in every state.csv",
          fn=_no_systematic_drift_in_production),
    Check(id="found-08", severity="should", section="§4.4.3-4.4.4, Table 1",
          question="Does the thermostat sample a defined ensemble — not "
                   "Berendsen or simple velocity rescaling?",
          measures="md_arms.py integrator/thermostat choice",
          fn=_thermostat_samples_an_ensemble),
    Check(id="found-09", severity="should", section="§4.1",
          question="Does each run receipt record the complete force field — "
                   "protein, water, ligand and engine version?",
          measures="keys of results/md_pilot/*/*/receipt.json",
          fn=_forcefield_recorded),
    Check(id="found-10", severity="should", section="checklist box 1",
          question="Do the run artefacts name the ensemble and both halves of "
                   "the state point (temperature and pressure)?",
          measures="keys of results/md_pilot/*/*/receipt.json",
          fn=_state_point_recorded),
    Check(id="found-11", severity="should", section="§4.1",
          question="Does the nonbonded cutoff treatment match the one the force "
                   "field was parameterised under?",
          measures="md_arms.py switchDistance vs the Sage offxml vdW section",
          fn=_cutoff_matches_forcefield),
    Check(id="found-12", severity="should", section="checklist box 3; §4.1",
          question="Does the artefact behind the objective metric record the "
                   "electrostatic cutoff it was computed at?",
          measures="header of results/score_comparison.csv",
          fn=_score_artefact_records_its_cutoff),
    Check(id="found-13", severity="should", section="checklist box 1; §5",
          question="Was the cost of each dynamics stage counted before "
                   "committing to it?",
          measures="results/cost_curve.csv rows and feasibility verdicts",
          fn=_cost_counted),
    Check(id="found-14", severity="should", section="§4.4.2, Table 1; §4.5.3",
          question="Are all reported read-outs configurational, given that a "
                   "Langevin thermostat and an MC barostat destroy dynamics?",
          measures="md_readouts.py channel names",
          fn=_no_dynamical_observables),
    Check(id="found-15", severity="should", section="checklist box 2",
          question="Has the force field implementation been validated against an "
                   "energy, force or observable from a prior publication?",
          measures="repo-wide search for known-answer / reference-value controls",
          fn=_forcefield_reproduces_reference),

    # --- consider ----------------------------------------------------------
    Check(id="found-16", severity="consider", section="§4.3.3",
          question="Do independent replicates start from different "
                   "configurations, not merely different velocities?",
          measures="md_arms.py replicate loop", fn=_replicates_differ_in_configuration),
    Check(id="found-17", severity="consider", section="§4.3.5",
          question="Is the trajectory save interval set against a measured "
                   "autocorrelation time?",
          measures="save_ps in receipts + repo-wide search for a correlation time",
          fn=_save_interval_justified),
    Check(id="found-18", severity="consider", section="checklist box 3",
          question="Is the Lennard-Jones cutoff inside the paper's 1-1.5 nm band?",
          measures="md_arms.py nonbondedCutoff", fn=_lj_cutoff_range),
]
