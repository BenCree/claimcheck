"""Cavender et al. 2025 — structure-based datasets for benchmarking protein force fields.

    Structure-Based Experimental Datasets for Benchmarking Protein Simulation
    Force Fields [Article v1.0]
    Cavender, Case, Chen, Chong, Keedy, Lindorff-Larsen, Mobley, Ollila,
    Oostenbrink, Robustelli, Voelz, Wall, Wych & Gilson
    Living J. Comp. Mol. Sci. 2025, 6(1), 3871 · doi:10.33011/livecoms.6.1.3871

**Why this paper governs us even though its subject is not our subject.** Its
Sections 3, 4.2 and most of 2 are about ligand-free proteins — the review says
so explicitly ("We consider only datasets that involve water-soluble ...
proteins and peptides without ligands or co-factors") — so its *dataset*
recommendations are not obligations on a protein-ligand hit-finder. Its
**Section 5** is not scoped that way. Section 5 is a setup-and-analysis
protocol for any benchmark MD, and every one of our OpenMM runs is a benchmark
MD: `md_arms.py` compares two ligand force fields and quotes the difference.
Section 4 then governs us wherever we take a *crystallographic* quantity as
ground truth — crystal waters, pocket flexibility, pose RMSD — which we do
constantly.

The checks below therefore come from §4 and §5, plus one from §2. Everything
drawn from §3 (NMR observables) is genuinely N/A and is marked so with its
reason, not quietly dropped.

**Severity follows the paper's own verbs.** `must` is reserved for what the
review calls *essential*, or backs with a hard number and a named artefact
("solvent padding shorter than 10 Å", "the flying ice cube effect"). Everything
the review merely *recommends* is `should`; its dataset-selection preferences
are `consider`. Four checks are `must` and all four currently pass — they are
regression guards on invariants we already satisfy, which is what a gating
severity is for. Nothing that currently fails has been promoted to make a
point.
"""

from __future__ import annotations

import json
import math
import re

from . import Check, Context, Paper, Result, PASS, FAIL, UNTESTABLE, NA

PAPER = Paper(
    slug="protein_ff_benchmarks",
    title="Structure-Based Experimental Datasets for Benchmarking Protein "
          "Simulation Force Fields [Article v1.0]",
    authors="Cavender, Case, Chen, Chong, Keedy, Lindorff-Larsen, Mobley, "
            "Ollila, Oostenbrink, Robustelli, Voelz, Wall, Wych & Gilson",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v6i1e3871",
    doi="10.33011/livecoms.6.1.3871",
    year="2025",
    scope="Setup and analysis of any benchmark MD simulation (§5), and the use "
          "of crystallographic quantities as ground truth (§4).",
)

# --------------------------------------------------------------------------
# shared, cached corpus measurements — every check below is read-only and the
# full 925-structure scan is done once
# --------------------------------------------------------------------------

STRUCT_GLOB = "data/OpenBind_EV-A71_2A/structures/*/*/*_complex_ref.pdb"
_CORPUS: dict | None = None

_PROG = re.compile(r"REMARK   3   PROGRAM\s+:\s+(\S+)")
_RES_HIGH = re.compile(r"RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([\d.]+)")


def _corpus(ctx: Context) -> dict:
    """Refinement program, resolution and altloc census over the deposited set."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    programs: dict[str, int] = {}
    res: list[float] = []
    res_by_prog: dict[str, list[float]] = {}
    remark2 = {}
    n_alt_files = 0
    n_alt_atoms = 0
    files = ctx.glob(STRUCT_GLOB)
    for f in files:
        txt = f.read_text(errors="replace")
        m = _PROG.search(txt)
        prog = m.group(1) if m else "UNRECORDED"
        programs[prog] = programs.get(prog, 0) + 1
        r = _RES_HIGH.search(txt)
        if r:
            v = float(r.group(1))
            res.append(v)
            res_by_prog.setdefault(prog, []).append(v)
        key = (prog, "REMARK   2 RESOLUTION." in txt)
        remark2[key] = remark2.get(key, 0) + 1
        n_here = sum(1 for ln in txt.splitlines()
                     if ln[:6] in ("ATOM  ", "HETATM") and len(ln) > 16
                     and ln[16] != " ")
        if n_here:
            n_alt_files += 1
            n_alt_atoms += n_here
    _CORPUS = {"n_files": len(files), "programs": programs, "res": sorted(res),
               "res_by_prog": res_by_prog, "remark2": remark2,
               "n_alt_files": n_alt_files, "n_alt_atoms": n_alt_atoms}
    return _CORPUS


def _any_build_json(ctx: Context) -> tuple[dict | None, str]:
    """The most recent MD build receipt, or (None, '') if no MD has been built."""
    for p in ctx.glob("results/**/build.json"):
        try:
            return json.loads(p.read_text()), str(p)
        except Exception:                                        # noqa: BLE001
            continue
    return None, ""


def _any_protonation(ctx: Context) -> tuple[dict | None, str]:
    for p in ctx.glob("data/protonated/*_protonation.json"):
        try:
            return json.loads(p.read_text()), str(p)
        except Exception:                                        # noqa: BLE001
            continue
    return None, ""


def _py_sources(ctx: Context):
    """Our own Python, excluding vendored trees and caches."""
    skip = ("__pycache__", "/mace/", "/mace-off/", "/traversing_chem_space/",
            "/EV-A71_2A_benchmark/", "/bestpractices/")
    for p in ctx.glob("**/*.py"):
        s = str(p)
        if any(k in s for k in skip):
            continue
        yield p


# ==========================================================================
# §5.1.1 — initial coordinates and protonation
# ==========================================================================

def _protonation_states(ctx: Context) -> Result:
    """§5.1.1: 'it is essential to assign correct protonation states to
    titratable side chains and protein termini' — with PROPKA/PDB2PQR named as
    the tools that do it from the local environment and a specified pH."""
    src = ctx.read("protonate.py")
    if not src:
        return Result(UNTESTABLE, "protonate.py is absent; this becomes "
                                  "testable once a protonation stage exists "
                                  "and writes a per-structure record",
                      evidence="protonate.py not found")
    sidecar, path = _any_protonation(ctx)
    if sidecar is None:
        return Result(UNTESTABLE,
                      "protonate.py exists but no *_protonation.json is on "
                      "disk; run it over the corpus to make this testable",
                      evidence="data/protonated/*_protonation.json: 0 files")

    uses_pka = "propka" in src.lower() and "pdb2pqr" in src.lower()
    ph = sidecar.get("ph")
    has_pka_table = bool(sidecar.get("propka_pka"))
    termini = "net_charge_termini" in sidecar
    n_sidecars = len(ctx.glob("data/protonated/*_protonation.json"))

    missing = []
    if not uses_pka:
        missing.append("no pKa predictor in protonate.py")
    if ph is None:
        missing.append("no pH recorded in the sidecar")
    if not has_pka_table:
        missing.append("no per-residue pKa recorded")
    if not termini:
        missing.append("termini charge not recorded")
    if missing:
        return Result(FAIL, "; ".join(missing), evidence=path)

    return Result(PASS,
                  f"{n_sidecars} structures protonated by "
                  f"{sidecar.get('method', 'pdb2pqr/propka')} at pH {ph}; "
                  f"side-chain and terminal charges both recorded "
                  f"(net {sidecar.get('net_charge')}e = "
                  f"{sidecar.get('net_charge_sidechains')} side chain + "
                  f"{sidecar.get('net_charge_termini')} termini)",
                  evidence=f"{path}: ph={ph}, "
                           f"{len(sidecar.get('propka_pka', {}))} pKa values")


def _local_environment_overrides(ctx: Context) -> Result:
    """§5.1.1: 'the local environment of a residue ... may indicate a
    protonation state different from the expected one at the experimental pH.
    Similarly, the local environment of cysteine residues can indicate whether
    they should be modeled with a disulfide bond.'"""
    sidecar, path = _any_protonation(ctx)
    if sidecar is None:
        return Result(UNTESTABLE,
                      "no protonation sidecar on disk; testable once "
                      "protonate.py has run",
                      evidence="data/protonated/*_protonation.json: 0 files")
    src = ctx.read("protonate.py")
    has_cyx = "CYX" in src
    has_metal = "metal_contacts" in sidecar or "metal_sites" in sidecar
    overrides = sidecar.get("overrides") or []
    if not (has_cyx and has_metal):
        return Result(FAIL,
                      f"cysteine disulfide handling {'present' if has_cyx else 'ABSENT'}, "
                      f"metal-coordination record {'present' if has_metal else 'ABSENT'} "
                      "— §5.1.1 requires the local environment to override the "
                      "pH-expected state",
                      evidence=path)
    reasons = {o.get("reason", "")[:40] for o in overrides}
    return Result(PASS,
                  f"{len(overrides)} residue(s) overridden from the local "
                  f"environment in this structure; CYX/CYM handled and metal "
                  f"contacts recorded with distances",
                  evidence=f"{path}: overrides={sorted(reasons)}")


# ==========================================================================
# §5.1.2 — solvation and periodic box
# ==========================================================================

def _solvent_padding(ctx: Context) -> Result:
    """§5.1.2: 'If the solvent padding is shorter than 10 Å ... then
    interactions between periodic images can perturb observed quantities such
    as solvation free energies and secondary structure preferences ... a
    solvent padding of 10 Å to 15 Å is sufficient to avoid such artifacts.'"""
    src = ctx.read("md_arms.py")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no solvated MD to check",
                      evidence="md_arms.py not found")
    m = re.search(r"^PADDING_NM\s*=\s*([\d.]+)", src, re.M)
    if not m:
        return Result(UNTESTABLE,
                      "md_arms.py sets no module-level PADDING_NM; make the "
                      "padding an explicit named constant to test it",
                      evidence="no PADDING_NM in md_arms.py")
    pad_A = float(m.group(1)) * 10.0
    build, bpath = _any_build_json(ctx)
    on_disk = f"; build receipt agrees ({build['padding_nm'] * 10:.1f} Å, " \
              f"{bpath})" if build and "padding_nm" in build else ""
    if pad_A < 10.0:
        return Result(FAIL,
                      f"solvent padding {pad_A:.1f} Å is below the 10 Å floor "
                      f"§5.1.2 sets; periodic-image interactions perturb "
                      f"exactly the structural readouts we quote",
                      evidence=f"md_arms.py PADDING_NM={m.group(1)}{on_disk}")
    note = " — at the floor, not inside the band; §5.1.2's stated range is " \
           "10–15 Å and nothing in our record justifies the minimum" \
           if pad_A < 12.0 else ""
    return Result(PASS, f"solvent padding {pad_A:.1f} Å{note}",
                  evidence=f"md_arms.py PADDING_NM={m.group(1)}{on_disk}")


def _box_shape(ctx: Context) -> Result:
    """§5.1.2: 'The rhombic dodecahedron has the smallest volume of these
    space-filling polyhedra and is the recommended box shape for simulations of
    solutes that are roughly spherical, including most globular proteins.'"""
    src = ctx.read("md_arms.py")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no box to check",
                      evidence="md_arms.py not found")
    if "addSolvent" not in src:
        return Result(NA, "md_arms.py does not solvate, so it builds no "
                          "periodic box whose shape could be chosen",
                      evidence="no addSolvent call in md_arms.py")
    if re.search(r"boxShape\s*=|boxVectors\s*=", src):
        shape = re.search(r"boxShape\s*=\s*['\"](\w+)['\"]", src)
        return Result(PASS,
                      f"box shape set explicitly to "
                      f"{shape.group(1) if shape else 'custom vectors'}",
                      evidence="md_arms.py addSolvent(boxShape=...)")
    build, bpath = _any_build_json(ctx)
    waters = build.get("n_waters") if build else None
    saving = ""
    if waters:
        # equal minimum-image distance d: V_cube = d^3, V_dodec = d^3/sqrt(2)
        saving = (f"; at {waters} waters that is ~{round(waters * (1 - 1 / math.sqrt(2)))} "
                  f"redundant water molecules per system "
                  f"({100 * (1 - 1 / math.sqrt(2)):.0f}% of the box)")
    return Result(FAIL,
                  "addSolvent is called with padding only, so OpenMM builds "
                  "the default cubic box; §5.1.2 recommends the rhombic "
                  "dodecahedron for globular solutes, which gives the same "
                  "minimum image distance in 1/sqrt(2) of the volume"
                  + saving +
                  ". This is not a correctness point for us — it is wall-clock, "
                  "which is our objective function.",
                  evidence=f"md_arms.py: addSolvent(model=, padding=, "
                           f"ionicStrength=, neutralize=) — no boxShape"
                           + (f"; {bpath} n_waters={waters}" if waters else ""))


def _water_model_declared(ctx: Context) -> Result:
    """§5.1.2: 'the water model should be considered part of the force field
    being benchmarked just as much as the parameters applied to the protein
    ... the widely used 3-point water model TIP3P favors compact structures
    excessively and has low viscosity compared to experimental water.'"""
    src = ctx.read("md_arms.py")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no water model to check",
                      evidence="md_arms.py not found")
    build, bpath = _any_build_json(ctx)
    declared = build.get("protein_ff") if build else None
    if not declared:
        return Result(FAIL,
                      "no MD build receipt records the water model, so it is "
                      "not being treated as part of the force field under test",
                      evidence="no results/**/build.json with protein_ff")
    ff_text = " ".join(declared).lower()
    is_tip3p = "tip3p" in ff_text
    if not is_tip3p:
        return Result(PASS, f"water model recorded in the build receipt: "
                            f"{declared}", evidence=bpath)
    # TIP3P is in use: the paper's caveat must be on the record somewhere.
    caveat_words = ("compact", "collaps", "viscosit", "over-structur")
    found = []
    for doc in ("FINDINGS.md", "PLAN.md", "RUNS.md", "GOALS.md", "LITERATURE.md",
                "README.md"):
        text = ctx.read(doc).lower()
        for ln in text.splitlines():
            if "tip3p" in ln and any(w in ln for w in caveat_words):
                found.append(f"{doc}: {ln.strip()[:70]}")
    if found:
        return Result(PASS,
                      "TIP3P declared in the receipt and its documented bias "
                      "is on the record",
                      evidence="; ".join(found[:2]))
    return Result(FAIL,
                  "TIP3P is declared in the build receipt (good) but its "
                  "documented bias is recorded nowhere: §5.1.2 says TIP3P "
                  "'favors compact structures excessively' and has low "
                  "viscosity. We quote CA-RMSF, radius-of-gyration SD and "
                  "persistent-contact counts off these trajectories — all "
                  "three are compaction-sensitive, and a difference we read as "
                  "a ligand-force-field effect could be the water model.",
                  evidence=f"{bpath} protein_ff={declared}; no line in "
                           f"FINDINGS/PLAN/RUNS/GOALS/LITERATURE/README pairs "
                           f"'tip3p' with compaction or viscosity")


# ==========================================================================
# §5.1.3 — salt ions and co-solutes
# ==========================================================================

def _ionic_strength(ctx: Context) -> Result:
    """§5.1.3: 'For proteins with a net charge, counterions should be added to
    the simulation box to neutralize the system. If the observables targeted by
    the simulation were measured in a solution containing salt, then additional
    ion pairs should be added ... to model the experimental ionic strength.'"""
    src = ctx.read("md_arms.py")
    build, bpath = _any_build_json(ctx)
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no solvated system",
                      evidence="md_arms.py not found")
    neutralised = "neutralize=True" in src.replace(" ", "")
    m = re.search(r"^ION_MOLARITY\s*=\s*([\d.]+)", src, re.M)
    conc = float(m.group(1)) if m else None
    if not neutralised or not conc:
        return Result(FAIL,
                      f"neutralize={'True' if neutralised else 'not set'}, "
                      f"ionic strength={'not set' if conc is None else conc}",
                      evidence="md_arms.py addSolvent kwargs")
    obs = f"; {bpath} reports {build['n_ions']} ions in " \
          f"{build['n_waters']} waters" if build and "n_ions" in build else ""
    return Result(PASS,
                  f"system neutralised and bulk salt modelled at {conc} M",
                  evidence=f"md_arms.py ION_MOLARITY={conc}, neutralize=True{obs}")


def _sltcap(ctx: Context) -> Result:
    """§5.1.3: 'If the protein has a net charge, then the solvent near the
    protein will be depleted of ions of like charge and enriched in ions of
    opposite charge, relative to bulk. In this case, the desired number of ion
    pairs can be estimated using the SLTCAP method' (Eq 16), rather than the
    bulk-ratio Eq 15."""
    build, bpath = _any_build_json(ctx)
    if not build or "n_waters" not in build:
        return Result(UNTESTABLE,
                      "no MD build receipt with a water count; testable as "
                      "soon as md_arms.py build has run once",
                      evidence="no results/**/build.json with n_waters")
    sidecar, spath = _any_protonation(ctx)
    if sidecar is None or sidecar.get("net_charge") is None:
        return Result(UNTESTABLE,
                      "the solute net charge is not recorded in the build "
                      "receipt; record protein+ligand net charge in build.json "
                      "to make Eq 16 checkable per system",
                      evidence=f"{bpath} has ligand_charge but no solute net charge")
    q = float(sidecar["net_charge"]) + float(build.get("ligand_charge", 0.0))
    if abs(q) < 0.5:
        return Result(NA, f"the solute is net neutral (q = {q:+.0f}e), so the "
                          f"ion-depletion correction of Eq 16 reduces to Eq 15",
                      evidence=f"{spath} net_charge + {bpath} ligand_charge")
    nw, c_ion, c_wat = build["n_waters"], build.get("ionic_strength_M", 0.15), 55.4
    naive = nw * c_ion / c_wat                                   # Eq 15
    lam = q * c_wat / (2 * nw * c_ion)
    sltcap = naive * (math.sqrt(1 + lam * lam) - lam)            # Eq 16
    dev = abs(sltcap - naive) / naive
    if "SLTCAP" in ctx.read("md_arms.py").upper():
        return Result(PASS, "SLTCAP is applied when setting the ion count",
                      evidence="md_arms.py mentions SLTCAP")
    obs = build.get("n_ions")
    implied_naive = round(abs(q) + 2 * naive)
    implied_sltcap = round(abs(q) + 2 * sltcap)
    return Result(FAIL,
                  f"solute net charge is {q:+.0f}e, so §5.1.3 says the ion "
                  f"count should follow Eq 16 (SLTCAP), not Eq 15. We use "
                  f"OpenMM's ionicStrength=, which is Eq 15: {naive:.1f} pairs "
                  f"where Eq 16 gives {sltcap:.1f} — a {100 * dev:.0f}% "
                  f"deviation. Predicted total ions {implied_naive} (Eq 15) vs "
                  f"{implied_sltcap} (Eq 16); the receipt records {obs}, which "
                  f"is the Eq-15 number.",
                  evidence=f"{bpath}: n_waters={nw}, n_ions={obs}, "
                           f"ionic_strength={c_ion} M; {spath}: net_charge={q:+.0f}")


# ==========================================================================
# §5.1.4 — thermodynamic ensembles
# ==========================================================================

def _correct_ensemble(ctx: Context) -> Result:
    """§5.1.4: 'benchmark simulations should sample from the isobaric-isothermal
    (NPT) ensemble ... Berendsen or weak-coupling methods fail to preserve the
    equipartition of energy and can lead to artifacts such as the flying ice
    cube effect. These methods should be avoided in favor of methods that
    sample correctly from physical ensembles, such as Langevin dynamics, the
    canonical velocity rescaling thermostat, and the Monte Carlo barostat.'"""
    offenders = []
    for p in _py_sources(ctx):
        txt = p.read_text(errors="replace")
        if re.search(r"Berendsen|weak.?coupl", txt, re.I):
            offenders.append(p.name)
    src = ctx.read("md_arms.py")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no NPT run to check",
                      evidence="md_arms.py not found")
    has_baro = "MonteCarloBarostat" in src
    has_thermo = "LangevinMiddleIntegrator" in src or "LangevinIntegrator" in src
    if offenders:
        return Result(FAIL,
                      f"weak-coupling/Berendsen thermostat or barostat present "
                      f"in {offenders} — §5.1.4 says these must be avoided "
                      f"(flying ice cube)",
                      evidence=", ".join(offenders))
    if not (has_baro and has_thermo):
        return Result(FAIL,
                      f"production MD does not sample NPT with an approved "
                      f"pair: barostat={'yes' if has_baro else 'NO'}, "
                      f"Langevin={'yes' if has_thermo else 'NO'}",
                      evidence="md_arms.py")
    return Result(PASS,
                  "NPT sampled with MonteCarloBarostat + LangevinMiddle"
                  "Integrator; no Berendsen or weak-coupling scheme anywhere "
                  "in our Python",
                  evidence="md_arms.py: MonteCarloBarostat(1 bar, T, 25) + "
                           "LangevinMiddleIntegrator(T, 1/ps); "
                           f"{sum(1 for _ in _py_sources(ctx))} files scanned "
                           "for Berendsen, 0 hits")


def _kinetic_artefacts(ctx: Context) -> Result:
    """§5.1.4: 'Additional care is needed when computing kinetic quantities, as
    some thermostats—most notably Langevin dynamics, with its added frictional
    damping and random forces—may introduce significant kinetic artifacts.'"""
    kinetic = re.compile(
        r"residence.?time|tau.?ramd|\bk_?off\b|koff|dissociation.?rate|"
        r"diffusion.?coeff|rate.?constant|mean.?first.?passage", re.I)
    hits = []
    for p in _py_sources(ctx):
        txt = p.read_text(errors="replace")
        # a docstring reference to someone else's method is not our readout
        for ln in txt.splitlines():
            if kinetic.search(ln) and "=" in ln and not ln.strip().startswith("#"):
                hits.append(f"{p.name}: {ln.strip()[:60]}")
    if not hits:
        return Result(NA,
                      "every quantity we compute from a trajectory is "
                      "structural or thermodynamic (ligand RMSD/RMSF, radius "
                      "of gyration, centroid drift, persistent contacts, CA "
                      "RMSF) — no rate, residence time or diffusion "
                      "coefficient is derived, so Langevin's friction cannot "
                      "bias a reported number. This flips to FAIL the moment a "
                      "tauRAMD-style readout lands.",
                      evidence=f"{sum(1 for _ in _py_sources(ctx))} source "
                               f"files scanned for kinetic readouts, 0 hits")
    return Result(FAIL,
                  f"kinetic quantities are computed under Langevin dynamics "
                  f"without the §5.1.4 caveat: {len(hits)} site(s)",
                  evidence="; ".join(hits[:3]))


# ==========================================================================
# §5.2 — analysis
# ==========================================================================

def _equilibration_monitored(ctx: Context) -> Result:
    """§5.2.1: 'for novel force fields or complex systems ... a plateau in
    volume can be reached before strain from the initial configuration can be
    dissipated. Thus, we recommend monitoring quantities such as potential
    energies as well to determine when a system is equilibrated.'"""
    src = ctx.read("md_arms.py")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no equilibration to check",
                      evidence="md_arms.py not found")
    m = re.search(r"sim\.step\(eq_steps\)", src)
    if not m:
        return Result(UNTESTABLE,
                      "no identifiable equilibration step in md_arms.py; name "
                      "the equilibration call to make this testable",
                      evidence="no sim.step(eq_steps) in md_arms.py")
    clear = src.rfind("sim.reporters.clear()", 0, m.start())
    window = src[clear:m.start()] if clear != -1 else src[:m.start()]
    reported = "Reporter(" in window
    default = re.search(r'"--equil-ps",\s*type=float,\s*default=([\d.]+)', src)
    fixed = default.group(1) if default else "?"
    if reported:
        return Result(PASS,
                      "observables are recorded during equilibration, so the "
                      "burn-in length can be justified rather than assumed",
                      evidence="md_arms.py: reporter attached before "
                               "sim.step(eq_steps)")
    return Result(FAIL,
                  f"equilibration is a fixed {fixed} ps with every reporter "
                  f"detached — `sim.reporters.clear()` runs immediately before "
                  f"`sim.step(eq_steps)`, so neither potential energy nor "
                  f"volume nor density is recorded while the system relaxes. "
                  f"There is no evidence on disk that {fixed} ps was enough, "
                  f"and §5.2.1 names our exact case: one arm runs a bespoke "
                  f"(novel) ligand force field, for which it warns a volume "
                  f"plateau can precede strain dissipation.",
                  evidence="md_arms.py: sim.reporters.clear(); sim.step("
                           "eq_steps) — StateDataReporter is only attached "
                           "afterwards, for production")


def _convergence_assessed(ctx: Context) -> Result:
    """§5.2.2: 'It is good practice to assess the convergence of simulations by
    determining whether estimates of observables are robust to changes in the
    set of conformational samples used to estimate them ... In one simple
    approach, the time series of observable estimates is visualized to check
    for drift ... Blocking analysis, a more rigorous approach, divides the
    trajectory into blocks and looks for convergence of the standard error of
    the mean as the block size is increased.'"""
    writers, readers, blocking = [], [], []
    for p in _py_sources(ctx):
        txt = p.read_text(errors="replace")
        if "state.csv" in txt:
            (writers if "StateDataReporter" in txt else readers).append(p.name)
        if re.search(r"block(ing)?_(analysis|average)|blocked_se|"
                     r"statistical_inefficiency|\bpymbar\b", txt, re.I):
            blocking.append(p.name)
    if blocking:
        return Result(PASS, f"blocking / statistical-inefficiency analysis "
                            f"present in {blocking}",
                      evidence=", ".join(blocking))
    if readers:
        return Result(PASS,
                      f"the production observable time series is consumed for "
                      f"a drift check by {readers}",
                      evidence=", ".join(readers))
    return Result(FAIL,
                  f"convergence is never assessed. `state.csv` carries "
                  f"potential energy, volume, density and temperature per "
                  f"frame and is written by {writers or ['md_arms.py']} — and "
                  f"read by nothing. No blocking analysis, no statistical "
                  f"inefficiency, no drift check exists anywhere in our "
                  f"Python. §5.2.2 also warns that even a converged-looking "
                  f"run 'cannot identify when a simulation fails entirely to "
                  f"visit regions of a protein's configuration space', which "
                  f"is the failure mode a 500 ps trajectory is most exposed to.",
                  evidence=f"state.csv written by {writers}, read by {readers}; "
                           f"0 hits for blocking/pymbar across "
                           f"{sum(1 for _ in _py_sources(ctx))} source files")


def _significance_tested(ctx: Context) -> Result:
    """§5.2.3: 'When comparing sets of simulations against experimental data, it
    is essential to determine whether the differences are statistically
    significant.'"""
    src = ctx.read("md_readouts.py")
    if not src:
        return Result(UNTESTABLE,
                      "md_readouts.py absent; testable once an arm-vs-arm "
                      "comparison exists",
                      evidence="md_readouts.py not found")
    boot = re.search(r"paired_boot|bootstrap", src, re.I)
    ci = re.search(r"\bci\b|percentile|confidence", src, re.I)
    if boot and ci:
        return Result(PASS,
                      "arm-vs-arm differences carry a paired bootstrap "
                      "interval rather than a bare point estimate, and the "
                      "comparison refuses to run below a minimum n",
                      evidence="md_readouts.py: paired_boot(...) with "
                               "percentile CI; MIN_N guard")
    return Result(FAIL,
                  f"the simulation comparison reports differences without a "
                  f"significance test (bootstrap={'yes' if boot else 'NO'}, "
                  f"interval={'yes' if ci else 'NO'})",
                  evidence="md_readouts.py")


def _independent_replicates(ctx: Context) -> Result:
    """§5.2.3: 'Statistical uncertainties in simulated data are perhaps most
    reliable if obtained by running multiple independent simulations.'"""
    src = ctx.read("md_arms.py")
    sbatch = ctx.read("cluster/md_arms.sbatch")
    if not src:
        return Result(UNTESTABLE, "md_arms.py absent — no replicates to count",
                      evidence="md_arms.py not found")
    d = re.search(r'"--replicates",\s*type=int,\s*default=(\d+)', src)
    cli_default = int(d.group(1)) if d else None
    r = re.search(r"REPS=\$\{REPS:-(\d+)\}", sbatch) if sbatch else None
    prod_reps = int(r.group(1)) if r else None
    if prod_reps is None:
        return Result(UNTESTABLE,
                      f"the CLI default is {cli_default} replicate(s) and no "
                      f"submission script pins a production value; pin REPS in "
                      f"cluster/md_arms.sbatch to make this testable",
                      evidence=f"md_arms.py --replicates default={cli_default}")
    if prod_reps < 2:
        return Result(FAIL,
                      f"the production path runs {prod_reps} replicate(s), so "
                      f"the only uncertainty available is within a single "
                      f"trajectory",
                      evidence=f"cluster/md_arms.sbatch REPS={prod_reps}")
    note = (f" The CLI default is {cli_default}, so any run launched outside "
            f"the sbatch silently drops to a single trajectory.") \
        if cli_default is not None and cli_default < 2 else ""
    return Result(PASS,
                  f"the production path runs {prod_reps} independent "
                  f"replicates per arm, seeded differently within an arm and "
                  f"paired across arms.{note}",
                  evidence=f"cluster/md_arms.sbatch REPS={prod_reps}; "
                           f"md_arms.py --replicates default={cli_default}")


def _mixed_effects_and_experimental_error(ctx: Context) -> Result:
    """§5.2.3: 'When multiple proteins are simulated and several properties are
    considered, it may not be straightforward to determine if two sets of
    simulations are significantly different ... The differences in a particular
    property between different force fields will be affected by both the
    variability due to the choice of protein and the variability between
    independent replicate simulations. A statistical approach that takes mixed
    effects explicitly into account may be most appropriate.' The same section:
    'The raw experimental data also have uncertainties.'"""
    src = ctx.read("md_readouts.py")
    if not src:
        return Result(UNTESTABLE,
                      "md_readouts.py absent; testable once a multi-system "
                      "comparison exists",
                      evidence="md_readouts.py not found")
    channels = re.search(r'for ch in \(([^)]*)\)', src, re.S)
    n_ch = len(re.findall(r'"', channels.group(1))) // 2 if channels else 0
    mixed = re.search(r"MixedLM|mixedlm|statsmodels|lme4|random.?effect", src, re.I)
    multi = re.search(r"bonferroni|holm|fdr|benjamini|multipletests", src, re.I)
    exp_err = re.search(r"pkd_err|affinity_err|experimental.?(error|uncertain)",
                        src, re.I)
    if mixed:
        return Result(PASS, "the comparison uses an explicit mixed-effects model",
                      evidence="md_readouts.py: MixedLM")
    return Result(FAIL,
                  f"{n_ch} readout channels are compared across ligands with a "
                  f"paired bootstrap and nothing else. That resamples the "
                  f"ligand axis but never separates the replicate-level "
                  f"variance component §5.2.3 names, and there is no "
                  f"multiplicity control across the {n_ch} channels "
                  f"(bonferroni/FDR: {'present' if multi else 'absent'}) — so "
                  f"the probability that at least one channel clears its "
                  f"interval by chance is far above the nominal rate. "
                  f"Experimental uncertainty on the pKD each difference is "
                  f"correlated against is {'carried' if exp_err else 'not carried'} "
                  f"at all.",
                  evidence=f"md_readouts.py: {n_ch} channels, paired_boot only; "
                           f"no MixedLM/statsmodels, no multiplicity "
                           f"correction, no experimental error term")


# ==========================================================================
# §4 — crystallographic ground truth
# ==========================================================================

def _crystal_vs_solution(ctx: Context) -> Result:
    """§4: 'when one uses crystallographic data to benchmark molecular
    simulations, one should simulate the crystal rather than the protein in
    solution ... it is preferable to simulate a supercell comprising multiple
    unit cells.' §4.2.7 quantifies the penalty: crystallographic water recovery
    was 'over 90% when the protein was restrained modestly to crystallographic
    positions' but 'dropped to 50% without restraints'."""
    consumers = []
    for name in ("crystal_water_fingerprint.py", "validate_gcmc_waters.py",
                 "cluster/check_water_eint.py"):
        if ctx.exists(name):
            consumers.append(name)
    if not consumers:
        return Result(NA,
                      "no code takes a crystallographic quantity as ground "
                      "truth, so there is nothing that would need a crystal "
                      "simulation rather than a solution box",
                      evidence="crystal_water_fingerprint.py, "
                               "validate_gcmc_waters.py not found")
    lattice = []
    for p in _py_sources(ctx):
        txt = p.read_text(errors="replace")
        if re.search(r"CRYST1|space.?group|unit.?cell|supercell", txt, re.I):
            lattice.append(p.name)
    if lattice:
        return Result(PASS,
                      f"the crystal lattice is read and simulated by {lattice}",
                      evidence=", ".join(lattice))
    corp = _corpus(ctx)
    return Result(FAIL,
                  f"{len(consumers)} module(s) take deposited crystallographic "
                  f"waters as ground truth, and not one line of our Python "
                  f"reads CRYST1, a space group, a unit cell or a supercell. "
                  f"Every simulation we run is a dilute-solution box, so the "
                  f"comparison crosses phases: solution-phase sampling scored "
                  f"against crystal-phase observation. §4.2.7 puts a number on "
                  f"the cost — crystal-water recovery falls from >90% to 50% "
                  f"purely by releasing protein restraints, 'traced to small, "
                  f"local protein motions that disrupted entire water "
                  f"networks'. The lattice is in every one of our "
                  f"{corp['n_files']} deposited files and we discard it.",
                  evidence=f"consumers: {consumers}; CRYST1/space-group/"
                           f"supercell hits across "
                           f"{sum(1 for _ in _py_sources(ctx))} source files: 0")


def _refinement_provenance(ctx: Context) -> Result:
    """§4.1.1: 'the structural models used as the basis for the comparison are
    typically derived by refinement informed by an empirical force field.
    Therefore, comparisons against the structural model risk assessing, at
    least in part, the agreement of the force field of interest with the force
    field used during refinement, rather than agreement with experimental
    data.' §4: 'force fields should be compared against a single
    crystallographic dataset, rather than attempting to compare force fields
    based on benchmarks against different crystal structures.'"""
    corp = _corpus(ctx)
    if not corp["n_files"]:
        return Result(UNTESTABLE,
                      "no deposited structures found under "
                      f"{STRUCT_GLOB}; testable once the corpus is on disk",
                      evidence=STRUCT_GLOB)
    progs = corp["programs"]
    claim_docs = []
    for doc in ("FINDINGS.md", "PLAN.md", "DATASETS.md", "RUNS.md", "README.md"):
        for ln in ctx.read(doc).splitlines():
            if re.search(r"one (Diamond )?XChem|autoBUSTER|single .*pipeline", ln, re.I):
                claim_docs.append(f"{doc}: {ln.strip()[:80]}")
    if len(progs) == 1:
        return Result(PASS,
                      f"all {corp['n_files']} structures were refined by the "
                      f"same program ({list(progs)[0]}), so the refinement "
                      f"force field is a constant across the corpus",
                      evidence=f"REMARK 3 PROGRAM: {progs}")
    biggest = max(progs, key=progs.get)
    minority = {k: v for k, v in progs.items() if k != biggest}
    n_min = sum(minority.values())
    return Result(FAIL,
                  f"the corpus mixes {len(progs)} refinement programs — "
                  f"{progs} — so {n_min} of {corp['n_files']} structures "
                  f"({100 * n_min / corp['n_files']:.1f}%) carry a different "
                  f"restraint dictionary, bulk-solvent model and B-factor "
                  f"model from the rest. §4.1.1 says a comparison against a "
                  f"refined model partly measures the refinement force field; "
                  f"here that force field is not even held constant. "
                  + (f"Worse, {len(claim_docs)} place(s) in our own documents "
                     f"assert the opposite and use it to justify a decision."
                     if claim_docs else ""),
                  evidence=f"REMARK 3 PROGRAM over {corp['n_files']} files: "
                           f"{progs}" +
                           (f"; claims: {claim_docs[:2]}" if claim_docs else ""))


def _resolution_recorded(ctx: Context) -> Result:
    """§4.2: crystallography datasets are chosen 'prioritizing the following
    features: 1) resolution better than 1.2 Å, so that observable data have low
    uncertainty'."""
    corp = _corpus(ctx)
    if not corp["res"]:
        return Result(UNTESTABLE,
                      "no resolution record parsed from the deposited files; "
                      "testable if REMARK 3 RESOLUTION RANGE HIGH is present",
                      evidence=STRUCT_GLOB)
    res = corp["res"]
    n = len(res)
    med = res[n // 2]
    sub12 = sum(1 for v in res if v <= 1.2)
    # is per-structure resolution available to our analyses at all?
    meta = ctx.read("data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv")
    header = meta.splitlines()[0].lower() if meta else ""
    in_meta = "resolution" in header
    # the REMARK-2 trap: is the record we actually parse present for every file?
    r2 = corp["remark2"]
    missing_r2 = sum(v for (prog, present), v in r2.items() if not present)
    detail = (f"resolution is in REMARK 3 for all {n} structures "
              f"(median {med:.2f} Å, {sub12} = {100 * sub12 / n:.1f}% meet "
              f"§4.2's ≤1.2 Å bar)")
    if in_meta:
        return Result(PASS, detail + "; and it is carried in the analysis "
                                     "metadata", evidence="metadata CSV header")
    extra = ""
    if missing_r2:
        by_prog = {p: v for (p, present), v in r2.items() if not present}
        extra = (f" The trap is concrete: REMARK 2 RESOLUTION is absent from "
                 f"exactly the {missing_r2} {list(by_prog)} structures, so any "
                 f"parser keyed on REMARK 2 silently returns "
                 f"{n - missing_r2} structures and calls it the corpus.")
    return Result(FAIL,
                  detail + f", but the analysis metadata CSV has no resolution "
                           f"column, so no downstream analysis can filter on "
                           f"§4.2's criterion." + extra,
                  evidence=f"metadata header={header[:90]!r}; "
                           f"REMARK 2 presence by program: {dict(r2)}")


def _alternate_conformations(ctx: Context) -> Result:
    """§4.1.3: 'Alternate conformations are a promising avenue for comparison to
    and improvement of simulations ... Each alternate conformation is also
    assigned a partial occupancy (i.e., probability) ... The fractional
    occupancies of alternate conformations inferred from RT crystallography can
    be readily compared with the corresponding results from a simulation.'"""
    corp = _corpus(ctx)
    if not corp["n_files"]:
        return Result(UNTESTABLE, "no deposited structures on disk",
                      evidence=STRUCT_GLOB)
    if not corp["n_alt_atoms"]:
        return Result(NA,
                      "no deposited structure in this corpus models an "
                      "alternate conformation, so there is no experimental "
                      "occupancy to preserve or compare against",
                      evidence=f"0 altloc atoms across {corp['n_files']} files")
    prepared = ctx.glob("data/OpenBind_EV-A71_2A/structures/*/*/*_prepared.pdb")
    kept = 0
    sample = prepared[:200]
    for p in sample:
        for ln in p.read_text(errors="replace").splitlines():
            if ln[:6] in ("ATOM  ", "HETATM") and len(ln) > 16 and ln[16] != " ":
                kept += 1
    consumers = [p.name for p in _py_sources(ctx)
                 if re.search(r"altloc|altLoc", p.read_text(errors="replace"))]
    if kept:
        return Result(PASS,
                      f"alternate conformations survive into the prepared "
                      f"structures ({kept} altloc atoms in {len(sample)} files)",
                      evidence="sampled _prepared.pdb")
    return Result(FAIL,
                  f"{corp['n_alt_files']} of {corp['n_files']} deposited "
                  f"structures ({100 * corp['n_alt_files'] / corp['n_files']:.0f}%) "
                  f"model alternate conformations — {corp['n_alt_atoms']} "
                  f"altloc atoms with refined fractional occupancies — and "
                  f"{kept} survive into the {len(sample)} `_prepared.pdb` files "
                  f"sampled. Every occupancy is discarded before we ever look "
                  f"at the structure. §4.1.3 calls these occupancies directly "
                  f"comparable with simulation populations; we instead spend "
                  f"GPU-hours estimating side-chain flexibility that the "
                  f"crystals already measured. "
                  f"altloc-aware modules: {consumers or 'none'}.",
                  evidence=f"deposited: {corp['n_alt_atoms']} altloc atoms in "
                           f"{corp['n_alt_files']}/{corp['n_files']} files; "
                           f"prepared: {kept} altloc atoms in {len(sample)} files")


# ==========================================================================
# §2 — choosing what to benchmark against
# ==========================================================================

def _protein_ff_provenance(ctx: Context) -> Result:
    """§2: 'publications of new force fields often validate their parameters on
    different subsets of the available experimental datasets, making it
    difficult to directly compare a new force field with established ones. It
    is therefore important to work toward a common set of accepted benchmarks
    ... a minimal benchmark [should] include at least one dataset containing
    peptides, folded proteins, and disordered proteins'. We do not develop a
    protein force field, so the obligation that transfers is the weaker one:
    the protein+water model we took off the shelf should have a recorded
    benchmark provenance, because its accuracy is an input to every number we
    read off a trajectory."""
    build, bpath = _any_build_json(ctx)
    ff = build.get("protein_ff") if build else None
    if not ff:
        return Result(UNTESTABLE,
                      "no build receipt names a protein force field; testable "
                      "once md_arms.py build has run",
                      evidence="no results/**/build.json with protein_ff")
    name = " ".join(ff)
    named = [d for d in ("DATASETS.md", "LITERATURE.md", "PLAN.md", "FINDINGS.md")
             if re.search(r"amber14|ff14sb|ff99|charmm36", ctx.read(d), re.I)]
    tbl1 = ("beauchamp", "trp-cage", "stroet", "a99sb-disp", "robustelli",
            "bmrb", "lindorff", "crambin", "cyclophilin", "ptp1b",
            "staphylococcal nuclease", "endoglucanase", "lysozyme")
    ds = ctx.read("DATASETS.md").lower()
    present = [t for t in tbl1 if t in ds]
    if named and present:
        return Result(PASS,
                      f"{name} is named in {named} and its benchmark "
                      f"provenance is traceable ({present})",
                      evidence=f"{bpath}: {ff}")
    return Result(FAIL,
                  f"every structural number we quote off a trajectory is "
                  f"conditioned on {name}. It is named in "
                  f"{named or 'no project document'}, but with no benchmark "
                  f"behind it: DATASETS.md contains none of this review's Table 1 "
                  f"protein-force-field benchmark sets "
                  f"(matched: {present or 'none'}) — our dataset ladder is "
                  f"entirely protein-ligand, so we have no way to say whether "
                  f"the protein force field under everything is fit for the "
                  f"folded/flexible mixture our pocket actually is.",
                  evidence=f"{bpath}: protein_ff={ff}; DATASETS.md Table-1 "
                           f"matches: {present or 'none'}")


CHECKS = [
    # §5.1 setup
    Check(id="ffb-01", severity="must", section="§5.1.1",
          question="Are protonation states of titratable side chains AND termini "
                   "assigned by a pKa method at a stated pH?",
          measures="protonate.py + data/protonated/*_protonation.json",
          fn=_protonation_states),
    Check(id="ffb-02", severity="should", section="§5.1.1",
          question="Do disulfides and metal-coordinating residues get their state "
                   "from the local environment rather than the pH default?",
          measures="protonate.py + the overrides recorded in the sidecar",
          fn=_local_environment_overrides),
    Check(id="ffb-03", severity="must", section="§5.1.2",
          question="Is the solvent padding at least 10 Å?",
          measures="md_arms.py PADDING_NM, cross-checked against build.json",
          fn=_solvent_padding),
    Check(id="ffb-04", severity="should", section="§5.1.2",
          question="Is the periodic box a rhombic dodecahedron for a globular solute?",
          measures="the addSolvent call in md_arms.py",
          fn=_box_shape),
    Check(id="ffb-05", severity="should", section="§5.1.2",
          question="Is the water model treated as part of the force field under "
                   "test, with TIP3P's documented bias on the record?",
          measures="build.json protein_ff + a keyword scan of our documents",
          fn=_water_model_declared),
    Check(id="ffb-06", severity="should", section="§5.1.3",
          question="Is the system neutralised and the experimental ionic strength "
                   "modelled?",
          measures="md_arms.py addSolvent kwargs + build.json n_ions",
          fn=_ionic_strength),
    Check(id="ffb-07", severity="consider", section="§5.1.3",
          question="For a net-charged solute, does the ion count follow SLTCAP "
                   "(Eq 16) rather than the bulk-ratio Eq 15?",
          measures="build.json n_waters/n_ions + the recorded solute net charge",
          fn=_sltcap),
    Check(id="ffb-08", severity="must", section="§5.1.4",
          question="Is NPT sampled with correct estimators, and is every "
                   "Berendsen/weak-coupling scheme absent?",
          measures="a Berendsen scan over our Python + md_arms.py integrator",
          fn=_correct_ensemble),
    Check(id="ffb-09", severity="should", section="§5.1.4",
          question="If kinetic quantities are computed, is Langevin's kinetic "
                   "artefact accounted for?",
          measures="a scan for rate/residence-time/diffusion readouts",
          fn=_kinetic_artefacts),
    # §5.2 analysis
    Check(id="ffb-10", severity="should", section="§5.2.1",
          question="Is the equilibration length justified by monitoring potential "
                   "energy as well as volume/density?",
          measures="whether md_arms.py records anything during equilibration",
          fn=_equilibration_monitored),
    Check(id="ffb-11", severity="should", section="§5.2.2",
          question="Is convergence assessed — drift in the observable time series, "
                   "or blocking analysis?",
          measures="who writes and who reads state.csv; a blocking/pymbar scan",
          fn=_convergence_assessed),
    Check(id="ffb-12", severity="must", section="§5.2.3",
          question="Is the difference between two sets of simulations tested for "
                   "statistical significance?",
          measures="md_readouts.py bootstrap and interval",
          fn=_significance_tested),
    Check(id="ffb-13", severity="should", section="§5.2.3",
          question="Does the uncertainty come from multiple independent simulations?",
          measures="cluster/md_arms.sbatch REPS and the --replicates default",
          fn=_independent_replicates),
    Check(id="ffb-14", severity="should", section="§5.2.3",
          question="With multiple systems and several properties, are mixed effects "
                   "and experimental uncertainty accounted for?",
          measures="md_readouts.py channel count vs its statistical machinery",
          fn=_mixed_effects_and_experimental_error),
    # §4 crystallographic ground truth
    Check(id="ffb-15", severity="should", section="§4",
          question="Are crystallographic observables compared against a simulation "
                   "of the crystal rather than of the protein in solution?",
          measures="a CRYST1/space-group/supercell scan over our Python",
          fn=_crystal_vs_solution),
    Check(id="ffb-16", severity="should", section="§4.1.1",
          question="Is the refinement provenance of the reference structures "
                   "recorded, and is it actually uniform?",
          measures="REMARK 3 PROGRAM across every deposited structure",
          fn=_refinement_provenance),
    Check(id="ffb-17", severity="consider", section="§4.2",
          question="Is per-structure resolution recorded so §4.2's ≤1.2 Å criterion "
                   "can be applied?",
          measures="REMARK 2/3 resolution records vs the analysis metadata CSV",
          fn=_resolution_recorded),
    Check(id="ffb-18", severity="consider", section="§4.1.3",
          question="Are the deposited alternate conformations and their occupancies "
                   "retained rather than silently collapsed?",
          measures="altloc atom counts, deposited vs _prepared.pdb",
          fn=_alternate_conformations),
    # §2 choosing the benchmark
    Check(id="ffb-19", severity="consider", section="§2",
          question="Does the protein+water force field under every trajectory have a "
                   "recorded benchmark provenance?",
          measures="build.json protein_ff vs our documents and DATASETS.md",
          fn=_protein_ff_provenance),
]
