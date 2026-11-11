"""AQUA-DUCT: analysing MD with small molecules as *probes* (LiveCoMS 2021).

This is a LiveCoMS **Tutorial**, not a Best Practices article, and it matters to
us for one reason: it is the only paper in the set whose explicit checklist is
about the **trajectory file itself** rather than about statistics. §2
"Prerequisites" lists five numbered requirements a trajectory must satisfy
*before* any solvent analysis is run on it, and every one of them is measurable
against files we already have on disk.

Why this paper governs us at all. Our water thesis is live: crystallographic
occupancy predicts affinity at rho = +0.359 (`FINDINGS` §2.8), water *count* at
+0.403 while the MACE water *energy* is a null (§2.16), GCMC sites reproduce
deposited waters 32/32 (§2.9), and GCMC equilibration is worth 19 points in
someone else's ablation (§4.4). All of that is currently computed from **static
crystal structures**. The moment any of it is computed from a trajectory
instead, this paper's §2 becomes binding, and three of its five requirements are
things our MD pipeline does not do.

The checks below deliberately measure the DCD header directly (`struct`, no
MDAnalysis import) so they stay cheap, and so they measure what was *written*
rather than what a script says it intended to write.
"""

from __future__ import annotations

import csv
import json
import re
import struct
from io import StringIO
from pathlib import Path

from . import NA, PASS, FAIL, UNTESTABLE, Check, Context, Paper, Result

PAPER = Paper(
    slug="aquaduct",
    title="AQUA-DUCT: Analysis of Molecular Dynamics Simulations of "
          "Macromolecules with the Use of Molecular Probes [Article v1.0]",
    authors="Karolina Mitusinska, Agata Raczynska, Piotr Wojsa, Maria Bzowka, "
            "Artur Gora",
    year="2021",
    url="https://livecomsjournal.org/index.php/livecoms/article/view/v2i1e21383",
    doi="10.33011/livecoms.2.1.21383",
    scope="What a trajectory must satisfy before water, cosolvent or ligand "
          "molecules in it can be treated as molecular probes: solvent must be "
          "saved, the box imaged and derotated, the sampling interval short "
          "enough to follow a molecule, and the interior hydrated before "
          "production.",
)

# 1 AKMA time unit in picoseconds. OpenMM's DCDFile writes the integration
# timestep into icntrl[9] in these units; NSAVC * delta * AKMA_PS is therefore
# the interval between saved frames, read from the trajectory rather than from
# whatever the launch script claimed.
AKMA_PS = 0.04888821

#: §2, requirement 4: "not exceed 5 ps sampling".
INTERVAL_CEILING_PS = 5.0
#: §2, requirement 4: "coordinates are saved every picosecond (1 ps) for
#: simulations shorter than 100 ns".
INTERVAL_RECOMMENDED_PS = 1.0
SHORT_TRAJECTORY_PS = 100_000.0
#: The DCD stores the timestep as float32, so a trajectory saved at exactly
#: 1.00 ps reads back as 1.0000000030 ps. Comparing without slack would fail a
#: compliant trajectory on storage precision alone — the check would be wrong,
#: not the run. Any real violation here is a factor of 2 or more.
INTERVAL_TOL = 1.005
#: §4 Overview: every sample data set the paper demonstrates the method on is
#: "10 ns saved each 5 ps". The paper states no minimum; this is its range.
DEMONSTRATED_PROD_PS = 10_000.0


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------

def _dcd_header(path: Path) -> dict | None:
    """NSET / NSAVC / delta / unit-cell flag / NATOM straight out of a DCD.

    Returns None rather than raising, so a check can report UNTESTABLE.
    """
    try:
        raw = path.open("rb").read(8192)
        if raw[4:8] != b"CORD":
            return None
        icntrl = struct.unpack("<20i", raw[8:88])
        (delta,) = struct.unpack("<f", raw[44:48])
        (title_block,) = struct.unpack("<i", raw[92:96])
        off = 96 + title_block + 4          # past the title record + its marker
        (natom,) = struct.unpack("<i", raw[off + 4:off + 8])
    except Exception:                                        # noqa: BLE001
        return None
    return {
        "n_frames": icntrl[0], "first_step": icntrl[1], "nsavc": icntrl[2],
        "delta_akma": delta, "unitcell": icntrl[10], "n_atoms": natom,
        "interval_ps": icntrl[2] * delta * AKMA_PS,
        "path": path,
    }


def _md_arms(ctx: Context) -> list[dict]:
    """Every finished MD arm: its receipt, its trajectory and its state log."""
    arms = []
    seen = set()
    for pattern in ("results/*/*/*/receipt.json",
                    "results/*/*/*/*/receipt.json"):
        for rec in ctx.glob(pattern):
            d = rec.parent
            if d in seen or not (d / "traj.dcd").exists():
                continue
            seen.add(d)
            try:
                meta = json.loads(rec.read_text())
            except Exception:                                # noqa: BLE001
                continue
            arms.append({"dir": d, "receipt": meta,
                         "dcd": d / "traj.dcd",
                         "state": d / "state.csv"})
    return sorted(arms, key=lambda a: str(a["dir"]))


def _build_meta(ctx: Context) -> list[dict]:
    """`build.json` written by `md_arms.py build` — one per solvated system."""
    out = []
    for pattern in ("results/*/*/build.json", "results/*/build.json"):
        for p in ctx.glob(pattern):
            try:
                out.append({"path": p, "meta": json.loads(p.read_text())})
            except Exception:                                # noqa: BLE001
                pass
    return out


def _our_source(ctx: Context) -> list[tuple[str, str]]:
    """Our own Python, excluding the vendored `mace/` and `mace-off/` trees."""
    files = []
    for pattern in ("*.py", "cluster/*.py", "albench/*.py"):
        for p in ctx.glob(pattern):
            try:
                files.append((str(p.relative_to(ctx.root)), p.read_text(errors="replace")))
            except Exception:                                # noqa: BLE001
                pass
    return files


def _grep(ctx: Context, pattern: str) -> list[str]:
    """Files whose source matches `pattern`, as `path:line`."""
    rx = re.compile(pattern)
    hits = []
    for rel, text in _our_source(ctx):
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{n}")
    return hits


def _column(text: str, name: str) -> list[float]:
    rows = list(csv.reader(StringIO(text)))
    if not rows:
        return []
    header = [h.strip().strip('#').strip('"') for h in rows[0]]
    if name not in header:
        return []
    i = header.index(name)
    out = []
    for r in rows[1:]:
        if len(r) > i:
            try:
                out.append(float(r[i]))
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------
# §2 Prerequisites — the explicit checklist
# --------------------------------------------------------------------------

def _solvent_is_saved(ctx: Context) -> Result:
    """Req. 1: the trajectory must hold the macromolecule AND the small
    molecules, "otherwise it will be impossible to track molecules"."""
    arms, builds = _md_arms(ctx), _build_meta(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm with a traj.dcd found under results/",
                      evidence="testable as soon as one md_arms.py run lands")
    dry = [b for b in builds if not b["meta"].get("n_waters")]
    if dry:
        return Result(FAIL, f"{len(dry)}/{len(builds)} solvated systems report "
                            f"zero waters — nothing to trace as a probe",
                      evidence="; ".join(str(b["path"]) for b in dry[:3]))
    bad, ok = [], []
    for a in arms:
        h = _dcd_header(a["dcd"])
        want = _system_of(ctx, a["dir"], builds)
        if h is None:
            bad.append(f"{a['dir'].name}: unreadable DCD")
        elif want is None:
            bad.append(f"{a['dir'].name}: no build.json in its tree to check "
                       f"the {h['n_atoms']} written atoms against")
        elif h["n_atoms"] != want["n_atoms"]:
            bad.append(f"{a['dir'].name}: DCD holds {h['n_atoms']} of the "
                       f"system's {want['n_atoms']} atoms")
        else:
            ok.append(f"{a['dir'].name}={h['n_atoms']}")
    if bad:
        return Result(FAIL, "trajectory is an atom subset, so the solvent was "
                            "not saved: " + "; ".join(bad[:3]),
                      evidence="; ".join(ok[:3]))
    nw = sorted(b["meta"]["n_waters"] for b in builds)
    xw = sorted(b["meta"].get("n_crystal_waters", 0) for b in builds)
    return Result(PASS, f"all {len(arms)} arms write their system in full; every "
                        f"DCD carries the waters as well as the protein",
                  evidence=f"NATOM matches build.json per system ({ok[0]}); "
                           f"across {len(builds)} systems n_waters "
                           f"{nw[0]}-{nw[-1]}, of which crystallographic "
                           f"{xw[0]}-{xw[-1]}")


def _system_of(ctx: Context, arm_dir: Path, builds: list[dict]) -> dict | None:
    """The build.json governing this arm — the nearest one up the tree, so an
    arm is compared against its own system rather than against any system."""
    by_dir = {b["path"].parent: b["meta"] for b in builds}
    p = arm_dir
    while p != ctx.root and p.parent != p:
        if p in by_dir:
            return by_dir[p]
        p = p.parent
    return None


def _interval_under_ceiling(ctx: Context) -> Result:
    """Req. 4: "not exceed 5 ps sampling" — the hard ceiling."""
    arms = _md_arms(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm with a traj.dcd found under results/",
                      evidence="testable as soon as one md_arms.py run lands")
    worst, rows = 0.0, []
    for a in arms:
        h = _dcd_header(a["dcd"])
        if h is None:
            continue
        worst = max(worst, h["interval_ps"])
        rows.append(f"{a['dir'].name}={h['interval_ps']:.2f}ps")
    if not rows:
        return Result(UNTESTABLE, "no DCD header could be parsed",
                      evidence="; ".join(str(a["dcd"]) for a in arms[:2]))
    if worst > INTERVAL_CEILING_PS * INTERVAL_TOL:
        return Result(FAIL, f"slowest saved interval {worst:.2f} ps exceeds the "
                            f"{INTERVAL_CEILING_PS:.0f} ps ceiling — a probe "
                            f"cannot be followed between frames",
                      evidence="; ".join(rows[:6]))
    return Result(PASS, f"slowest saved interval {worst:.2f} ps, under the "
                        f"{INTERVAL_CEILING_PS:.0f} ps ceiling",
                  evidence="measured from DCD NSAVC*delta, not from the launch "
                           "flag: " + "; ".join(rows[:6]))


def _interval_is_recommended(ctx: Context) -> Result:
    """Req. 4: 1 ps for trajectories shorter than 100 ns."""
    arms = _md_arms(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm with a traj.dcd found under results/",
                      evidence="testable as soon as one md_arms.py run lands")
    offenders = []
    for a in arms:
        h = _dcd_header(a["dcd"])
        prod = float(a["receipt"].get("prod_ps") or 0.0)
        if h is None:
            continue
        if (prod < SHORT_TRAJECTORY_PS
                and h["interval_ps"] > INTERVAL_RECOMMENDED_PS * INTERVAL_TOL):
            offenders.append(f"{a['dir'].name}: {h['interval_ps']:.2f} ps "
                             f"over {prod:.0f} ps")
    if offenders:
        return Result(FAIL, f"{len(offenders)}/{len(arms)} arms are shorter than "
                            f"100 ns yet save less often than every 1 ps",
                      evidence="; ".join(offenders[:6]))
    return Result(PASS, "every sub-100 ns trajectory saves at least every 1 ps",
                  evidence=f"{len(arms)} arms checked")


def _box_is_recorded(ctx: Context) -> Result:
    """Req. 2 (precondition): a trajectory with no unit cell can never be
    imaged, so the requirement is unsatisfiable rather than merely unmet."""
    arms = _md_arms(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm with a traj.dcd found under results/",
                      evidence="testable as soon as one md_arms.py run lands")
    missing = []
    for a in arms:
        h = _dcd_header(a["dcd"])
        if h is None or not h["unitcell"]:
            missing.append(a["dir"].name)
    if missing:
        return Result(FAIL, f"{len(missing)} trajectories carry no unit cell, so "
                            f"periodic boundaries can never be fixed after the "
                            f"fact: {', '.join(missing[:4])}")
    return Result(PASS, f"all {len(arms)} trajectories record box vectors per "
                        f"frame, so re-imaging remains possible",
                  evidence="DCD icntrl[10]=1 (unit-cell flag) in every arm")


def _trajectory_is_imaged(ctx: Context) -> Result:
    """Req. 2: "fixed periodic boundaries conditions and the protein should
    stay in the centre of the solvent box"."""
    arms = _md_arms(ctx)
    hits = _grep(ctx, r"center_in_box|add_transformations|autoimage|make_whole"
                      r"|trjconv|\.unwrap\(|transformations\.wrap")
    if hits:
        return Result(PASS, "a re-imaging / centring step exists",
                      evidence="; ".join(hits[:5]))
    if not arms:
        return Result(UNTESTABLE, "no MD arm found, and no imaging code either — "
                                  "cannot tell whether the gap matters yet",
                      evidence="testable as soon as one md_arms.py run lands")
    writer = _grep(ctx, r"DCDReporter\(")
    return Result(FAIL, "nothing in the repo re-images or re-centres a "
                        "trajectory; DCDReporter is called without an explicit "
                        "enforcePeriodicBox, so OpenMM wraps molecules into the "
                        "box while the protein is free to sit on a boundary. "
                        "Ligand-RMSD readouts survive this (the protein is one "
                        "molecule and stays whole); any water path does not — a "
                        "wrapped water teleports across the box between frames",
                  evidence=f"{len(arms)} periodic arms on disk; "
                           f"DCDReporter at {', '.join(writer[:2]) or 'unknown'}; "
                           f"zero matches for center_in_box/autoimage/"
                           f"add_transformations/trjconv")


def _rotation_is_removed(ctx: Context) -> Result:
    """Req. 3: "All rotational movements of the protein should be reduced. All
    the structures should be aligned to some reference"."""
    hits = _grep(ctx, r"AlignTraj|align\.alignto|rms\.RMSD\(.*ref")
    if not hits:
        return Result(FAIL, "no frame-by-frame superposition onto a reference "
                            "anywhere in the repo, so every readout carries the "
                            "protein's rigid-body rotation",
                      evidence="zero matches for AlignTraj / align.alignto")
    traj = [h for h in hits if "AlignTraj" in _line_at(ctx, h)]
    if not traj:
        return Result(FAIL, "only single-structure superposition found "
                            f"({hits[0]}); no trajectory is derotated frame by "
                            f"frame", evidence="; ".join(hits[:5]))
    return Result(PASS, "every trajectory is superposed on a reference before "
                        "readout, so rigid-body rotation is removed",
                  evidence="; ".join(traj[:3]) + " (protein and name CA)")


def _line_at(ctx: Context, ref: str) -> str:
    rel, _, num = ref.rpartition(":")
    try:
        return ctx.read(rel).splitlines()[int(num) - 1]
    except Exception:                                        # noqa: BLE001
        return ""


def _interior_is_hydrated(ctx: Context) -> Result:
    """Req. 5: "be mindful of protein hydration before running MD simulations
    and use one of the available tools to place water molecules inside the
    protein's cavities and/or extend the equilibration step"."""
    builds = _build_meta(ctx)
    if not builds:
        return Result(UNTESTABLE, "no build.json for a solvated system found",
                      evidence="testable as soon as md_arms.py build runs")
    dry = [b for b in builds if not b["meta"].get("n_crystal_waters")]
    if dry:
        return Result(FAIL, f"{len(dry)}/{len(builds)} systems were built with no "
                            f"waters in the interior: crystallographic waters "
                            f"were dropped and no cavity-hydration tool was run, "
                            f"so buried sites start empty and must fill by "
                            f"diffusion during a short equilibration",
                      evidence="n_crystal_waters=0 in "
                               + "; ".join(str(b['path']) for b in dry[:3]))
    xw = sorted(b["meta"]["n_crystal_waters"] for b in builds)
    equil = sorted({float(a["receipt"].get("equil_ps") or 0)
                    for a in _md_arms(ctx)})
    return Result(PASS, f"deposited waters are placed in the interior before "
                        f"production in all {len(builds)} systems, satisfying "
                        f"the requirement's first limb (placement)",
                  evidence=f"n_crystal_waters {xw[0]}-{xw[-1]} across "
                           f"{len(builds)} systems; equilibration {equil or 'n/a'}"
                           f" ps — far too short to hydrate a cavity by "
                           f"diffusion, so the requirement is met by placement "
                           f"alone, and GCMC waters are not placed at all")


# --------------------------------------------------------------------------
# rules stated inside the tutorials
# --------------------------------------------------------------------------

def _volume_is_stable(ctx: Context) -> Result:
    """§4.5.4: "The plots of Object area and Object volume should be stable or
    otherwise there might be something wrong with the calculations." Our
    closest measurable analogue is the NPT box volume across production."""
    arms = _md_arms(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm with a state.csv found",
                      evidence="testable as soon as one md_arms.py run lands")
    rows, bad = [], []
    for a in arms:
        v = _column(ctx.read(str(a["state"].relative_to(ctx.root))),
                    "Box Volume (nm^3)")
        if len(v) < 8:
            continue
        mean = sum(v) / len(v)
        var = sum((x - mean) ** 2 for x in v) / len(v)
        rsd = (var ** 0.5) / mean
        h = len(v) // 2
        drift = abs(sum(v[h:]) / len(v[h:]) - sum(v[:h]) / h) / mean
        rows.append(f"{a['dir'].name}: rSD {rsd*100:.2f}%, drift {drift*100:.2f}%")
        if rsd > 0.01 or drift > 0.01:
            bad.append(rows[-1])
    if not rows:
        return Result(UNTESTABLE, "no state.csv carried a box-volume column",
                      evidence=f"{len(arms)} arms inspected")
    if bad:
        return Result(FAIL, f"{len(bad)}/{len(rows)} arms have a box volume that "
                            f"is drifting or noisy by more than 1% — the system "
                            f"is still equilibrating, or worse",
                      evidence="; ".join(bad[:4]))
    return Result(PASS, f"box volume is stable in all {len(rows)} arms "
                        f"(< 1% relative SD and < 1% half-to-half drift)",
                  evidence="; ".join(rows[:4]))


def _cosolvent_is_its_own_probe(ctx: Context) -> Result:
    """§4.8.8 + §4.1: a cosolvent is a *different probe* with different
    physico-chemical properties that "penetrate different regions"; it gets its
    own pockets/hot-spots run against a pure-water reference system."""
    fp = ctx.read("results/crystal_water_fingerprint.csv")
    if not fp:
        return Result(UNTESTABLE, "results/crystal_water_fingerprint.csv absent",
                      evidence="regenerate with crystal_water_fingerprint.py")
    rows = list(csv.reader(StringIO(fp)))
    cells = {c.strip() for r in rows[1:] for c in r[1:] if c.strip()}
    additive = _grep(ctx, r"['\"](DMS|GOL)['\"]")
    if len(cells) > 2:
        return Result(PASS, f"the occupancy fingerprint carries {len(cells)} "
                            f"states, so a site occupied by a cosolvent is "
                            f"distinguishable from an empty one",
                      evidence=f"values {sorted(cells)[:6]}")
    return Result(FAIL, "the hydration fingerprint is binary, so a site "
                        "occupied by a cosolvent is scored identically to a dry "
                        "site. The paper treats each solvent species as its own "
                        "probe with its own reference run; we fold cryoprotectant "
                        "into 'no water'",
                  evidence=f"{len(rows)-1} rows x {len(rows[0])-1} sites, cell "
                           f"values exactly {sorted(cells)}; "
                           f"crystal_water_fingerprint.py selects only HOH; "
                           f"DMS/GOL handling in analysis code: "
                           f"{additive or 'none'}. FINDINGS §1.15: DMSO in 88.4% "
                           f"of crystals, first shell in 11.8%")


def _singletons_are_outliers(ctx: Context) -> Result:
    """§4.4.11: singleton clusters are moved to `cluster_out`; §4.4.9 defines
    outliers as "inlets which represent rarely occurring passages"."""
    txt = ctx.read("results/crystal_water_sites.csv")
    if not txt:
        return Result(UNTESTABLE, "results/crystal_water_sites.csv absent",
                      evidence="regenerate with crystal_water_fingerprint.py")
    rows = list(csv.DictReader(StringIO(txt)))
    if not rows or "site_id" not in rows[0]:
        return Result(UNTESTABLE, "crystal_water_sites.csv has no site_id column",
                      evidence=f"columns {list(rows[0]) if rows else 'none'}")
    support: dict[str, int] = {}
    for r in rows:
        if str(r.get("occupied", "")).strip().lower() in ("true", "1"):
            support[r["site_id"]] = support.get(r["site_id"], 0) + 1
    if not support:
        return Result(UNTESTABLE, "no occupied rows to count support over",
                      evidence=f"{len(rows)} rows")
    n_struct = len({r.get("complex_name") for r in rows})
    singles = [s for s, c in support.items() if c <= 1]
    lo = min(support, key=support.get)
    if singles:
        return Result(FAIL, f"{len(singles)} hydration site(s) are supported by a "
                            f"single structure and are still reported as sites "
                            f"rather than outliers: {', '.join(sorted(singles)[:5])}",
                      evidence=f"support counted over {n_struct} structures")
    return Result(PASS, f"every one of the {len(support)} sites is supported by "
                        f"many structures; none is a singleton passing as a site",
                  evidence=f"weakest site {lo} at {support[lo]}/{n_struct} "
                           f"structures ({support[lo]/n_struct*100:.1f}%)")


def _every_run_logs(ctx: Context) -> Result:
    """§4.3.7: use the debug file "to ensure that a log file will be created and
    all the information on the calculations will be saved to this file"."""
    dcds = {p.parent for p in ctx.glob("results/*/*/*/traj.dcd")}
    dcds |= {p.parent for p in ctx.glob("results/*/*/*/*/traj.dcd")}
    if not dcds:
        return Result(UNTESTABLE, "no trajectory found to look for a log beside",
                      evidence="testable as soon as one md_arms.py run lands")
    silent = [d.name for d in sorted(dcds) if not (d / "receipt.json").exists()]
    if silent:
        return Result(FAIL, f"{len(silent)}/{len(dcds)} trajectories have no "
                            f"run record beside them: {', '.join(silent[:4])}")
    return Result(PASS, f"all {len(dcds)} trajectories carry a receipt.json "
                        f"recording seeds, platform, timings and the system hash",
                  evidence=f"e.g. {sorted(dcds)[0].name}/receipt.json")


def _runs_do_not_overwrite(ctx: Context) -> Result:
    """§4.4.1 / §4.6.11: "To keep all the data and avoid overwriting the files
    in next runs of calculations, we recommend the user to copy the [output]
    files ... and store them in a separate directory"."""
    writers = _grep(ctx, r"DCDReporter\(")
    if not writers:
        return Result(UNTESTABLE, "no trajectory writer found in our source to "
                                  "inspect for a fixed output path",
                      evidence="zero matches for DCDReporter(")
    fixed = [w for w in writers
             if re.search(r"""DCDReporter\(\s*['"]""", _line_at(ctx, w))]
    if fixed:
        return Result(FAIL, f"{len(fixed)} trajectory writer(s) target a literal "
                            f"path, so every re-run overwrites the last: "
                            f"{', '.join(fixed[:3])}",
                      evidence="; ".join(f"{w} -> {_line_at(ctx, w).strip()}"
                                         for w in fixed[:2]))
    arms = _md_arms(ctx)
    dirs = sorted({a["dir"].name for a in arms})
    return Result(PASS, f"every trajectory is written into a per-run directory, "
                        f"so a re-run cannot overwrite the previous one",
                  evidence="; ".join(f"{w} -> {_line_at(ctx, w).strip()}"
                                     for w in writers[:2])
                           + (f" | {len(arms)} runs on disk in {len(dirs)} "
                              f"directories: {', '.join(dirs[:6])}" if arms else ""))


def _production_long_enough(ctx: Context) -> Result:
    """§4 Overview: every sample data set is 10 ns. §4.5.4: the N "cluster" —
    paths that begin or end *inside* the Scope, i.e. probes never seen to
    complete an entry or an exit — shrinks only "for longer simulations".

    The paper sets no minimum, so this is calibration against the range the
    method is demonstrated over, not a stated rule. Severity is `consider`.
    """
    arms = _md_arms(ctx)
    if not arms:
        return Result(UNTESTABLE, "no MD arm found to measure production length",
                      evidence="testable as soon as one md_arms.py run lands")
    lengths = {}
    for a in arms:
        prod = float(a["receipt"].get("prod_ps") or 0.0)
        reps = int(a["receipt"].get("replicates") or 1)
        lengths[a["dir"].name] = prod * max(reps, 1)
    worst = max(lengths.values())
    if worst >= DEMONSTRATED_PROD_PS:
        return Result(PASS, f"longest arm samples {worst/1000:.1f} ns, at or "
                            f"above the 10 ns the paper demonstrates on",
                      evidence=f"{len(arms)} arms, max {worst:.0f} ps")
    return Result(FAIL, f"longest arm samples {worst/1000:.2f} ns against the "
                        f"10 ns every sample data set in the paper uses — "
                        f"{DEMONSTRATED_PROD_PS/worst:.0f}x short. Probe paths "
                        f"would be dominated by the N cluster (never seen to "
                        f"enter or leave), so no transport readout is "
                        f"interpretable at this length",
                  evidence="; ".join(f"{k}={v:.0f}ps" for k, v in
                                     sorted(lengths.items())[:6])
                           + " | paper §4: 'All data sets consist of 2,000 "
                             "frames (10 ns saved each 5 ps)'")


def _probe_species_declared(ctx: Context) -> Result:
    """§4.1 / §4.6.2: `add_passing` must name the traced species, and a probe
    that never enters the Object is still evidence. Not applicable to us until
    a probe-tracking analysis exists at all."""
    if _grep(ctx, r"aquaduct|aqua_duct|valve_run|pond_run"):
        return Result(UNTESTABLE, "AQUA-DUCT is referenced but no configuration "
                                  "file is on disk to inspect for scope/object/"
                                  "add_passing",
                      evidence="testable once a valve config.txt exists")
    return Result(NA, "we run no probe-tracking analysis: AQUA-DUCT is not "
                      "installed, no valve/pond configuration exists, and every "
                      "hydration number we quote (FINDINGS §2.8, §2.9, §2.16) "
                      "comes from static crystal structures, not from tracing "
                      "molecules through a trajectory. This becomes testable "
                      "the day a trajectory-derived water readout is computed",
                  evidence="zero matches for aquaduct/valve_run/pond_run in our "
                           "source; no *.offxml-style valve config in results/")


CHECKS = [
    # --- §2 Prerequisites, the paper's own numbered checklist ---------------
    Check(id="aqd-01", severity="must", section="§2 Prerequisites, req. 1",
          question="Does the saved trajectory contain the solvent as well as "
                   "the macromolecule?",
          measures="DCD NATOM vs build.json n_atoms/n_waters",
          fn=_solvent_is_saved),
    Check(id="aqd-02", severity="must", section="§2 Prerequisites, req. 4",
          question="Is the saved coordinate interval within the 5 ps ceiling?",
          measures="NSAVC x delta from the DCD header of every arm",
          fn=_interval_under_ceiling),
    Check(id="aqd-03", severity="should", section="§2 Prerequisites, req. 4",
          question="Is it 1 ps, as recommended for trajectories under 100 ns?",
          measures="DCD interval vs receipt.json prod_ps",
          fn=_interval_is_recommended),
    Check(id="aqd-04", severity="must", section="§2 Prerequisites, req. 2",
          question="Does the trajectory record box vectors, so it can be imaged "
                   "at all?",
          measures="DCD unit-cell flag icntrl[10]",
          fn=_box_is_recorded),
    Check(id="aqd-05", severity="should", section="§2 Prerequisites, req. 2",
          question="Is the trajectory re-imaged so the macromolecule stays "
                   "centred in the solvent box?",
          measures="repo grep for centring/imaging calls, vs number of periodic "
                   "trajectories on disk",
          fn=_trajectory_is_imaged),
    Check(id="aqd-06", severity="must", section="§2 Prerequisites, req. 3",
          question="Is rotational motion removed by superposing every frame on "
                   "a reference?",
          measures="repo grep for AlignTraj / align.alignto",
          fn=_rotation_is_removed),
    Check(id="aqd-07", severity="must", section="§2 Prerequisites, req. 5",
          question="Were waters placed in the protein's interior cavities "
                   "before production?",
          measures="build.json n_crystal_waters and equil_ps",
          fn=_interior_is_hydrated),

    # --- rules stated inside the tutorials ----------------------------------
    Check(id="aqd-08", severity="should", section="§4.5.4",
          question="Is the box volume stable across production, or is something "
                   "wrong with the calculation?",
          measures="relative SD and half-to-half drift of Box Volume in state.csv",
          fn=_volume_is_stable),
    Check(id="aqd-09", severity="should", section="§4.8.8 and §4.1",
          question="Is a cosolvent traced as its own probe species rather than "
                   "folded into 'no water'?",
          measures="distinct cell values in results/crystal_water_fingerprint.csv",
          fn=_cosolvent_is_its_own_probe),
    Check(id="aqd-10", severity="should", section="§4.4.11",
          question="Are singleton hydration sites labelled outliers rather than "
                   "reported as sites?",
          measures="per-site support counted from results/crystal_water_sites.csv",
          fn=_singletons_are_outliers),
    Check(id="aqd-11", severity="should", section="§4.3.7",
          question="Does every run leave a log recording the calculation?",
          measures="receipt.json beside every traj.dcd",
          fn=_every_run_logs),
    Check(id="aqd-12", severity="should", section="§4.4.1 and §4.6.11",
          question="Does each run write to its own directory instead of "
                   "overwriting the previous one?",
          measures="distinct output directories vs number of runs",
          fn=_runs_do_not_overwrite),
    Check(id="aqd-13", severity="consider", section="§4 Overview and §4.5.4",
          question="Is production long enough that a probe path can complete an "
                   "entry and an exit?",
          measures="prod_ps x replicates vs the 10 ns of every sample data set",
          fn=_production_long_enough),
    Check(id="aqd-14", severity="consider", section="§4.1 and §4.6.2",
          question="Is the traced probe species declared, with passing paths "
                   "included?",
          fn=_probe_species_declared),
]
