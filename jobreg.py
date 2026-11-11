#!/usr/bin/env python
"""Job registry — register, reconcile and verify cluster jobs.

    python jobreg.py submit cluster/foo.sbatch --expect '/work/.../out/*.csv' \
        --check 'python check_foo.py' --question "does X move hits@25?"
    python jobreg.py status          # reconcile registry against the live queue
    python jobreg.py verify 67266    # run the recorded acceptance check
    python jobreg.py ls / show ID

WHY THIS EXISTS. Every gate below is a failure this project actually had, not a
hypothetical:

* **A finished job is not a result** (GOALS rule 3). `submit` REFUSES without an
  `--expect` and a `--check`, so the acceptance criterion is written down before
  the job runs rather than reconstructed afterwards from what it happened to
  produce. Two checks in RUNS.md were written against files their runs never
  create, and would have reported 0/56 on a healthy run.
* **Exit 0 having written nothing** is the characteristic failure here. `status`
  classifies COMPLETED-but-no-output as `FAIL` instead of trusting the return
  code. Job 67279 exited COMPLETED printing `ok=8 fail=0` — a success line for
  14% of its work.
* **A job with no `--expect` is UNTESTABLE, not failed.** The two used to share
  one status and the count read 142 when 30 things had failed and 112 could not
  be checked at all. `status` reports two axes now — the scheduler's lifecycle
  and the acceptance outcome — and the outcome vocabulary is imported from
  `bestpractices` rather than invented here, so `UNTESTABLE` means the same
  thing as it does in the invariant registry.
* **Does the output already exist?** `submit` refuses when the expected outputs
  are already on disk. The full 493 cross-dock (job 66901) finished at 03:47 and
  was written up that evening as "not launched"; the relaunch would have
  rewritten a complete 48-chunk dataset under a 40-chunk split. Its pre-launch
  validation passed every check it had — because every check described the
  *inputs*. FINDINGS §32.
* **No inline `--wrap`.** A script path is required, so the command is on disk
  and rerunnable. Job 67279 was an inline wrap submitted from ~, which is why it
  was a one-shot nobody could re-run.
* **Script identity is recorded** (path + md5). A duplicate `code/cluster/`
  tree was created tonight; jobs pointed at the copy, so edits to the canonical
  script would never have reached them.
* **Drift is reported.** `status` lists live jobs missing from the registry —
  `check_crashes.sh` has been saying "RUNS.md marks 4 running; thira queue holds
  18" and nothing consumed that.

The registry is an append-only JSONL event log, not a rewritten state file, so
several agents submitting at once cannot clobber each other's entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
REGISTRY = REPO / "results" / "jobs" / "registry.jsonl"
CANONICAL_REMOTE_DIR = "/work/ben/al_mace/cluster"
DEFAULT_REMOTE = "thira"

# Terminal states as SLURM reports them. CANCELLED is deliberately terminal but
# not a failure: cancelling on purpose is a disposition, and treating it as a
# crash is what made an earlier resume's action list mostly noise.
SLURM_OK = {"COMPLETED"}
# THE OUTCOME VOCABULARY IS NOT DEFINED HERE. `bestpractices/__init__.py`
# already declares PASS / FAIL / UNTESTABLE / N/A for this repository, CLAUDE.md
# already records "UNTESTABLE is never a pass", and ten other files already use
# it. This module used to carry nine strings of its own -- SILENT_FAIL,
# REJECTED, AWAITING_VERIFY, ACCEPTED, SUPERSEDED, ... -- none of them validated
# and none shared, which is how 112 jobs that nobody could check came to be
# printed with the same marker as 30 that failed.
#
# The four-outcome split is not this project's invention either: it is
# JUnit/pytest's `failure` against `error`, and GNU Automake's `FAIL` against
# `ERROR` (exit 99). Importing it costs nothing and means a status here means
# the same thing as a status in the invariant registry.
sys.path.insert(0, str(REPO))
from bestpractices import PASS, FAIL, UNTESTABLE, NA   # noqa: E402

SLURM_BAD = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}
SLURM_LIVE = {"PENDING", "RUNNING", "SUSPENDED", "REQUEUED", "CONFIGURING"}
# CANCELLED is terminal but is NOT a failure, and this cost a false positive on
# the night this file was written: job 67436 was cancelled at 00:00:00 elapsed
# and replaced, and it surfaced as SILENT_FAIL because "cancelled" fell through
# to "completed, wrote nothing". `check_crashes.sh` already carried a comment
# about exactly this trap -- six scancelled tasks once made up most of a resume's
# action list -- and this tool reproduced the bug anyway. An explicit CANCELLED
# stamp IS a disposition. sacct reports it as "CANCELLED by <uid>", so match on
# the prefix, not equality.
SLURM_CANCELLED_PREFIX = "CANCELLED"


# ----------------------------------------------------------------- helpers


def run(cmd: list[str] | str, timeout: int = 120) -> tuple[int, str]:
    """Run a command, returning (rc, combined output). Never raises."""
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - a registry must not die on this
        return 1, str(exc)


def ssh(remote: str, cmd: str, timeout: int = 120) -> tuple[int, str]:
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", remote, cmd],
        timeout=timeout,
    )


def append_event(ev: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    ev.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    line = json.dumps(ev, sort_keys=True) + "\n"
    # O_APPEND with a single small write is atomic enough for concurrent agents;
    # a rewritten JSON state file would not be.
    fd = os.open(REGISTRY, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def load_state() -> dict[str, dict]:
    """Fold the event log into per-job state. Later events win."""
    jobs: dict[str, dict] = {}
    if not REGISTRY.exists():
        return jobs
    for ln in REGISTRY.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue  # a corrupt line must not blind the whole registry
        jid = str(ev.get("job_id", ""))
        if not jid:
            continue
        jobs.setdefault(jid, {"job_id": jid, "events": []})
        jobs[jid]["events"].append(ev)
        if ev.get("event") == "submit":
            jobs[jid].update(
                {k: v for k, v in ev.items() if k not in ("event", "ts")}
            )
            jobs[jid]["submitted"] = ev.get("ts")
        elif ev.get("event") == "verify":
            jobs[jid]["verdict"] = ev.get("verdict")
            jobs[jid]["verified"] = ev.get("ts")
            jobs[jid]["verify_output"] = ev.get("output", "")
        elif ev.get("event") == "note":
            jobs[jid].setdefault("notes", []).append(ev.get("text", ""))
            # A note that records an acceptance PASS *after* a failing verify
            # supersedes that verdict. Without this, jobs 67394/67435/67444
            # read REJECTED permanently: `cmd_verify` used to run the recorded
            # check LOCALLY with no remote dispatch, so it returned fail for
            # every cluster job whatever its output. The bug was fixed the same
            # day; the three verdicts were never re-run, and `classify()` read
            # only the last verify event — so the registry asserted three
            # successful jobs had failed, in the tool whose whole purpose is
            # that a return code is not an outcome.
            txt = (ev.get("text") or "").upper()
            if "ACCEPTANCE CHECK PASS" in txt:
                jobs[jid]["note_pass"] = ev.get("text", "")
                jobs[jid]["note_pass_ts"] = ev.get("ts")
        elif ev.get("event") == "supersede":
            jobs[jid]["superseded_by"] = ev.get("by", "")
            jobs[jid]["supersede_reason"] = ev.get("reason", "")
    return jobs


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- queries


def slurm_state(job_id: str, remote: str) -> tuple[str, str]:
    """Aggregate state for a job or job array -> (state, detail).

    Arrays are the normal case here, so a single job id maps to many task rows.
    Any live task keeps the whole thing live; otherwise any bad task makes it
    bad. Reporting an array as COMPLETED while 30 of its tasks are still
    pending is exactly the misreading that produced tonight's wrong ETA.
    """
    rc, out = ssh(
        remote,
        f"sacct -j {shlex.quote(job_id)} -X -n -P -o State 2>/dev/null",
    )
    states = [s.split()[0] for s in out.splitlines() if s.strip()] if rc == 0 else []
    if not states:
        return "UNKNOWN", "no sacct record"
    live = [s for s in states if s in SLURM_LIVE]
    bad = [s for s in states if s in SLURM_BAD]
    ok = [s for s in states if s in SLURM_OK]
    cancelled = [s for s in states if s.startswith(SLURM_CANCELLED_PREFIX)]
    detail = (f"{len(ok)} ok, {len(bad)} bad, {len(live)} live, "
              f"{len(cancelled)} cancelled of {len(states)}")
    if live:
        return "RUNNING", detail
    if bad:
        return "FAILED", detail
    if ok:
        return "COMPLETED", detail
    if cancelled:
        # Every task cancelled and none succeeded: a deliberate stop, not a
        # silent failure. Reported so it is visible, but never as [ACTION].
        return "CANCELLED", detail
    return states[0], detail


def outputs_present(expects: list[str], remote: str | None) -> tuple[bool, str]:
    """True if every expected glob matches at least one NON-EMPTY file.

    Non-empty matters: a zero-byte SDF stub already read as a finished docking
    once, and five array tasks 'succeeded' on it.
    """
    if not expects:
        return False, "no --expect recorded"
    parts = []
    all_ok = True
    for g in expects:
        if remote:
            # -s tests existence AND non-zero size.
            rc, out = ssh(
                remote,
                f"n=0; for f in {g}; do [ -s \"$f\" ] && n=$((n+1)); done; echo $n",
            )
            n = int(out.strip() or 0) if rc == 0 and out.strip().isdigit() else 0
        else:
            n = sum(1 for p in Path("/").glob(g.lstrip("/")) if p.is_file() and p.stat().st_size > 0)
        parts.append(f"{g} -> {n}")
        if n == 0:
            all_ok = False
    return all_ok, "; ".join(parts)


def classify(job: dict, remote: str) -> tuple[str, str | None, str]:
    """-> (lifecycle, outcome, detail).

    TWO AXES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. `lifecycle` is what the
    scheduler says (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED/UNKNOWN);
    `outcome` is what the acceptance check says (PASS/FAIL/UNTESTABLE/N/A, from
    `bestpractices`). Collapsing them into one string is what this function used
    to do, and it is why a job nobody could check read the same as a job that
    broke. A running job HAS no outcome yet; that is `None`, not a verdict.

    THE ORDER BELOW IS THE ARGUMENT, so it is worth reading as one:

      superseded            -> N/A. Replaced work is not a failure, and N/A must
                               carry its reason -- `bestpractices.Result`
                               enforces that, so the reason is built here.
      recorded pass         -> PASS.
      live or cancelled     -> no outcome. Not resolved, not a verdict.
      scheduler says FAILED -> FAIL, **whether or not an --expect was declared**.
                               sacct's own record is evidence; a job that failed
                               did not become uncheckable by nobody having
                               written down what it should produce.
      no sacct record       -> UNTESTABLE. Nothing left to check against.
      no --expect recorded  -> UNTESTABLE. This is the one that matters: a check
                               with nothing to assert cannot fail, and a check
                               that cannot fail has not checked anything
                               (`bestpractices/__init__.py`, and FINDINGS 3.7,
                               where a metal-geometry check reported
                               "closest inf A, PASS" with no hydrogens to test).
      outputs absent        -> FAIL. The headline case: exit 0, nothing produced.
      acceptance failed     -> FAIL.
      otherwise             -> no outcome. Outputs are there and nobody has run
                               the check; that is work pending, not a result.
    """
    if job.get("superseded_by"):
        return ("SUPERSEDED", NA,
                f"superseded by {job['superseded_by']}: "
                f"{job.get('supersede_reason') or 'no reason recorded'}")
    if job.get("verdict") == "pass":
        return "COMPLETED", PASS, "acceptance check passed"
    # A later note recording a PASS overrides an earlier failing verdict. Ordered
    # BEFORE the slurm path so it applies whatever the queue now says -- the jobs
    # this exists for are months old.
    #
    # CORRECTED 2026-08-04: this comment used to add "and long gone from
    # `sacct`". That is false, and something rested on it -- ERRORS.md #92 calls 8
    # FAILED jobs "permanently undispositionable" on exactly that belief.
    # Measured: sacct on thira still answers for job 55456, submitted
    # 2026-01-15, and still carries WorkDir AND SubmitLine. A job with no log
    # file is not a job with no recipe. See `jobreg.py recover`.
    if job.get("note_pass") and job.get("verdict") == "fail":
        return ("COMPLETED", PASS,
                "verify failed but a later note records a PASS: "
                + job["note_pass"][:110])

    lifecycle, detail = slurm_state(job["job_id"], remote)
    if lifecycle in ("RUNNING", "CANCELLED"):
        return lifecycle, None, detail
    if lifecycle == "FAILED":
        return lifecycle, FAIL, detail
    if lifecycle == "UNKNOWN":
        return lifecycle, UNTESTABLE, "no sacct record, so nothing to check against"

    expects = job.get("expect", [])
    if not expects:
        return (lifecycle, UNTESTABLE,
                "no --expect recorded, so no check here can fail; "
                "re-register with `jobreg.py adopt <id> --expect ... --check ...` "
                "if the outputs are still worth claiming")
    have, odetail = outputs_present(expects, job.get("remote"))
    if not have:
        return lifecycle, FAIL, f"exit 0 but {odetail}"
    if job.get("verdict") == "fail":
        return lifecycle, FAIL, f"acceptance check failed; outputs: {odetail}"
    return lifecycle, None, f"outputs present, not yet verified ({odetail})"


# ---------------------------------------------------------------- commands


def cmd_submit(a: argparse.Namespace) -> int:
    script = Path(a.script)
    if not script.is_file():
        print(f"REFUSED: {script} is not a file. Inline --wrap is not accepted —")
        print("         the command must be on disk to be rerunnable.")
        return 2
    if not a.expect:
        print("REFUSED: --expect is required. Name the output(s) the run must")
        print("         produce, or 'finished' cannot be distinguished from 'worked'.")
        return 2
    if not a.check:
        print("REFUSED: --check is required. A finished job is not a result")
        print("         (GOALS rule 3); write the acceptance check up front.")
        return 2

    have, detail = outputs_present(a.expect, a.remote if not a.local else None)
    if have and not (a.resume or a.force):
        print("REFUSED: the expected outputs ALREADY EXIST —")
        print(f"         {detail}")
        print("  This is the check that would have caught job 66901 being")
        print("  relaunched after it had already finished (FINDINGS §32).")
        print("  Pass --resume if the job skips completed work, or --force to")
        print("  overwrite deliberately.")
        return 3

    remote_script = a.remote_script or f"{CANONICAL_REMOTE_DIR}/{script.name}"
    if not a.local and not remote_script.startswith(CANONICAL_REMOTE_DIR):
        print(f"WARNING: {remote_script} is outside {CANONICAL_REMOTE_DIR}.")
        print("         Jobs pointing at a duplicate script tree do not pick up")
        print("         edits to the canonical one.")

    if a.dry_run:
        print(f"[dry-run] would submit {remote_script}")
        print(f"[dry-run] expect: {a.expect}")
        print(f"[dry-run] check : {a.check}")
        return 0

    # Ship the script so the registered md5 is what actually runs.
    if not a.local:
        rc, out = run(["scp", "-q", str(script), f"{a.remote}:{remote_script}"])
        if rc != 0:
            print(f"FAILED to copy script: {out}")
            return 1

    env = " ".join(a.env or [])
    sb = f"{env} sbatch --parsable {' '.join(a.sbatch_arg or [])} {remote_script}"
    rc, out = ssh(a.remote, f"cd {CANONICAL_REMOTE_DIR} && {sb}") if not a.local else run(sb)
    if rc != 0 or not out.strip():
        print(f"SUBMIT FAILED: {out}")
        return 1
    job_id = out.strip().splitlines()[-1].split(";")[0].strip()

    append_event(
        {
            "event": "submit",
            "job_id": job_id,
            "script": str(script.relative_to(REPO)) if script.is_relative_to(REPO) else str(script),
            "script_md5": md5(script),
            "remote_script": None if a.local else remote_script,
            "remote": None if a.local else a.remote,
            "expect": a.expect,
            "check": a.check,
            "question": a.question,
            "sbatch_arg": a.sbatch_arg or [],
            "env": a.env or [],
        }
    )
    print(f"submitted {job_id} and registered")
    print(f"  expect: {a.expect}")
    print(f"  check : {a.check}")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    jobs = load_state()
    rows = []
    for jid, job in sorted(jobs.items(), key=lambda kv: kv[0]):
        if (job.get("verdict") == "pass" or job.get("superseded_by")) and not a.all:
            continue  # accepted and replaced work is done; show only what needs a decision
        lifecycle, outcome, detail = classify(job, a.remote)
        rows.append((jid, lifecycle, outcome,
                     job.get("question") or job.get("script", ""), detail))

    if rows:
        # SORTED BY OUTCOME, worst first. A list where 112 uncheckable entries
        # sit above 30 failures is a list that trains you to scroll past the
        # failures -- the mechanism that killed `check_crashes.sh` four times.
        order = {FAIL: 0, UNTESTABLE: 1, None: 2, NA: 3, PASS: 4}
        rows.sort(key=lambda r: (order.get(r[2], 9), r[0]))
        print(f"{'JOB':>12}  {'OUTCOME':<11} {'LIFECYCLE':<10} WHAT")
        for jid, lifecycle, outcome, what, detail in rows:
            mark = {FAIL: "!!", UNTESTABLE: " ?"}.get(outcome, "  ")
            shown = outcome if outcome is not None else "-"
            print(f"{mark}{jid:>10}  {shown:<11} {lifecycle:<10} {str(what)[:60]}")
            if a.verbose or outcome == FAIL:
                print(f"{'':14}{detail}")
    else:
        print("registry: nothing outstanding")

    # Drift: work running that nobody registered. This is the number
    # check_crashes.sh has been printing with no consumer.
    # %F, not %A. For an array task %A is that task's own allocation id
    # (67330), while %F is the array's parent (67266) — the id the registry
    # and sacct use. Keying on %A reported every task of every array as
    # unregistered, which is precisely the cry-wolf noise that trains people
    # to stop reading the alert.
    rc, out = ssh(a.remote, 'squeue -u $USER -h -o "%F"')
    if rc == 0:
        live = {ln.strip() for ln in out.splitlines() if ln.strip()}
        unreg = sorted(live - set(jobs))
        if unreg:
            print(f"\nUNREGISTERED live jobs ({len(unreg)}): {' '.join(unreg[:12])}")
            print("  These have no recorded expectation or acceptance check.")
            print("  Register with: jobreg.py adopt <id> --expect ... --check ...")
    # THREE LINES, NOT ONE NUMBER. The single "needing attention" count summed
    # FAIL and UNTESTABLE together and read 142 when 30 things had failed and
    # 112 could not be checked. Those need different actions -- one is a bug,
    # the other is a missing --expect -- so they are counted apart. UNTESTABLE
    # is printed on its own line and never folded into a pass, per CLAUDE.md.
    n = {k: sum(1 for r in rows if r[2] == k)
         for k in (FAIL, UNTESTABLE, NA, PASS, None)}
    live = sum(1 for r in rows if r[1] == "RUNNING")
    canc = sum(1 for r in rows if r[1] == "CANCELLED")
    unver = sum(1 for r in rows if r[2] is None and r[1] not in ("RUNNING", "CANCELLED"))
    print(f"\n  {FAIL:<11} {n[FAIL]:>4}   a check ran and did not pass")
    print(f"  {UNTESTABLE:<11} {n[UNTESTABLE]:>4}   nothing was declared to check, "
          f"so no check here can fail")
    print(f"  {NA:<11} {n[NA]:>4}   superseded, with a recorded reason"
          + ("" if a.all else "   (filtered from this view; use --all)"))
    print(f"  {PASS:<11} {n[PASS]:>4}   accepted"
          + ("" if a.all else "   (filtered from this view; use --all)"))
    print(f"  {'pending':<11} {unver:>4}   outputs present, acceptance check not run")
    print(f"  {'running':<11} {live:>4}")
    print(f"  {'cancelled':<11} {canc:>4}")
    # THE ROWS MUST ADD UP. A tally that does not account for every line it just
    # printed invites the reader to assume the remainder is fine, which is the
    # same mistake in miniature as the count this replaced. Asserted, not hoped.
    total = n[FAIL] + n[UNTESTABLE] + n[NA] + n[PASS] + unver + live + canc
    if total != len(rows):
        print(f"  !! {total} counted against {len(rows)} listed — the tally is "
              f"missing a category and cannot be trusted")
    else:
        print(f"  {'':11} {'':>4}   = {len(rows)} listed")
    return 0


def cmd_adopt(a: argparse.Namespace) -> int:
    """Register a job that was submitted outside the registry."""
    if not a.expect or not a.check:
        print("REFUSED: adopt still requires --expect and --check.")
        return 2
    append_event(
        {
            "event": "submit",
            "job_id": a.job_id,
            "script": a.script or "(adopted - submitted outside jobreg)",
            "script_md5": "",
            "remote": None if a.local else a.remote,
            "expect": a.expect,
            "check": a.check,
            "question": a.question,
            "adopted": True,
        }
    )
    print(f"adopted {a.job_id}")
    return 0


def cmd_recover(a: argparse.Namespace) -> int:
    """Reconstruct what a job actually ran, from sacct, when no log survives.

    The registry's own failure mode is a job submitted outside it: `status`
    can then say only "unregistered". ERRORS.md #92 concluded from that that 8
    FAILED jobs were "permanently undispositionable — ad-hoc srun probes with
    no log file at all". The log part is true; the conclusion is not. SLURM's
    accounting keeps `SubmitLine` and `WorkDir` long after the log is gone, and
    those two fields ARE the recipe — verified back to 2026-01-15 on thira.

    This prints them so a disposition can be written from evidence, and so an
    `adopt` can be filled in rather than guessed at.
    """
    fields = "JobID,JobName,State,ExitCode,Elapsed,Submit,WorkDir,SubmitLine"
    query = (f"sacct -u $USER -X -n -P -o {fields} "
             f"{'-S ' + shlex.quote(a.since) if a.since else ''} 2>/dev/null")
    if a.job_id:
        query = (f"sacct -j {shlex.quote(a.job_id)} -X -n -P -o {fields} "
                 f"2>/dev/null")
    rc, out = ssh(a.remote, query, timeout=180)
    if rc != 0:
        print(f"sacct unavailable on {a.remote} (rc={rc}) — cannot recover.")
        return 1
    rows = [r.split("|") for r in out.splitlines() if r.strip()]
    if a.state:
        rows = [r for r in rows if len(r) > 2 and r[2].startswith(a.state)]
    if not rows:
        print("no sacct record matched — this really is unrecoverable.")
        return 1
    for r in rows:
        r += [""] * (8 - len(r))
        jid, name, state, ec, el, sub, wd, line = r[:8]
        print(f"\n  job {jid}  {name}  {state} ({ec})  elapsed {el}  submitted {sub}")
        print(f"    WorkDir    {wd or '(not recorded)'}")
        print(f"    SubmitLine {line or '(not recorded)'}")
        if not line:
            # Say which half is missing. "No recipe" and "no log" are different
            # findings and only one of them is terminal.
            print("    -> no SubmitLine: this one genuinely has no recipe.")
    print(f"\n{len(rows)} job(s) recovered from sacct on {a.remote}.")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    jobs = load_state()
    targets = list(jobs) if a.job_id == "all" else [a.job_id]
    rc_all = 0
    for jid in targets:
        job = jobs.get(jid)
        if not job:
            print(f"{jid}: not registered")
            rc_all = 1
            continue
        lifecycle, _outcome, detail = classify(job, a.remote)
        if lifecycle in ("RUNNING", "UNKNOWN") and not a.force:
            print(f"{jid}: {lifecycle} — not verifying yet ({detail})")
            continue
        check = job.get("check")
        if not check:
            print(f"{jid}: no check recorded")
            rc_all = 1
            continue
        # WHERE the check runs is part of the check. This ran the recorded
        # command on paros unconditionally, so every cluster job whose check
        # touched /work/... returned FAIL regardless of its outputs -- Q22's
        # three jobs all "failed" with 490/490/48 files sitting on disk. A
        # check that always fails is exactly as useless as one that always
        # passes, and it is worse than having none, because it teaches you to
        # ignore the verdict.
        #
        # A job registered with a remote runs its check there, unless the
        # recorded command already dispatches itself (starts with ssh) or the
        # caller overrides with --local.
        remote_host = job.get("remote")
        run_remote = bool(remote_host) and not a.local and not check.lstrip().startswith("ssh ")
        where = f"on {remote_host}" if run_remote else "locally"
        print(f"{jid}: running check {where}: {check}")
        if run_remote:
            rc, out = ssh(remote_host, f"cd {CANONICAL_REMOTE_DIR} && {check}",
                          timeout=a.timeout)
        else:
            rc, out = run(check, timeout=a.timeout)
        verdict = "pass" if rc == 0 else "fail"
        append_event(
            {
                "event": "verify",
                "job_id": jid,
                "verdict": verdict,
                "rc": rc,
                "output": out[-4000:],
            }
        )
        print(f"{jid}: {verdict.upper()} (rc={rc})")
        # `--verbose` PRINTS THE WHOLE CHECK OUTPUT, and the 20-line tail is
        # only for the terse form. Measured on 2026-08-25: job 72203's
        # acceptance report ended "267/369 runs accepted" and named ONE of the
        # 102 refused runs, because the tail cut the rest. Every verify rule in
        # `workflow/rules/` pipes `--verbose` into the file that IS the
        # acceptance record, so the names of 101 refused runs existed nowhere on
        # disk. A verdict whose evidence is truncated cannot be acted on.
        if a.verbose:
            print(out.rstrip("\n"))
        elif rc != 0:
            print("\n".join(out.splitlines()[-20:]))
        rc_all = rc_all or rc
    return rc_all


def cmd_show(a: argparse.Namespace) -> int:
    job = load_state().get(a.job_id)
    if not job:
        print(f"{a.job_id}: not registered")
        return 1
    lifecycle, outcome, detail = classify(job, a.remote)
    print(f"job {a.job_id}   {outcome or '-'}   {lifecycle}   {detail}")
    for k in ("question", "script", "script_md5", "remote_script", "remote",
              "expect", "check", "submitted", "verified", "verdict"):
        if job.get(k):
            print(f"  {k:14} {job[k]}")
    for n in job.get("notes", []):
        print(f"  note           {n}")
    if job.get("verify_output"):
        print("  --- last check output ---")
        print("\n".join("  " + l for l in job["verify_output"].splitlines()[-15:]))
    return 0


def cmd_ls(a: argparse.Namespace) -> int:
    jobs = load_state()
    if not jobs:
        print("registry is empty")
        return 0
    print(f"{'JOB':>12}  {'SUBMITTED':<20} {'VERDICT':<8} QUESTION")
    for jid, job in sorted(jobs.items()):
        print(
            f"{jid:>12}  {str(job.get('submitted', ''))[:19]:<20} "
            f"{str(job.get('verdict', '-')):<8} {str(job.get('question', ''))[:60]}"
        )
    return 0


def cmd_note(a: argparse.Namespace) -> int:
    append_event({"event": "note", "job_id": a.job_id, "text": a.text})
    print(f"noted on {a.job_id}")
    return 0


def cmd_supersede(a: argparse.Namespace) -> int:
    """Mark a job replaced, so it stops asking to be verified."""
    append_event(
        {"event": "supersede", "job_id": a.job_id, "by": a.by, "reason": a.reason or ""}
    )
    print(f"{a.job_id} superseded by {a.by}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--remote", default=DEFAULT_REMOTE)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="pre-flight, submit and register")
    s.add_argument("script")
    s.add_argument("--expect", action="append", help="glob the run MUST produce (repeatable)")
    s.add_argument("--check", help="command that must exit 0 for acceptance")
    s.add_argument("--question", help="what this run decides")
    s.add_argument("--sbatch-arg", action="append")
    s.add_argument("--env", action="append", help="VAR=value passed to sbatch")
    s.add_argument("--remote-script", help=f"default {CANONICAL_REMOTE_DIR}/<name>")
    s.add_argument("--resume", action="store_true", help="job skips completed work")
    s.add_argument("--force", action="store_true", help="overwrite existing outputs")
    s.add_argument("--local", action="store_true", help="submit here, not on --remote")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("adopt", help="register an already-submitted job")
    s.add_argument("job_id")
    s.add_argument("--expect", action="append")
    s.add_argument("--check")
    s.add_argument("--question")
    s.add_argument("--script")
    s.add_argument("--local", action="store_true")
    s.set_defaults(fn=cmd_adopt)

    s = sub.add_parser("recover",
                       help="reconstruct WorkDir+SubmitLine from sacct for a job "
                            "whose log is gone")
    s.add_argument("job_id", nargs="?", help="omit to sweep with --since/--state")
    s.add_argument("--since", default="2026-01-15",
                   help="sacct -S window when sweeping (default 2026-01-15, the "
                        "earliest record thira still holds)")
    s.add_argument("--state", help="keep only states with this prefix, e.g. FAILED")
    s.set_defaults(fn=cmd_recover)

    s = sub.add_parser("status", help="reconcile registry against the live queue")
    s.add_argument("-a", "--all", action="store_true", help="include accepted")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("verify", help="run the recorded acceptance check")
    s.add_argument("job_id", help="job id or 'all'")
    s.add_argument("--force", action="store_true", help="verify even if still running")
    s.add_argument("--local", action="store_true",
                   help="run the check here even for a job registered with a remote")
    s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("show")
    s.add_argument("job_id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("ls")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("note")
    s.add_argument("job_id")
    s.add_argument("text")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("supersede", help="mark a job replaced by another")
    s.add_argument("job_id")
    s.add_argument("--by", required=True, help="job id that replaced it")
    s.add_argument("--reason")
    s.set_defaults(fn=cmd_supersede)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
