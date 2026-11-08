"""Experiment runner shared by all notebooks: job queue, checkpoints, ledger, time budget, verification.

A job is a function fn(ctx, cfg) -> metrics dict. The runner owns everything around it:

  TODO -> RUNNING -> DONE | INTERRUPTED | FAILED          (DEFERRED: a required job is not done yet)
                                                          (BLOCKED: its config or code changed)

Guarantees:
  * A finished job is never repeated, and DONE is written last, after hashing every artefact.
  * Checkpoints are written to a temporary file, verified, then renamed; a bad latest falls back to previous.
  * A job resumes only under the same config hash and code hash.
  * The run stops cleanly before the session limit, and on SIGTERM.
  * One job failing never stops the others.
  * Outputs record hardware, never who ran the notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KEEP_CKPT_AFTER_DONE = {"best.bin"}
MAX_ATTEMPTS = 3


class Interrupted(Exception):
    """The session budget ended or a stop signal arrived; the job resumes next session."""


class CheckpointCorrupt(Exception):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def code_hash(root: Path) -> str:
    """Hash of the research code. Every notebook must report the same value."""
    h = hashlib.sha256()
    for sub in ("src/amlh", "scripts"):
        for f in sorted((root / sub).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=str).encode())


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def environment() -> dict:
    """Hardware and software only. Deliberately records no user, account or host identity."""
    env = {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        env["torch"] = None
    try:
        with open("/proc/meminfo") as fh:
            env["ram_gb"] = round(int(fh.readline().split()[1]) / 1e6, 1)
    except OSError:
        env["ram_gb"] = None
    for mod in ("numpy", "polars", "lightgbm", "numba", "torch_geometric"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    return env


@dataclass
class JobSpec:
    job_id: str
    fn: Callable | str
    cfg: dict
    tier: str = "P0"
    requires: list[str] = field(default_factory=list)
    est_seconds: float = 0.0


class Context:
    """What a job sees: its folder, checkpoints, logs and the stop check."""

    def __init__(self, job_dir: Path, runner: "Runner", spec: JobSpec):
        self.dir = job_dir
        self.runner = runner
        self.spec = spec
        self.ckpt_dir = job_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def check(self, need_seconds: float = 0.0):
        """Call between units of work. Raises Interrupted if the next unit might not finish."""
        self.runner.heartbeat(self.spec.job_id)
        self.runner.n_checks += 1
        if self.runner.kill_after and self.runner.n_checks >= self.runner.kill_after:
            os._exit(137)  # the pilot's forced kill: no cleanup, as when a session ends abruptly
        if self.runner.stop_requested or self.runner.remaining() < need_seconds:
            raise Interrupted(f"remaining {self.runner.remaining():.0f}s, need {need_seconds:.0f}s")

    def log(self, row: dict):
        path = self.dir / "log.jsonl"
        with open(path, "a") as fh:
            fh.write(json.dumps({"ts": time.time(), **row}, default=str) + "\n")

    def save_checkpoint(self, write: Callable[[Path], None], name: str = "latest.bin", meta: dict | None = None):
        """write(path) serialises the state. The hash is verified before the old checkpoint is replaced."""
        tmp = self.ckpt_dir / (name + ".tmp")
        write(tmp)
        digest = sha256_file(tmp)
        if sha256_file(tmp) != digest:
            raise CheckpointCorrupt(name)
        target = self.ckpt_dir / name
        if name == "latest.bin" and target.exists():
            os.replace(target, self.ckpt_dir / "previous.bin")
            prev_side = self.ckpt_dir / "latest.bin.json"
            if prev_side.exists():
                os.replace(prev_side, self.ckpt_dir / "previous.bin.json")
        os.replace(tmp, target)
        atomic_write_json(self.ckpt_dir / (name + ".json"), {"sha256": digest, "meta": meta or {}, "ts": time.time()})
        self.runner.after_checkpoint()

    def load_checkpoint(self, read: Callable[[Path], Any], names=("latest.bin", "previous.bin")):
        """Returns the newest checkpoint whose hash verifies and which read() can parse, or None."""
        for name in names:
            path, side = self.ckpt_dir / name, read_json(self.ckpt_dir / (name + ".json"))
            if not path.exists() or side is None:
                continue
            if sha256_file(path) != side["sha256"]:
                self.log({"event": "checkpoint_corrupt", "name": name})
                continue
            try:
                return read(path)
            except Exception as exc:  # a verified file that cannot be parsed is also unusable
                self.log({"event": "checkpoint_unreadable", "name": name, "error": repr(exc)})
        return None

    def save_state(self, state: dict):
        """Small JSON progress for jobs whose unit of work is not a model."""
        atomic_write_json(self.dir / "state.json", state)
        self.runner.after_checkpoint()

    def load_state(self, default=None):
        return read_json(self.dir / "state.json", default)


class Runner:
    def __init__(self, out_dir: Path, notebook: str, budget_seconds: float, code_root: Path | None = None,
                 code_version: str | None = None, registry: dict | None = None, mirror=None, retries: int = MAX_ATTEMPTS):
        self.out = Path(out_dir)
        self.jobs_dir = self.out / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.notebook = notebook
        self.started = time.time()
        self.budget = budget_seconds
        self.stop_requested = False
        self.code = code_version or (code_hash(code_root) if code_root else "unversioned")
        self.registry = registry or {}
        self.mirror = mirror
        self.retries = retries
        self._last_beat = 0.0
        self.kill_after = int(os.environ.get("AMLH_KILL_AFTER_CHECKS", "0"))
        self.n_checks = 0
        self.session_outcomes: dict[str, str] = {}

    def install_signal_handlers(self):
        def handler(signum, frame):
            self.stop_requested = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except ValueError:  # not the main thread
                pass

    def remaining(self) -> float:
        return self.budget - (time.time() - self.started)

    def ledger(self, event: str, job_id: str = "", **extra):
        with open(self.out / "ledger.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": time.time(), "event": event, "job": job_id, "code": self.code, **extra}, default=str) + "\n")

    def heartbeat(self, job_id: str):
        now = time.time()
        if now - self._last_beat > 60:
            self._last_beat = now
            path = self.jobs_dir / job_id / "status.json"
            st = read_json(path, {})
            st["heartbeat"] = now
            atomic_write_json(path, st)

    def after_checkpoint(self):
        if self.mirror is not None:
            self.mirror.maybe_sync()

    def write_manifest(self, extra: dict | None = None):
        atomic_write_json(self.out / "RUN_MANIFEST.json", {
            "notebook": self.notebook, "code_hash": self.code, "environment": environment(),
            "budget_seconds": self.budget, **(extra or {})})

    def restore(self, roots: list[Path]):
        """Pull finished jobs and the newest unfinished progress from earlier sessions' outputs."""
        restored = 0
        for root in roots:
            root = Path(root)
            for jobs in sorted(root.rglob("jobs")):
                if not jobs.is_dir() or jobs == self.jobs_dir or "aml_pilot" in jobs.parts:
                    continue
                for src in jobs.iterdir():
                    dst = self.jobs_dir / src.name
                    if not src.is_dir() or verify_job(dst)[0]:
                        continue
                    st_src = read_json(src / "status.json", {})
                    if not verify_job(src)[0] and st_src.get("status") in ("FAILED", "BLOCKED"):
                        continue
                    if verify_job(src)[0] or self._newer(src, dst):
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        restored += 1
        self.ledger("restore", n=restored, roots=[str(r) for r in roots])
        return restored

    @staticmethod
    def _newer(src: Path, dst: Path) -> bool:
        def stamp(p):
            side = read_json(p / "ckpt" / "latest.bin.json")
            state = p / "state.json"
            vals = [side["ts"]] if side else []
            if state.exists():
                vals.append(state.stat().st_mtime)
            return max(vals) if vals else 0.0
        return stamp(src) > stamp(dst)

    def _resolve(self, fn):
        return self.registry[fn] if isinstance(fn, str) else fn

    def run_job(self, spec: JobSpec) -> str:
        d = self.jobs_dir / spec.job_id
        d.mkdir(parents=True, exist_ok=True)
        chash = config_hash(spec.cfg)
        if verify_job(d)[0]:
            return "DONE"
        st = read_json(d / "status.json", {})
        if st and st.get("config_hash") not in (None, chash):
            self.ledger("blocked", spec.job_id, reason="config changed")
            return self._set(d, spec, "BLOCKED", chash, st, reason=f"config hash {st.get('config_hash')} != {chash}")
        has_progress = (d / "ckpt" / "latest.bin").exists() or (d / "state.json").exists()
        if st and has_progress and st.get("code_hash") not in (None, self.code):
            self.ledger("blocked", spec.job_id, reason="code changed")
            return self._set(d, spec, "BLOCKED", chash, st, reason=f"code hash {st.get('code_hash')} != {self.code}")
        missing = [r for r in spec.requires if not verify_job(self.jobs_dir / r)[0]]
        if missing:
            return self._set(d, spec, "DEFERRED", chash, st, reason=f"waiting for {missing}")
        if st.get("attempts", 0) >= self.retries and st.get("status") == "FAILED":
            return "FAILED"
        if spec.est_seconds and self.remaining() < spec.est_seconds:
            return self._set(d, spec, "TODO", chash, st, reason="not enough session time left")
        atomic_write_json(d / "config.json", {"job_id": spec.job_id, "tier": spec.tier, "cfg": spec.cfg})
        st["attempts"] = st.get("attempts", 0) + 1
        self._set(d, spec, "RUNNING", chash, st)
        self.ledger("start", spec.job_id, attempt=st["attempts"], resumed=has_progress)
        t0 = time.time()
        ctx = Context(d, self, spec)
        try:
            metrics = self._resolve(spec.fn)(ctx, spec.cfg) or {}
            atomic_write_json(d / "metrics.json", metrics)
            finish_job(d, {"job_id": spec.job_id, "code_hash": self.code, "config_hash": chash,
                           "seconds_last_session": time.time() - t0})
            self.ledger("done", spec.job_id, seconds=time.time() - t0)
            self._set(d, spec, "DONE", chash, st)
            if self.mirror is not None:
                self.mirror.maybe_sync()
            return "DONE"
        except Interrupted as exc:
            st["attempts"] -= 1  # running out of session time is not a failure
            self.ledger("interrupted", spec.job_id, reason=str(exc))
            return self._set(d, spec, "INTERRUPTED", chash, st, reason=str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            self.ledger("failed", spec.job_id, error=repr(exc), attempt=st["attempts"])
            status = "FAILED" if st["attempts"] >= self.retries else "TODO"
            return self._set(d, spec, status, chash, st, reason=repr(exc), traceback=tb)

    def _set(self, d: Path, spec: JobSpec, status: str, chash: str, st: dict, **extra) -> str:
        st.update({"status": status, "job_id": spec.job_id, "tier": spec.tier, "config_hash": chash, "code_hash": self.code,
                   "updated": time.time(), **extra})
        atomic_write_json(d / "status.json", st)
        self.session_outcomes[spec.job_id] = status
        return status

    def run(self, jobs: list[JobSpec], tiers=("P0", "P1", "P2")) -> dict:
        """Runs in tier order, retrying failures, until the queue or the session budget ends."""
        queue = sorted([j for j in jobs if j.tier in tiers], key=lambda j: (j.tier, jobs.index(j)))
        self.ledger("session_start", n_jobs=len(queue), budget=self.budget)
        for _ in range(self.retries + 1):
            pending = 0
            for spec in queue:
                if self.stop_requested or self.remaining() <= 0:
                    break
                outcome = self.run_job(spec)
                pending += outcome in ("TODO", "DEFERRED")
            if pending == 0 or self.stop_requested or self.remaining() <= 0:
                break
        return self.summarise(queue)

    def summarise(self, queue: list[JobSpec]) -> dict:
        rows = []
        for spec in queue:
            d = self.jobs_dir / spec.job_id
            ok, why = verify_job(d)
            st = read_json(d / "status.json", {})
            rows.append({"job_id": spec.job_id, "tier": spec.tier, "status": "DONE" if ok else st.get("status", "TODO"),
                         "reason": st.get("reason"), "verified": ok, "detail": why if not ok else ""})
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        summary = {"notebook": self.notebook, "code_hash": self.code, "counts": counts, "jobs": rows,
                   "session_seconds": time.time() - self.started, "complete": all(r["verified"] for r in rows)}
        atomic_write_json(self.out / "summary.json", summary)
        lines = [f"# {self.notebook}: execution summary", "", f"- Code version: `{self.code}`",
                 f"- Session time: {summary['session_seconds'] / 3600:.2f} h", f"- Complete: **{summary['complete']}**",
                 f"- Job counts: {counts}", "", "| Job | Tier | Status | Note |", "|---|---|---|---|"]
        lines += [f"| {r['job_id']} | {r['tier']} | {r['status']} | {(r['reason'] or r['detail'] or '')[:90]} |" for r in rows]
        (self.out / "summary.md").write_text("\n".join(lines) + "\n")
        self.ledger("session_end", counts=counts)
        return summary


def _artefacts(job_dir: Path) -> list[Path]:
    skip_names = {"DONE", "status.json"}
    files = []
    for p in sorted(job_dir.rglob("*")):
        if not p.is_file() or p.name in skip_names or p.name.endswith(".tmp"):
            continue
        rel = p.relative_to(job_dir)
        if rel.parts[0] == "ckpt" and p.name.split(".json")[0] not in KEEP_CKPT_AFTER_DONE:
            continue
        files.append(p)
    return files


def finish_job(job_dir: Path, info: dict):
    """Drops resume-only checkpoints, then writes DONE last with a hash of every remaining artefact."""
    ck = job_dir / "ckpt"
    if ck.exists():
        for p in ck.iterdir():
            if p.name.split(".json")[0] not in KEEP_CKPT_AFTER_DONE:
                p.unlink()
    files = {str(p.relative_to(job_dir)).replace("\\", "/"): sha256_file(p) for p in _artefacts(job_dir)}
    atomic_write_json(job_dir / "DONE", {**info, "finished": time.time(), "files": files})


def verify_job(job_dir: Path) -> tuple[bool, str]:
    done = read_json(Path(job_dir) / "DONE")
    if done is None:
        return False, "no DONE marker"
    for rel, digest in done["files"].items():
        p = Path(job_dir) / rel
        if not p.exists():
            return False, f"missing {rel}"
        if sha256_file(p) != digest:
            return False, f"hash mismatch {rel}"
    return True, ""


class KaggleDatasetMirror:
    """Copies the run folder to a private Kaggle dataset, at most once per interval, so a hard session kill
    loses little. Requires the kaggle CLI and a token held in Kaggle Secrets, never in the notebook's code.
    A failed sync is logged and never stops the experiment."""

    def __init__(self, run_dir: Path, slug: str, interval_seconds: float = 3600.0, staging: Path | None = None, command=("kaggle",)):
        self.run_dir = Path(run_dir)
        self.slug = slug
        self.interval = interval_seconds
        self.staging = Path(staging) if staging else self.run_dir.parent / "mirror_staging"
        self.command = list(command)
        self.last = 0.0
        self.created = False
        self.failures = 0

    def maybe_sync(self, force: bool = False) -> bool:
        if not force and time.time() - self.last < self.interval:
            return False
        self.last = time.time()
        try:
            if self.staging.exists():
                shutil.rmtree(self.staging)
            shutil.copytree(self.run_dir, self.staging / self.run_dir.name)
            atomic_write_json(self.staging / "dataset-metadata.json", {"title": self.slug.split("/")[-1], "id": self.slug,
                                                                       "licenses": [{"name": "other"}]})
            base = self.command + ["datasets"]
            args = ["version", "-p", str(self.staging), "-m", f"sync {int(self.last)}", "-r", "tar", "-q"]
            res = subprocess.run(base + args, capture_output=True, text=True, timeout=1200)
            if res.returncode != 0 and not self.created:
                res = subprocess.run(base + ["create", "-p", str(self.staging), "-r", "tar", "-q"], capture_output=True, text=True, timeout=1200)
            self.created = res.returncode == 0 or self.created
            ok = res.returncode == 0
        except (OSError, subprocess.SubprocessError, shutil.Error):
            ok = False
        self.failures += 0 if ok else 1
        return ok


def assert_expected_rows(n_rows: int, expected: int, what: str):
    if n_rows != expected:
        raise ValueError(f"{what}: expected {expected} rows, got {n_rows}")
