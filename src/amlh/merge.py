"""Phase 3: verify five notebooks' outputs, unseal test scores once, and give a verdict on each frozen hypothesis.

Thresholds come from DECISIONS.md D-001, D-008 and D-012. A hypothesis with missing data is reported as pending, never
as passed. The report says what is complete, partial, failed or missing, and never rounds that into a success.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .datasets import Stream
from .evaluate import attack_outcomes, c3b, equal_alert_threshold, paired_diff_ci, heldout_metrics
from .jobs_gbt import first_alert, get_stream, policy_value, read_scores
from .metrics import attempt_outcomes  # noqa: F401
from .plans import CHECKPOINTS, FRAUDGT_MULTI, PLANS, SEEDS
from .prepare import DOCUMENTED_DAYS
from .runner import read_json, verify_job
from .sealing import UnsealLog, is_sealed


# Published minority-class F1 on HI-Small, as cited in the literature review. Which FraudGT variant its configuration
# file corresponds to is not stated in the repository, so that comparison is approximate.
PUBLISHED_F1 = {("HI-Small", "gin"): 0.6479, ("HI-Small", "pna"): 0.6816, ("HI-Small", FRAUDGT_MULTI): 0.7613}


@dataclass
class Collected:
    jobs: dict[str, Path] = field(default_factory=dict)
    manifests: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def collect(roots: list[Path]) -> Collected:
    """Finds every verified job under the given output folders. A job failing verification is reported, not used."""
    c = Collected()
    for root in roots:
        for manifest in Path(root).rglob("RUN_MANIFEST.json"):
            if "aml_pilot" in manifest.parts:  # pilot runs are short and never count as results
                continue
            c.manifests.append(read_json(manifest, {}))
            for d in sorted((manifest.parent / "jobs").iterdir()):
                ok, why = verify_job(d)
                if ok:
                    c.jobs.setdefault(d.name, d)
                elif (d / "DONE").exists():
                    c.problems.append(f"{d.name}: {why}")
    return c


def plan_status(c: Collected) -> dict:
    out = {}
    for name, fn in PLANS.items():
        rows = {"P0": [0, 0], "P1": [0, 0], "P2": [0, 0]}
        for j in fn():
            rows[j.tier][1] += 1
            rows[j.tier][0] += j.job_id in c.jobs
        out[name] = rows
    return out


def checklist(c: Collected) -> list[tuple[str, bool | None, str]]:
    """The verification checklist of NOTEBOOK_METHODOLOGY.md, with None meaning it could not be checked yet."""
    items = []
    codes = {m.get("code_hash") for m in c.manifests}
    items.append(("All notebooks ran the same code", len(codes) == 1 if codes else None, ", ".join(sorted(str(x) for x in codes))))
    done_codes = {read_json(d / "DONE", {}).get("code_hash") for d in c.jobs.values()}
    items.append(("Every finished job has that code", len(done_codes) <= 1 if done_codes else None, ", ".join(sorted(str(x) for x in done_codes))))
    items.append(("No finished job fails its hash check", not c.problems, "; ".join(c.problems[:5])))
    data = {}
    for m in c.manifests:
        for k, v in (m.get("data_hashes") or {}).items():
            data.setdefault(k, set()).add(v)
    bad = [k for k, v in data.items() if len(v) > 1]
    items.append(("Prepared data agrees across notebooks", (not bad) if data else None, ", ".join(bad)))
    return items


def _job(c: Collected, nb: str, ds: str, proto: str, H: str, seed: int, family="wall", tag=""):
    return c.jobs.get(f"{nb}-gbt-{family}-{ds}-{proto}-H{H}{('-' + tag) if tag else ''}-s{seed}")


def load_scores(s: Stream, d: Path) -> tuple[np.ndarray, np.ndarray]:
    return read_scores(d, "scores_val", s, s.va), read_scores(d, "scores_test", s, s.te)


def sweep_results(c: Collected, s: Stream, nb="nb4", family="wall", seeds=SEEDS) -> dict:
    rows = {}
    for h in ["0", "1", "60", "600", "3600", "21600", "86400", "259200", "inf"]:
        for k in seeds:
            d = _job(c, nb, s.name, s.protocol, h, k, family)
            if d is not None:
                sv, st = load_scores(s, d)
                rows[(h, k)] = {"row": heldout_metrics(s, sv, st, h, unsealed=True), "val": sv, "test": st}
    return rows


def _mean(rows: dict, h: str, f) -> float | None:
    vals = [f(r["row"]) for (hh, _), r in rows.items() if hh == h]
    return float(np.mean(vals)) if vals else None


def verdicts_for(c: Collected, s: Stream, rows: dict) -> dict:
    out = {}
    pr = lambda r: r["test_core"]["pr_auc"]  # noqa: E731
    f1 = lambda r: r["test"]["f1"]  # noqa: E731
    cf1 = lambda r: r["test_core"]["f1"]  # noqa: E731
    if _mean(rows, "inf", f1) is not None:
        gap = 100 * (_mean(rows, "inf", f1) - _mean(rows, "inf", cf1))
        out["C1"] = {"value": gap, "pass": gap > 5, "note": "points of F1 added by the tail at the whole-window view"}
    seeds = sorted({k for _, k in rows})
    if all(("inf", k) in rows and ("0", k) in rows for k in seeds) and seeds:
        d = [pr(rows[("inf", k)]["row"]) - pr(rows[("0", k)]["row"]) for k in seeds]
        out["C2"] = {"value": float(np.mean(d)), "pass": min(d) > 0.05, "per_seed": d}
    if all(("1", k) in rows and ("0", k) in rows for k in seeds) and seeds:
        d = [pr(rows[("1", k)]["row"]) - pr(rows[("0", k)]["row"]) for k in seeds]
        out["C4"] = {"value": float(np.mean(d)), "pass": min(d) > 0.02, "per_seed": d}
    scores = {key: (r["val"], r["test"]) for key, r in rows.items()}
    res = c3b(s, scores, seeds, unsealed=True) if seeds else None
    if res is not None:
        out["C3b"] = {"value": res["diff"], "ci": res["ci"], "pass": res["confirmed"], "H_star": res["H_star"], "attempts": res["attempts"]}
    near = [h for h in ("1", "60", "600", "3600") if all((h, k) in rows for k in seeds)]
    if near and all(("86400", k) in rows for k in seeds):
        def per_attempt(h):
            per = []
            for k in seeds:
                from .metrics import threshold_for_budget
                r = rows[(h, k)]
                from .evaluate import heldout_outcomes
                per.append(heldout_outcomes(s, r["test"], h, threshold_for_budget(r["val"], s.val_days, 500), unsealed=True)["value_share"])
            return np.mean(per, axis=0)
        table = {h: per_attempt(h) for h in near}
        best = max(table, key=lambda h: table[h].mean())
        m, lo, hi = paired_diff_ci(table[best], per_attempt("86400"))
        out["C3"] = {"value": m, "ci": [lo, hi], "pass": bool(lo > 0 and (_mean(rows, "86400", pr) or 0) >= (_mean(rows, best, pr) or 1)), "best": best,
                     "note": "descriptive: close to true by construction"}
    out.update(policy_verdict(c, s, seeds))
    return out


def policy_verdict(c: Collected, s: Stream, seeds) -> dict:
    H = np.array([int(x) for x in CHECKPOINTS], dtype=np.int64)
    from .datasets import attempts_in_window
    aid, apos = attempts_in_window(s, s.te)
    ada, fix = [], []
    for k in seeds:
        d = c.jobs.get(f"nb4-policy-{s.name}-{s.protocol}-s{k}")
        if d is None:
            continue
        info = read_json(d / "metrics.json")["budgets"]["500"]
        st = [read_scores(_job(c, "nb4", s.name, s.protocol, h, k), "scores_test", s, s.te) for h in CHECKPOINTS]
        ada.append(policy_value(s, s.te, first_alert(st, info["taus"]), H, aid, apos))
        fix.append(policy_value(s, s.te, first_alert(st, info["fixed_taus"]), H, aid, apos))
    if not ada:
        return {}
    m, lo, hi = paired_diff_ci(np.mean(ada, axis=0), np.mean(fix, axis=0))
    return {"C5": {"value": m, "ci": [lo, hi], "pass": bool(lo > 0), "note": "adaptive minus best fixed at 500 alerts/day; selection on validation"}}


def attack_table(c: Collected, s: Stream, family: str, H: str, W: int, seeds=SEEDS, nb="nb5", budget="500") -> dict:
    """Per-condition, per-attempt outcomes averaged over seeds."""
    table: dict[str, dict] = {}
    for k in seeds:
        d = c.jobs.get(f"{nb}-attack-{family}-{s.name}-H{H}-s{k}") if nb == "nb5" else c.jobs.get(f"nb3-tdgnn-{s.name}-H{H}-s{k}-attack")
        if d is None:
            continue
        tau = read_json(d / "metrics.json")["tau"][budget]
        for f in d.glob(f"attack_W{W}_*.parquet"):
            cond = f.stem.split(f"W{W}_", 1)[1]
            df = pl.read_parquet(f)
            o = attack_outcomes(df["t"].to_numpy(), df["usd"].to_numpy(), df["pattern_id"].to_numpy(), df["score"].to_numpy().astype(np.float64), tau, H,
                                int(s.t.max()))
            table.setdefault(cond, {"recall": [], "detected": [], "value_share": [], "completion_h": []})
            for key in ("recall", "value_share", "completion_h"):
                table[cond][key].append(o[key])
            table[cond]["detected"].append(o["detected"].astype(float))
    return {cond: {k: np.mean(v, axis=0) for k, v in vals.items()} for cond, vals in table.items()}


def suppression(table: dict, attack: str, metric: str = "detected") -> dict | None:
    """Loss beyond the attack's own matched control, paired on attempts."""
    ctrl = f"shift_{attack}"
    if attack not in table or ctrl not in table:
        return None
    m, lo, hi = paired_diff_ci(table[ctrl][metric], table[attack][metric])
    return {"loss": m, "ci": [lo, hi]}


def timing_verdicts(c: Collected, s: Stream) -> dict:
    """C6 and C8, as frozen."""
    out = {}
    strict = attack_table(c, s, "wall", "0", 21600)
    tie = attack_table(c, s, "wall", "1", 21600)
    a, b = suppression(strict, "burst"), suppression(tie, "burst")
    if a and b:
        out["C6"] = {"strict_loss": a, "same_minute_loss": b, "pass": bool(a["ci"][0] > 0 and b["loss"] <= 0.5 * a["loss"])}
    for H in ("0", "1", "3600"):
        wall, logical = attack_table(c, s, "wall", H, 21600), attack_table(c, s, "logical", H, 21600)
        res = {}
        ok = True
        for attack in ("stretch", "blend"):
            w, l = suppression(wall, attack), suppression(logical, attack)
            if not (w and l):
                ok = None
                break
            lw = wall[f"shift_{attack}"]["detected"] - wall[attack]["detected"]
            ll = logical[f"shift_{attack}"]["detected"] - logical[attack]["detected"]
            m, lo, hi = paired_diff_ci(lw, ll)
            res[attack] = {"wall_loss": w["loss"], "logical_loss": l["loss"], "difference": m, "ci": [lo, hi]}
            ok = ok and l["loss"] <= 0.5 * w["loss"] and lo > 0
        if ok is None:
            continue
        pw = [read_job_pr(c, s, "wall", H, k) for k in SEEDS]
        pl_ = [read_job_pr(c, s, "logical", H, k) for k in SEEDS]
        if None in pw or None in pl_:
            continue
        cost = float(np.mean(pl_) / max(np.mean(pw), 1e-12))
        res.update({"clean_pr_ratio": cost})
        out[f"C8@H={H}"] = {**res, "pass": bool(ok and cost >= 0.7)}
    return out


def read_job_pr(c: Collected, s: Stream, family: str, H: str, seed: int) -> float | None:
    nb = "nb4" if family == "wall" else "nb5"
    d = _job(c, nb, s.name, s.protocol, H, seed, family)
    if d is None:
        return None
    sv, st = load_scores(s, d)
    return heldout_metrics(s, sv, st, H, unsealed=True)["test_core"]["pr_auc"]


def tail_invariance(c: Collected, ds: str, seeds=SEEDS) -> dict:
    """Causality check: at strict past, core scores must not depend on whether the tail days are in the file."""
    out = {}
    a, _ = get_stream({"dataset": ds, "protocol": "released"})
    released = {k: _job(c, "nb4", ds, "released", "0", k) for k in seeds}
    held = {k: load_scores(a, d) for k, d in released.items() if d is not None}
    b, _ = get_stream({"dataset": ds, "protocol": "documented"})
    for k, (_, st) in held.items():
        d = _job(c, "nb4", ds, "documented", "0", k)
        if d is None:
            continue
        doc = read_scores(d, "scores_test", b, b.te)
        out[k] = bool(len(doc) == int(a.core_mask.sum()) and np.allclose(st[a.core_mask], doc, atol=1e-5, rtol=1e-5))
    get_stream({"dataset": ds, "protocol": "released"})
    return out


def export_for_analysis(root: Path, s: Stream, rows: dict):
    """Writes results in the layout scripts/08_analyze.py reads, so tables and figures regenerate from these runs."""
    tag = f"gbt_{s.name.lower().replace('-', '_')}_{s.protocol}"
    d = Path(root) / "results" / tag
    d.mkdir(parents=True, exist_ok=True)
    out = [{"H": h, "seed": k, **r["row"]} for (h, k), r in rows.items()]
    (d / "results.json").write_text(json.dumps({"rows": out}, indent=1, default=str))
    for (h, k), r in rows.items():
        np.savez_compressed(d / f"scores_H{h}_s{k}.npz", val=r["val"].astype(np.float32), test=r["test"].astype(np.float32))


def official_audit(c: Collected, s: Stream) -> dict:
    """Gate G1: the published models' own scores, split by region and restricted by cadence. A stock model is scored on the
    whole split graph, so its horizon is the whole window. Cadence scores use the bucket's end as an upper bound on the wait.
    Multi-GNN's official code evaluates in training mode, so its jobs have a second, deterministic set of scores in evaluation
    mode (files ending _eval). Both are reported, the official one first."""
    out = {}
    for jid, d in sorted(c.jobs.items()):
        if f"-{s.name}-" not in jid or not ("-mgnn-" in jid or "-fraudgt-" in jid):
            continue
        cfg = read_json(d / "config.json")["cfg"]
        for suffix, tag in (("", ""), ("_eval", " [eval mode]")):
            if not (d / f"scores_test{suffix}.parquet").exists():
                continue
            val_dir = d if (d / f"scores_val{suffix}.parquet").exists() else c.jobs.get(cfg.get("train_job", ""))
            if val_dir is None or not (val_dir / f"scores_val{suffix}.parquet").exists():
                continue
            try:
                sv, st = read_scores(val_dir, f"scores_val{suffix}", s, s.va), read_scores(d, f"scores_test{suffix}", s, s.te)
            except ValueError as exc:
                out[jid + tag] = {"problem": str(exc)}
                continue
            H = str(cfg["cadence"]) if "cadence" in cfg else "inf"
            row = heldout_metrics(s, sv, st, H, unsealed=True)
            df = pl.read_parquet(d / f"scores_test{suffix}.parquet", columns=["event_id", "sampled"]).join(
                pl.DataFrame({"event_id": s.event_id[s.te], "y": s.y[s.te].astype(np.int8), "p": st}), on="event_id")
            kept = df.filter(pl.col("sampled"))
            flagged = kept["p"].to_numpy() >= 0.5
            truth = kept["y"].to_numpy() == 1
            tp = int((flagged & truth).sum())
            official = 2 * tp / max(int(flagged.sum()) + int(truth.sum()), 1)
            out[jid + tag] = {"H": H, "official_style_f1": official, "coverage": float(df["sampled"].mean()), "train_cadence": cfg.get("train_cadence"),
                              "published_reference": PUBLISHED_F1.get((s.name, cfg.get("model") or cfg.get("config"))) if not suffix else None, "row": row}
    return out


def leakage_check(c: Collected, s: Stream) -> dict:
    """Shuffled-label runs must collapse to prevalence. A failure here invalidates positive-horizon results."""
    out = {}
    prevalence = float(s.y[s.te][s.core_mask].mean())
    for jid, d in c.jobs.items():
        if "shuffled" not in jid or f"-{s.name}-" not in jid:
            continue
        sv, st = load_scores(s, d)
        pr = heldout_metrics(s, sv, st, read_json(d / "config.json")["cfg"]["H"], unsealed=True)["test_core"]["pr_auc"]
        out[jid] = {"core_pr_auc": pr, "prevalence": prevalence, "pass": pr < 10 * prevalence}
    return out


def merge(roots: list[Path], out: Path, datasets: list[str], analysis_root: Path | None = None) -> dict:
    """Unseals only datasets whose planned P0 and P1 sweep jobs have finished, or that were never sealed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c = collect(roots)
    log = UnsealLog(out / "unseal.jsonl")
    code = next(iter({m.get("code_hash") for m in c.manifests}), "unknown")
    report = {"status": plan_status(c), "checklist": checklist(c), "datasets": {}, "not_unsealed": {}}
    planned = [j.job_id for j in PLANS["NB4"]()]
    for ds in datasets:
        wanted = [j for j in planned if f"-gbt-wall-{ds}-" in j and "fold" not in j and "shuffled" not in j]
        finished = [j for j in wanted if j in c.jobs]
        complete = len(finished) == len(wanted) and bool(wanted)
        if is_sealed(ds) and not complete:
            report["not_unsealed"][ds] = f"{len(finished)} of {len(wanted)} sweep jobs finished"
            continue
        status = log.record(ds, code, len(finished), complete) if is_sealed(ds) else "not sealed"
        s, _ = get_stream({"dataset": ds, "protocol": "released"})
        rows = sweep_results(c, s)
        v = verdicts_for(c, s, rows)
        v.update(timing_verdicts(c, s))
        report["datasets"][ds] = {"unseal": status, "verdicts": v, "leakage": leakage_check(c, s), "official": official_audit(c, s),
                                  "per_horizon": {f"{h}/s{k}": r["row"] for (h, k), r in rows.items()}}
        if analysis_root is not None:
            export_for_analysis(analysis_root, s, rows)
        if ds in DOCUMENTED_DAYS:
            report["datasets"][ds]["tail_invariance"] = tail_invariance(c, ds)
            if analysis_root is not None:
                d, _ = get_stream({"dataset": ds, "protocol": "documented"})
                export_for_analysis(analysis_root, d, sweep_results(c, d))
    (out / "merged.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "MERGE_REPORT.md").write_text(render(report))
    return report


def render(report: dict) -> str:
    lines = ["# Merge report", "", "## Plan status (finished / planned)", "", "| Notebook | P0 | P1 | P2 |", "|---|---|---|---|"]
    for nb, rows in report["status"].items():
        lines.append(f"| {nb} | {rows['P0'][0]}/{rows['P0'][1]} | {rows['P1'][0]}/{rows['P1'][1]} | {rows['P2'][0]}/{rows['P2'][1]} |")
    lines += ["", "## Verification", ""]
    for name, ok, detail in report["checklist"]:
        lines.append(f"- [{'x' if ok else ' '}] {name}{' (not checked yet)' if ok is None else ''}{': ' + detail if detail else ''}")
    if report["not_unsealed"]:
        lines += ["", "## Still sealed", ""] + [f"- {k}: {v}" for k, v in report["not_unsealed"].items()]
    for ds, res in report["datasets"].items():
        lines += ["", f"## {ds} (unsealing: {res['unseal']})", ""]
        if res["leakage"]:
            failed = [k for k, v in res["leakage"].items() if not v["pass"]]
            lines.append(f"**Leakage check: {'FAILED for ' + ', '.join(failed) if failed else 'passed'}.**")
            lines.append("")
        if res.get("tail_invariance"):
            lines.append(f"Strict-past scores identical without the tail: {res['tail_invariance']}.")
            lines.append("")
        if res.get("official"):
            lines += ["### Published models (Gate G1)", "", "| Job | H | Official-style F1 | Published | F1 at val threshold | Core F1 | Tail F1 | Core PR-AUC | Stoppable value at 500/day | Sampler coverage |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for jid, a in res["official"].items():
                if "problem" in a:
                    lines.append(f"| {jid} | {a['problem']} | | | | | | | | |")
                    continue
                r = a["row"]
                tail = r.get("test_tail", {}).get("f1")
                sv = r["attempts"].get("budget_500", {}).get("value_share_mean")
                num = lambda x: f"{x:.4f}" if x is not None else ""  # noqa: E731
                lines.append(f"| {jid} | {a['H']} | {num(a['official_style_f1'])} | {num(a['published_reference'])} | {num(r['test']['f1'])} | {num(r['test_core']['f1'])} | {num(tail)} | {num(r['test_core']['pr_auc'])} | {num(sv)} | {num(a['coverage'])} |")
            lines.append("")
        lines += ["| Hypothesis | Value | CI | Verdict |", "|---|---|---|---|"]
        for hid, v in sorted(res["verdicts"].items()):
            value = v.get("value")
            ci = v.get("ci")
            lines.append(f"| {hid} | {value:.3f} | {ci if ci else ''} | {'PASS' if v.get('pass') else 'FAIL'} |" if value is not None else f"| {hid} | see merged.json | | {'PASS' if v.get('pass') else 'FAIL'} |")
    return "\n".join(lines) + "\n"


__all__ = ["merge", "collect", "checklist", "equal_alert_threshold"]
