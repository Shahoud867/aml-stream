"""The frozen job list for the five notebooks (NOTEBOOK_METHODOLOGY.md, DECISIONS.md D-001, D-008, D-012).

P0 is what the 15 November milestone needs; P1 and P2 may finish later. Order inside a tier is the running order.
"""
from __future__ import annotations

from .runner import JobSpec

H_GRID = ["0", "1", "60", "600", "3600", "21600", "86400", "259200", "inf"]
SEEDS = [0, 1, 2]
BUDGETS = [100, 500, 2000]
CHECKPOINTS = ["1", "600", "3600", "21600", "86400"]
CONDITIONS = ["none", "burst", "stretch", "blend", "shift_burst", "shift_stretch", "shift_blend"]
ALL_W = [3600, 21600, 86400]
ATTACK_H = ["0", "1", "3600", "86400"]
ROLLING_FOLDS = [(4, 5, 7), (5, 6, 8), (6, 7, 9), (7, 8, 10)]
ABLATIONS = {"nofwd": "^fwd_", "nomotif": "^m_", "nopair": "_pair_", "nohour": "^hour$", "nofmt": "^fmt$", "nocur": "^cur_|same_cur"}
# Rough limits so a job is not started when it cannot finish. The pilot replaces these with measurements.
EST = {"HI-Small": 600, "LI-Small": 800, "SAML-D": 1200, "HI-Medium": 4800, "LI-Medium": 4800, "Ethereum": 3000}
PROTOCOLS = {"HI-Small": ["released", "documented"], "LI-Small": ["released", "documented"], "HI-Medium": ["released", "documented"],
             "LI-Medium": ["released", "documented"], "SAML-D": ["released"], "Ethereum": ["released"]}
FOLDER = {"HI-Small": "Small_HI", "LI-Small": "Small_LI", "HI-Medium": "Medium_HI", "LI-Medium": "Medium_LI"}
FRAUDGT_MULTI = "AML-Small-HI/AML-Small-HI-Multi-SparseNodeGT.yaml"


def _sweep_id(nb, ds, proto, H, seed, family="wall", tag=""):
    return f"{nb}-gbt-{family}-{ds}-{proto}-H{H}{('-' + tag) if tag else ''}-s{seed}"


def _sweep(nb, ds, proto, H, seed, tier, family="wall", tag="", **extra):
    cfg = {"dataset": ds, "protocol": proto, "H": H, "seed": seed, "family": family, "neg_rate": 0.25, **extra}
    return JobSpec(_sweep_id(nb, ds, proto, H, seed, family, tag), "gbt_sweep", cfg, tier, est_seconds=EST[ds])


def _sweep_grid(nb, ds, tier):
    return [_sweep(nb, ds, p, h, s, tier) for p in PROTOCOLS[ds] for h in H_GRID for s in SEEDS]


def _policy(nb, ds, tier):
    jobs = []
    for s in SEEDS:
        sweep = [_sweep_id(nb, ds, "released", h, s) for h in CHECKPOINTS]
        jobs.append(JobSpec(f"{nb}-policy-{ds}-released-s{s}", "alert_policy", {
            "dataset": ds, "protocol": "released", "seed": s, "checkpoints": CHECKPOINTS, "sweep_jobs": sweep, "budgets": BUDGETS}, tier, requires=sweep))
    return jobs


def _mgnn(nb, ds, model, seed, tier, cadence=None):
    cfg = {"dataset": ds, "folder": FOLDER[ds], "model": model, "seed": seed}
    if cadence:
        cfg["train_cadence"] = cadence
    tag = f"-cad{cadence}" if cadence else ""
    return JobSpec(f"{nb}-mgnn-{model}-{ds}{tag}-s{seed}", "mgnn_train", cfg, tier)


def _replay(train: JobSpec, cadence, tier):
    return JobSpec(f"{train.job_id}-replay{cadence}", "mgnn_replay", {**train.cfg, "train_job": train.job_id, "cadence": cadence}, tier, requires=[train.job_id])


def nb1() -> list[JobSpec]:
    """Official Multi-GIN+EU on HI-Small."""
    jobs = [JobSpec("nb1-format-check", "mgnn_format_check", {"dataset": "HI-Small"}, "P0")]
    trained = [_mgnn("nb1", "HI-Small", "gin", s, "P0") for s in (1, 2, 3)]
    jobs += trained
    jobs += [_replay(t, c, "P0") for t in trained for c in (86400, 21600, 3600)]
    # Retrained at a cadence and then scored at that cadence, so training and scoring see the same view (W5).
    retrained = [_mgnn("nb1", "HI-Small", "gin", 1, "P1", cad) for cad in (86400, 3600)]
    retrained += [_mgnn("nb1", "HI-Small", "gin", s, "P1", 86400) for s in (2, 3)]
    jobs += retrained + [_replay(t, t.cfg["train_cadence"], "P1") for t in retrained]
    jobs += [JobSpec(f"{t.job_id}-trainnorm", "mgnn_train_norm", {**t.cfg, "train_job": t.job_id}, "P1", requires=[t.job_id]) for t in trained]
    return jobs


def nb2() -> list[JobSpec]:
    """Official models, breadth. FraudGT trains for at most 20 hours; a shorter run is reported as truncated."""
    pna = [_mgnn("nb2", "HI-Small", "pna", s, "P0") for s in (1, 2)]
    li = [_mgnn("nb2", "LI-Small", "gin", s, "P0") for s in (1, 2)]
    jobs = pna + li + [_replay(t, c, "P0") for t in pna + li for c in (86400, 3600)]
    jobs += [_mgnn("nb2", "LI-Small", "pna", s, "P1") for s in (1, 2)]
    jobs += [JobSpec("nb2-fraudgt-multi-HI-Small-s42", "fraudgt_train", {"dataset": "HI-Small", "config": FRAUDGT_MULTI, "seed": 42, "max_train_hours": 20}, "P1")]
    jobs += [_mgnn("nb2", "HI-Medium", "gin", 1, "P2")]
    return jobs


def _tdgnn(ds, H, seed, tier, tag="", **extra):
    cfg = {"dataset": ds, "H": H, "seed": seed, **extra}
    return JobSpec(f"nb3-tdgnn-{ds}-H{H}{('-' + tag) if tag else ''}-s{seed}", "tdgnn_train", cfg, tier)


def _tdgnn_attack(train: JobSpec, Ws, tier):
    cfg = {"dataset": train.cfg["dataset"], "H": train.cfg["H"], "seed": train.cfg["seed"], "train_job": train.job_id, "Ws": Ws, "conditions": CONDITIONS}
    return JobSpec(f"{train.job_id}-attack", "tdgnn_attack", cfg, tier, requires=[train.job_id])


def nb3() -> list[JobSpec]:
    """TD-GNN across horizons. Horizon set is deliberately smaller than the boosted-tree grid at P0."""
    p0_h = ["0", "1", "3600", "86400", "inf"]
    jobs = [_tdgnn("HI-Small", h, 0, "P0", "shuffled", shuffle_labels=True) for h in ("0", "3600")]
    train = {(ds, h, s): _tdgnn(ds, h, s, "P0") for ds in ("HI-Small", "LI-Small") for h in p0_h for s in SEEDS}
    jobs += list(train.values())
    jobs += [_tdgnn_attack(train[("HI-Small", h, s)], [21600], "P0") for h in ("0", "1", "3600") for s in SEEDS]
    jobs += [_tdgnn("HI-Small", h, s, "P1") for h in H_GRID if h not in p0_h for s in SEEDS]
    for flag, tag in (("shared_direction", "nodir"), ("one_hop", "onehop"), ("no_count", "nocount")):
        jobs += [_tdgnn("HI-Small", h, s, "P1", tag, **{flag: True}) for h in ("0", "1", "3600") for s in SEEDS]
    jobs += [_tdgnn_attack(train[("LI-Small", h, s)], [21600], "P1") for h in ("0", "1", "3600") for s in SEEDS]
    t = train[("HI-Small", "1", 0)]
    jobs.append(JobSpec("nb3-tdgnn-service", "tdgnn_service", {"dataset": "HI-Small", "H": "1", "seed": 0, "train_job": t.job_id}, "P1", requires=[t.job_id]))
    return jobs


def nb4() -> list[JobSpec]:
    """Boosted-tree confirmation. Stage 0, the leakage check, runs first."""
    jobs = [_sweep("nb4", "HI-Small", "released", h, 0, "P0", tag="shuffled", shuffle_labels=True) for h in ("0", "1", "3600", "inf")]
    jobs += _sweep_grid("nb4", "HI-Small", "P0") + _policy("nb4", "HI-Small", "P0")
    jobs += _sweep_grid("nb4", "LI-Small", "P0") + _policy("nb4", "LI-Small", "P0")
    jobs += _sweep_grid("nb4", "SAML-D", "P1") + _policy("nb4", "SAML-D", "P1")
    jobs += _sweep_grid("nb4", "HI-Medium", "P1") + _policy("nb4", "HI-Medium", "P1")
    jobs += _sweep_grid("nb4", "LI-Medium", "P1") + _policy("nb4", "LI-Medium", "P1")
    for fold in ROLLING_FOLDS:
        tag = "fold" + "_".join(str(x) for x in fold)
        jobs += [_sweep("nb4", "HI-Small", "documented", h, 0, "P2", tag=tag, fold=list(fold)) for h in ("0", "1", "600", "3600", "86400")]
    jobs += [_sweep("nb4", "Ethereum", "released", h, s, "P2") for h in ("0", "1", "3600", "86400", "inf") for s in SEEDS]
    return jobs


def _attack(ds, family, H, seed, Ws, tier):
    cfg = {"dataset": ds, "protocol": "released", "family": family, "H": H, "seed": seed, "Ws": Ws, "conditions": CONDITIONS, "neg_rate": 0.25}
    return JobSpec(f"nb5-attack-{family}-{ds}-H{H}-s{seed}", "gbt_attack", cfg, tier, est_seconds=EST[ds])


def nb5() -> list[JobSpec]:
    """Timing robustness and attribution."""
    jobs = [_sweep("nb5", "HI-Small", "released", h, s, "P0", family=f) for f in ("logical", "combined") for h in ATTACK_H for s in SEEDS]
    jobs += [_attack("HI-Small", f, h, s, ALL_W, "P0") for f in ("wall", "logical", "combined") for h in ATTACK_H for s in SEEDS]
    for tag, drop in ABLATIONS.items():
        horizons = ["3600"] if tag == "nofwd" else ["0", "1", "3600"]
        jobs += [_sweep("nb5", "HI-Small", "released", h, s, "P0", tag=tag, drop=drop) for h in horizons for s in SEEDS]
    jobs += [_sweep("nb5", "LI-Small", "released", h, s, "P1", family="logical") for h in ATTACK_H for s in SEEDS]
    jobs += [_attack("LI-Small", f, h, s, [21600], "P1") for f in ("wall", "logical") for h in ATTACK_H for s in SEEDS]
    jobs += [JobSpec(f"nb5-gfp-{ds}-b{b}", "gfp_baseline", {"dataset": ds, "protocol": "released", "batch": b, "seed": 0}, "P1") for ds in ("HI-Small", "LI-Small") for b in (128, 2048)]
    jobs += [JobSpec("nb5-service-cpu", "service_profile", {"dataset": "HI-Small", "protocol": "released", "H": "1"}, "P1")]
    jobs += [_sweep("nb5", "HI-Medium", "released", h, s, "P1", family="logical") for h in ("0", "1", "3600") for s in SEEDS]
    jobs += [_attack("HI-Medium", f, h, s, [21600], "P1") for f in ("wall", "logical") for h in ("0", "1", "3600") for s in SEEDS]
    return jobs


PLANS = {"NB1": nb1, "NB2": nb2, "NB3": nb3, "NB4": nb4, "NB5": nb5}


def _chains(jobs: list[JobSpec]) -> list[list[JobSpec]]:
    """Groups jobs linked by `requires`: a replay must run where the checkpoint it reads was written."""
    parent = {j.job_id: j.job_id for j in jobs}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for j in jobs:
        for r in j.requires:
            parent[root(r)] = root(j.job_id)
    groups: dict[str, list[JobSpec]] = {}
    for j in jobs:
        groups.setdefault(root(j.job_id), []).append(j)
    return list(groups.values())


def plan(notebook: str, shard: tuple[int, int] | None = None) -> list[JobSpec]:
    """shard (i, n) takes every n-th dependency chain, so a second GPU or spare quota elsewhere can take work with no overlap."""
    jobs = PLANS[notebook]()
    if shard is None:
        return jobs
    i, n = shard
    keep = {j.job_id for k, chain in enumerate(_chains(jobs)) if k % n == i for j in chain}
    return [j for j in jobs if j.job_id in keep]


def pilot(notebook: str) -> list[JobSpec]:
    """Short runs on HI-Small that prove each code path and measure time. Sealed datasets are only prepared, never modelled."""
    if notebook == "NB1":
        gin = JobSpec("pilot-nb1-mgnn-gin", "mgnn_train", {"dataset": "HI-Small", "folder": "Small_HI", "model": "gin", "seed": 1, "n_epochs": 2})
        return [JobSpec("pilot-nb1-format-check", "mgnn_format_check", {"dataset": "HI-Small"}), gin, _replay(gin, 86400, "P0"),
                JobSpec("pilot-nb1-trainnorm", "mgnn_train_norm", {**gin.cfg, "train_job": gin.job_id}, requires=[gin.job_id])]
    if notebook == "NB2":
        pna = JobSpec("pilot-nb2-mgnn-pna", "mgnn_train", {"dataset": "HI-Small", "folder": "Small_HI", "model": "pna", "seed": 1, "n_epochs": 2})
        fraud = {"dataset": "HI-Small", "config": FRAUDGT_MULTI, "seed": 42, "score_limit": 20000,
                 "opts": {"optim.max_epoch": 2, "train.iter_per_epoch": 16, "val.iter_per_epoch": 8}}
        return [pna, _replay(pna, 86400, "P0"), JobSpec("pilot-nb2-fraudgt", "fraudgt_train", fraud)]
    if notebook == "NB3":
        td = JobSpec("pilot-nb3-tdgnn", "tdgnn_train", {"dataset": "HI-Small", "H": "3600", "seed": 0, "epochs": 2})
        attack = {"dataset": "HI-Small", "H": "3600", "seed": 0, "train_job": td.job_id, "Ws": [21600], "conditions": ["none", "blend", "shift_blend"]}
        return [td, JobSpec("pilot-nb3-attack", "tdgnn_attack", attack, requires=[td.job_id]),
                JobSpec("pilot-nb3-service", "tdgnn_service", {"dataset": "HI-Small", "H": "3600", "seed": 0, "train_job": td.job_id, "sizes": [1, 100, 1000], "repeats": 10}, requires=[td.job_id])]
    if notebook == "NB4":
        base = {"dataset": "HI-Small", "protocol": "released", "seed": 0, "family": "wall", "neg_rate": 0.25}
        return [JobSpec("pilot-nb4-shuffled-H0", "gbt_sweep", {**base, "H": "0", "shuffle_labels": True}), JobSpec("pilot-nb4-sweep-H0", "gbt_sweep", {**base, "H": "0"}),
                JobSpec("pilot-nb4-sweep-H3600", "gbt_sweep", {**base, "H": "3600"}), JobSpec("pilot-nb4-sweep-Hinf", "gbt_sweep", {**base, "H": "inf"}),
                JobSpec("pilot-nb4-prepare-HI-Medium", "prepare_check", {"dataset": "HI-Medium"}), JobSpec("pilot-nb4-prepare-SAML-D", "prepare_check", {"dataset": "SAML-D"})]
    if notebook == "NB5":
        base = {"dataset": "HI-Small", "protocol": "released", "seed": 0, "neg_rate": 0.25}
        attack = {**base, "family": "wall", "H": "0", "Ws": [21600], "conditions": ["none", "burst", "blend", "shift_burst", "shift_blend"]}
        return [JobSpec("pilot-nb5-logical", "gbt_sweep", {**base, "family": "logical", "H": "0"}), JobSpec("pilot-nb5-attack", "gbt_attack", attack),
                JobSpec("pilot-nb5-gfp", "gfp_baseline", {"dataset": "HI-Small", "protocol": "released", "batch": 2048, "seed": 0}),
                JobSpec("pilot-nb5-service-cpu", "service_profile", {"dataset": "HI-Small", "protocol": "released", "H": "1"})]
    raise KeyError(notebook)


# Which pilot job the forced kill interrupts, and at which of its stop checks: after a checkpoint, so resume has something to do.
KILL_POINT = {"NB1": ("pilot-nb1-mgnn-gin", 2), "NB2": ("pilot-nb2-mgnn-pna", 2), "NB3": ("pilot-nb3-tdgnn", 2), "NB4": ("pilot-nb4-sweep-H3600", 1), "NB5": ("pilot-nb5-attack", 3)}


def registry() -> dict:
    """Imported lazily, because each notebook needs only some of these dependencies."""
    from .jobs_gbt import REGISTRY as gbt
    out = dict(gbt)
    try:
        from .jobs_gnn import REGISTRY as gnn
        out.update(gnn)
    except ImportError:
        pass
    try:
        from .mgnn import REGISTRY as mg
        out.update(mg)
    except ImportError:
        pass
    from .extras import REGISTRY as ex
    from .fraudgt import REGISTRY as fg
    out.update(ex)
    out.update(fg)
    return out
