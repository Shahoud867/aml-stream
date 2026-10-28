"""TD-GNN jobs: training with resume, resumable scoring, attacks on its own checkpoints, and GPU service time.

The neighbour sampler runs on CPU and the model on the GPU when one exists.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .design import horizon_seconds
from .evaluate import heldout_metrics, validation_metrics
from .features import build_graph
from .jobs_gbt import attacked_attempts, get_stream, retimed, write_attack_scores, write_scores
from .metrics import average_precision_score, best_f1_threshold
from .runner import Context, Interrupted
from .sampler import sample_batch
from .sealing import is_sealed, score_health

DEFAULTS = dict(epochs=6, patience=2, neg_rate=0.05, batch=1024, eval_batch=2048, val_neg_rate=0.05, d=64, lr=2e-3,
                pos_weight=10.0, sp=8, sf=8, s2p=3, s2f=3)


class StreamTensors:
    """Holds one stream's event attributes on the device and turns sampled positions into model inputs."""

    def __init__(self, s, g, H: int, device, mu: float, sigma: float, cfg: dict, t=None, view_end=None, order=None):
        import torch
        self.torch = torch
        self.g = g
        self.H = H
        self.device = device
        self.cfg = cfg
        o = order if order is not None else np.arange(s.n)
        self.view_end = view_end if view_end is not None else s.view_end
        tt = t if t is not None else s.t
        lu = torch.from_numpy(np.log1p(s.usd[o]).astype(np.float32))
        self.log_usd = ((lu - mu) / sigma).to(device)
        self.cur_p = torch.from_numpy(s.cur_p[o].astype(np.int64)).to(device)
        self.cur_r = torch.from_numpy(s.cur_r[o].astype(np.int64)).to(device)
        self.fmt = torch.from_numpy(s.fmt[o].astype(np.int64)).to(device)
        self.same = (self.cur_p == self.cur_r).float()
        self.is_self = torch.from_numpy((s.src[o] == s.dst[o]).astype(np.float32)).to(device)
        self.t = torch.from_numpy(tt.astype(np.float64)).to(device)

    def edge_inputs(self, pos: np.ndarray, t_seed):
        torch = self.torch
        mask = torch.from_numpy(pos >= 0).to(self.device)
        p = torch.from_numpy(np.where(pos >= 0, pos, 0)).to(self.device)
        dt = (self.t[p] - t_seed).float()
        cont = torch.stack([self.log_usd[p], torch.sign(dt) * torch.log1p(dt.abs() / 60.0) / 5.0, (dt >= 0).float(), self.same[p], self.is_self[p]], -1)
        return cont * mask.unsqueeze(-1), self.cur_p[p], self.cur_r[p], self.fmt[p], mask

    def forward(self, model, seeds: np.ndarray):
        torch, c, g = self.torch, self.cfg, self.g
        hop1, _, hop2 = sample_batch(seeds, self.H, g["t"], g["src"], g["dst"], self.view_end, g["out_ptr"], g["out_idx"],
                                     g["in_ptr"], g["in_idx"], c["sp"], c["sf"], c["s2p"], c["s2f"])
        ts = self.t[torch.from_numpy(seeds).to(self.device)]
        c0, cp0, cr0, f0, _ = self.edge_inputs(seeds, ts)
        seed_e = model.enc(c0, cp0, cr0, f0)
        c1, cp1, cr1, f1, m1 = self.edge_inputs(hop1, ts.view(-1, 1, 1, 1))
        e1 = model.enc(c1, cp1, cr1, f1) * m1.unsqueeze(-1)
        c2, cp2, cr2, f2, m2 = self.edge_inputs(hop2, ts.view(-1, 1, 1, 1, 1, 1))
        e2 = model.enc(c2, cp2, cr2, f2) * m2.unsqueeze(-1)
        return model(seed_e, e1, m1, e2, m2)


def _build_model(s, c):
    from .gnn import TemporalDirectedGNN
    n_cur = int(max(s.cur_p.max(), s.cur_r.max())) + 1
    return TemporalDirectedGNN(n_cur, int(s.fmt.max()) + 1, c["d"], shared_direction=c.get("shared_direction", False),
                               one_hop=c.get("one_hop", False), no_count=c.get("no_count", False))


def _norm_stats(s):
    lu = np.log1p(s.usd[s.split == 0])
    return float(lu.mean()), float(lu.std())


def _predict(model, st: StreamTensors, rows: np.ndarray, batch: int, autocast, check=None) -> np.ndarray:
    torch = st.torch
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    i = 0
    bs = batch
    with torch.no_grad():
        while i < len(rows):
            if check is not None:
                check()
            part = rows[i:i + bs]
            try:
                with autocast():
                    out[i:i + len(part)] = torch.sigmoid(st.forward(model, part)).float().cpu().numpy()
                i += len(part)
            except RuntimeError as exc:  # inference may shrink its batch; it changes no result
                if "out of memory" in str(exc).lower() and bs > 16:
                    torch.cuda.empty_cache()
                    bs //= 2
                else:
                    raise
    return out


def tdgnn_train(ctx: Context, cfg: dict) -> dict:
    import contextlib

    import torch
    import torch.nn.functional as F
    c = {**DEFAULTS, **cfg}
    s, g = get_stream(cfg)
    H = horizon_seconds(cfg["H"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    mu, sigma = _norm_stats(s)
    st = StreamTensors(s, g, H, device, mu, sigma, c)
    model = _build_model(s, c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tr = s.tr
    tr_pos, tr_neg = tr[s.y[tr] == 1], tr[s.y[tr] == 0]
    yt = s.y.copy()
    if cfg.get("shuffle_labels"):
        shuffled = s.y[tr].copy()
        np.random.default_rng(cfg["seed"] + 1000).shuffle(shuffled)
        yt[tr] = shuffled
        tr_pos, tr_neg = tr[yt[tr] == 1], tr[yt[tr] == 0]
    va = s.va
    r_val = np.random.default_rng(777)
    va_neg = va[s.y[va] == 0]
    val_stop = np.sort(np.r_[va[s.y[va] == 1], r_val.choice(va_neg, int(len(va_neg) * c["val_neg_rate"]), replace=False)])

    state = {"epoch": 0, "best_pr": -1.0, "best_epoch": -1, "epoch_seconds": 0.0, "stopped": False}

    def read_ckpt(path):
        return torch.load(path, map_location=device, weights_only=False)
    saved = ctx.load_checkpoint(read_ckpt)
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if use_amp and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        ctx.log({"event": "resumed", "epoch": state["epoch"]})

    pw = torch.tensor(c["pos_weight"], device=device)
    while state["epoch"] < c["epochs"] and not state["stopped"]:
        ctx.check(state["epoch_seconds"] * 1.2)
        t0 = time.time()
        model.train()
        seeds = np.r_[tr_pos, rng.choice(tr_neg, int(len(tr_neg) * c["neg_rate"]), replace=False)]
        rng.shuffle(seeds)
        if c.get("max_train_seeds"):
            seeds = seeds[:c["max_train_seeds"]]
        total, nb = 0.0, 0
        for i in range(0, len(seeds), c["batch"]):
            part = seeds[i:i + c["batch"]]
            target = torch.from_numpy(yt[part].astype(np.float32)).to(device)
            with autocast():
                loss = F.binary_cross_entropy_with_logits(st.forward(model, part), target, pos_weight=pw)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {state['epoch']}")
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            nb += 1
            if ctx.runner.stop_requested:
                raise Interrupted("stop requested")
        sv = _predict(model, st, val_stop, c["eval_batch"], autocast)
        pr = float(average_precision_score(s.y[val_stop], sv)) if s.y[val_stop].sum() else 0.0
        ctx.log({"event": "epoch", "epoch": state["epoch"], "loss": total / max(nb, 1), "val_stop_pr": pr, "seconds": time.time() - t0})
        if pr > state["best_pr"]:
            state.update(best_pr=pr, best_epoch=state["epoch"])
            ctx.save_checkpoint(lambda p: torch.save(model.state_dict(), p), name="best.bin", meta={"epoch": state["epoch"]})
        elif state["epoch"] - state["best_epoch"] >= c["patience"]:
            state["stopped"] = True
        state["epoch"] += 1
        state["epoch_seconds"] = time.time() - t0
        ctx.save_checkpoint(lambda p, sd=state: torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": dict(sd), "rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if use_amp else None}, p), meta={"epoch": state["epoch"]})

    best = ctx.load_checkpoint(read_ckpt, names=("best.bin",))
    if best is None:
        raise RuntimeError("no best checkpoint")
    model.load_state_dict(best)
    t1 = time.time()

    def scored(name, rows):
        if not (ctx.dir / f"{name}.parquet").exists():
            ctx.check(0)
            write_scores(ctx, name, s, rows, _predict(model, st, rows, c["eval_batch"], autocast, check=ctx.check))
    scored("scores_val", va)
    scored("scores_test", s.te)
    t_inf = time.time() - t1
    from .jobs_gbt import read_scores
    s_val, s_te = read_scores(ctx.dir, "scores_val", s, va), read_scores(ctx.dir, "scores_test", s, s.te)
    m = {"best_epoch": state["best_epoch"], "epochs_run": state["epoch"], "device": str(device), "infer_seconds": round(t_inf, 1),
         "throughput_eps": float((len(va) + len(s.te)) / max(t_inf, 1e-9)), "test_health": score_health(s_te), **validation_metrics(s, s_val)}
    if not is_sealed(s.name):
        m["test_results"] = heldout_metrics(s, s_val, s_te, cfg["H"])
    return m


def _load_trained(ctx: Context, cfg: dict, s, device):
    import torch
    c = {**DEFAULTS, **cfg}
    model = _build_model(s, c).to(device)
    path = ctx.runner.jobs_dir / cfg["train_job"] / "ckpt" / "best.bin"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    return model.eval(), c


def tdgnn_attack(ctx: Context, cfg: dict) -> dict:
    """Same design as gbt_attack, on the checkpoint written by this notebook's own training job."""
    import contextlib

    import torch

    from .jobs_gbt import read_scores
    from .metrics import threshold_for_budget
    s, g = get_stream(cfg)
    H = horizon_seconds(cfg["H"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else contextlib.nullcontext
    model, c = _load_trained(ctx, cfg, s, device)
    mu, sigma = _norm_stats(s)
    s_val = read_scores(ctx.runner.jobs_dir / cfg["train_job"], "scores_val", s, s.va)
    tau = {str(b): threshold_for_budget(s_val, s.val_days, b) for b in cfg.get("budgets", [500])}
    state = ctx.load_state({"done": []})
    attempts = attacked_attempts(s)
    pos = np.concatenate(list(attempts.values()))
    for W, cond in itertools.product(cfg["Ws"], cfg["conditions"]):
        key = f"W{W}_{cond}"
        if key in state["done"]:
            continue
        ctx.check(120)
        t2, order, inv, gr = retimed(s, attempts, g, W, cond, cfg["seed"])
        st = StreamTensors(s, gr, H, device, mu, sigma, c, t=t2[order], view_end=s.view_end[order], order=order)
        scores = _predict(model, st, inv[pos], c["eval_batch"], autocast, check=ctx.check)
        write_attack_scores(ctx, f"attack_{key}", s, attempts, t2, pos, scores)
        state["done"].append(key)
        ctx.save_state(state)
    return {"n_attempts": len(attempts), "n_edges": int(len(pos)), "conditions": state["done"], "tau": tau}


def tdgnn_service(ctx: Context, cfg: dict) -> dict:
    """Scoring time per batch on this session's device. Must run with nothing else on the GPU."""
    import contextlib

    import torch
    s, g = get_stream(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else contextlib.nullcontext
    model, c = _load_trained(ctx, cfg, s, device)
    mu, sigma = _norm_stats(s)
    st = StreamTensors(s, g, horizon_seconds(cfg["H"]), device, mu, sigma, c)
    rng = np.random.default_rng(0)
    te = s.te
    profile = []
    with torch.no_grad():
        for n in cfg.get("sizes", [1, 10, 100, 1000, 10000]):
            ctx.check(60)
            times = []
            for r, start in enumerate(rng.integers(0, max(len(te) - n, 1), cfg.get("repeats", 30) + 5)):
                part = te[start:start + n]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with autocast():
                    st.forward(model, part)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if r >= 5:
                    times.append(time.perf_counter() - t0)
            profile.append({"n": n, "median_s": float(np.median(times)), "p95_s": float(np.percentile(times, 95)), "events_per_s": float(n / np.median(times))})
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return {"device": name, "H": cfg["H"], "profile": profile}


REGISTRY = {"tdgnn_train": tdgnn_train, "tdgnn_attack": tdgnn_attack, "tdgnn_service": tdgnn_service}

__all__ = ["REGISTRY", "StreamTensors", "best_f1_threshold", "build_graph"]
