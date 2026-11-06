"""Runs the official IBM Multi-GNN and audits it. NOT YET EXECUTED: the pilot gate must pass before any result counts.

The official training code has no resume, so this module repeats its loop around the official get_data, get_model,
get_loaders and evaluate_hetero, adding epoch checkpoints. Selection follows the official rule: epoch 0 is never
selected, and later epochs must beat the best validation F1. Differences from the official loop:
  * test data is scored only after training, not every epoch (same selection, far less GPU time, and sealed data stays sealed);
  * every scored edge gets a score. Official evaluation drops seed edges the sampler did not return on Small_HI, so the
    `sampled` column keeps that behaviour available and the merge step reports both.

Only reverse-message-passing (hetero) models are supported, which is what the audited configurations use.
"""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from .mgnn_format import format_transactions
from .prepare import DOCUMENTED_DAYS  # noqa: F401
from .runner import Context, Interrupted
from .sealing import is_sealed

TO = ("node", "to", "node")
REV = ("node", "rev_to", "node")


class MgnnPaths:
    code: Path = Path("third_party/Multi-GNN")
    work: Path = Path("cache/mgnn")


def _official():
    code = str(Path(MgnnPaths.code).resolve())
    if code not in sys.path:
        sys.path.insert(0, code)
    os.chdir(code)  # the official code reads model_settings.json from the working directory
    import data_loading
    import train_util
    import training
    from util import create_parser
    # The official AddEgoIds defines __call__ only, but BaseTransform in PyG 2.7.0 also declares an abstract forward, so it cannot be
    # instantiated. Clearing the flag keeps the official __call__ as the code that runs, unchanged.
    train_util.AddEgoIds.__abstractmethods__ = frozenset()
    return data_loading, train_util, training, create_parser


def _args(cfg: dict):
    *_, create_parser = _official()
    argv = ["--data", cfg["folder"], "--model", cfg["model"], "--testing", "--seed", str(cfg["seed"]),
            "--n_epochs", str(cfg.get("n_epochs", 100)), "--batch_size", str(cfg.get("batch_size", 8192))]
    argv += ["--emlps", "--reverse_mp", "--ego", "--ports"]
    argv += ["--num_neighs", *[str(x) for x in cfg.get("num_neighs", [100, 100])]]
    args = create_parser().parse_args(argv)
    # The official parser gives --num_neighs no type, so values passed on the command line arrive as strings and the
    # sampler rejects them. Its own default is a list of ints, which is what the official runs used.
    args.num_neighs = [int(x) for x in args.num_neighs]
    return args


def ensure_formatted(cfg: dict, raw_csv: Path) -> Path:
    """Official code reads {aml_data}/{folder}/formatted_transactions.csv."""
    folder = Path(MgnnPaths.work) / cfg["folder"]
    out = folder / "formatted_transactions.csv"
    if not out.exists():
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"formatted.{os.getpid()}.tmp"  # two workers on one machine may reach this together
        format_transactions(raw_csv, tmp, cfg.get("max_rows"))
        os.replace(tmp, out)
    return out


def load_data(cfg: dict, raw_csv: Path):
    """get_data computes port numbers in pure Python, so its result is cached per session.
    Also records the normalisation statistics each split graph used, so training-set normalisation can be tested."""
    import torch
    data_loading, _, _, _ = _official()
    ensure_formatted(cfg, raw_csv)
    cache = Path(MgnnPaths.work) / f"{cfg['folder']}_{cfg['model']}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    recorded = []
    original = data_loading.z_norm

    def recording_z_norm(x):
        recorded.append((x.mean(0).clone(), x.std(0).clone()))
        return original(x)
    data_loading.z_norm = recording_z_norm
    try:
        tr, val, te, tr_i, val_i, te_i = data_loading.get_data(_args(cfg), {"paths": {"aml_data": str(Path(MgnnPaths.work).resolve())}})
    finally:
        data_loading.z_norm = original
    edge_id = pl.read_csv(Path(MgnnPaths.work) / cfg["folder"] / "formatted_transactions.csv", columns=["EdgeID"])["EdgeID"].to_numpy()
    blob = {"data": (tr, val, te), "idx": (tr_i, val_i, te_i), "z": recorded[1:], "edge_id": edge_id}  # recorded[0] is the node feature
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, cache)
    return blob


def _config_obj(args):
    _, train_util, _, _ = _official()
    keys = ("n_gnn_layers", "n_hidden", "dropout", "final_dropout", "lr", "w_ce1", "w_ce2", "n_heads")
    return type("Cfg", (), {**{k: train_util.extract_param(k, args) for k in keys}, "model": args.model, "epochs": args.n_epochs})


def _set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def official_mode(cfg: dict) -> bool:
    """The official code never calls model.eval() or model.train(), so its models, which use batch-norm and a final dropout of 0.5,
    are evaluated in PyTorch's default training mode. Following that is what reproduces the published number. cfg eval_mode 'eval' switches it off."""
    return cfg.get("eval_mode", "official") == "official"


def _strip_id(batch):
    batch[TO].edge_attr = batch[TO].edge_attr[:, 1:]
    batch[REV].edge_attr = batch[REV].edge_attr[:, 1:]


def _train_batch(model, optimizer, loss_fn, batch, seed_ids, device) -> float:
    """One step, with the same seed-edge selection as the official train_hetero."""
    import torch
    optimizer.zero_grad()
    mask = torch.isin(batch[TO].edge_attr[:, 0].detach().cpu(), seed_ids)
    _strip_id(batch)
    batch.to(device)
    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[TO]
    pred, truth = out[mask.to(out.device)], batch[TO].y[mask.to(out.device)]
    loss = loss_fn(pred, truth)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    loss.backward()
    optimizer.step()
    return float(loss)


def subgraph_until(data, times: np.ndarray, cut: int):
    """A copy of a split graph holding only edges up to time cut, with stock features. Port numbers count only
    earlier edges, so slicing gives the values recomputing them on the prefix would give."""
    import torch
    keep = torch.from_numpy(times <= cut)
    d = copy.copy(data)
    for et in (TO, REV):
        d[et].edge_index = data[et].edge_index[:, keep]
        d[et].edge_attr = data[et].edge_attr[keep]
    d[TO].y = data[TO].y[keep]
    if "timestamps" in data[TO]:
        d[TO].timestamps = data[TO].timestamps[keep]
    return d, np.flatnonzero(times <= cut)


def score_edges(model, data, seed_pos: np.ndarray, args, transform, device, check=None, train_mode: bool = False):
    """Scores every seed edge. Returns (graph position ids, probability, whether the sampler returned it)."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader
    seeds = torch.as_tensor(seed_pos, dtype=torch.long)
    es = data[TO]
    loader = LinkNeighborLoader(data, num_neighbors=args.num_neighs, edge_label_index=(TO, es.edge_index[:, seeds]),
                                edge_label=es.y[seeds], batch_size=args.batch_size, shuffle=False, transform=transform)
    ids, probs, sampled = [], [], []
    model.train() if train_mode else model.eval()
    with torch.no_grad():
        for batch in loader:
            if check is not None:
                check()
            bs = batch[TO]
            seed_ids = es.edge_attr[seeds[bs.input_id.cpu()], 0].cpu()
            batch_ids = bs.edge_attr[:, 0].cpu()
            mask = torch.isin(batch_ids, seed_ids)
            missing = ~torch.isin(seed_ids, batch_ids)
            was_sampled = torch.ones(int(mask.sum()), dtype=torch.bool)
            kept = batch_ids[mask]
            if missing.any():
                # Same add-back as the official code applies to Small_J and Small_Q, used here for full coverage.
                miss_ids = seed_ids[missing]
                miss_pos = seeds[bs.input_id.cpu()][missing]
                node_map = {int(v): i for i, v in enumerate(batch["node"].n_id.tolist())}
                add_index = torch.tensor([[node_map[int(a)] for a in row] for row in es.edge_index[:, miss_pos]], dtype=torch.long)
                bs.edge_index = torch.cat([bs.edge_index, add_index], 1)
                bs.edge_attr = torch.cat([bs.edge_attr, es.edge_attr[miss_pos]], 0)
                mask = torch.cat([mask, torch.ones(len(miss_ids), dtype=torch.bool)])
                kept = torch.cat([kept, miss_ids])
                was_sampled = torch.cat([was_sampled, torch.zeros(len(miss_ids), dtype=torch.bool)])
            _strip_id(batch)
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[TO]
            p = torch.softmax(out[mask.to(out.device)], -1)[:, 1]
            ids.append(kept.numpy().astype(np.int64))
            probs.append(p.float().cpu().numpy())
            sampled.append(was_sampled.numpy())
    return _dedupe(np.concatenate(ids), np.concatenate(probs), np.concatenate(sampled))


def _dedupe(ids: np.ndarray, p: np.ndarray, sampled: np.ndarray):
    """A seed edge can be returned twice when reached at two hops; keep one row per edge, with the mean score."""
    uniq, inv = np.unique(ids, return_inverse=True)
    mean = np.bincount(inv, weights=p) / np.bincount(inv)
    was = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(was, inv, sampled)
    return uniq, mean, was


def _write(ctx: Context, name: str, event_ids: np.ndarray, p: np.ndarray, sampled: np.ndarray, expected: int):
    if len(event_ids) != expected:
        raise ValueError(f"{name}: scored {len(event_ids)} of {expected} edges")
    pl.DataFrame({"event_id": event_ids, "score": p.astype(np.float32), "sampled": sampled}).write_parquet(ctx.dir / f"{name}.parquet")


def _model(cfg, blob, args, device):
    import torch
    from torch_geometric.nn import to_hetero
    _, train_util, training, _ = _official()
    tr, _, te = blob["data"]
    transform = train_util.AddEgoIds()
    tr_i, val_i, te_i = blob["idx"]
    tr_loader, _, _ = train_util.get_loaders(tr, blob["data"][1], te, tr_i, val_i, te_i, transform, args)
    sample = next(iter(tr_loader))
    config = _config_obj(args)
    model = to_hetero(training.get_model(sample, config, args), te.metadata(), aggr="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    _strip_id(sample)
    sample.to(device)
    with torch.no_grad():
        model(sample.x_dict, sample.edge_index_dict, sample.edge_attr_dict)  # materialises lazy parameters, as the official summary call does
    return model, optimizer, transform, tr_loader, config


def _add_ids(blob):
    _, train_util, _, _ = _official()
    if not blob.get("ids_added"):
        train_util.add_arange_ids(list(blob["data"]))
        blob["ids_added"] = True


def mgnn_train(ctx: Context, cfg: dict) -> dict:
    """Trains one official model, with epoch checkpoints, and scores validation and test."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader
    from .jobs_gbt import Paths
    _, train_util, _, _ = _official()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Measured: training batches (batch_size 8192, num_neighs [100,100], reverse_mp+emlps+ego) take ~7s each on an
        # RTX 3080, dominating the epoch. TF32 matmuls are an Ampere-only precision trade (not a change to the frozen
        # algorithm or its hyperparameters): error stays far inside normal seed-to-seed spread. See DECISIONS.md D-018.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args = _args(cfg)
    _set_seed(cfg["seed"])
    blob = load_data(cfg, Paths.raw[cfg["dataset"]] / f"{cfg['dataset']}_Trans.csv")
    _add_ids(blob)
    tr, val, te = blob["data"]
    tr_i, val_i, te_i = blob["idx"]
    model, optimizer, transform, tr_loader, config = _model(cfg, blob, args, device)
    weight = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    _, val_loader, _ = train_util.get_loaders(tr, val, te, tr_i, val_i, te_i, transform, args)
    state = {"epoch": 0, "best_val_f1": 0.0, "best_epoch": -1, "epoch_seconds": 0.0}

    def read_ckpt(path):
        return torch.load(path, map_location=device, weights_only=False)
    saved = ctx.load_checkpoint(read_ckpt)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["py_rng"])
        np.random.set_state(saved["np_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        ctx.log({"event": "resumed", "epoch": state["epoch"]})

    cadence = cfg.get("train_cadence")
    tr_times = tr[TO].timestamps.numpy().astype(np.int64) if cadence else None
    while state["epoch"] < config.epochs:
        ctx.check(state["epoch_seconds"] * 1.2)
        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        if cadence:
            buckets = tr_times // cadence
            for b in np.random.permutation(np.unique(buckets)):
                d, kept = subgraph_until(tr, tr_times, int((b + 1) * cadence - 1))
                pos_of = np.full(len(tr_times), -1, dtype=np.int64)
                pos_of[kept] = np.arange(len(kept))
                seeds = torch.as_tensor(pos_of[buckets == b], dtype=torch.long)
                loader = LinkNeighborLoader(d, num_neighbors=args.num_neighs, edge_label_index=(TO, d[TO].edge_index[:, seeds]),
                                            edge_label=d[TO].y[seeds], batch_size=args.batch_size, shuffle=True, transform=transform)
                for batch in loader:
                    ids = d[TO].edge_attr[seeds[batch[TO].input_id.cpu()], 0].cpu()
                    total += _train_batch(model, optimizer, loss_fn, batch, ids, device)
                    nb += 1
        else:
            inds = tr_i.detach().cpu()
            for batch in tr_loader:
                ids = tr_loader.data[TO].edge_attr.detach().cpu()[inds[batch[TO].input_id.detach().cpu()], 0]
                total += _train_batch(model, optimizer, loss_fn, batch, ids, device)
                nb += 1
                if ctx.runner.stop_requested:
                    raise Interrupted("stop requested")
        if not official_mode(cfg):
            model.eval()
        val_f1 = float(train_util.evaluate_hetero(val_loader, val_i, model, val, device, args))
        if state["epoch"] > 0 and val_f1 > state["best_val_f1"]:
            state.update(best_val_f1=val_f1, best_epoch=state["epoch"])
            ctx.save_checkpoint(lambda p: torch.save(model.state_dict(), p), name="best.bin", meta={"epoch": state["epoch"]})
        state["epoch"] += 1
        state["epoch_seconds"] = time.time() - t0
        ctx.log({"event": "epoch", "epoch": state["epoch"] - 1, "loss": total / max(nb, 1), "val_f1": val_f1, "seconds": state["epoch_seconds"]})
        ctx.save_checkpoint(lambda p: torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": dict(state), "py_rng": random.getstate(), "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, p))
    best = ctx.load_checkpoint(read_ckpt, names=("best.bin",))
    if best is not None:
        model.load_state_dict(best)
    else:
        ctx.log({"event": "no_improvement_after_epoch_0", "note": "scoring the last epoch"})
        ctx.save_checkpoint(lambda p: torch.save(model.state_dict(), p), name="best.bin")
    file_of_val = np.concatenate([tr_i.numpy(), val_i.numpy()])
    edge_id = blob["edge_id"]
    official = official_mode(cfg)
    uv, pv, sv = score_edges(model, val, val_i.numpy(), args, transform, device, check=ctx.check, train_mode=official)
    _write(ctx, "scores_val", edge_id[file_of_val[uv]], pv, sv, len(val_i))
    ut, pt, st = score_edges(model, te, te_i.numpy(), args, transform, device, check=ctx.check, train_mode=official)
    _write(ctx, "scores_test", edge_id[ut], pt, st, len(te_i))
    result = {"model": cfg["model"], "dataset": cfg["dataset"], "seed": cfg["seed"], "train_cadence": cadence, "eval_mode": "official" if official else "eval",
              **state, "device": str(device), "coverage_val": float(sv.mean()), "coverage_test": float(st.mean())}

    def argmax_f1(p, sampled, y):
        flagged = (p >= 0.5) & sampled
        return float(2 * (flagged & y).sum() / max(flagged.sum() + (y & sampled).sum(), 1))
    if official:
        # Official scoring keeps dropout and batch-norm batch statistics on, so each score depends on chance and on its batch.
        # A second pass in evaluation mode gives scores that depend only on the weights, for the horizon analysis and the resume check.
        uv2, pv2, sv2 = score_edges(model, val, val_i.numpy(), args, transform, device, check=ctx.check)
        _write(ctx, "scores_val_eval", edge_id[file_of_val[uv2]], pv2, sv2, len(val_i))
        ut2, pt2, st2 = score_edges(model, te, te_i.numpy(), args, transform, device, check=ctx.check)
        _write(ctx, "scores_test_eval", edge_id[ut2], pt2, st2, len(te_i))
    if not is_sealed(cfg["dataset"]):  # the number to compare with the published F1: argmax, on the edges official evaluation keeps
        y = te[TO].y[torch.as_tensor(ut)].numpy() == 1
        result["official_style_test_f1"] = argmax_f1(pt, st, y)
        if official:
            result["official_style_test_f1_eval_mode"] = argmax_f1(pt2, st2, te[TO].y[torch.as_tensor(ut2)].numpy() == 1)
    return result


def _load_trained(ctx: Context, cfg: dict, device):
    import torch
    from .jobs_gbt import Paths
    args = _args(cfg)
    blob = load_data(cfg, Paths.raw[cfg["dataset"]] / f"{cfg['dataset']}_Trans.csv")
    _add_ids(blob)
    model, _, transform, _, _ = _model(cfg, blob, args, device)
    path = ctx.runner.jobs_dir / cfg["train_job"] / "ckpt" / "best.bin"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    return model, blob, args, transform


def mgnn_replay(ctx: Context, cfg: dict) -> dict:
    """A trained model's scores in time buckets, where the graph holds only edges up to each bucket's end. With stock weights
    this is a diagnostic: a model trained on a whole-window view degrades partly from the mismatch. With weights from
    train_cadence the training and scoring views match. Validation gets the same treatment, so thresholds match the view."""
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, blob, args, transform = _load_trained(ctx, cfg, device)
    tr_i, val_i, te_i = (x.numpy() for x in blob["idx"])
    file_of_val = np.concatenate([tr_i, val_i])
    cad = cfg["cadence"]
    state = ctx.load_state({"done": []})
    buckets = 0
    for name, data, seeds, to_file in (("scores_val", blob["data"][1], val_i, file_of_val), ("scores_test", blob["data"][2], te_i, None)):
        times = data[TO].timestamps.numpy().astype(np.int64)
        bucket = times[seeds] // cad
        keys = [f"{name}_{int(b)}" for b in np.unique(bucket)]
        buckets += len(keys)
        if (ctx.dir / f"{name}.parquet").exists():
            continue
        for b, key in zip(np.unique(bucket), keys):
            if key in state["done"]:
                continue
            ctx.check(120)
            d, kept = subgraph_until(data, times, int((b + 1) * cad - 1))
            pos_of = np.full(len(times), -1, dtype=np.int64)
            pos_of[kept] = np.arange(len(kept))
            i_, p_, s_ = score_edges(model, d, pos_of[seeds[bucket == b]], args, transform, device, check=ctx.check, train_mode=official_mode(cfg))
            np.savez_compressed(ctx.dir / f"bucket_{key}.npz", ids=i_, p=p_, s=s_)  # ids are positions in the whole split graph, kept in edge_attr
            state["done"].append(key)
            ctx.save_state(state)
        parts = [np.load(ctx.dir / f"bucket_{key}.npz") for key in keys]
        ids = np.concatenate([x["ids"] for x in parts])
        rows = ids if to_file is None else to_file[ids]
        _write(ctx, name, blob["edge_id"][rows], np.concatenate([x["p"] for x in parts]), np.concatenate([x["s"] for x in parts]), len(seeds))
        for key in keys:
            (ctx.dir / f"bucket_{key}.npz").unlink()
    return {"cadence": cad, "buckets": buckets, "train_job": cfg["train_job"], "train_cadence": cfg.get("train_cadence")}


def mgnn_train_norm(ctx: Context, cfg: dict) -> dict:
    """Stock weights with test features standardised by training statistics, removing the per-snapshot z_norm."""
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, blob, args, transform = _load_trained(ctx, cfg, device)
    te = blob["data"][2]
    (m_tr, s_tr), _, (m_te, s_te) = blob["z"]
    if args.ports:  # the official hetero builder swaps the last two columns in place, after normalising
        for v in (m_tr, s_tr, m_te, s_te):
            v[[-1, -2]] = v[[-2, -1]]
    fix = lambda x: (x * s_te + m_te - m_tr) / torch.where(s_tr == 0, torch.ones_like(s_tr), s_tr)  # noqa: E731
    d = copy.copy(te)
    for et in (TO, REV):
        attr = te[et].edge_attr
        d[et].edge_attr = torch.cat([attr[:, :1], fix(attr[:, 1:])], 1)
    i_, p_, s_ = score_edges(model, d, blob["idx"][2].numpy(), args, transform, device, check=ctx.check, train_mode=official_mode(cfg))
    _write(ctx, "scores_test", blob["edge_id"][i_], p_, s_, len(blob["idx"][2]))
    return {"variant": "train_normalisation", "train_job": cfg["train_job"]}


LAUNCHER = """
import sys, runpy
try:
    import datatable
    used = "real"
except ImportError:
    from amlh.datatable_shim import install
    install()
    used = "shim"
sys.stderr.write("DATATABLE=" + used + chr(10))
script, data = sys.argv[1], sys.argv[2]
sys.argv = [script, data]
runpy.run_path(script, run_name="__main__")
"""


def run_official_formatter(script: Path, head: Path, install: bool = True) -> tuple[int, str, str]:
    """Runs the official formatter script. Uses the real datatable when it can be installed and amlh.datatable_shim when it cannot,
    and says which. Returns (exit code, stderr, 'real' or 'shim')."""
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datatable"], check=False, capture_output=True)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    res = subprocess.run([sys.executable, "-c", LAUNCHER, str(script), str(head)], capture_output=True, text=True, env=env)
    used = "shim" if "DATATABLE=shim" in res.stderr else "real"
    return res.returncode, res.stderr, used


def mgnn_format_check(ctx: Context, cfg: dict) -> dict:
    """Pilot gate: the port must match the official formatter. Runs both on the first rows of the raw file."""
    from .jobs_gbt import Paths
    raw = Paths.raw[cfg["dataset"]] / f"{cfg['dataset']}_Trans.csv"
    n = cfg.get("rows", 200000)
    work = ctx.dir / "check"
    work.mkdir(exist_ok=True)
    head = work / "raw_head.csv"
    with open(raw) as fin, open(head, "w") as fout:
        for i, line in enumerate(fin):
            if i > n:
                break
            fout.write(line)
    code, stderr, used = run_official_formatter(Path(MgnnPaths.code).resolve() / "format_kaggle_files.py", head, cfg.get("install_datatable", True))
    official = work / "formatted_transactions.csv"
    if code != 0 or not official.exists():
        raise RuntimeError(f"official formatter failed: {stderr[-800:]}")
    ported = work / "ported.csv"
    format_transactions(head, ported)
    a, b = pl.read_csv(official), pl.read_csv(ported)
    # Measured on HI-Small (D-020): the official datatable path and the Python port write the two amount columns through
    # different float-to-text formatting, so about 28% of rows differ at ~1e-16 relative (e.g. 10.299999999999999 vs
    # 10.300000) -- text-formatting noise, not a port bug. Every other column (ids, timestamps, currencies, the label)
    # must still match exactly: only the two amount columns get a tolerance, and it is far tighter than the noise seen.
    AMOUNT_COLS, RTOL, ATOL = {"Amount Sent", "Amount Received"}, 1e-9, 1e-6
    diffs = {}
    for c in a.columns:
        if c in AMOUNT_COLS:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            diffs[c] = int((np.abs(x - y) > ATOL + RTOL * np.abs(y)).sum())
        else:
            diffs[c] = int((a[c] != b[c]).sum())
    for f in (head, official, ported):
        f.unlink()
    work.rmdir()
    result = {"rows": a.height, "differences": diffs, "match": all(v == 0 for v in diffs.values()), "datatable": used}
    if not result["match"]:
        raise ValueError(f"port differs from official formatter: {json.dumps(diffs)}")
    return result


def _isolated(name: str):
    """What the runner calls: the same function, run in its own process so the official preprocessing's memory is returned when it ends."""
    def job(ctx: Context, cfg: dict) -> dict:
        from .isolate import run_isolated
        return run_isolated(ctx, cfg, "amlh.mgnn", name)
    job.__name__ = name
    return job


REGISTRY = {name: _isolated(name) for name in ("mgnn_train", "mgnn_replay", "mgnn_train_norm", "mgnn_format_check")}
