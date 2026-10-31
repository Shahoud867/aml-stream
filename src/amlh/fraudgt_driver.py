"""Runs inside its own process with the published FraudGT code on the path. NOT YET EXECUTED: the pilot must prove it.

Repeats FraudGT's custom_train loop around its own train_epoch, eval_epoch, loaders, model and logger, adding resume.
Differences from that loop, each needed for the audit:
  * test data is scored only after training (same selection, far less GPU time);
  * model selection is unchanged, including FraudGT's validation F1 on a shuffled subset of edges;
  * every validation and test edge then gets a score from an unshuffled pass, and the `sampled` column records which
    edges the sampler returned by itself, because FraudGT's own evaluation drops the rest.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
import types
from pathlib import Path

import numpy as np
import polars as pl

from .isolate import OUT_OF_TIME, ChildRunner
from .mgnn import _dedupe
from .mgnn_format import format_transactions
from .runner import Context, Interrupted, JobSpec, atomic_write_json
from .sealing import is_sealed

BATCH_ONLY = ("edge_index", "edge_label", "edge_label_index", "input_id", "e_id")


def add_back(bs, node_ids, full, wanted):
    """Puts seed edges the sampler did not return into the batch, as Multi-GNN does on Small_J and Small_Q.
    Returns the positions it added. bs and full are the batch's and the split graph's stores for the scored edge type."""
    import torch
    missing = wanted[~torch.isin(wanted, bs.e_id)]
    if len(missing) == 0:
        return missing
    ordered, perm = torch.sort(node_ids)
    ends = full.edge_index[:, missing]
    at = torch.searchsorted(ordered, ends).clamp(max=len(ordered) - 1)
    if (ordered[at] != ends).any():
        raise ValueError("a seed edge's endpoint is not in its batch")
    n_old = bs.edge_index.size(1)
    for key in list(bs.keys()):
        value = bs[key]
        if key not in BATCH_ONLY and torch.is_tensor(value) and value.size(0) == n_old and key in full:
            bs[key] = torch.cat([value, full[key][missing]], 0)
    bs.edge_index = torch.cat([bs.edge_index, perm[at]], 1)
    bs.e_id = torch.cat([bs.e_id, missing], 0)
    return missing


def probability(pred):
    """The positive-class probability, for either output shape FraudGT's loss accepts."""
    import torch
    if pred.ndim > 1 and pred.size(1) > 1:
        return torch.softmax(pred, -1)[:, 1]
    return torch.sigmoid(pred.view(-1))


def score_split(model, dataset, split: str, ctx: Context, cfg, sampler_transform, limit: int | None, device):
    """One unshuffled pass over a split's seed edges. Returns (graph positions, probabilities, returned by the sampler)."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.utils import mask_to_index
    task = cfg.dataset.task_entity
    data = dataset[split]
    seeds = mask_to_index(data[task].split_mask)
    if limit:
        seeds = seeds[:limit]
    loader = LinkNeighborLoader(data=data, num_neighbors=cfg.train.neighbor_sizes, edge_label_index=(task, data[task].edge_index[:, seeds]),
                                edge_label=data[task].y[seeds], batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, shuffle=False,
                                transform=sampler_transform)
    ids, probs, sampled = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ctx.check()
            bs = batch[task]
            # The head selects seed edges through split_inds[input_id], which is the same as seeds[input_id] here.
            wanted = seeds[bs.input_id.cpu()]
            added = add_back(bs, batch[task[0]].n_id.cpu(), data[task], wanted)
            e_id = bs.e_id.cpu()
            mask = torch.isin(e_id, wanted)
            batch.split = split
            batch.to(device)
            pred, _ = model(batch)
            if pred.size(0) != int(mask.sum()):
                raise ValueError(f"model returned {pred.size(0)} predictions for {int(mask.sum())} seed edges")
            kept = e_id[mask]
            ids.append(kept.numpy().astype(np.int64))
            probs.append(probability(pred).float().cpu().numpy())
            sampled.append((~torch.isin(kept, added)).numpy())
    uniq, p, was = _dedupe(np.concatenate(ids), np.concatenate(probs), np.concatenate(sampled))
    if len(uniq) != len(seeds):
        raise ValueError(f"{split}: scored {len(uniq)} of {len(seeds)} edges")
    return uniq, p, was


def map_to_rows(positions: np.ndarray, split_store, formatted: pl.DataFrame) -> np.ndarray:
    """Split graphs are prefixes of the time-sorted file, so a graph position is a row of the formatted file. Checked, not assumed."""
    src = split_store.edge_index[0, positions].numpy()
    dst = split_store.edge_index[1, positions].numpy()
    if (formatted["from_id"].to_numpy()[positions] != src).any() or (formatted["to_id"].to_numpy()[positions] != dst).any():
        raise ValueError("graph positions do not match the formatted file")
    return formatted["EdgeID"].to_numpy()[positions]


def format_beside_raw(path):
    """Stands in for FraudGT's formatter, which needs datatable. Writes where FraudGT's process() reads it back: beside the raw file.
    The port is checked against Multi-GNN's formatter by mgnn_format_check, and FraudGT's own formatter matches it line for line by reading, not by test."""
    format_transactions(Path(path), Path(path).parent / "formatted_transactions.csv")


def stub_datatable():
    """FraudGT imports datatable only for its formatter, which the port replaces, and wheels do not exist for every Python."""
    try:
        import datatable  # noqa: F401
    except ImportError:
        stub = types.ModuleType("datatable")
        stub.f = stub.join = stub.sort = None
        sys.modules["datatable"] = stub


def main(spec_path: str):
    spec = json.loads(Path(spec_path).read_text())
    job_dir = Path(spec["job_dir"])
    runner = ChildRunner(spec["deadline"])
    ctx = Context(job_dir, runner, JobSpec(spec["job_id"], "fraudgt_train", {}))

    def stop(signum, frame):
        runner.stop_requested = True
    signal.signal(signal.SIGTERM, stop)

    stub_datatable()
    code = spec["code"]
    sys.path.insert(0, code)
    os.chdir(code)
    sys.argv = ["fraudgt", "--cfg", spec["config"], "--repeat", "1", "--gpu", "0", *spec["overrides"]]
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import fraudGT  # noqa: F401 (registers its modules)
    from torch_geometric import seed_everything

    from fraudGT.datasets import aml_dataset
    from fraudGT.graphgym.cmd_args import parse_args
    from fraudGT.graphgym.config import cfg, dump_cfg, load_cfg, set_cfg
    from fraudGT.graphgym.loader import create_loader
    from fraudGT.graphgym.logger import setup_printing
    from fraudGT.graphgym.model_builder import create_model
    from fraudGT.graphgym.optimizer import create_optimizer, create_scheduler
    from fraudGT.graphgym.utils.epoch import is_eval_epoch
    from fraudGT.logger import create_logger
    from fraudGT.sampler.custom_sampler import AddEgoIdsForLinkNeighbor
    from fraudGT.train.custom_train import eval_epoch, train_epoch
    from fraudGT.utils import custom_set_out_dir, custom_set_run_dir, new_optimizer_config, new_scheduler_config

    data_dir = Path(spec["data_dir"])
    formatted_csv = data_dir / "AML" / "formatted_transactions.csv"  # beside the raw file, where FraudGT's process() reads it back
    aml_dataset.format_dataset = format_beside_raw

    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    dump_cfg(cfg)
    torch.set_num_threads(cfg.num_threads)
    custom_set_run_dir(cfg, 0)
    setup_printing()
    cfg.run_id = 0
    seed_everything(cfg.seed)
    cfg.device = "cuda:0"
    device = torch.device(cfg.device)

    loaders, dataset = create_loader(returnDataset=True)
    loggers = create_logger()
    model = create_model(dataset=dataset)
    optimizer = create_optimizer(model.named_parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    from fraudGT.graphgym.utils.comp_budget import params_count
    cfg.params = params_count(model)

    state = {"epoch": 0, "best_val": -1.0, "best_epoch": -1, "train_seconds": 0.0, "epoch_seconds": 0.0, "truncated": False}

    def read_ckpt(path):
        return torch.load(path, map_location=device, weights_only=False)
    saved = ctx.load_checkpoint(read_ckpt)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        np.random.set_state(saved["np_rng"])
        ctx.log({"event": "resumed", "epoch": state["epoch"]})

    metric = cfg.metric_best
    budget = spec["max_train_hours"] * 3600
    try:
        while state["epoch"] < cfg.optim.max_epoch:
            if state["train_seconds"] >= budget:
                state["truncated"] = True
                break
            ctx.check(state["epoch_seconds"] * cfg.train.eval_period * 1.2)
            t0 = time.time()
            epoch = state["epoch"]
            train_epoch(epoch, loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
            row = {"event": "epoch", "epoch": epoch, "train_loss": loggers[0].write_epoch(epoch).get("loss")}
            if is_eval_epoch(epoch, 0):
                eval_epoch(loggers[1], loaders[1], model, split="val")
                perf = loggers[1].write_epoch(epoch)
                if metric not in perf:
                    raise KeyError(f"FraudGT's logger reports {sorted(perf)}, without {metric}")
                row["val"] = perf[metric]
                if perf[metric] > state["best_val"]:
                    state.update(best_val=perf[metric], best_epoch=epoch)
                    ctx.save_checkpoint(lambda p: torch.save(model.state_dict(), p), name="best.bin", meta={"epoch": epoch})
            scheduler.step()
            state["epoch"] = epoch + 1
            state["epoch_seconds"] = time.time() - t0
            state["train_seconds"] += state["epoch_seconds"]
            ctx.log({**row, "seconds": state["epoch_seconds"]})
            ctx.save_checkpoint(lambda p: torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": dict(state),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "np_rng": np.random.get_state()}, p), meta={"epoch": epoch})

        best = ctx.load_checkpoint(read_ckpt, names=("best.bin",))
        if best is None:
            raise RuntimeError("training produced no best checkpoint")
        model.load_state_dict(best)
        result = {"config": spec["config_name"], "dataset": spec["dataset"], "seed": cfg.seed, "device": torch.cuda.get_device_name(0), **state}
        task = cfg.dataset.task_entity
        n_test = int(dataset["test"][task].split_mask.sum())
        # What FraudGT's own evaluation can see: its test loader stops after val.iter_per_epoch shuffled batches.
        steps = cfg.val.iter_per_epoch
        result["official_eval_seed_fraction"] = float(min(1.0, steps * cfg.train.batch_size / n_test)) if steps > 0 else 1.0
        if not is_sealed(spec["dataset"]):
            ctx.check()
            eval_epoch(loggers[2], loaders[2], model, split="test")
            result["official_style_test"] = {k: v for k, v in loggers[2].write_epoch(state["epoch"]).items() if isinstance(v, (int, float))}
        formatted = pl.read_csv(formatted_csv, columns=["EdgeID", "from_id", "to_id"])
        transform = AddEgoIdsForLinkNeighbor() if cfg.train.add_ego_id else None
        for name, split in (("scores_val", "val"), ("scores_test", "test")):
            if (job_dir / f"{name}.parquet").exists():
                continue
            pos, p, sampled = score_split(model, dataset, split, ctx, cfg, transform, spec.get("score_limit"), device)
            pl.DataFrame({"event_id": map_to_rows(pos, dataset[split][task], formatted), "score": p.astype(np.float32), "sampled": sampled}).write_parquet(job_dir / f"{name}.parquet")
            result[f"coverage_{split}"] = float(sampled.mean())
        shutil.copy(Path(cfg.out_dir) / "config.yaml", job_dir / "effective_config.yaml")
        atomic_write_json(job_dir / "result.json", result)
    except Interrupted as exc:
        ctx.log({"event": "out_of_time", "reason": str(exc)})
        sys.exit(OUT_OF_TIME)


if __name__ == "__main__":
    main(sys.argv[1])
