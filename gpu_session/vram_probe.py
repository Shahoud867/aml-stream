"""Estimate how much GPU memory one official Multi-GNN training step needs, without a GPU.

It builds the real data and model, takes real batches from the official loader, runs forward and backward on the CPU, and
reports (a) the size of each sampled batch and (b) the peak process memory growth during the step. The CPU figure includes
the sampler's own temporaries, so it errs on the high side. A GPU adds about 0.4 GB for the CUDA context and some allocator
slack, which is added in the verdict. This is an estimate, not a measurement on the target GPU.

  python gpu_session/vram_probe.py --model gin --dataset HI-Small --batches 12
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gin", choices=["gin", "pna"])
    ap.add_argument("--dataset", default="HI-Small", choices=["HI-Small", "LI-Small"])
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--cache", default=str(ROOT.parent / "aml_cache"))
    ap.add_argument("--official", default=str(ROOT.parent / "aml_official" / "Multi-GNN"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import psutil
    import torch

    from amlh import mgnn
    from amlh.plans import FOLDER

    mgnn.MgnnPaths.code = Path(a.official)
    mgnn.MgnnPaths.work = Path(a.cache) / "mgnn"
    mgnn.MgnnPaths.work.mkdir(parents=True, exist_ok=True)
    cfg = {"dataset": a.dataset, "folder": FOLDER[a.dataset], "model": a.model, "seed": 1, "batch_size": a.batch_size}
    t0 = time.time()
    blob = mgnn.load_data(cfg, Path(a.data) / f"{a.dataset}_Trans.csv")
    print(f"data ready in {time.time() - t0:.0f} s", flush=True)
    mgnn._add_ids(blob)
    args = mgnn._args(cfg)
    model, optimizer, transform, loader, config = mgnn._model(cfg, blob, args, torch.device("cpu"))
    weight = torch.FloatTensor([config.w_ce1, config.w_ce2])
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    params = sum(p.numel() for p in model.parameters())
    proc = psutil.Process()
    rows = []
    model.train()
    for i, batch in enumerate(loader):
        if i >= a.batches:
            break
        n_nodes = sum(int(batch[t].num_nodes) for t in batch.node_types)
        n_edges = sum(int(batch[e].edge_index.shape[1]) for e in batch.edge_types)
        batch_bytes = sum(v.numel() * v.element_size() for s in batch.stores for v in s.values() if torch.is_tensor(v))
        base = proc.memory_info().rss
        peak = [base]
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.01)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        ids = batch[mgnn.TO].edge_attr[:, 0].detach().clone()
        mask = torch.isin(ids, ids[: max(1, a.batch_size)])
        mgnn._strip_id(batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[mgnn.TO]
        loss = loss_fn(out[mask[: out.shape[0]]], batch[mgnn.TO].y[mask[: out.shape[0]]])
        loss.backward()
        optimizer.step()
        stop.set()
        th.join()
        grown = (peak[0] - base) / 2**30
        rows.append({"nodes": n_nodes, "edges": n_edges, "batch_gb": round(batch_bytes / 2**30, 3), "step_peak_growth_gb": round(grown, 2)})
        print(i, rows[-1], flush=True)
    worst = max(r["step_peak_growth_gb"] for r in rows)
    est = worst + 0.4 + 0.15 * worst
    verdict = "FITS 10 GB (estimate)" if est < 8.5 else ("MARGINAL on 10 GB" if est < 10 else "DOES NOT FIT 10 GB (estimate)")
    res = {"model": a.model, "dataset": a.dataset, "params": params, "batches": rows, "worst_step_growth_gb": worst, "estimated_gpu_peak_gb": round(est, 1), "verdict": verdict}
    print(json.dumps({k: v for k, v in res.items() if k != "batches"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
