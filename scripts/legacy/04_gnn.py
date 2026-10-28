"""Train and evaluate the temporal GNN at one decision horizon (retrained per horizon)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.features import INF_H, build_graph  # noqa: E402
from amlh.gnn import TemporalDirectedGNN  # noqa: E402
from amlh.metrics import attempt_outcomes, best_f1_threshold, event_metrics, summarize_attempts, threshold_for_budget  # noqa: E402
from amlh.sampler import sample_batch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI-Small")
ap.add_argument("--max-day", type=int, default=None)
ap.add_argument("--H", default="0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--neg-rate", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--eval-batch", type=int, default=2048)
ap.add_argument("--val-neg-rate", type=float, default=0.05)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos-weight", type=float, default=10.0)
ap.add_argument("--sp", type=int, default=8)
ap.add_argument("--sf", type=int, default=8)
ap.add_argument("--s2p", type=int, default=3)
ap.add_argument("--s2f", type=int, default=3)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--tag", required=True)
ap.add_argument("--max-train-seeds", type=int, default=None)
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
P = ROOT / f"data/processed/{args.dataset}"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
if args.max_day is not None:
    ev = ev.filter(pl.col("t") < 86400 * (args.max_day + 1))
t = ev["t"].to_numpy()
src = ev["src"].to_numpy().astype(np.int64)
dst = ev["dst"].to_numpy().astype(np.int64)
usd = ev["usd"].to_numpy()
y = ev["label"].to_numpy().astype(np.int64)
N = len(t)
i_day, j_day = card["split_days"]
day = t // 86400
split = np.where(day < i_day, 0, np.where(day < j_day, 1, 2)).astype(np.int8)
view_end = np.where(split == 0, i_day * 86400 - 1, np.where(split == 1, j_day * 86400 - 1, t.max()))
g = build_graph(t, src, dst, usd)
H = INF_H if args.H == "inf" else int(args.H)

log_usd = torch.from_numpy(np.log1p(usd).astype(np.float32))
mu, sigma = log_usd[split == 0].mean(), log_usd[split == 0].std()
log_usd_n = (log_usd - mu) / sigma
cur_p = torch.from_numpy(ev["cur_paid_id"].to_numpy().astype(np.int64))
cur_r = torch.from_numpy(ev["cur_recv_id"].to_numpy().astype(np.int64))
fmt = torch.from_numpy(ev["fmt_id"].to_numpy().astype(np.int64))
same_cur = (cur_p == cur_r).float()
is_self = torch.from_numpy((src == dst).astype(np.float32))
t_t = torch.from_numpy(t.astype(np.float64))
n_cur = int(max(cur_p.max(), cur_r.max())) + 1
n_fmt = int(fmt.max()) + 1


def edge_features(pos: np.ndarray, t_seed: torch.Tensor):
    """pos may contain -1 padding; t_seed broadcasts to pos. Returns cont, cur_p, cur_r, fmt, mask."""
    mask = torch.from_numpy(pos >= 0)
    p = torch.from_numpy(np.where(pos >= 0, pos, 0))
    dt = (t_t[p] - t_seed).float()
    cont = torch.stack([log_usd_n[p], torch.sign(dt) * torch.log1p(dt.abs() / 60.0) / 5.0,
                        (dt >= 0).float(), same_cur[p], is_self[p]], -1)
    cont = cont * mask.unsqueeze(-1)
    return cont, cur_p[p], cur_r[p], fmt[p], mask


def forward_batch(model, seeds):
    hop1, _, hop2 = sample_batch(seeds, H, g["t"], g["src"], g["dst"], view_end, g["out_ptr"], g["out_idx"],
                                 g["in_ptr"], g["in_idx"], args.sp, args.sf, args.s2p, args.s2f)
    ts = t_t[torch.from_numpy(seeds)]
    c0, cp0, cr0, f0, _ = edge_features(seeds, ts)
    seed_e = model.enc(c0, cp0, cr0, f0)
    c1, cp1, cr1, f1, m1 = edge_features(hop1, ts.view(-1, 1, 1, 1))
    e1 = model.enc(c1, cp1, cr1, f1) * m1.unsqueeze(-1)
    c2, cp2, cr2, f2, m2 = edge_features(hop2, ts.view(-1, 1, 1, 1, 1, 1))
    e2 = model.enc(c2, cp2, cr2, f2) * m2.unsqueeze(-1)
    return model(seed_e, e1, m1, e2, m2)


@torch.no_grad()
def predict(model, rows):
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), args.eval_batch):
        s = rows[i:i + args.eval_batch]
        out[i:i + len(s)] = torch.sigmoid(forward_batch(model, s)).numpy()
    return out


tr_pos = np.where((split == 0) & (y == 1))[0]
tr_neg = np.where((split == 0) & (y == 0))[0]
val_rows = np.where(split == 1)[0]
te_rows = np.where(split == 2)[0]
# Early stopping uses all validation positives and a fixed negative sample; thresholds use full validation.
r_val = np.random.default_rng(777)
val_s = np.sort(np.r_[val_rows[y[val_rows] == 1],
                      r_val.choice(val_rows[y[val_rows] == 0], int((y[val_rows] == 0).sum() * args.val_neg_rate), replace=False)])

model = TemporalDirectedGNN(n_cur, n_fmt, args.d)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
n_params = sum(p.numel() for p in model.parameters())
print(f"H={args.H} seed={args.seed} params={n_params} train_pos={len(tr_pos)} neg/epoch={int(len(tr_neg) * args.neg_rate)} "
      f"val_stop={len(val_s)} val={len(val_rows)} test={len(te_rows)}", flush=True)
best = (-1.0, None, 0)
log = []
for ep in range(args.epochs):
    t0 = time.time()
    model.train()
    seeds = np.r_[tr_pos, rng.choice(tr_neg, int(len(tr_neg) * args.neg_rate), replace=False)]
    rng.shuffle(seeds)
    if args.max_train_seeds:
        seeds = seeds[:args.max_train_seeds]
    total, nb = 0.0, 0
    pw = torch.tensor(args.pos_weight)
    for i in range(0, len(seeds), args.batch):
        s = seeds[i:i + args.batch]
        logits = forward_batch(model, s)
        loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[s]).float(), pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
        nb += 1
    t1 = time.time()
    sv = predict(model, val_s)
    m = event_metrics(y[val_s], sv, best_f1_threshold(y[val_s], sv)[0])
    print(f"ep {ep} loss={total / nb:.4f} val_stop PR={m['pr_auc']:.4f} F1={m['f1']:.4f} "
          f"train={t1 - t0:.0f}s eval={time.time() - t1:.0f}s", flush=True)
    log.append({"epoch": ep, "loss": total / nb, "val_pr": m["pr_auc"], "val_f1": m["f1"], "s": round(time.time() - t0, 1)})
    if m["pr_auc"] > best[0]:
        best = (m["pr_auc"], {k: v.clone() for k, v in model.state_dict().items()}, ep)
    elif ep - best[2] >= args.patience:
        break

model.load_state_dict(best[1])
t0 = time.time()
s_val = predict(model, val_rows)
s_te = predict(model, te_rows)
t_inf = time.time() - t0
tau, f1v = best_f1_threshold(y[val_rows], s_val)
core = t[te_rows] < 86400 * 10
row = {"model": "tdgnn", "H": args.H, "seed": args.seed, "best_epoch": best[2], "tau": tau, "val_f1": f1v,
       "val_pr_auc": float(event_metrics(y[val_rows], s_val, tau)["pr_auc"]), "t_infer_s": round(t_inf, 1),
       "throughput_eps": len(val_rows + 0) / max(t_inf, 1e-9) if False else (len(val_rows) + len(te_rows)) / max(t_inf, 1e-9),
       "params": n_params, "log": log, "args": vars(args)}
row["test"] = event_metrics(y[te_rows], s_te, tau)
if (~core).any():
    row["test_core"] = event_metrics(y[te_rows][core], s_te[core], tau)
    row["test_tail"] = event_metrics(y[te_rows][~core], s_te[~core], tau)
mem = pl.read_parquet(P / "pattern_members.parquet").filter(pl.col("pos") < N)
mem = mem.with_columns(pl.Series("mt", t[mem["pos"].to_numpy()]))
first = mem.group_by("pattern_id").agg(pl.col("mt").min().alias("first"))
elig = first.filter(pl.col("first") >= int(t[te_rows[0]]))["pattern_id"].to_list()
m_eval = mem.filter(pl.col("pattern_id").is_in(elig))
A_id, A_pos = m_eval["pattern_id"].to_numpy(), m_eval["pos"].to_numpy()
pos_to_te = np.full(N, -1, dtype=np.int64)
pos_to_te[te_rows] = np.arange(len(te_rows))
batch_end = int(t.max()) if args.H == "inf" else None
H_eff = 0 if args.H == "inf" else H
res = {}
for b in (100, 500, 2000):
    tb = threshold_for_budget(s_val, j_day - i_day, b)
    o = attempt_outcomes(A_id, t[A_pos], usd[A_pos], s_te[pos_to_te[A_pos]], tb, H_eff, batch_end)
    res[f"budget_{b}"] = summarize_attempts(o)
row["attempts"] = res
out = ROOT / "results" / args.tag
out.mkdir(parents=True, exist_ok=True)
(out / f"H{args.H}_s{args.seed}.json").write_text(json.dumps(row, indent=1))
np.savez_compressed(out / f"scores_H{args.H}_s{args.seed}.npz", val=s_val, test=s_te)
tc = row.get("test_core", row["test"])
print(f"RESULT H={args.H} seed={args.seed} test F1={row['test']['f1']:.4f} PR={row['test']['pr_auc']:.4f} "
      f"core F1={tc['f1']:.4f} PR={tc['pr_auc']:.4f} val F1={f1v:.4f} infer {row['throughput_eps']:.0f} ev/s", flush=True)
