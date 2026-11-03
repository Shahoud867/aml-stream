"""Generates every table, figure and in-text number in the report from results/ and logs.
Nothing in the report's preliminary-evidence section is typed by hand: prose reads LaTeX macros written here.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.metrics import attempt_outcomes  # noqa: E402

OUT = ROOT / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# Palette: validated reference categorical slots; text tokens; recessive grid.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": SURFACE, "lines.linewidth": 1.6, "lines.markersize": 5,
})
macros: dict[str, str] = {}


def macro(name: str, value: str):
    macros[name] = value


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


H_ORDER = ["0", "1", "60", "600", "3600", "21600", "86400", "259200", "inf"]
H_LABEL = {"0": "strict\npast", "1": "same\nminute", "60": "1 min", "600": "10 min", "3600": "1 h", "21600": "6 h",
           "86400": "24 h", "259200": "72 h", "inf": "whole\nwindow"}
H_TEX = {"0": "strict past", "1": "same minute", "60": "1 min", "600": "10 min", "3600": "1 h", "21600": "6 h",
         "86400": "24 h", "259200": "72 h", "inf": "whole window"}


def load_sweep(tag):
    p = ROOT / "results" / tag / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def agg(rows, H, f):
    v = np.array([f(r) for r in rows if r["H"] == H], dtype=float)
    return v.mean() if len(v) else np.nan, v.std() if len(v) else np.nan, v


def split_setup(dataset, max_day=None):
    P = ROOT / f"data/processed/{dataset}"
    card = json.loads((P / "data_card.json").read_text())
    ev = pl.read_parquet(P / "events.parquet", columns=["pos", "t", "usd", "label", "src", "dst"])
    if max_day is not None:
        ev = ev.filter(pl.col("t") < 86400 * (max_day + 1))
    t = ev["t"].to_numpy()
    i, j = card["split_days"]
    day = t // 86400
    split = np.where(day < i, 0, np.where(day < j, 1, 2))
    return dict(card=card, t=t, usd=ev["usd"].to_numpy(), y=ev["label"].to_numpy(), split=split,
                te=np.where(split == 2)[0], va=np.where(split == 1)[0], P=P, N=len(t), val_days=j - i)


def attempt_arrays(S):
    """Attempts whose first member lies in the test window; all members in the file are scored."""
    t, te = S["t"], S["te"]
    mem = pl.read_parquet(S["P"] / "pattern_members.parquet").filter(pl.col("pos") < S["N"])
    mem = mem.with_columns(pl.Series("mt", t[mem["pos"].to_numpy()]))
    first = mem.group_by("pattern_id").agg(pl.col("mt").min().alias("f"))
    elig = first.filter(pl.col("f") >= int(t[te[0]]))["pattern_id"].to_list()
    m = mem.filter(pl.col("pattern_id").is_in(elig))
    idx = np.full(S["N"], -1, dtype=np.int64)
    idx[te] = np.arange(len(te))
    typ = pl.read_parquet(S["P"] / "attempts.parquet")
    return m["pattern_id"].to_numpy(), m["pos"].to_numpy(), idx, dict(zip(typ["pattern_id"].to_list(), typ["typology"].to_list()))


def sv_for(S, A, score_te, H, budget, mode):
    """Stoppable-value outcomes at an alert budget. mode='val' sets the threshold from validation scores (deployable);
    mode='test' equalises the realised alert count across horizons on the test window (sensitivity analysis)."""
    aid, apos, idx, _ = A
    s_te, s_va = score_te
    if mode == "val":
        k = int(round(budget * S["val_days"]))
        tau = float(np.partition(s_va, len(s_va) - k)[len(s_va) - k])
    else:
        k = int(round(budget * 2))  # the core spans two calendar days
        tau = float(np.partition(s_te, len(s_te) - k)[len(s_te) - k])
    batch_end = int(S["t"].max()) if H == "inf" else None
    H_eff = 0 if H == "inf" else int(H)
    return attempt_outcomes(aid, S["t"][apos], S["usd"][apos], s_te[idx[apos]], tau, H_eff, batch_end)


def boot_diff(a, b, n=4000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


# ---------------------------------------------------------------- tail audit
audit = json.loads((ROOT / "results/tail_audit.json").read_text())
PUB = {"HI-Small": ("63.23", "68.16"), "LI-Small": ("27.30", "33.07"), "HI-Medium": ("65.70", "65.67"), "LI-Medium": ("28.16", "33.23")}
DOC_DAYS = {"HI-Small": 10, "LI-Small": 10, "HI-Medium": 16, "LI-Medium": 16}
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Dataset & Days in file & Documented & Tail events & Tail illicit & Share of test & Date-rule & XGB+GFs & Multi-PNA+EU \\",
         r" & & days & & rate (\%) & illicit (\%) & F1 (\%) & (publ.) & (publ.) \\", r"\midrule"]
for k in ("HI-Small", "LI-Small", "HI-Medium", "LI-Medium"):
    a = audit[k]
    lines.append(f"{k} & {a['days']} & {DOC_DAYS[k]} & {a['tail_events']:,} & {a['tail_illicit_rate_pct']:.1f} & "
                 f"{a['tail_share_of_test_illicit_pct']:.1f} & {a['trivial_flag_tail_f1_pct']:.2f} & {PUB[k][0]} & {PUB[k][1]} \\\\")
    s = k.replace("-", "")
    macro(f"Tail{s}Fone", f"{a['trivial_flag_tail_f1_pct']:.2f}")
    macro(f"Tail{s}Share", f"{a['tail_share_of_test_illicit_pct']:.1f}")
    macro(f"Tail{s}Rate", f"{a['tail_illicit_rate_pct']:.1f}")
    macro(f"Tail{s}Events", f"{a['tail_events']:,}".replace(",", "{,}"))
    macro(f"Tail{s}Days", str(a["days"]))
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_tail.tex").write_text("\n".join(lines))

a = audit["HI-Small"]
n = np.array(a["daily_n"], float)
ill = np.array(a["daily_illicit"], float)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
d = np.arange(len(n))
tail = d >= a["tail_start_day"]
ax[0].bar(d, n, width=0.72, color=C1)
ax[0].set_yscale("log")
ax[0].set_title("Transfers per day (log scale)")
ax[0].set_xlabel("Day of file")
ax[1].bar(d, 100 * ill / n, width=0.72, color=C1)
ax[1].set_title("Illicit share of each day's transfers (%)")
ax[1].set_xlabel("Day of file")
for x in ax:
    x.axvline(a["tail_start_day"] - 0.5, color=MUTED, lw=1, ls="--")
    x.grid(axis="x", visible=False)
ax[0].text(a["tail_start_day"] - 0.1, n.max() * 0.5, "documented\nspan ends", color=MUTED, fontsize=7.5, va="top")
ax[1].text(a["tail_start_day"] - 0.8, 62, "documented\nspan ends", color=MUTED, fontsize=7.5, va="top", ha="right")
fig.tight_layout()
fig.savefig(OUT / "fig_tail.pdf")
fig.savefig(OUT / "fig_tail.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- horizon sweep
tables = {}
for tag, dataset, max_day, name in [("gbt_hi_small_released", "HI-Small", None, "HIS"),
                                    ("gbt_hi_small_documented", "HI-Small", 9, "HISD"),
                                    ("gbt_li_small_released", "LI-Small", None, "LIS"),
                                    ("gbt_li_small_documented", "LI-Small", 9, "LISD")]:
    res = load_sweep(tag)
    if res is None:
        continue
    n_seeds = max(len([r for r in res["rows"] if r["H"] == h]) for h in H_ORDER)
    rows = [r for r in res["rows"] if len([x for x in res["rows"] if x["H"] == r["H"]]) == n_seeds]
    hs = [h for h in H_ORDER if any(r["H"] == h for r in rows)]
    tables[name] = dict(rows=rows, hs=hs, dataset=dataset, max_day=max_day, tag=tag)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Horizon $H$ & Test F1 & Core F1 & Core PR-AUC & Detected & SV$_{100}$ & SV$_{500}$ & SV$_{2000}$ \\", r"\midrule"]
    for h in hs:
        m_f1 = agg(rows, h, lambda r: r["test"]["f1"])
        m_cf = agg(rows, h, lambda r: r.get("test_core", r["test"])["f1"])
        m_cp = agg(rows, h, lambda r: r.get("test_core", r["test"])["pr_auc"])
        det = agg(rows, h, lambda r: r["attempts"]["budget_500"]["detected_frac"])
        sv = [agg(rows, h, lambda r, b=b: r["attempts"][f"budget_{b}"]["value_share_mean"]) for b in (100, 500, 2000)]
        lines.append(f"{H_TEX[h]} & {100 * m_f1[0]:.1f}$\\pm${100 * m_f1[1]:.1f} & {100 * m_cf[0]:.1f}$\\pm${100 * m_cf[1]:.1f} & "
                     f"{m_cp[0]:.3f} & {100 * det[0]:.0f}\\% & {sv[0][0]:.3f} & {sv[1][0]:.3f} & {sv[2][0]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"tab_horizon_{name}.tex").write_text("\n".join(lines))
    for h, lab in zip(("0", "1", "3600", "86400", "inf"), ("Strict", "Tie", "Hour", "Day", "Inf")):
        macro(f"{name}F{lab}", f"{100 * agg(rows, h, lambda r: r['test']['f1'])[0]:.1f}")
        macro(f"{name}CoreF{lab}", f"{100 * agg(rows, h, lambda r: r.get('test_core', r['test'])['f1'])[0]:.1f}")
        macro(f"{name}CorePR{lab}", f"{agg(rows, h, lambda r: r.get('test_core', r['test'])['pr_auc'])[0]:.3f}")
        macro(f"{name}SV{lab}", f"{agg(rows, h, lambda r: r['attempts']['budget_500']['value_share_mean'])[0]:.3f}")

# Horizon figure, HI-Small released: quality on one axis, value on another panel (never a dual axis).
if "HIS" in tables:
    T = tables["HIS"]
    rows, hs = T["rows"], T["hs"]
    x = np.arange(len(hs))
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 4.5), sharex=True)
    for label, key, col in (("Full test split", "test", C1), ("Core only (tail removed)", "test_core", C2)):
        vals = np.array([agg(rows, h, lambda r: r[key]["f1"])[0] for h in hs]) * 100
        sd = np.array([agg(rows, h, lambda r: r[key]["f1"])[1] for h in hs]) * 100
        ax[0].errorbar(x, vals, yerr=sd, marker="o", color=col, label=label, capsize=2, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax[0].annotate(label, (x[-1], vals[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=MUTED, va="center")
    ax[0].set_ylabel("Minority F1 (%)")
    ax[0].set_title("Quality rises with look-ahead, then plateaus near one hour")
    ax[0].legend(loc="lower right", fontsize=7.5)
    for b, col in ((100, C3), (500, C1), (2000, C2)):
        vals = [agg(rows, h, lambda r, b=b: r["attempts"][f"budget_{b}"]["value_share_mean"])[0] for h in hs]
        ax[1].plot(x, vals, marker="o", color=col, label=f"{b} alerts/day", markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax[1].set_ylabel("Stoppable value share")
    ax[1].set_title("Stoppable value falls once the wait exceeds about an hour")
    ax[1].legend(loc="upper right", fontsize=7.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([H_LABEL[h] for h in hs], fontsize=8)
    ax[1].set_xlabel("Decision horizon H")
    fig.tight_layout()
    fig.savefig(OUT / "fig_horizon.pdf")
    fig.savefig(OUT / "fig_horizon.png", dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------- per-attempt paired analysis from saved scores
paired = {}
for name, T in tables.items():
    S = split_setup(T["dataset"], T["max_day"])
    A = attempt_arrays(S)
    d = ROOT / "results" / T["tag"]
    seeds = sorted({r["seed"] for r in T["rows"]})
    outcome = {}
    for h in T["hs"]:
        for seed in seeds:
            f = d / f"scores_H{h}_s{seed}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for mode in ("val", "test"):
                for b in (100, 500, 2000):
                    o = sv_for(S, A, (z["test"].astype(np.float64), z["val"].astype(np.float64)), h, b, mode)
                    outcome[(h, seed, mode, b)] = o
    paired[name] = outcome
    n_att = None
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Horizon $H$ & SV$_{500}$ (val.\ threshold) & Transfers (val.) & SV$_{500}$ (equal alerts) & Detected (equal alerts) \\", r"\midrule"]
    per_h = {}
    for h in T["hs"]:
        get = lambda mode, key: np.mean([outcome[(h, s, mode, 500)][key].astype(float).mean() for s in seeds if (h, s, mode, 500) in outcome])  # noqa: E731
        if not any((h, s, "val", 500) in outcome for s in seeds):
            continue
        per_h[h] = (get("val", "value_share"), get("test", "value_share"), get("test", "detected"))
        lines.append(f"{H_TEX[h]} & {get('val', 'value_share'):.3f} & {get('val', 'transfer_share'):.3f} & {get('test', 'value_share'):.3f} & {100 * get('test', 'detected'):.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"tab_sv_{name}.tex").write_text("\n".join(lines))
    # C3-style contrast: best H <= 1 h vs 24 h, paired on attempts and averaged over seeds.
    for mode in ("val", "test"):
        near = [h for h in ("1", "60", "600", "3600") if (h, seeds[0], mode, 500) in outcome]
        if not near or ("86400", seeds[0], mode, 500) not in outcome:
            continue
        sv_near = {h: np.mean([outcome[(h, s, mode, 500)]["value_share"] for s in seeds], axis=0) for h in near}
        best = max(sv_near, key=lambda h: sv_near[h].mean())
        late = np.mean([outcome[("86400", s, mode, 500)]["value_share"] for s in seeds], axis=0)
        m, lo, hi = boot_diff(sv_near[best], late)
        macro(f"{name}CThree{mode.capitalize()}Best", H_TEX[best])
        macro(f"{name}CThree{mode.capitalize()}Diff", f"{m:.3f}")
        macro(f"{name}CThree{mode.capitalize()}CI", f"[{lo:.3f}, {hi:.3f}]")
        macro(f"{name}Attempts", str(len(late)))
        # Same contrast for strict-past vs the best near horizon: is any waiting worthwhile?
        if ("0", seeds[0], mode, 500) in outcome:
            strict = np.mean([outcome[("0", s, mode, 500)]["value_share"] for s in seeds], axis=0)
            m2, lo2, hi2 = boot_diff(sv_near[best], strict)
            macro(f"{name}WaitGain{mode.capitalize()}", f"{m2:+.3f}")
            macro(f"{name}WaitGain{mode.capitalize()}CI", f"[{lo2:+.3f}, {hi2:+.3f}]")
    # Typology table at B = 500, deployable thresholds.
    if name == "HIS":
        typ = A[3]
        ids = outcome[("0", seeds[0], "val", 500)]["ids"]
        labels = np.array([typ[int(i)] for i in ids])
        cols = [h for h in ("0", "1", "3600", "86400") if (h, seeds[0], "val", 500) in outcome]
        lines = [r"\begin{tabular}{lr" + "rr" * len(cols) + "}", r"\toprule",
                 "Typology & $n$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{H_TEX[h]}}}" for h in cols) + r" \\",
                 " & & " + " & ".join("Det. & SV" for _ in cols) + r" \\", r"\midrule"]
        for t_ in sorted(set(labels)):
            mask = labels == t_
            cells = []
            for h in cols:
                det = np.mean([outcome[(h, s, "val", 500)]["detected"][mask].mean() for s in seeds])
                sv = np.mean([outcome[(h, s, "val", 500)]["value_share"][mask].mean() for s in seeds])
                cells += [f"{100 * det:.0f}\\%", f"{sv:.2f}"]
            lines.append(f"{t_.title()} & {mask.sum()} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "tab_typology.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- ablations
abl = {}
base_doc = tables.get("HISD")
ABL_NAMES = {"nofwd": "A1: no forward block", "nomotif": "A2: no motifs", "nopair": "A3: no pair block",
             "nohour": "A4: no hour of day", "nofmt": "A5: no payment format", "nocur": "A5b: no currencies"}
if base_doc:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Ablation & Horizon & Baseline PR-AUC & Ablated PR-AUC & Difference \\", r"\midrule"]
    any_row = False
    for key, lab in ABL_NAMES.items():
        r = load_sweep(f"abl_{key}")
        if r is None:
            continue
        for h in sorted({x["H"] for x in r["rows"]}, key=H_ORDER.index):
            b = agg(base_doc["rows"], h, lambda x: x["test"]["pr_auc"])
            v = agg(r["rows"], h, lambda x: x["test"]["pr_auc"])
            lines.append(f"{lab} & {H_TEX[h]} & {b[0]:.3f} & {v[0]:.3f} & {v[0] - b[0]:+.3f} \\\\")
            macro(f"Abl{key}H{h.replace('inf', 'I')}".replace("0", "Z").replace("1", "O").replace("3", "T").replace("6", "S"), f"{v[0] - b[0]:+.3f}")
            any_row = True
    lines += [r"\bottomrule", r"\end{tabular}"]
    if any_row:
        (OUT / "tab_ablation.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- adaptive alerting
for tag, name in (("gbt_hi_small_released", "HIS"), ("gbt_hi_small_documented", "HISD"),
                  ("gbt_li_small_released", "LIS"), ("gbt_li_small_documented", "LISD")):
    p = ROOT / "results" / tag / "cascade.json"
    if not p.exists():
        continue
    c = json.loads(p.read_text())["rows"]
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule", r"Budget & Adaptive SV & Best fixed SV & Difference & Adaptive detected & Fixed detected \\", r"\midrule"]
    for b in (100, 500, 2000):
        rs = [r for r in c if r["budget"] == b]
        ada = np.array([r["adaptive"]["per_attempt_value"] for r in rs]).mean(axis=0)
        fix = np.array([r["best_fixed_on_val"]["per_attempt_value"] for r in rs]).mean(axis=0)
        m, lo, hi = boot_diff(ada, fix)
        lines.append(f"{b} & {ada.mean():.3f} & {fix.mean():.3f} & {m:+.3f} [{lo:+.3f}, {hi:+.3f}] & "
                     f"{100 * np.mean([r['adaptive']['detected'] for r in rs]):.0f}\\% & {100 * np.mean([r['best_fixed_on_val']['detected'] for r in rs]):.0f}\\% \\\\")
        macro(f"{name}PolicyDiff{b}".replace("100", "H").replace("500", "F").replace("2000", "K"), f"{m:+.3f}")
        macro(f"{name}PolicyCI{b}".replace("100", "H").replace("500", "F").replace("2000", "K"), f"[{lo:+.3f}, {hi:+.3f}]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"tab_policy_{name}.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- timing attack
p = ROOT / "results/attack_hi_small/attack.json"
if p.exists():
    res = json.loads(p.read_text())
    rs = res["results"]
    hs = sorted({r["H"] for r in rs}, key=H_ORDER.index)
    Ws = sorted({r["W"] for r in rs})
    seeds = sorted({r["seed"] for r in rs})

    def arr(H, W, cond, key):
        out = []
        for s in seeds:
            row = next(r for r in rs if r["H"] == H and r["W"] == W and r["seed"] == s and r["cond"] == cond)
            ks = sorted(row["per_attempt"], key=int)
            out.append([float(row["per_attempt"][k][key]) for k in ks])
        return np.mean(out, axis=0)

    lines = [r"\begin{tabular}{rlrrrrrl}", r"\toprule",
             r"$W$ & Horizon & Clean det. & Attack det. & Control det. & Recall clean & Recall attack & Suppression beyond control \\", r"\midrule"]
    effect = {}
    for attack, ctrl in (("burst", "shift_burst"), ("stretch", "shift_stretch")):
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{attack.title()} vs matched-displacement control}}}} \\\\")
        for W in Ws:
            for H in hs:
                clean = arr(H, W, "none", "det")
                atk = arr(H, W, attack, "det")
                con = arr(H, W, ctrl, "det")
                rc, ra = arr(H, W, "none", "recall"), arr(H, W, attack, "recall")
                m, lo, hi = boot_diff(con, atk)
                rm, rlo, rhi = boot_diff(arr(H, W, ctrl, "recall"), ra)
                effect[(attack, W, H)] = dict(det=(m, lo, hi), recall=(rm, rlo, rhi), clean=clean.mean(), atk=atk.mean(), con=con.mean(),
                                              rc=rc.mean(), ra=ra.mean(), rcon=arr(H, W, ctrl, "recall").mean(),
                                              compl_clean=np.median(arr(H, W, "none", "completion_h")),
                                              compl_atk=np.median(arr(H, W, attack, "completion_h")))
                lines.append(f"{W // 3600}\\,h & {H_TEX[H]} & {100 * clean.mean():.0f}\\% & {100 * atk.mean():.0f}\\% & {100 * con.mean():.0f}\\% & "
                             f"{rc.mean():.3f} & {ra.mean():.3f} & {100 * m:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_attack.tex").write_text("\n".join(lines))
    (OUT / "attack_effects.json").write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in effect.items()}, indent=1, default=float))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax_, attack in zip(axes, ("burst", "stretch")):
        width = 0.8 / len(hs)
        cols = [C1, C2, C3, "#eda100"]
        for k, H in enumerate(hs):
            vals = [100 * (effect[(attack, W, H)]["rcon"] - effect[(attack, W, H)]["ra"]) for W in Ws]
            err = np.array([[100 * (effect[(attack, W, H)]["recall"][0] - effect[(attack, W, H)]["recall"][1]) for W in Ws],
                            [100 * (effect[(attack, W, H)]["recall"][2] - effect[(attack, W, H)]["recall"][0]) for W in Ws]])
            ax_.bar(np.arange(len(Ws)) + k * width, vals, width * 0.92, color=cols[k % 4], label=H_TEX[H].replace("\n", " "),
                    yerr=err, error_kw=dict(lw=0.7, ecolor=MUTED))
        ax_.axhline(0, color=MUTED, lw=0.8)
        ax_.set_xticks(np.arange(len(Ws)) + width * (len(hs) - 1) / 2)
        ax_.set_xticklabels([f"W = {W // 3600} h" for W in Ws])
        ax_.set_title(f"{attack.title()} attack")
        ax_.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall lost beyond control (points)")
    axes[1].legend(fontsize=7.5, title="Detector view", title_fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_attack.pdf")
    fig.savefig(OUT / "fig_attack.png", dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------- service and queueing
p = ROOT / "results/service_hi_small/service.json"
if p.exists():
    sv = json.loads(p.read_text())
    lines = [r"\begin{tabular}{rrrr}", r"\toprule", r"Batch size $n$ & Median (ms) & p95 (ms) & Events/s \\", r"\midrule"]
    for r in sv["profile"]:
        lines.append(f"{r['n']:,} & {1000 * r['median_s']:.1f} & {1000 * r['p95_s']:.1f} & {r['events_per_s']:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_service.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{rrrrl}", r"\toprule", r"Offered load (ev/s) & p50 (ms) & p95 (ms) & p99 (ms) & Stable? \\", r"\midrule"]
    for r in sv["table"]:
        lines.append(f"{r['rate']:,.0f} & {1000 * r['p50']:.1f} & {1000 * r['p95']:.1f} & {1000 * r['p99']:.1f} & {'yes' if r['stable'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_queue.tex").write_text("\n".join(lines))
    macro("MaxStable", f"{sv['max_stable_rate']:,.0f}".replace(",", "{,}"))
    macro("BenchRate", f"{sv['benchmark_mean_rate_eps']:.1f}")
    macro("BenchPeak", f"{sv['benchmark_peak_minute_rate_eps']:.0f}")

# ---------------------------------------------------------------- GNN
g_rows = []
for f in sorted((ROOT / "results/gnn_hi_small_released").glob("H*_s*.json")):
    g_rows.append(json.loads(f.read_text()))
if g_rows:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Horizon & GNN test F1 & GNN core F1 & GNN core PR-AUC & GBT core PR-AUC & GNN SV$_{500}$ & GBT SV$_{500}$ \\", r"\midrule"]
    gb = tables.get("HIS")
    for h in H_ORDER:
        rs = [r for r in g_rows if r["H"] == h]
        if not rs:
            continue
        m = lambda f: np.mean([f(r) for r in rs])  # noqa: E731
        lines.append(f"{H_TEX[h]} & {100 * m(lambda r: r['test']['f1']):.1f} & {100 * m(lambda r: r['test_core']['f1']):.1f} & "
                     f"{m(lambda r: r['test_core']['pr_auc']):.3f} & {agg(gb['rows'], h, lambda r: r['test_core']['pr_auc'])[0]:.3f} & "
                     f"{m(lambda r: r['attempts']['budget_500']['value_share_mean']):.3f} & {agg(gb['rows'], h, lambda r: r['attempts']['budget_500']['value_share_mean'])[0]:.3f} \\\\")
        lab = {"0": "Strict", "1": "Tie", "3600": "Hour", "inf": "Inf"}.get(h)
        if lab:
            macro(f"GNNF{lab}", f"{100 * m(lambda r: r['test']['f1']):.1f}")
            macro(f"GNNCoreF{lab}", f"{100 * m(lambda r: r['test_core']['f1']):.1f}")
            macro(f"GNNCorePR{lab}", f"{m(lambda r: r['test_core']['pr_auc']):.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_gnn.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- same-minute mechanism (model-free)
_ev = pl.read_parquet(ROOT / "data/processed/HI-Small/events.parquet", columns=["t", "src", "dst", "label", "split"])
_core = _ev.filter((pl.col("split") == 2) & (pl.col("t") < 864000))
_recv = _core.group_by(["t", "dst"]).len().rename({"dst": "src", "len": "src_in_same"})
_c = _core.with_columns(pl.len().over(["t", "src"]).alias("same_src")).join(_recv, on=["t", "src"], how="left").fill_null(0)
for lab, name in ((1, "Illicit"), (0, "Benign")):
    g = _c.filter(pl.col("label") == lab)
    macro(f"SameMinInflow{name}", pct(float((g["src_in_same"] > 0).mean())))
    macro(f"SameMinSrc{name}", pct(float((g["same_src"] > 1).mean())))

# ---------------------------------------------------------------- leakage sanity (partial run)
_sp = ROOT / "results/sanity.json"
if _sp.exists():
    _s = json.loads(_sp.read_text())
    macro("SanityPrev", f"{_s['prevalence']['core']:.4f}")
    macro("SanityShuffledStrict", f"{_s['shuffled']['0']['core_pr_auc']:.4f}")
    macro("SanityHorizonsDone", str(len(_s["shuffled"])))

# ---------------------------------------------------------------- adaptive policy, partial log (stopped run)
import re  # noqa: E402

_lp = ROOT / "logs/cascade_hi_released.log"
if _lp.exists():
    lines = [r"\begin{tabular}{rrrrl}", r"\toprule", r"Budget & Adaptive SV & Best fixed SV & Difference & 95\% CI \\", r"\midrule"]
    pat = re.compile(r"seed=(\d+) B=(\d+): adaptive value=([\d.]+).*?\| fixed.*?value=([\d.]+).*?diff=([+-][\d.]+) CI\[([+-][\d.]+),([+-][\d.]+)\]")
    n = 0
    for m in pat.finditer(_lp.read_text()):
        lines.append(f"{m.group(2)} & {m.group(3)} & {m.group(4)} & {m.group(5)} & [{m.group(6)}, {m.group(7)}] \\\\")
        n += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    if n:
        (OUT / "tab_policy_partial.tex").write_text("\n".join(lines))
    macro("PolicyRows", str(n))

# ---------------------------------------------------------------- hypothesis contrasts on the discovery set
if "HIS" in tables:
    _r = tables["HIS"]["rows"]
    _m = lambda h, f: agg(_r, h, f)[0]  # noqa: E731
    _f = lambda r: r["test"]["f1"]  # noqa: E731
    _cf = lambda r: r["test_core"]["f1"]  # noqa: E731
    _cp = lambda r: r["test_core"]["pr_auc"]  # noqa: E731
    macro("COneGap", f"{100 * (_m('inf', _f) - _m('inf', _cf)):.1f}")
    macro("CTwoGap", f"{_m('inf', _cp) - _m('0', _cp):.3f}")
    macro("CFourGap", f"{_m('1', _cp) - _m('0', _cp):.3f}")
    macro("HourToInfPR", f"{_m('inf', _cp) - _m('3600', _cp):+.3f}")
    _bud = lambda h: agg(_r, h, lambda r: r["attempts"]["budget_500"]["alerts_per_day_core"])[0]  # noqa: E731
    macro("AlertsStrict", f"{_bud('0'):.0f}")
    macro("AlertsHour", f"{_bud('3600'):.0f}")
(OUT / "macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(f"wrote {len(macros)} macros; files: {sorted(p.name for p in OUT.iterdir())}")
if DOCS:
    target = DOCS / "generated"
    target.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        if f.suffix in (".tex", ".pdf"):
            shutil.copy(f, target / f.name)
    print("copied to", target)
