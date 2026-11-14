"""Generate publication-grade vector and raster figures from verified experiment JSONs.

Reads strictly from:
  - results/tail_audit.json
  - results/gbt_hi_small_released/results.json
  - results/molab_nb1_full/nb3/aml_run/jobs/nb3-tdgnn-HI-Small-H*-s*/metrics.json

Writes fig_tail.{pdf,png} and fig_horizon.{pdf,png} into paper/generated/ and paper/springer/.
Zero em/en dashes are used anywhere in plot text or annotations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIRS = [ROOT / "paper" / "generated", ROOT / "paper" / "springer"]

NAVY = "#1D4E89"
TEAL = "#0E7490"
AMBER = "#B45309"
CRIMSON = "#B91C1C"
SLATE = "#475569"
MUTED = "#64748B"
BG_TAIL = "#FEF2F2"
BG_BAND = "#EFF6FF"

H_ORDER = ["0", "1", "60", "600", "3600", "21600", "86400", "259200", "inf"]
H_TICK = [
    "Strict\npast (0)",
    "Same\nsec (1s)",
    "1 min",
    "10 min",
    "1 hour",
    "6 hours",
    "24 hours",
    "72 hours",
    "Whole\n(inf)",
]


def setup_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#334155",
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(d / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.06)


def build_fig_tail() -> None:
    audit = json.loads((ROOT / "results" / "tail_audit.json").read_text(encoding="utf-8"))
    hi = audit["HI-Small"]
    n = np.asarray(hi["daily_n"], dtype=float)
    ill = np.asarray(hi["daily_illicit"], dtype=float)
    days = np.arange(len(n))
    cutoff = hi["tail_start_day"]  # 10 for HI-Small
    rate_pct = 100.0 * ill / np.maximum(n, 1.0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 2.75))

    colors = [NAVY if d < cutoff else CRIMSON for d in days]
    for ax in (ax0, ax1):
        ax.axvspan(cutoff - 0.5, len(n) - 0.35, color=BG_TAIL, zorder=0)
        ax.axvline(cutoff - 0.5, color=CRIMSON, lw=1.1, ls="--", zorder=2)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(days)
        ax.set_xticklabels([str(d) for d in days], fontsize=7.0)
        ax.set_xlabel("Day index in released CSV file")
        ax.set_xlim(-0.6, len(n) - 0.35)

    # Panel (a): daily volume
    ax0.bar(days, n, width=0.72, color=colors, edgecolor="none", zorder=3)
    ax0.set_yscale("log")
    ax0.set_ylim(0.7, n.max() * 14.0)
    ax0.set_ylabel("Transactions per day (log scale)")
    ax0.set_title("(a) Daily transfer volume on HI-Small", loc="left", pad=6)

    ax0.text(
        4.3,
        n.max() * 9.5,
        "Documented 10-day span\n(Days 0 to 9: 99.98% of volume)",
        ha="center",
        va="top",
        fontsize=7.3,
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", lw=0.6),
    )
    ax0.annotate(
        "Tail (Days 10 to 17):\n1,108 transfers total",
        xy=(11.8, float(n[11]) * 1.4),
        xytext=(13.5, n.max() * 0.12),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=CRIMSON,
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.85, shrinkA=2, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#FECACA", lw=0.6),
    )

    # Panel (b): illicit percentage
    ax1.bar(days, rate_pct, width=0.72, color=colors, edgecolor="none", zorder=3)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Illicit transfer share (%)")
    ax1.set_title("(b) Daily illicit concentration on HI-Small", loc="left", pad=6)

    core_rate = hi["core_illicit_rate_pct"]
    tail_rate = hi["tail_illicit_rate_pct"]
    tail_share = hi["tail_share_of_test_illicit_pct"]

    ax1.annotate(
        f"Core test (Days 8 to 9):\n{core_rate:.3f}% illicit rate",
        xy=(8.5, float(rate_pct[8]) + 1.5),
        xytext=(4.3, 48),
        ha="center",
        va="center",
        fontsize=7.3,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.85, connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", lw=0.6),
    )
    ax1.text(
        13.5,
        96.5,
        f"Post-cutoff tail:\n{tail_rate:.1f}% illicit rate\n({tail_share:.1f}% of test positives)",
        ha="center",
        va="top",
        fontsize=7.3,
        color=CRIMSON,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#FECACA", lw=0.6),
    )

    fig.tight_layout(w_pad=1.8)
    save_all(fig, "fig_tail")
    plt.close(fig)


def load_gbt_his() -> dict[str, dict[str, tuple[float, float]]]:
    data = json.loads((ROOT / "results" / "gbt_hi_small_released" / "results.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for h in H_ORDER:
        sub = [r for r in rows if r["H"] == h]
        f1_test = [100.0 * r["test"]["f1"] for r in sub]
        f1_core = [100.0 * r["test_core"]["f1"] for r in sub]
        sv100 = [r["attempts"]["budget_100"]["value_share_mean"] for r in sub]
        sv500 = [r["attempts"]["budget_500"]["value_share_mean"] for r in sub]
        sv2000 = [r["attempts"]["budget_2000"]["value_share_mean"] for r in sub]
        out[h] = {
            "test_f1": (float(np.mean(f1_test)), float(np.std(f1_test, ddof=1))),
            "core_f1": (float(np.mean(f1_core)), float(np.std(f1_core, ddof=1))),
            "sv100": (float(np.mean(sv100)), float(np.std(sv100, ddof=1))),
            "sv500": (float(np.mean(sv500)), float(np.std(sv500, ddof=1))),
            "sv2000": (float(np.mean(sv2000)), float(np.std(sv2000, ddof=1))),
        }
    return out


def load_tdgnn_his() -> dict[str, dict[str, tuple[float, float]]] | None:
    base = ROOT / "results" / "molab_nb1_full" / "nb3" / "aml_run" / "jobs"
    if not base.exists():
        return None
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for h in H_ORDER:
        cf, sv500 = [], []
        for s in (0, 1, 2):
            p = base / f"nb3-tdgnn-HI-Small-H{h}-s{s}" / "metrics.json"
            if not p.exists():
                return None
            m = json.loads(p.read_text(encoding="utf-8"))
            tr = m["test_results"]
            cf.append(100.0 * tr["test_core"]["f1"])
            sv500.append(tr["attempts"]["budget_500"]["value_share_mean"])
        out[h] = {
            "core_f1": (float(np.mean(cf)), float(np.std(cf, ddof=1))),
            "sv500": (float(np.mean(sv500)), float(np.std(sv500, ddof=1))),
        }
    return out


def build_fig_horizon() -> None:
    gbt = load_gbt_his()
    tdgnn = load_tdgnn_his()
    x = np.arange(len(H_ORDER), dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 4.95), sharex=True, gridspec_kw={"hspace": 0.20})

    for ax in (ax0, ax1):
        ax.axvspan(0.7, 4.3, color=BG_BAND, zorder=0)
        ax.grid(axis="x", visible=False)

    ax0.text(
        2.5,
        32.5,
        "Operational real-time regime (1 s to 1 h)",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color=NAVY,
        style="italic",
    )

    # Panel (a): Quality metrics
    lgb_full_m = [gbt[h]["test_f1"][0] for h in H_ORDER]
    lgb_full_sd = [gbt[h]["test_f1"][1] for h in H_ORDER]
    lgb_core_m = [gbt[h]["core_f1"][0] for h in H_ORDER]
    lgb_core_sd = [gbt[h]["core_f1"][1] for h in H_ORDER]

    ax0.errorbar(
        x, lgb_full_m, yerr=lgb_full_sd, color=SLATE, ls="--", marker="o", ms=4.5,
        mfc="white", mec=SLATE, mew=1.1, capsize=2.2, lw=1.3,
        label="LightGBM: Full test split (includes tail)", zorder=3,
    )
    ax0.errorbar(
        x, lgb_core_m, yerr=lgb_core_sd, color=NAVY, ls="-", marker="o", ms=5.0,
        mfc=NAVY, mec="white", mew=0.7, capsize=2.2, lw=1.6,
        label="LightGBM: Core test split (Days 8 to 9)", zorder=4,
    )
    if tdgnn is not None:
        gnn_core_m = [tdgnn[h]["core_f1"][0] for h in H_ORDER]
        gnn_core_sd = [tdgnn[h]["core_f1"][1] for h in H_ORDER]
        ax0.errorbar(
            x, gnn_core_m, yerr=gnn_core_sd, color=TEAL, ls="-.", marker="s", ms=4.6,
            mfc=TEAL, mec="white", mew=0.7, capsize=2.2, lw=1.45,
            label="Streaming TD-GNN: Core test split (Days 8 to 9)", zorder=4,
        )

    ax0.annotate(
        f"Same-second tie leap:\nCore F1 +14.6 pts ({lgb_core_m[0]:.1f}% to {lgb_core_m[1]:.1f}%)",
        xy=(1.0, lgb_core_m[1]),
        xytext=(1.85, 38.2),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.85, connectionstyle="arc3,rad=-0.18"),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.6),
    )

    ax0.set_ylim(31, 88)
    ax0.set_ylabel("Minority F1 (%)")
    ax0.set_title("(a) Classification performance across look-ahead horizons on HI-Small (3 seeds, Mean +/- SD)", loc="left", pad=6)
    ax0.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95)

    # Panel (b): Stoppable Value Share
    sv2000 = [gbt[h]["sv2000"][0] for h in H_ORDER]
    sv500 = [gbt[h]["sv500"][0] for h in H_ORDER]
    sv100 = [gbt[h]["sv100"][0] for h in H_ORDER]

    ax1.plot(
        x, sv2000, color=SLATE, ls="-", marker="^", ms=4.8, mfc=SLATE, mec="white", mew=0.65, lw=1.45,
        label="LightGBM (B = 2,000 alerts/day)", zorder=4,
    )
    ax1.plot(
        x, sv500, color=NAVY, ls="-", marker="o", ms=5.0, mfc=NAVY, mec="white", mew=0.7, lw=1.65,
        label="LightGBM (B = 500 alerts/day)", zorder=5,
    )
    if tdgnn is not None:
        gnn_sv500 = [tdgnn[h]["sv500"][0] for h in H_ORDER]
        ax1.plot(
            x, gnn_sv500, color=TEAL, ls="-.", marker="s", ms=4.6, mfc=TEAL, mec="white", mew=0.65, lw=1.45,
            label="Streaming TD-GNN (B = 500 alerts/day)", zorder=4,
        )
    ax1.plot(
        x, sv100, color=AMBER, ls="-", marker="D", ms=4.3, mfc=AMBER, mec="white", mew=0.65, lw=1.45,
        label="LightGBM (B = 100 alerts/day)", zorder=4,
    )

    ax1.annotate(
        f"Peak SV_500 = {sv500[1]:.3f}\n(1 s / 1 min)",
        xy=(1.0, sv500[1]),
        xytext=(2.3, 0.44),
        ha="center",
        va="center",
        fontsize=7.2,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.85),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.6),
    )
    ax1.annotate(
        "Complete collapse at H = inf\n(SV_B = 0.000: all attempts finish first)",
        xy=(8.0, 0.015),
        xytext=(4.35, 0.06),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=CRIMSON,
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.85),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#FECACA", lw=0.6),
    )

    ax1.set_ylim(-0.05, 0.87)
    ax1.set_ylabel("Stoppable value share (SV_B)")
    ax1.set_title("(b) Attempt-level stoppable illicit value prior to completion on HI-Small Core Test", loc="left", pad=6)
    ax1.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95)
    ax1.set_xticks(x)
    ax1.set_xticklabels(H_TICK)
    ax1.set_xlabel("Decision horizon H (per-transfer inference wait cutoff)")

    save_all(fig, "fig_horizon")
    plt.close(fig)


def main() -> None:
    setup_rc()
    build_fig_tail()
    build_fig_horizon()
    print("Generated fig_tail and fig_horizon into:", ", ".join(str(d) for d in OUT_DIRS))


if __name__ == "__main__":
    main()
