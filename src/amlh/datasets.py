"""Loads a prepared dataset into the arrays every experiment uses, under one protocol."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .prepare import DOCUMENTED_DAYS


@dataclass
class Stream:
    name: str
    protocol: str
    t: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    usd: np.ndarray
    y: np.ndarray
    event_id: np.ndarray
    paid: np.ndarray
    recv: np.ndarray
    cur_p: np.ndarray
    cur_r: np.ndarray
    fmt: np.ndarray
    split: np.ndarray
    view_end: np.ndarray
    i_day: int
    j_day: int
    core_end_t: int
    members: pl.DataFrame
    attempts: pl.DataFrame
    card: dict

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def tr(self) -> np.ndarray:
        return np.where(self.split == 0)[0]

    @property
    def va(self) -> np.ndarray:
        return np.where(self.split == 1)[0]

    @property
    def te(self) -> np.ndarray:
        return np.where(self.split == 2)[0]

    @property
    def val_days(self) -> int:
        return self.j_day - self.i_day

    @property
    def core_mask(self) -> np.ndarray:
        """Over test rows: True where background traffic still exists."""
        return self.t[self.te] < self.core_end_t

    @property
    def core_days(self) -> int:
        """Calendar days of the test core, the D in the equal-alert budget B x D."""
        te_t = self.t[self.te][self.core_mask]
        return int(te_t.max() // 86400 - te_t.min() // 86400 + 1) if te_t.size else 0


def load_stream(root: Path, name: str, protocol: str = "released", fold: tuple[int, int, int] | None = None) -> Stream:
    """protocol: 'released' keeps all days; 'documented' cuts at the documented span.
    fold (train_end_day, val_end_day, test_end_day) gives a rolling-origin split inside the kept days."""
    P = Path(root) / name
    card = json.loads((P / "data_card.json").read_text())
    ev = pl.read_parquet(P / "events.parquet")
    # AMLworld spans are known constants: a stale or incomplete data card must never turn the tail into core.
    doc = card.get("documented_days") or DOCUMENTED_DAYS.get(name)
    limit = None
    if protocol == "documented":
        if doc is None:
            raise ValueError(f"{name} has no documented span, so the documented protocol does not apply")
        limit = doc * 86400
    elif protocol != "released":
        raise ValueError(protocol)
    i_day, j_day = card["split_days"]
    if fold is not None:
        i_day, j_day, k_day = fold
        limit = min(limit, k_day * 86400) if limit else k_day * 86400
    if limit is not None:
        ev = ev.filter(pl.col("t") < limit)  # a time prefix, so canonical positions stay valid
    t = ev["t"].to_numpy()
    day = t // 86400
    split = np.where(day < i_day, 0, np.where(day < j_day, 1, 2)).astype(np.int8)
    view_end = np.where(split == 0, i_day * 86400 - 1, np.where(split == 1, j_day * 86400 - 1, t.max()))
    core_end = doc * 86400 if doc else int(t.max()) + 1
    mem = pl.read_parquet(P / "pattern_members.parquet").filter(pl.col("pos") < len(t))
    return Stream(
        name=name, protocol=protocol, t=t, src=ev["src"].to_numpy().astype(np.int64), dst=ev["dst"].to_numpy().astype(np.int64),
        usd=ev["usd"].to_numpy(), y=ev["label"].to_numpy().astype(np.int8), event_id=ev["event_id"].to_numpy(),
        paid=ev["amt_paid"].to_numpy(), recv=ev["amt_recv"].to_numpy(),
        cur_p=ev["cur_paid_id"].to_numpy().astype(np.float32), cur_r=ev["cur_recv_id"].to_numpy().astype(np.float32),
        fmt=ev["fmt_id"].to_numpy().astype(np.float32), split=split, view_end=view_end,
        i_day=i_day, j_day=j_day, core_end_t=core_end, members=mem,
        attempts=pl.read_parquet(P / "attempts.parquet"), card=card)


def attempts_in_window(s: Stream, rows: np.ndarray, members_inside_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Attempts whose first member falls in the window spanned by rows. Returns (attempt ids, positions).
    With members_inside_only, later members outside the window are dropped (used on validation)."""
    if rows.size == 0 or s.members.height == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lo, hi = int(s.t[rows[0]]), int(s.t[rows[-1]])
    pos = s.members["pos"].to_numpy()
    pid = s.members["pattern_id"].to_numpy()
    mt = s.t[pos]
    order = np.argsort(pid, kind="stable")
    uniq, start = np.unique(pid[order], return_index=True)
    first = np.minimum.reduceat(mt[order], start)
    keep = set(uniq[(first >= lo) & (first <= hi)].tolist())
    mask = np.fromiter((p in keep for p in pid), dtype=bool, count=len(pid))
    if members_inside_only:
        mask &= (mt >= lo) & (mt <= hi)
    return pid[mask], pos[mask]
