"""Port of IBM/Multi-GNN format_kaggle_files.py without the datatable dependency.

Dictionaries fill in raw row order, as in the original, so ids match. The pilot compares this with the official
formatter (mgnn.mgnn_format_check). Output is sorted by Timestamp with a stable sort, like the original's sort(3).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def _lookup(name, collection):
    if name in collection:
        return collection[name]
    collection[name] = len(collection)
    return collection[name]


def format_transactions(in_path: Path, out_path: Path, max_rows: int | None = None) -> dict:
    currency, payment_format, account = {}, {}, {}
    first_ts = -1.0
    cols = {k: [] for k in ("EdgeID", "from_id", "to_id", "Timestamp", "Amount Sent", "Sent Currency", "Amount Received",
                            "Received Currency", "Payment Format", "Is Laundering")}
    with open(in_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for i, r in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            d = datetime.strptime(r[0], "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
            ts = d.timestamp()
            if first_ts == -1.0:
                first_ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() - 10
            cur_recv = _lookup(r[6], currency)
            cur_paid = _lookup(r[8], currency)
            fmt = _lookup(r[9], payment_format)
            from_id = _lookup(r[1] + r[2], account)
            to_id = _lookup(r[3] + r[4], account)
            for k, v in zip(cols, (i, from_id, to_id, int(ts - first_ts), float(r[7]), cur_paid, float(r[5]), cur_recv, fmt, int(r[10]))):
                cols[k].append(v)
    df = pl.DataFrame(cols).sort("Timestamp", maintain_order=True)
    df.write_csv(out_path, float_precision=6)
    return {"rows": df.height, "accounts": len(account), "currencies": len(currency), "formats": len(payment_format)}
