"""Raw files -> canonical event table, split labels and laundering attempts.

Conventions (see SPEC.md):
  * t is int64 seconds since midnight of the earliest calendar date in the file.
  * Canonical order is (t, event_id); event_id is the raw CSV row index.
  * The split reproduces Multi-GNN's data_loading.py rule: whole days, brute-force
    search for the (i, j) day cut whose proportions are closest to 60/20/20.

Everything runs in polars expressions so that 32M-row files do not create Python objects per row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

RAW_COLUMNS = [
    "timestamp", "from_bank", "from_acct", "to_bank", "to_acct",
    "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "label",
]
USD = "US Dollar"
TYPOLOGIES = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER",
              "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
DOCUMENTED_DAYS = {"HI-Small": 10, "LI-Small": 10, "HI-Medium": 16, "LI-Medium": 16}
ALL_STRINGS = {c: pl.String for c in RAW_COLUMNS}


def multignn_split_days(daily_totals: np.ndarray, split_per=(0.6, 0.2, 0.2)) -> tuple[int, int]:
    """Same brute-force rule as IBM/Multi-GNN data_loading.py: minimise the maximum
    relative error of the three proportions over all day cuts i < j."""
    n_days = len(daily_totals)
    cum = np.concatenate([[0], np.cumsum(daily_totals)]).astype(float)
    total = cum[-1]
    best, best_score = None, np.inf
    for i in range(1, n_days):
        for j in range(i + 1, n_days):
            props = ((cum[i]) / total, (cum[j] - cum[i]) / total, (total - cum[j]) / total)
            score = max(abs(v - p) / p for v, p in zip(props, split_per))
            if score < best_score:
                best, best_score = (i, j), score
    return best


def _usd_rates(df: pl.DataFrame) -> tuple[dict, dict]:
    """USD per unit of each currency, estimated on the train split only."""
    tr = df.filter(pl.col("split") == 0)
    rates, cv = {USD: 1.0}, {USD: 0.0}
    currencies = sorted(set(df["cur_paid"].unique()) | set(df["cur_recv"].unique()))
    for c in currencies:
        if c == USD:
            continue
        a = tr.filter((pl.col("cur_paid") == c) & (pl.col("cur_recv") == USD) & (pl.col("amt_paid") > 0))
        ratio = (a["amt_recv"] / a["amt_paid"]).to_numpy()
        if len(ratio) == 0:
            b = tr.filter((pl.col("cur_paid") == USD) & (pl.col("cur_recv") == c) & (pl.col("amt_recv") > 0))
            ratio = (b["amt_paid"] / b["amt_recv"]).to_numpy()
        if len(ratio) == 0:
            rates[c], cv[c] = float("nan"), float("nan")
        else:
            rates[c] = float(np.median(ratio))
            cv[c] = float(np.std(ratio) / np.mean(ratio))
    return rates, cv


def parse_patterns(path: Path) -> list[tuple[str, list[str]]]:
    attempts, cur, typ = [], None, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            typ = line.split(" - ", 1)[1].split(":")[0].strip().upper()
            if typ not in TYPOLOGIES:
                raise ValueError(f"unknown typology {typ!r}")
            cur = []
        elif line.startswith("END LAUNDERING ATTEMPT"):
            attempts.append((typ, cur))
            cur = None
        elif cur is not None and line:
            cur.append(line)
    return attempts


def _node_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Accounts numbered by first appearance in canonical order, src before dst.
    Expects src_key and dst_key columns; df must already be in canonical order."""
    n = df.height
    long = pl.concat([
        df.select(pl.col("src_key").alias("key"), (pl.int_range(n, dtype=pl.Int64) * 2).alias("o")),
        df.select(pl.col("dst_key").alias("key"), (pl.int_range(n, dtype=pl.Int64) * 2 + 1).alias("o")),
    ])
    first = long.group_by("key").agg(pl.col("o").min()).sort("o").with_row_index("node").select("key", pl.col("node").cast(pl.Int32))
    del long
    df = df.join(first.rename({"key": "src_key", "node": "src"}), on="src_key", how="left", maintain_order="left")
    df = df.join(first.rename({"key": "dst_key", "node": "dst"}), on="dst_key", how="left", maintain_order="left")
    return df


def _finish(df: pl.DataFrame, out: Path, documented_days: int | None, extra_card: dict) -> tuple[pl.DataFrame, dict]:
    """Shared tail of preparation. df holds event_id, t, src_key, dst_key, amounts, currencies, fmt, label."""
    df = df.sort(["t", "event_id"]).with_row_index("pos")
    df = _node_ids(df)
    t = df["t"].to_numpy()
    day = (t // 86400).astype(int)
    daily = np.bincount(day)
    i, j = multignn_split_days(daily)
    split = np.where(day < i, 0, np.where(day < j, 1, 2)).astype(np.int8)
    df = df.with_columns(pl.Series("split", split))

    rates, rate_cv = _usd_rates(df)
    missing = [c for c, r in rates.items() if not np.isfinite(r)]
    rates_filled = {c: (1.0 if not np.isfinite(r) else r) for c, r in rates.items()}
    df = df.with_columns(
        (pl.col("amt_paid") * pl.col("cur_paid").replace_strict(rates_filled, return_dtype=pl.Float64)).alias("usd"))

    vocab = {}
    for col in ("cur_paid", "cur_recv", "fmt"):
        values = sorted(df.filter(pl.col("split") == 0)[col].unique().to_list())
        vocab[col] = values
        mapping = {v: k for k, v in enumerate(values)}
        df = df.with_columns(
            pl.col(col).replace_strict(mapping, default=len(values), return_dtype=pl.Int16).alias(col + "_id"))

    events = df.select(
        pl.col("pos").cast(pl.Int32), pl.col("event_id").cast(pl.Int64), "t",
        pl.col("src").cast(pl.Int32), pl.col("dst").cast(pl.Int32),
        "amt_paid", "amt_recv", "usd", "cur_paid_id", "cur_recv_id", "fmt_id", "label", "split")
    events.write_parquet(out / "events.parquet")
    n_nodes = int(max(events["src"].max(), events["dst"].max())) + 1
    tie_sizes = np.bincount(np.unique(t, return_inverse=True)[1])
    by_split = events.group_by("split").agg(pl.len().alias("n"), pl.col("label").sum().alias("illicit"),
                                            pl.col("t").min().alias("t_min"), pl.col("t").max().alias("t_max")).sort("split")
    card = {
        "n_events": events.height, "n_accounts": n_nodes,
        "n_illicit": int(events["label"].sum()), "illicit_rate": float(events["label"].mean()),
        "n_days": int(len(daily)), "daily_totals": daily.tolist(),
        "daily_illicit": np.bincount(day, weights=events["label"].to_numpy(), minlength=len(daily)).astype(int).tolist(),
        "split_days": [int(i), int(j)], "documented_days": documented_days, "by_split": by_split.to_dicts(),
        "timestamp_unique": int(len(tie_sizes)), "tie_size_median": float(np.median(tie_sizes)),
        "tie_size_max": int(tie_sizes.max()), "n_self_loops": int((events["src"] == events["dst"]).sum()),
        "usd_rates": rates, "usd_rate_cv": rate_cv, "rate_missing": missing, "vocab": vocab,
    }
    card.update(extra_card)
    return events, card


def prepare(raw_csv: Path, patterns: Path, out: Path, documented_days: int | None = None) -> dict:
    """AMLworld: one CSV of transfers and one attempts file."""
    out.mkdir(parents=True, exist_ok=True)
    attempts = parse_patterns(patterns)
    rows = [(pid, typ, ln) for pid, (typ, lines) in enumerate(attempts) for ln in lines]
    pat = pl.DataFrame(rows, schema=[("pattern_id", pl.Int32), ("typology", pl.String), ("line", pl.String)], orient="row")
    pat = pat.with_row_index("line_order").with_columns(pl.int_range(pl.len()).over("line").alias("occ"))

    raw = pl.scan_csv(raw_csv, has_header=True, new_columns=RAW_COLUMNS, schema_overrides=ALL_STRINGS, infer_schema=False).with_row_index("event_id")
    # One-to-one assignment of pattern lines to raw rows, in appearance order.
    hits = (raw.select("event_id", pl.concat_str(RAW_COLUMNS, separator=",").alias("line"))
            .join(pat.lazy().select("line").unique(), on="line", how="semi")
            .sort("event_id")
            .with_columns(pl.int_range(pl.len()).over("line").alias("occ"))
            .collect(engine="streaming"))
    matched = pat.join(hits, on=["line", "occ"], how="left").sort("line_order")
    unmatched = int(matched["event_id"].null_count())
    members = matched.filter(pl.col("event_id").is_not_null()).select(pl.col("pattern_id"), pl.col("event_id").cast(pl.UInt32))
    del hits, matched

    df = raw.select(
        "event_id", pl.col("timestamp").str.to_datetime("%Y/%m/%d %H:%M").alias("dt"),
        pl.col("amt_paid").cast(pl.Float64), pl.col("amt_recv").cast(pl.Float64), "cur_paid", "cur_recv", "fmt",
        pl.col("label").cast(pl.Int8),
        (pl.col("from_bank") + "|" + pl.col("from_acct")).alias("src_key"),
        (pl.col("to_bank") + "|" + pl.col("to_acct")).alias("dst_key")).collect(engine="streaming")
    t0 = df["dt"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    df = df.with_columns((pl.col("dt") - pl.lit(t0)).dt.total_seconds().cast(pl.Int64).alias("t")).drop("dt")

    events, card = _finish(df, out, documented_days, {"t0": str(t0)})
    del df
    members = members.join(events.select(pl.col("event_id").cast(pl.UInt32), "pos", "t", "usd", "label", "split"), on="event_id")
    agg = members.group_by("pattern_id").agg(
        pl.col("t").min().alias("t_first"), pl.col("t").max().alias("t_last"),
        pl.col("split").min().alias("split_first"), pl.col("split").max().alias("split_last"),
        pl.col("usd").sum().alias("usd_total"))
    att = pl.DataFrame([(pid, typ, len(lines)) for pid, (typ, lines) in enumerate(attempts)],
                       schema=[("pattern_id", pl.Int32), ("typology", pl.String), ("n_lines", pl.Int32)], orient="row")
    n_matched = members.group_by("pattern_id").agg(pl.len().cast(pl.Int32).alias("n_matched"))
    att = att.join(n_matched, on="pattern_id", how="left").join(agg, on="pattern_id", how="left")
    members.write_parquet(out / "pattern_members.parquet")
    att.write_parquet(out / "attempts.parquet")
    card.update({
        "attempts": len(attempts), "pattern_lines": pat.height, "pattern_lines_unmatched": unmatched,
        "events_in_patterns": int(members["event_id"].n_unique()),
        "illicit_events_in_patterns": int(members.filter(pl.col("label") == 1)["event_id"].n_unique()),
        "pattern_events_labelled_benign": int(members.filter(pl.col("label") == 0)["event_id"].n_unique()),
    })
    (out / "data_card.json").write_text(json.dumps(card, indent=1))
    return card


def _components(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Connected component label of each edge, by union-find over its endpoints."""
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in u.tolist()])
    return np.unique(roots, return_inverse=True)[1]


def prepare_saml(raw_csv: Path, out: Path) -> dict:
    """SAML-D. One-second timestamps; one amount column, so USD values are not available.

    Laundering_type encodes the label (`Normal_*` versus suspicious types), so it never enters the event
    table. It is used only to build approximate attempts: connected components of illicit transfers of the
    same type. SAML-D has no attempt identifiers, so stoppable value on it is labelled approximate.
    """
    out.mkdir(parents=True, exist_ok=True)
    raw = pl.scan_csv(raw_csv, infer_schema=False).with_row_index("event_id")
    df = raw.select(
        "event_id",
        (pl.col("Date") + " " + pl.col("Time")).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("dt"),
        pl.col("Amount").cast(pl.Float64).alias("amt_paid"), pl.col("Amount").cast(pl.Float64).alias("amt_recv"),
        pl.col("Payment_currency").alias("cur_paid"), pl.col("Received_currency").alias("cur_recv"),
        pl.col("Payment_type").alias("fmt"), pl.col("Is_laundering").cast(pl.Int8).alias("label"),
        pl.col("Sender_account").alias("src_key"), pl.col("Receiver_account").alias("dst_key"),
        pl.col("Laundering_type").alias("_type")).collect(engine="streaming")
    types = df.select("event_id", "_type")
    df = df.drop("_type")
    t0 = df["dt"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    df = df.with_columns((pl.col("dt") - pl.lit(t0)).dt.total_seconds().cast(pl.Int64).alias("t")).drop("dt")
    # No exchange rates exist in this file, so amounts are used as recorded. Overwriting usd afterwards
    # keeps the shared code path from deriving a rate out of two copies of the same amount column.
    events, card = _finish(df, out, None, {
        "t0": str(t0), "note": "usd column holds raw amounts because SAML-D has one amount column and no rates"})
    events = events.with_columns(pl.col("amt_paid").alias("usd"))
    events.write_parquet(out / "events.parquet")

    ill = events.filter(pl.col("label") == 1).join(types, on="event_id", how="left").select("event_id", "pos", "t", "usd", "label", "split", "src", "dst", "_type")
    comp = np.full(ill.height, -1, dtype=np.int64)
    next_id = 0
    typ_col = ill["_type"].to_numpy()
    for typ in np.unique(typ_col):
        idx = np.where(typ_col == typ)[0]
        labels = _components(ill["src"].to_numpy()[idx], ill["dst"].to_numpy()[idx])
        comp[idx] = labels + next_id
        next_id += int(labels.max()) + 1
    members = ill.with_columns(pl.Series("pattern_id", comp).cast(pl.Int32)).select(
        "pattern_id", pl.col("event_id").cast(pl.UInt32), "pos", "t", "usd", "label", "split")
    tname = ill.with_columns(pl.Series("pattern_id", comp).cast(pl.Int32)).group_by("pattern_id").agg(pl.col("_type").first().alias("typology"))
    agg = members.group_by("pattern_id").agg(
        pl.len().cast(pl.Int32).alias("n_matched"), pl.col("t").min().alias("t_first"), pl.col("t").max().alias("t_last"),
        pl.col("split").min().alias("split_first"), pl.col("split").max().alias("split_last"), pl.col("usd").sum().alias("usd_total"))
    att = tname.join(agg, on="pattern_id").with_columns(pl.col("n_matched").alias("n_lines"))
    members.write_parquet(out / "pattern_members.parquet")
    att.write_parquet(out / "attempts.parquet")
    card.update({"attempts": att.height, "attempts_approximate": True, "illicit_events_in_patterns": ill.height,
                 "events_in_patterns": ill.height, "pattern_lines_unmatched": 0})
    (out / "data_card.json").write_text(json.dumps(card, indent=1))
    return card


def ensure_prepared(name: str, raw_dir: Path, cache_root: Path) -> Path:
    """Idempotent and deterministic: each notebook prepares what it needs, with no shared state."""
    out = cache_root / name
    if (out / "data_card.json").exists() and (out / "attempts.parquet").exists():
        return out
    if name == "SAML-D":
        prepare_saml(raw_dir / "SAML-D.csv", out)
    elif name == "Ethereum":
        from .extras import prepare_ethereum
        prepare_ethereum(raw_dir / "MulDiGraph.pkl", out)
    else:
        prepare(raw_dir / f"{name}_Trans.csv", raw_dir / f"{name}_Patterns.txt", out, DOCUMENTED_DAYS.get(name))
    return out
