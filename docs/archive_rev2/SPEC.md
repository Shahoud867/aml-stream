# SPEC.md — Data Contracts and Definitions (Phase 3, streaming AML)

| | |
|---|---|
| **Version** | Draft v0.1 — for **Gate S0** review |
| **Author / reviewer** | Shahoud (author) · Saif (reviewer) |
| **Status** | DRAFT. Becomes binding when both of you sign §14 |
| **Location** | Drafted in the docs folder; moves to the `aml-stream` repository root at task H-01 |
| **Sources** | Phase 2 Revision 2 (Steps 1–10), `PHASE3_PLAN.md` §4–§6 and §10–§11, `ROADMAP.md` §10 |

**Purpose.** Two people implement against this document independently: the fast code (Shahoud: prep and eval; Saif: models and runner) and the brute-force reference tests (each person for the other's code). Both must produce **the same numbers from this text alone**. Anything this document leaves ambiguous is a bug in the document; fix the text, don't guess.

**Change control.** After Gate S0, any change needs (1) a PR touching only `SPEC.md`, (2) approval from both of you, (3) a version bump, and (4) a `DECISIONS.md` entry listing affected files and runs. Derived files built under an older spec version are invalid unless the change log says otherwise.

**Words.** MUST / MUST NOT = required (tests enforce them). SHOULD = recommended. "Verify at H-06" = an expectation about the raw data that must be checked when the data is first formatted (§15).

---

## 1. Conventions

| Clause | Rule |
|---|---|
| 1.1 Time | All times are **int64 seconds**. `t` = seconds since `T_ALIGN`, defined as **00:00:00 of the calendar date of the earliest raw timestamp**. Raw timestamps are parsed as naive local times; no timezone conversion. |
| 1.2 Canonical order | Events are ordered by **`(t, event_id)` ascending**. `pos` is the 0-based position in this order. Every derived array is indexed by `pos`. |
| 1.3 Tie group | A **tie group** is a maximal set of events with equal `t`. `tie_id` (int32) numbers tie groups in canonical order. A tie group is **never split** across update units (§5.2). |
| 1.4 IDs | `event_id` (int64) = 0-based row index in the **raw** Kaggle CSV (header excluded) (**D-1**). `node_id` (int32) = account index assigned by first appearance in canonical order, `src` before `dst` within an event. Accounts are keyed by `(bank, account)` string pairs. |
| 1.5 Floats | Accumulate sums and means in **float64**; store as **float32** unless stated otherwise. |
| 1.6 Logs | `log1p` = natural log of (1 + x). For x < 0 the input is invalid: raise an error, never clip silently. |
| 1.7 Missing | Empty sets give 0 for counts, sums and spans (§6.4). Undefined ratios are **NaN**, stored as NaN and excluded explicitly with a reported count (§6.6). |
| 1.8 Determinism | Integer outputs MUST be byte-identical across machines (Windows HP, macOS, Linux Kaggle/molab, Windows lab PC). Float outputs MUST match within `rtol = 1e-6, atol = 1e-6`. Any sort used for output MUST have a full tie-break (no reliance on sort stability). |
| 1.9 Tolerance in tests | Reference-vs-fast comparisons (Tier 1) use exact equality for integers and the 1.8 tolerance for floats. |
| 1.10 Paths | Code MUST use `pathlib`; no hard-coded OS separators. |
| 1.11 Split codes | `split ∈ {0 = train, 1 = val, 2 = test}`. |
| 1.12 Directions | Relative to an account `w`: an event is **out** for `w` if `w == src`, **in** if `w == dst`. A **self-loop** (`src == dst`) is **both** out and in for that account, and its counterparty is the account itself. |

---

## 2. Source data (raw)

| File | Source | Expected content (**verify at H-06**) |
|---|---|---|
| `HI-Small_Trans.csv` | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | Columns: `Timestamp, From Bank, Account, To Bank, Account(.1), Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`. Timestamp format `YYYY/MM/DD HH:MM` (minute resolution). |
| `HI-Small_Patterns.txt` | Same dataset | Blocks starting `BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[: <details>]` and ending `END LAUNDERING ATTEMPT - <TYPOLOGY>`; each line inside is one transaction in the raw CSV column order. |
| Multi-GNN code | `github.com/IBM/Multi-GNN` | `format_kaggle_files.py` (formatter) and the Small_HI loading/split code |

Raw files MUST be recorded in `data/checksums.txt` (sha256, lowercase, `sha256sum` format) and never modified.

---

## 3. Canonical tables (`data/processed/`)

### 3.1 `events.parquet` — one row per transaction, stored in canonical order
| Column | Type | Definition |
|---|---|---|
| `pos` | int32 | Canonical position (= row index of this file) |
| `event_id` | int64 | Raw CSV row index (1.4) |
| `t` | int64 | Seconds since `T_ALIGN` (1.1) |
| `tie_id` | int32 | Tie-group index (1.3) |
| `src`, `dst` | int32 | `node_id` of sender and receiver |
| `is_self` | int8 | 1 if `src == dst` |
| `amt_paid`, `amt_recv` | float64 | Raw amounts |
| `cur_paid`, `cur_recv` | int16 | Currency codes (3.4) |
| `fmt` | int8 | Payment-format code (3.4) |
| `amount_usd` | float64 | `amt_paid × rate[cur_paid]` (3.5, **D-2**) |
| `label` | int8 | `Is Laundering` (0/1) |
| `split` | int8 | 1.11 (§4) |
| `n_patterns` | int8 | Number of laundering attempts the event belongs to (§3.3) |
| `first_pattern_id` | int32 | Smallest `pattern_id` the event belongs to; −1 if none |
| `mgnn_edge_id` | int64 | Multi-GNN `EdgeID` for this event (cross-reference only) |

### 3.2 `accounts.parquet`
`node_id:int32, bank:str, account:str` — one row per account.

### 3.3 `attempts.parquet` and `pattern_members.parquet`
- **Parsing.** `pattern_id` (int32) numbers attempts in file order from 0. `typology` = text after `BEGIN LAUNDERING ATTEMPT - `, up to the first `:` or end of line, stripped and upper-cased.
- **Typology codes:** `FAN-OUT=0, FAN-IN=1, GATHER-SCATTER=2, SCATTER-GATHER=3, CYCLE=4, RANDOM=5, BIPARTITE=6, STACK=7`, with `−1` for none. An unknown string is an error: stop and update this spec.
- **Matching.** A pattern line matches the raw CSV row whose 11 raw fields are all equal (string comparison after stripping whitespace). If several raw rows match one pattern line, assign them to the pattern lines **in order of appearance** (one-to-one). Pattern lines left unmatched are logged with a count in the data card; they MUST NOT be invented.
- `attempts.parquet`: `pattern_id, typology, n_lines, n_matched, t_first, t_last, split_first, split_last`. Times and splits are taken over matched events only.
- `pattern_members.parquet`: `pattern_id, pos, event_id` — **the authoritative membership table** (many-to-many).

### 3.4 Vocabularies (`vocab.json`)
- Currency and payment-format codes come from the values present in the **train split**, sorted lexicographically, numbered from 0. One extra code `OTHER = K` covers values seen only in val/test.
- `vocab.json` records the lists, `K`, and how many val/test events map to `OTHER`.

### 3.5 Exchange rates (`rates.json`) — **D-2**
Amounts in different currencies cannot be summed directly. For each currency `c`, `rate[c]` is USD per unit of `c`, estimated on the **train split only**:
1. `c = "US Dollar"` (exact string — verify at H-06): `rate = 1`.
2. Otherwise the median of `amt_recv / amt_paid` over train events with `cur_paid = c`, `cur_recv = USD` and `amt_paid > 0`.
3. Otherwise the median of `amt_paid / amt_recv` over train events with `cur_paid = USD`, `cur_recv = c` and `amt_recv > 0`.
4. Otherwise `rate = 1`, flagged `rate_missing = true`. The data card MUST list every such currency and its event count.

The data card MUST report, for each currency, the coefficient of variation of the ratios used. The simulator is expected to use fixed rates (verify), so CVs should be near 0.

### 3.6 Scalers (`scalers.json`)
`mu` and `sigma` of `log1p(amt_paid)` and `log1p(amt_recv)` over the **train split**. If `sigma == 0`, use 1.

---

## 4. Splits

- **4.1** `split` MUST equal the assignment produced by Multi-GNN's released Small_HI loading code. The procedure (commit hash, function, parameters) and the resulting boundaries go in `DATA_CARD.md`. **Expected (verify at H-06):** whole-day boundaries, giving roughly 64/19/17 by transaction count, which matches FraudGT's reported split.
- **4.2** MUST hold: `max t (train) < min t (val)` and `max t (val) < min t (test)`. If the released split puts one tie group into two splits, record it in the data card, use `≤` for that boundary only, and treat the two parts as separate tie groups for unit formation.
- **4.3** Splits label evaluation only. History is never truncated: val and test events see train history (§5.4).

---

## 5. Update units and visibility (core semantics)

### 5.1 Granularity tokens
| Token | Meaning |
|---|---|
| `fin` | Finest: each tie group is one unit |
| `b<n>` | Micro-batch of at least *n* events (e.g. `b10`, `b100`, `b200`, `b1000`) |
| `d<m>m` | Time bucket of *m* minutes aligned to `T_ALIGN` (e.g. `d1m`, `d10m`, `d60m`, `d1440m`) |
| suffix `_p<s>` | Bucket edges shifted by phase `s` seconds (a multiple of 60, `0 ≤ s < Δ`) |
| suffix `_j<seed>` | Independently jittered bucket edges (5.3), RNG seed `seed` |

B1's periodic recomputation every *R* uses the same `d<m>m` tokens (R = 10 min → `d10m`; R = 24 h → `d1440m`).

### 5.2 Unit assignment
- **`fin`:** `unit_id = tie_id`.
- **`b<n>`:** walk tie groups in canonical order, adding each to the current unit. Close the unit when (a) it holds ≥ *n* events, or (b) the next tie group belongs to a different split. Units therefore never cross split boundaries, and a large tie group can make a unit exceed *n*.
- **`d<m>m`:** with `Δ = 60m`, edges `E_k = T + kΔ`, where `T = 0`, or `T = s` with `_p<s>`; `unit_id = floor((t − T) / Δ)`, renumbered densely from 0 in canonical order. Empty buckets get no `unit_id`.
- `unit_id` is int32, dense, non-decreasing in `pos`.

### 5.3 Jittered edges (`_j<seed>`, defence D2)
For every *k*, draw `φ_k` uniformly from `{0, 60, …, Δ − 60}` using `numpy.random.default_rng(seed)`, in increasing *k*, covering the whole time range. Edges are `E_k = kΔ + φ_k`. Edges stay strictly increasing; unit lengths lie in `[60, 2Δ − 60]`, with mean Δ.

### 5.4 Commit time, close time, visible history
For unit *k*:
- **Commit time `τ_k`:** the left edge `E_k` for time buckets; the `t` of the unit's first event for `fin` and `b<n>`.
- **Close time `c_k`:** the right edge `E_{k+1}` for time buckets; the `t` of the unit's last event for `fin` and `b<n>`. A unit can only produce alerts at `c_k` (data time).
- **Visible history `V_k = { e′ : t_{e′} < τ_k }`** — exactly the events in earlier units (guaranteed by 1.3 and 5.2; tested).

**5.5 Visibility rule (D-3).** Every history-derived input for an event in unit *k* MUST use only `V_k`. This covers neighbours, port index, causal features and merged aggregates. **The one exception is the intra-block context (§6.4)**, which by design also uses earlier events *inside* unit *k*. This keeps the IBC on/off contrast clean: without IBC, a model has no intra-unit information at all.

> **Refines Phase 2.** Phase 2 Step 4 defines the port index over "earlier transfers" without reference to units. Under 5.5 it is granularity-dependent. Record this under "Changes from Phase 2" in the report.

---

## 6. Derived files (per granularity *g*)

Every derived file has a JSON sidecar: `{file, spec_version, gran, start_pos, rows, dtype, shape, sha256, data_version, code_git_hash, built_on, built_at}`. Rows cover `pos ≥ start_pos`: `start_pos = 0` for training granularities, and the first val `pos` for sweep-only granularities.

### 6.1 `units_<g>.parquet`
`pos, unit_id:int32, tau:int64, close:int64`.

### 6.2 `nbr_<g>.npy` — neighbour slots
- Shape `int32 [rows, 2, 2, S_STORE]`, `S_STORE = 20` (**D-5**).
- Axis 1: endpoint (`0 = src u`, `1 = dst v`). Axis 2: direction relative to that endpoint (`0 = in`, `1 = out`). Axis 3: recency rank (0 = most recent).
- **Entry definition.** For the event at `pos` in unit *k*, endpoint *w* and direction δ: the `pos` values of the up to 20 events in `V_k` that involve *w* in direction δ, ordered by `(t, pos)` descending. Pad with **−1**.
- **Model use:**
  - DMT-GNN: top `S_in = S_out = 10` of each list.
  - B3 (TGN) and A2: per endpoint, merge the in and out lists by `(t, pos)` descending, drop duplicates (self-loops appear in both), and keep the top 20.

### 6.3 `port_<g>.npy`
`int32 [rows]` = the number of events in `V_k` with the same ordered pair `(src, dst)` as the scored event. The model feature is `log1p(port)`.

### 6.4 `ibc_<g>.parquet` — intra-block context (Phase 2 Step 4)
For the event `e = (u, v, t)` in unit `U`, endpoint *w* and direction δ:

  `P^δ_w(e) = { e′ ∈ U : t_{e′} < t, e′ involves w in direction δ }`  *(strictly earlier, same unit)*

  `ψ(P) = [ log1p|P| , log1p Σ_{e′∈P} amount_usd(e′) , log1p n_cp(P) , log1p( t − min_{e′∈P} t_{e′} ) ]`

  with `ψ(∅) = [0, 0, 0, 0]`. `n_cp` = distinct counterparties: `dst` for out-sets, `src` for in-sets, and the account itself for a self-loop.

  `n_uv(e) = |{ e′ ∈ U : t_{e′} < t, src(e′) = u, dst(e′) = v }|`

  Columns, in this order: `pos`, then 17 float32 columns —
  `ibc_u_in_{cnt,amt,ncp,span}, ibc_u_out_{…}, ibc_v_in_{…}, ibc_v_out_{…}, ibc_uv_cnt` (the last = `log1p n_uv`).

- **MUST:** all columns are exactly 0 under `fin` (tested).
- **MUST:** events with equal `t` never contribute to each other (tested).

### 6.5 `pair_cum.parquet` — supports B4 (merged) (**D-4**)
`pos, pair_cnt_incl:int32, pair_amt_usd_incl:float32` = the cumulative count and USD sum of all events with the same ordered `(src, dst)` up to and **including** this event, in canonical order. This file is granularity-independent.

**B4 semantics.** For each `(w, δ)` slot list from §6.2, keep the **first occurrence** of each counterparty (its most recent visible transfer `e′`). Represent that counterparty by the aggregate edge `[log1p pair_cnt_incl(e′), log1p pair_amt_usd_incl(e′), t_{e′}]`. Because `e′` is that pair's most recent visible event, this equals the running aggregate over the pair's whole visible history. The counterparty set is therefore identical to DMT-GNN's; only per-transfer detail is replaced by the aggregate (a controlled comparison).

### 6.6 `exposure_<g>.parquet` — model-free diagnostic (Gate G0)
- `P_w(e)` = 6.4's set with any direction.
- `H_w(e) = { e′ involving w : t − 86400 ≤ t_{e′} < t }`.
- `num = |P_u(e) ∪ P_v(e)|` and `den = |H_u(e) ∪ H_v(e) ∪ P_u(e) ∪ P_v(e)|`, counting distinct events (**D-6**: including `P` in the denominator guarantees `ρ ≤ 1`).
- `rho = num / den`; NaN if `den = 0`.
- Columns: `pos, num:int32, den:int32, rho:float32`.
- **G0 statistic:** mean `rho` over illicit vs benign events in the **train split** at `d10m`, NaNs excluded and counted.

### 6.7 `feat_causal_<g>.parquet` — B1 features (**D-12**)
For the event in unit *k*, endpoint `w ∈ {u, v}`, direction δ and window `W ∈ {3600, 86400, 604800}`, the set is `{ e′ ∈ V_k : τ_k − W ≤ t_{e′} < τ_k, e′ involves w in δ }`. Windows are anchored at `τ_k`, **not** at the event's own time: between refreshes the table is stale, which is exactly the periodic-refresh semantics. Under `fin`, `τ_k = t`, giving continuous mode.

- Per set: `cnt` (float32), `amt = log1p Σ amount_usd`, `ncp` (distinct counterparties). Named `cf_{u|v}_{in|out}_{1h|24h|7d}_{cnt|amt|ncp}`: 36 columns.
- Plus `cf_pair_uv` = `port_<g>` and `cf_pair_vu` = the number of events in `V_k` from `v` to `u`.
- **B1 input** = event features (§7.1) + these 38 columns.

### 6.8 `xe.npy` — event features (Phase 2 Step 4, D1)
`float32 [N, d_x]`. Columns:
1. `z(log1p amt_paid)`;
2. `z(log1p amt_recv)` (3.6);
3. one-hot `cur_paid` (K+1);
4. one-hot `cur_recv` (K+1);
5. one-hot `fmt`.

The port index is **not** here because it depends on the granularity (6.3). The column list is stored in the sidecar.

---

## 7. Model semantics (interface for Saif's code and Shahoud's Tier 2 tests)

- **7.1 Inputs per event:** `xe[pos]`, neighbour slots (6.2), `log1p port_<g>[pos]`, `ibc_<g>[pos]` (only if `use_ibc`), time deltas `t − t_{e′}` for the time encoding φ.
- **7.2 Score-before-update.** For unit *k*, every event is scored using node memory `s(τ_k⁻)`, the state after unit *k−1* is fully processed. Memory MUST be byte-identical before and after scoring (tested).
- **7.3 Memory update.** After scoring unit *k*, every account touched in the unit gets one message, built from its **most recent event in the unit** by `(t, pos)`, using the pre-unit memories of both endpoints. Then `s ← GRU(message, s)` (Phase 2 Step 5). Self-loops update the account once.
- **7.4 Training regimes.** `fine` = units `b200`; `block` = units `d10m`. Memory is reset to zero at the start of each epoch and detached between units. The loss is computed per unit. A unit MAY be split into chunks for the backward pass, provided memory updates happen only after the whole unit (no semantic change, **D-11**).
- **7.5 Continuation.** Validation and test replay continue from the memory state at the end of the train split. A sweep at granularity *g* replays val and test at *g*.
- **7.6 Flags.**

  | Flag | Effect |
  |---|---|
  | `merged` | B4: 6.5 |
  | `shared_direction` | A2: one parameter set and the merged list from 6.2 |
  | `use_memory = false` | A3: `s ≡ 0` |
  | `use_ibc` | IBC inputs on |

  TGN (B3) = single-stage attention over the merged list of 20.
- **7.7 Seeds.** Model seeds `{42, 101, 202}` control initialisation and dropout only. Event order is always canonical.

---

## 8. Score files and thresholds

- **8.1** `scores_<g>.parquet` (in the run folder): `pos:int32, event_id:int64, split:int8, unit_id:int32, close:int64, score:float32`. `score` = sigmoid probability. Rows = **all val and test events**, sorted by `pos`.
- **8.2 Threshold.** τ = the score value that maximises minority-class F1 on the **val** scores at the model's **training granularity** (`b200` or `d10m`; B1: `fin`). Flag if `score ≥ τ`. On ties, take the **highest** such τ. Stored in `threshold.json` as `{tau, f1_val, gran}`. τ is **never** re-tuned per granularity, condition or defence.

---

## 9. Metrics and statistics (test split unless stated)

| Clause | Metric | Definition |
|---|---|---|
| 9.1 | PR-AUC | Average precision, `Σ_n (R_n − R_{n−1}) P_n` (`sklearn.metrics.average_precision_score`) |
| 9.2 | Minority F1 | F1 of class 1 at τ (8.2) |
| 9.3 | P@100/day | Day = `floor(t / 86400)`. The top 100 by score within each day (ties broken by `pos` ascending); precision = positives / min(100, n_day); averaged over test days |
| 9.4 | Typology recall | Over `pattern_members` events whose attempt has typology *y*: fraction flagged |
| 9.5 | Staleness penalty | `ΔPR-AUC(g) = PR-AUC(fin) − PR-AUC(g)`, same model and seed |
| 9.6 | Class-conditional drop | `mean(score_fin − score_g)` over illicit events, minus the same over benign events |
| 9.7 | Recovered fraction | `1 − ΔPR-AUC_IBC(g) / ΔPR-AUC_noIBC(g)` (block-trained models); undefined if the denominator ≤ 0, reported as such |
| 9.8 | Detection delay | Attempts with `split_first = split_last = test` and ≥ 1 matched event. `D_A = min over flagged e in A of close(unit(e)) − t_first(A)`; undetected if none flagged. Report the % detected, the median `D_A` among detected attempts, and the % detected within 600 s / 3600 s / 86400 s (undetected attempts count as not within) |
| 9.9 | End-to-end delay | As 9.8, with `close(unit(e)) + L_unit(e)`, where `L` is the simulated latency of that unit (§11) |
| 9.10 | Bootstrap (event metrics) | Test time is cut into consecutive 1-hour bins (blocks). For 2000 replicates: resample seeds with replacement, then resample hour bins with replacement (the **same** bins for both models: paired); recompute the metric difference. CI = 2.5th/97.5th percentiles. `p = 2·min(Pr(Δ ≤ 0), Pr(Δ ≥ 0))`. Holm within each family (RQ2: IBC on/off × models; RQ3: DMT vs B3, B4, A2, A3; E-ADV: §10.6) (**D-10**) |
| 9.11 | Bootstrap (attempt metrics) | 9.8, 9.9 and §10 resample **attempts** (paired across conditions), 2000 replicates |

All metric code MUST pass the Tier 4 comparisons against `sklearn` on random inputs.

---

## 10. E-ADV — adversarial timing study (`PHASE3_PLAN.md` §5)

- **10.1 Scope.** Attacked attempts = attempts with `split_first = split_last = test` and ≥ 2 matched events. Only their events are re-timed; every other event is unchanged. All attempts are attacked simultaneously in one perturbed stream.
- **10.2 Invariants (MUST, tested).**
  - Only `t` changes. `src`, `dst`, amounts, currencies, format, label and membership are identical.
  - Shifts stay within budget: `|t′ − t| ≤ W`, with `W ∈ {3600, 21600}`.
  - `t′` is a multiple of 60 (minute grid).
  - `t′` stays within `[min t(test), max t(test)]`.
  - **Order preservation:** for any two events of the same attempt, `t(e1) ≤ t(e2) ⇒ t′(e1) ≤ t′(e2)`. This implies the funds-flow precedence (an inflow before an outflow from the same account stays before or level), and holds by construction because every map below is non-decreasing.
- **10.3 Conditions.** Each attempt *A* has sorted original times `t_1 ≤ … ≤ t_n`, `clipW(x, t) = min(max(x, t − W), t + W)`, and `clamp` = clip to the test range.
  - **C0** (original): `t′ = t`.
  - **C1** (packed, attacker knows the grid, A-full): for Δ, choose the target bucket `j*` = the bucket `[L_j, L_j + Δ)` on the phase-0 grid that maximises `|{e ∈ A : [t_e − W, t_e + W] ∩ [L_j, L_j + Δ) ≠ ∅}|`, with ties going to the smallest *j*. Then `t′ = clamp(clipW(min(max(t, L_{j*}), L_{j*} + Δ − 60), t))` — project into the bucket, then respect the budget. Non-decreasing in *t*.
  - **C2** (control: same displacement, no packing): `m_A` = the mean of `|t′ − t|` under C1 for the same *(A, W, Δ)`, rounded to a multiple of 60. Draw `σ_A ∈ {−1, +1}` with `default_rng(seed_adv)` in `pattern_id` order. `t′ = clamp(t + σ_A · m_A)` — a rigid shift that keeps every internal gap.
  - **C3** (anti-packed): `t_c` = the median of `t_1..t_n` (lower median). `t′ = clamp(t − W)` if `t < t_c`, `t` if `t = t_c`, `clamp(t + W)` if `t > t_c`.
- **10.4 Knowledge variants.** A-full = the detector uses phase-0 units (`d<m>m`). A-partial = the C1 streams are built on the phase-0 grid but evaluated with `d<m>m_p<s>`, where `s` is drawn uniformly from `{0, 60, …, Δ − 60}` with `default_rng(seed_phase)`, one draw per evaluation run.
- **10.5 Rebuild.** After re-timing, re-sort the full stream canonically (`pos` changes for test events only) and rebuild `units`, `nbr`, `port`, `ibc` and `feat_causal` for the test range. Each perturbed stream is a separate data version: `events_adv-<C>-w<W>-d<m>m.parquet`. The memory state at the val/test boundary is identical across conditions for a given (model, seed, Δ) and MAY be reused.
- **10.6 Measures.** An attempt counts as detected if any of its events is flagged (τ from 8.2).
  - **Suppression rate** `SR(C) = |detected(C0) \ detected(C)| / |detected(C0)|`.
  - **Headline:** `SR(C1) − SR(C2)`, with an attempt-level paired bootstrap CI (9.11).
  - Also: ΔPR-AUC on attacked events, Δ detection delay, the breakdown by typology (H6), and residual `SR(C1) − SR(C2)` under D1 (IBC) and D2 (`_j<seed>` units, seeds {0, 1, 2}) (H7).
  - Holm across the E-ADV family.
- **10.7 Ethics note.** Synthetic data only; defences reported alongside the attack (`PHASE3_PLAN.md` §1.5).

---

## 11. Service time and queue simulation

- **11.1 Profile protocol (lab GPU, or the fallback named in the report).**
  - Unit sizes `n ∈ {1, 10, 100, 1000, 10000}`, each made of *n* consecutive test events, starting from the memory state at the test boundary.
  - Measure score + memory update with the data already resident on the GPU and `torch.cuda.synchronize()` around the timed region.
  - 20 warm-up iterations, then 50 timed repeats.
  - Output `service_times.csv`: `model, n, repeat, ms, device, dtype, git_hash`.
- **11.2 Service function.** `S(n)` = linear interpolation of the **median** in log–log space between measured sizes (main results); the p95 version is used for sensitivity.
- **11.3 Simulator.** One FIFO queue, one worker. When the worker is free and the queue is non-empty:
  - if the queue holds ≥ 1024 events, take 1024 now;
  - otherwise take `min(queue, 1024)` at `min(arrival of oldest waiting + 5 ms, time the queue reaches 1024)`.

  Batch completion = start + `S(n)`. Latency per event = completion − arrival.

  **Arrivals:**
  - **Poisson:** event order = canonical test order, inter-arrivals ~ Exp(λ), `λ ∈ {1e2, 1e3, 1e4, 1e5}` per second, `default_rng(seed_sim)`.
  - **Block:** all events of each 12 s or 600 s data-time window arrive at the window's end.

  Warm-up: the first 10% of arrivals are excluded from the statistics. Sustainable λ = the largest tested λ whose queue length over the last 50% of the run has a non-positive least-squares slope.
- **11.4 Tier 4 check.** With exponential `S` (mean 1/μ), batch size 1 and Poisson λ < μ, the simulated mean waiting time MUST be within 5% of the M/M/1 value `λ / (μ(μ − λ))` over 10⁶ arrivals.
- **11.5 Live validation.** The same arrival sequence and batching policy on the real GPU worker, at 2 values of λ. Report the simulated vs measured p50/p95/p99 and their relative errors; discrepancies are reported, not tuned away.

---

## 12. Run folders, configs, jobs, manifests

- **12.1** `data/processed/manifest.json` lists every processed and derived file with its sha256. `data_version` = the first 12 hex characters of the sha256 of the manifest.
- **12.2 Run folder:** `results/runs/<model>-<regime>[-ibc]-s<seed>-<yyyymmdd-hhmm>/`, containing:
  - `config.yaml`, `git_hash.txt`, `device.txt`, `data_version.txt`, `spec_version.txt`;
  - `train_log.csv`, `threshold.json`, `scores_<g>.parquet` (one per evaluated granularity);
  - `adv/<C>_w<W>_<g>/scores.parquet`.

  Checkpoints live in `ckpt/` (never synced to the shared results folder).
- **12.3 Job CSV:** `job_id, kind{train|sweep|adv|profile}, config, seed, gran, priority, status{TODO|RUNNING|RESUME|DONE|FAILED}, platform_hint`.
- **12.4 Checkpoint content (runner):** model, optimiser, memory, any neighbour state, all RNG states (python, numpy, torch, cuda), epoch, unit index, best validation metric, early-stopping counter. Written at least every 10 minutes **atomically** (write to a temporary file, then rename).

---

## 13. Test obligations (clause → tier → writer)

| Clause | Test | Tier | Writer |
|---|---|---|---|
| 1.2, 1.3, 5.2 | Canonical order; tie groups never split; units dense and monotone | 2 | Saif |
| 4.2 | Split boundary inequalities | 2 | Saif |
| 5.4 | `V_k` = events of earlier units, for every token, including `_p` and `_j` | 2 | Saif |
| 6.2, 6.3 | Brute-force neighbours and port index | 1 | **Saif** |
| 6.4 | Brute-force IBC; all-zero under `fin`; equal-`t` isolation | 1 + 2 | **Saif** |
| 6.5 | Brute-force pair cumulants; B4 running-aggregate equivalence | 1 | **Saif** |
| 6.6, 6.7 | Brute-force exposure and causal features | 1 | **Saif** |
| 3.3 | Pattern parsing and one-to-one matching on a fixture | 1 | **Saif** |
| 7.2, 7.3 | Memory unchanged during scoring; one message per touched account | 2 | **Shahoud** |
| 7.4, 12.4 | Resume == no-resume; chunking == no chunking | 2 | **Shahoud** |
| 7.x | Shuffled labels collapse; reversed time degrades; memory off + S=0 ≈ features only | 3 | **Shahoud** |
| 9.x | Metrics vs `sklearn`; bootstrap coverage on synthetic data with a known effect | 4 | Shahoud |
| 10.2, 10.3 | Invariants; non-decreasing maps; C2 matches C1 displacement | 2 | **Saif** |
| 11.4 | M/M/1 check | 4 | Shahoud |

Each person writes the tests for the **other** person's code, from this document only. Fixtures come from `tests/fixtures/make_stream.py`: small synthetic streams with ties, self-loops, multi-currency transfers and planted patterns.

---

## 14. Open decisions for Gate S0 (sign-off required)

Each decision has a recommendation. Saif either accepts it or writes an alternative here, and both of you sign below.

| ID | Decision | Recommended | Why it matters |
|---|---|---|---|
| **D-1** | Identity of an event | Raw CSV row index as `event_id`; Multi-GNN `EdgeID` kept as a cross-reference column | The raw file is the only order that never changes |
| **D-2** | How amounts are summed | Convert to USD with rates estimated on train only (3.5) | Summing euros, yuan and bitcoin as raw numbers would make every sum-based feature meaningless |
| **D-3** | Visibility | All history inputs use `V_k` only; IBC is the single exception (5.5) | Otherwise the port index silently leaks intra-unit information and the IBC on/off comparison (RQ2) is contaminated |
| **D-4** | B4 (merged) definition | Same counterparty set as DMT-GNN; each counterparty replaced by its running aggregate over visible history (6.5) | Makes B4 a controlled comparison: only per-transfer detail differs |
| **D-5** | Neighbour budget | Store 20 per direction; DMT uses 10 + 10; B3/A2 use the merged top 20 | Equal sampling budgets (Phase 2 Step 6) |
| **D-6** | Exposure denominator | Include in-unit events, so ρ ≤ 1 (6.6) | Avoids ratios above 1 for very short histories |
| **D-7** | E-ADV maps | C1 projection; C2 rigid shift with matched mean displacement; C3 median split (10.3) | All three keep fund-flow order by construction; C2 isolates packing from displacement |
| **D-8** | Bucket alignment | Midnight of the first day; jitter on the minute grid (5.2, 5.3) | Clock-aligned units; integer arithmetic only |
| **D-9** | Threshold | Max-F1 on val at the training granularity, highest τ on ties, never re-tuned (8.2) | Prevents threshold tuning from hiding staleness |
| **D-10** | Bootstrap blocks | 1-hour blocks as in Phase 2; attempt-level resampling for delay and E-ADV (9.10, 9.11) | Attempts, not hours, are the unit of analysis for delay and attacks |
| **D-11** | Gradient chunking | Allowed if memory updates only after the full unit | Lets block-regime units fit in GPU memory |
| **D-12** | B1 periodic semantics | Windows anchored at the commit time `τ_k` (6.7) | Matches a table refreshed every R; continuous mode falls out as `fin` |

Sign-off: Shahoud ______ (date) · Saif ______ (date) → set **Version 1.0** and update the status line.

---

## 15. Verify at H-06 (record results in `DATA_CARD.md`)

1. Raw CSV columns and timestamp format match §2; timestamp resolution is 1 minute (tie-group size distribution).
2. Event and account counts vs FraudGT (5,078,345 transactions; 515,088 accounts); illicit rate (0.07% vs 0.102% are both reported in the literature).
3. Multi-GNN split rule and boundaries (whole days?); proportions by count; whether any tie group straddles a boundary (4.2).
4. Multi-GNN `EdgeID` ↔ raw row index mapping (identical or not).
5. Patterns file format; typology strings; matched / unmatched counts; events in more than one attempt; laundering events in no attempt.
6. Currency strings, including the exact USD name; `rate_missing` currencies; CV of the rates (3.5).
7. Number of self-loops.
8. Payment-format values.

Anything that contradicts this spec → stop, update the spec (§0 change control), then continue.
