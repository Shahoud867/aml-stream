# SPEC.md: Definitions and Data Contracts

**Version 0.2, 1 November 2026.** Replaces v0.1 (saved in `docs/archive_rev2/`). The old update-unit semantics, intra-block context, exposure statistic and E-ADV conditions are retired; `DECISIONS.md` D-002, D-003 and D-005 give the reasons.

Two people must get the same numbers from this text alone. Anything ambiguous is a bug in this file. Each MUST is enforced by a test named here, unless marked as not yet built.

---

## 1. Conventions

| Clause | Rule |
|---|---|
| 1.1 Time | `t` is int64 seconds since midnight of the file's earliest date. AMLworld has 1-minute resolution. Raw timestamps are parsed as naive times |
| 1.2 Order | Canonical order is `(t, event_id)`. `event_id` is the raw CSV row index (header excluded). `pos` is the position in canonical order. All derived arrays index by `pos` |
| 1.3 Accounts | Keyed by `bank|account`. `node_id` is numbered by first appearance in canonical order, `src` before `dst` |
| 1.4 Self-loops | A self-loop appears in both the in and out lists of its account |
| 1.5 Splits | `0` train, `1` validation, `2` test |

---

## 2. Data

Source: Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.

| File | Data rows | SHA-256 |
|---|---|---|
| `HI-Small_Trans.csv` | 5,078,345 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `LI-Small_Trans.csv` | 6,924,049 | `f7a9940339c78b5d1476071505b5867c0f5319640b9c4279d13f531bf965319d` |
| `HI-Medium_Trans.csv` | 31,898,238 | `3126afb8155e7c8815d62bc5370549a7b5ae6bf7dfe872b3cd7813e66a3d7ff5` |
| `LI-Medium_Trans.csv` | 31,251,483 | `0fc89584453c97472b1bfde7ca746e21d7167140c22a6c4875272b197a4a3910` |
| `HI-Small_Patterns.txt` | 370 attempts, 3,209 lines, 0 unmatched | `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b` |
| `LI-Small_Patterns.txt` | not yet parsed | `734c15b7ff4be1a04a3529bd1524b92c1e8b62289d3ce4881423596ee47a208f` |
| `HI-Medium_Patterns.txt` | not yet parsed | `d612cd7ed6f7761c0e36f3b788dfbae68aa0842c77145b192c777c22daba5c9f` |
| `LI-Medium_Patterns.txt` | not yet parsed | `6aa4c4041894eb272339ed10c862a4ee1ac53a469326ed9dd3dd9e35da0e625c` |

**MUST** verify every download by row count and hash. One endpoint truncated a file while returning success.

**Facts from the HI-Small audit:** 38% of illicit transfers belong to no listed attempt. No transfer belongs to more than one attempt.

---

## 3. Split and protocols

**Split rule** (`multignn_split_days`, the same as Multi-GNN's `data_loading.py`): whole days; brute-force search over cuts `i < j` that minimises the maximum relative error from 60/20/20. HI-Small gives days 0–5, 6–7 and 8–17, or 64.0/19.0/17.0% of transfers.

| Protocol | Meaning |
|---|---|
| **Released** | All days in the file, as published work does |
| **Documented** | Later days removed before evaluation (`--max-day 9` for Small) |

**Test regions:**
- **Core:** test events before the documented span ends.
- **Tail:** test events after it.

| Dataset | Documented days | Tail starts | Days in file |
|---|---|---|---|
| HI-Small | 10 | 2022-09-11 | 18 |
| LI-Small | 10 | 2022-09-11 | 17 |
| HI-Medium | 16 | 2022-09-17 | 28 |
| LI-Medium | 16 | 2022-09-17 | 27 |

---

## 4. Decision horizon and view

For event `e` at time `t`, with `end(e)` the last second of `e`'s split:

| H | View `V(e, H)` |
|---|---|
| `0` (strict past) | Events with `t' < t` |
| `> 0` | Events `e' != e` with `t' <= min(t + H, end(e))` |

- The decision for `e` is taken at `t + H`.
- `H = 1` second makes same-minute events visible, giving an upper bound on real time. `H = 0` gives a lower bound. **Report real-time performance as this interval, never as a single number.**
- `H = infinity` reproduces the whole-split graph.
- Grid: `0, 1, 60, 600, 3600, 21600, 86400, 259200, infinity`.

**MUST:** rewriting any event outside the view leaves features and samples unchanged.
Tests: `test_engine_matches_reference`, `test_strict_causal_ignores_future_and_ties`, `test_horizon_view_ignores_beyond_horizon`, `test_split_clip`, `test_sampler_never_exceeds_view`, `test_sampler_future_invariant`.

**MUST:** a detector is retrained at each H using that H's view.

---

## 5. Features (`src/amlh/features.py`)

| Block | Count | Content |
|---|---|---|
| Base | 17 | Log amounts, USD, currencies, format, same-currency, self-loop, ratios, hour |
| Past | 72 | Per endpoint and direction, W in {6, 24, 72} h: count, log-sum USD, distinct counterparties, time gap, std of log-amount, std of event time |
| Forward | 24 | Same statistics over the forward part of the view; zero at H = 0 |
| Pair | 8 | Past and forward count and value, for `(u, v)` and `(v, u)` |
| Motif | 5 | Two-paths, three-cycles closed and opened by `e`, convergent siblings out and in |

**Caps are part of the definition:** 256 distinct counts, 128 path and cycle scans, 32 siblings, 16 downstream accounts, 8 sibling neighbours. Reference tests lift the caps, so capping itself is not reference-checked.

**USD conversion:** the median exchange ratio on the training split only.

### Logical-clock family (`src/amlh/logical.py`)
For each `k` in {8, 32, 128}:
- **Per endpoint and direction, over the last `k` events:** count, log-sum USD, distinct counterparties, std of log USD.
- **Per endpoint, over its last `k` merged in and out events:** whether any was an inflow, and signed log of net USD.

Also:
- **Per endpoint:** log of all prior events on the account.
- **At H > 0:** the same four set statistics over the next 32 visible events.
- **Pair and motif blocks,** computed over all visible history instead of a 72 h window.
- **Base:** the event's own attributes only. No wall-clock ratios and no hour of day.

Visibility still comes from `V(e, H)`: same-minute events count as prior at H > 0 and stay invisible at H = 0, so a burst remains a real threat to a strict-past logical detector.

**MUST:** the family is unchanged by any order-preserving re-timing, with no new ties, of an attempt on accounts with no other traffic (`test_order_preserving_retiming_changes_nothing_for_a_mule`).

---

## 6. Detectors

**LightGBM (B1):** `learning_rate 0.05, num_leaves 63, min_data_in_leaf 20, feature_fraction 0.8, bagging_fraction 0.8, lambda_l2 1.0`, up to 800 rounds, early stopping at 50 on validation average precision. All positives plus a fixed sample of negatives (rate 0.25). Seeds {0, 1, 2}. Use LightGBM **4.6.0**; 4.7.0 crashed in testing.

**TD-GNN (B3, `src/amlh/gnn.py`, `sampler.py`):** two hops. Up to 8 past and 8 forward first-hop events per endpoint and direction, and 3 + 3 at the second hop, nearest in time and obeying the **seed's** view. Pooling: mean, max and log-count. Hidden size 64. Weighted BCE. **Trained on the same view it is evaluated with.**

**Official Multi-GNN (B2):** official model, data and evaluation code. `src/amlh/mgnn.py` repeats the official training loop to add epoch checkpoints, scores test only after training, and adds region metrics and cadence replay and retraining. A model retrained at a cadence is scored at that cadence, on validation and test. Like the official code, it evaluates in PyTorch's training mode (`DECISIONS.md` D-017), and a second scoring pass in evaluation mode is written beside it. **Never executed yet.** Never present a re-implementation as Multi-GNN.

**FraudGT audit (`src/amlh/fraudgt.py`, `fraudgt_driver.py`):** published code and configuration at a pinned commit, on HI-Small, 1 seed, at most 20 training hours. Model selection is unchanged. Test scores come from one unshuffled pass over every edge, and its own subset evaluation is reported beside them. **Never executed yet.**

**Date rule (B0):** flags every tail transfer. A diagnostic, not a detector.

---

## 7. Metrics

| Metric | Definition |
|---|---|
| Minority F1 | Threshold maximises F1 on validation; highest such threshold on ties. Never re-tuned on test |
| PR-AUC | Average precision |
| Budget threshold | Score giving `B` alerts/day on validation. `B` in {100, 500, 2000} |
| **Stoppable value** | First alert time `T = min(t_e + H)` over flagged members of attempt `A`. `SV = sum(usd : t_e > T) / sum(usd)`. Undetected attempts score 0. Whole-window alerts fire at the window's end |
| Transfer share | The same with transfer counts |
| Attempts in scope | First transfer falls in the test window |

**MUST** report the equal-alert analysis alongside validation thresholds: alert rates shift between validation and test (at a nominal 500/day, 944 at strict past against 575 at 1 h on HI-Small). Equal-alert: the top `B x D` scores on the test window, with `D` the test core's length in calendar days (2 for HI-Small). `src/amlh/evaluate.py` computes `D` per dataset. `scripts/08_analyze.py` still assumes `D = 2`, which is correct only for the Small datasets it covers.

**Sealed scoring (`DECISIONS.md` D-012):** on LI-Small, HI-Medium, LI-Medium and SAML-D, notebooks show validation metrics and test-score health checks only. Test metrics are computed once, per dataset, in the merge step.

**Limits:** SV assumes an alert stops later transfers in the attempt, and it can count the same funds across hops.

**Statistics:** paired bootstrap over attempts for attempt metrics, 2,000+ replicates. Paired moving-block bootstrap for event metrics. Holm correction within each family. Seeds shown individually.

---

## 8. Adaptive alert policy (`alert_policy` in `src/amlh/jobs_gbt.py`)

Checkpoints `H_1 < ... < H_m` = `1, 600, 3600, 21600, 86400`. Alert at the first `j` where `s_j(e) >= tau_j`. Budget fractions `alpha_j` sum to 1 and are spent in order, each `tau_j` set on validation so that transfers alerting first at `j` use `alpha_j` of the budget. `alpha` maximises validation stoppable value on a 0.1 simplex grid.

**MUST:** test data influences neither `alpha` nor `tau`.

---

## 9. Timing attack (`src/amlh/perturb.py`)

Attacked attempts: first event in the test window, at least 2 events. `W` in {1, 6, 24} h.

| Condition | Map |
|---|---|
| `burst` | `t' = clip(c, t - W, t + W)`, with `c` the attempt median |
| `stretch` | `t' = t + W(2r - 1)`, with `r` the rank in [0, 1], then snapped to the grid and made non-decreasing |
| `blend` | Stretch in logical time: `t'_i` is the earliest minute where the count of the attempt's accounts' other events reaches an evenly spaced target, then clipped to budget and test range and made non-decreasing. Exact formula in `DECISIONS.md` D-012 |
| `shift_burst`, `shift_stretch`, `shift_blend` | Rigid shift by the mean displacement of the named attack, with random sign per attempt. **The control** |

**MUST (tested in `tests/test_perturb.py`):**
- Only times change.
- `|t' - t| <= W`, on the minute grid, inside the test range.
- Non-decreasing within an attempt, which preserves funds-flow order.
- Controls keep internal gaps.

Thresholds stay fixed from clean validation, and features are recomputed from the perturbed stream. Report suppression **relative to the matched control**, not relative to clean.

---

## 10. Queue simulation (`src/amlh/queue_sim.py`)

One FIFO worker. Take up to 1,024 waiting events; if fewer are waiting, wait until the oldest has aged 5 ms or the batch fills. Service time is a log-log interpolation of the measured median. Latency = completion - arrival.

**MUST:** mean sojourn time matches M/M/1 `1 / (mu - lambda)` within 5% over 10^6 arrivals (`test_mm1_mean_sojourn`).

Name the device in every latency figure.

---

## 11. Run outputs

Each job writes `jobs/<job_id>/`, with `config.json`, `status.json`, `log.jsonl`, `metrics.json`, `scores_val.parquet` and `scores_test.parquet` (`event_id`, `score`), and `DONE`, which holds a hash of every artefact and is written last. `src/amlh/merge.py` verifies these, unseals, gives verdicts and exports results that `scripts/08_analyze.py` turns into tables and figures.

**Rule:** a horizon enters a table only when all its seeds finished.
