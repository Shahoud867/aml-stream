# Research Plan: Decision Horizons in Graph-Based Anti-Money Laundering

**Version 4, 1 November 2026.** Replaces v3, which is saved in `docs/archive_rev2/`.
**Milestone:** 15 November 2026 is a progress date, confirmed by the instructor. **Goal:** a defensible paper at a well-known conference.

Full literature review and methodology: `paper/cn_phase2.tex`, Revision 3. Definitions: `SPEC.md`. Schedule: `ROADMAP.md`. Reasons for each decision: `DECISIONS.md`.

---

## 1. The idea in one paragraph

AML papers ask how accurate a detector is. We ask **what it knows when it decides, and whether that decision is early enough to stop the money.** Published results on the main benchmark let each test transaction see transactions that happen after it, on a test set that runs past the end of normal traffic, and they use a metric that ignores when an alert fires. We make the decision horizon explicit, measure what it costs and yields, and propose detection based on evidence an adversary cannot change by re-timing transactions.

---

## 2. How much of this is established

Each statement has one of four labels. Keep these labels when writing the paper.

| Label | Statement |
|---|---|
| **Verified** | Multi-GNN's test graph holds all transactions, and its loaders have no time restriction. MEGA uses the same construction, and FraudGT's dataset class builds its test graph the same way |
| **Verified** | The AMLworld files run past their documented span: 18, 17, 28 and 27 days against 10, 10, 16 and 16. Normal traffic stops at exactly the documented boundary |
| **Measured, all 4 datasets** | A rule that only flags the tail scores 48.18%, 26.15%, 45.37% and 28.76% F1. This is a diagnostic, not a detector |
| **Measured, HI-Small only** | F1 rises from 54.4% (strict past) to 73.5% (1 h), then plateaus |
| **Measured, HI-Small only** | Stoppable value at 500 alerts/day: 0.61 at 1 h, 0.44 at 24 h, 0 at the whole window. With alerts held equal, 0.43 at strict past and 0.58 at 10 min |
| **Hypothesis** | All of the above generalises to other datasets, another simulator and real GNNs |
| **Hypothesis** | Timing-only attacks work, and logical-clock evidence resists them |
| **Proposal** | Timing-invariant detection (§4) |

**The biggest open question:** the official Multi-GNN has not been run. Everything said about published models is an inference from their code until it has.

**A second caveat:** the shuffled-label leakage check has only finished at H = 0. Results at H > 0 carry that caveat until it finishes.

---

## 3. What we claim, and what we do not

| Contribution | Why it is new | Closest prior work | Current strength |
|---|---|---|---|
| **A. AMLworld audit and clean protocol** | We found no report of the tail effect | Daniluk & Dabrowski (a trivial baseline beats TGB: a precedent, on another benchmark) | **Strong** |
| **B. Horizon curves for graph AML, at equal views** | Quality against an explicit horizon, retrained per horizon | Nguyen et al., AISTATS 2022 (future data helps in card fraud: one card's sequence, no cost side) | **Promising** |
| **C. F1 and stoppable value disagree** | Shows waiting has a cost that F1 hides | FRAUDability and Shen & Kurshan use money metrics, but not for this question | **Promising** |
| **D. Timing-only threat model** | Every attack we found changes the graph | Heijmans et al., Oct 2026 (structural; names timestamps as future work). **Time-sensitive** | Open, not yet run |
| **E. Timing-invariant detection** | A defence by design, needing no attack examples | Last-k sampling exists; using it as a timing defence does not appear to | **Highest potential, least evidence** |
| **F. Adaptive multi-horizon alerts** | Spending a budget across decision points | Alpha-spending; Shen & Kurshan adjust thresholds over time, not per-transfer horizon | **Weak so far** (one seed) |

**We do not claim:** the first multigraph GNN, the first streaming AML system, the best accuracy, the first proof that future data helps, a new money-based metric, or the first adversarial study of AML GNNs.

**What we retired, and why:**
- **Staleness as the main question:** a detector alerting when a batch closes may legally use that whole batch, so "coarser means worse" came from TGN's rules, not from causality.
- **Intra-block context:** AnchorRM covers the learned version, and using the whole batch is stronger anyway.
- **Random boundaries as a defence:** a one-minute burst against a one-hour unit is split only 1 time in 60.
- **DMT-GNN as the headline:** it is incremental next to MEGA, DIAM and SALT-GNN. It returns only if Gate G2 finds a gap worth closing.

---

## 4. The real-world innovation: timing-invariant detection

**Observation.** Laundering's one free variable is timing. Amounts and counterparties are constrained by the scheme, but submission time is entirely the launderer's choice. Yet detectors lean heavily on wall-clock windows such as "transfers in the last 24 hours". That is evidence an attacker can move.

**Proposal.** Build evidence on what re-timing cannot change:
1. **Logical-clock windows:** an account's last *k* events, not its last *W* hours (after Lamport's logical clocks).
2. **Event order:** did an inflow precede this outflow?
3. **Conservation of funds:** net value passing through an account across those events.

**Why it could work.** A sibling transfer leaves a last-*k* window only if *k* other events of that account fall between them. For a dedicated mule account with little other traffic, no amount of stretching removes the evidence.

**Why it might fail.**
- Wall-clock gaps carry real signal, so the clean-data cost could be too high.
- High-traffic accounts have background events that can push siblings out of the window.
- Burst attacks raise a separate problem: same-minute events have unknown order.

**What we test (C8):** wall-clock, logical-clock and combined detectors, on clean and attacked data. The result is a **price of robustness**, and a high price is still a reportable finding.

**Why this could matter beyond the paper.** It needs no attack data and no adversarial training, so a bank could adopt it by changing how features are defined.

---

## 5. Research questions and frozen hypotheses

- **RQ1, decision horizon:** how do F1, PR-AUC and stoppable value change with H? Do they agree on the best H?
- **RQ2, benchmark integrity:** how much do the post-simulation days contribute, and do rankings change without them and at equal views?
- **RQ3, adaptive alerting:** does spending a budget across horizons beat the best fixed one?
- **RQ4, timing robustness:** can a timing-only adversary suppress detection beyond a control, and does logical-clock evidence resist this?

The confirmation thresholds are frozen in `DECISIONS.md` (D-001, D-008, D-012). **HI-Small suggested these hypotheses, so it cannot confirm them.**

| ID | Claim | HI-Small (discovery) | Confirmation |
|---|---|---|---|
| C1 | The tail inflates F1 by more than 5 points | 11.0 points | Pending |
| C2 | Look-ahead is worth more than 0.05 core PR-AUC | 0.199 | Pending |
| C3 | F1 and stoppable value disagree on the best H (descriptive: close to true by construction) | 0.179, CI [0.133, 0.229] | Pending |
| **C3b** | **At equal alerts, a short wait beats strict past on stoppable value** (primary timing claim) | +0.148, CI [+0.095, +0.205]; exploratory | Pending |
| C4 | Real-time is an interval wider than 0.02 PR-AUC | 0.157 | Pending |
| C5 | Adaptive beats best fixed at 500/day | One seed; CI includes zero | Pending |
| C6 | Bursts evade strict-past detectors; same-minute view blunts this | Not run | Pending |
| C7 | Equal views change GNN vs boosted trees (two-sided) | Not run | Pending |
| C8 | Logical-clock evidence resists both `stretch` and `blend` at acceptable cost | Not built | Pending |

---

## 6. Experiments

| ID | Experiment | Answers | Status |
|---|---|---|---|
| E1 | Official Multi-GNN by test region and cadence replay | RQ2 | **Not run; most important** |
| E2 | Leakage checks at positive horizons (shuffled labels) | Validity | **Only H = 0 done; must finish before claims at H > 0** |
| E3 | LightGBM sweep on LI-Small, HI-Medium, LI-Medium | RQ1, RQ2 | Not run |
| E4 | SAML-D sweep (independent simulator) | RQ1 | Not obtained |
| E5 | TD-GNN at several horizons | RQ1, C7 | Code tested; no full run |
| E6 | GFP + XGBoost on the same split; FraudGT region audit on HI-Small | RQ2 | Not run |
| E7 | Adaptive alert policy, all seeds | RQ3 | One seed |
| E8 | Burst, stretch and `blend` attacks with matched controls | RQ4 | Burst and stretch code tested, not run; `blend` not written |
| E9 | Logical-clock features, then detector comparison | RQ4, C8 | **To build** |
| E10 | Ablations A1–A10 | Brief item 6 | Not run |
| E11 | Service times and queue simulation | CN angle | Simulator tested; no profile |
| E12 | Rolling-origin folds | Robustness | After 15 Nov |
| E13 | Real Ethereum window | External validity | After 15 Nov |

Where each one runs is in `ROADMAP.md` §3.

**Controls that must not be cut:** the matched-displacement shift in E8, per-horizon retraining, thresholds set only on validation, test sets used once, and sealed scoring on confirmatory datasets (`DECISIONS.md` D-012).

---

## 7. Threats to validity

| Threat | Why it matters | What we do |
|---|---|---|
| **Simulator artefact** | Same-minute bursts may reflect AMLworld's scheduler | SAML-D, ablations of hour and format, real Ethereum. No real-world effect sizes claimed |
| **Feature-model specific** | The GNN may behave differently | E1 tests the official code; Gate G1 records the result either way |
| **Few attempts** | Only 86 test attempts on HI-Small | HI-Medium, rolling-origin folds |
| **Alert-rate shift** | At a nominal 500/day, strict past raises 944 and 1 h raises 575 | Equal-alert analysis always reported |
| **Stoppable value is an accounting measure** | Assumes alerts stop later transfers; can count funds across hops | Stated plainly; transfer count reported too; hop-aware version planned |
| **Attacking all attempts at once** | Favours the attack | Stated; per-attempt attacks planned |

---

## 8. Publication strategy

**Two focused papers are more defensible than one overloaded paper.**

| Paper | Content | Target | Status of dates (checked 1 Nov 2026) |
|---|---|---|---|
| **A** | Audit, horizon framework, stoppable value | **KDD 2027 Datasets & Benchmarks** | Conference 1–5 Aug 2027. Cycle 1 closed. A February 2027 deadline is mentioned for the Research Track; **no second-cycle call for this track was published yet** |
| A, fallback | Same | **ACM ICAIF 2027** | GFP, FraudGT and LAS-GNN appeared at ICAIF. ICAIF 2026 closed in August, with 8 pages including references. **The 2027 call is not published; August is an inference** |
| **B** | Timing-only attack, timing-invariant detection | **Financial Cryptography 2028** | FC 2027 closed 17 Sep 2026. **2028 dates are not published.** Only if it holds beyond one simulator |

Alternatives: ECML PKDD Applied Data Science, Learning on Graphs, and the Temporal Graph Learning workshop.

Not recommended now: main ML research tracks, which expect a new method with large effects on real data, and networking flagships, where the systems part is too small to carry a paper.

**Minimum evidence for submission:** Multi-GNN audited; curves for two model families on two simulators; hypotheses confirmed on unseen data; robust across folds; code and audit tool released; literature search repeated.

**Before submission:** contact the AMLworld maintainers about the tail. Decide on a preprint, because the timing-attack space may close soon.

---

## 9. Ethics

Synthetic data only. No real institution is involved, and the work gives no operational guidance against any deployed system. Defences are reported with attacks.
