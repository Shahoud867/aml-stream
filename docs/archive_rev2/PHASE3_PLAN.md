# Phase 3 Execution Plan: Streaming AML — Freshness, Timing Attacks and Defences

**Deadline:** Sunday 15 November 2026 · **Plan revised:** Sunday 25 October 2026 (v3: adversarial study + verification regime) · **Time available:** 21 days

Aligned with **Phase 2 Revision 2** (`cn_phase2.tex`). Where the two documents differ, update both. Section numbers below refer to Phase 2.

**What changed in v3:**
1. Added the **adversarial timing study (E-ADV)** as the headline differentiator, gated at G2.
2. Added a full **verification regime** (§6) — reference implementations, property tests, sanity checks, and a numbers-provenance rule.
3. Re-ordered the build so **all code, including optional studies, exists in week 1**, making the G2 decision purely about GPU hours and analysis time.

---

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Primary question (RQ1) | **Freshness–latency:** how detection quality (overall and by class), delay, latency and throughput change as update units get coarser |
| Mitigation (RQ2) | **Intra-block context (IBC):** exact statistics of earlier same-account transfers inside the current update unit |
| Secondary (RQ3) | **Multi-edge value:** DMT-GNN vs merged vs TGN |
| **Headline differentiator (E-ADV)** | **Is the refresh interval an attack surface?** Timing-only adversary; two defences measured. Gated at G2, code built in week 1 |
| Core data | IBM HI-Small, Multi-GNN preprocessing and split. LI-Small / Ethereum = winter work, not Phase 3 |
| Seeds / tuning | 3 seeds; 8-config grid with successive halving |
| Baselines | B1 XGBoost + causal features (continuous + periodic); B2 Multi-GNN (1–2 seeds, Linux); B3 TGN (± IBC); B4 merged. FraudGT + Multi-GNN published numbers, not re-run. MEGA-GNN dropped |
| GPU allocation | **Kaggle** unattended for training; **lab GPU** (2–3 h/day) for timing, profiling, block runs, sweeps, E-ADV inference; **laptops** for all precompute and evaluation |
| Verification | **No full run starts until Tier 1–3 tests pass** (§6). Every reported number traced to a log + commit hash |
| Primary success criterion | A complete, verified report satisfying all 7 brief components. Publication is a winter extension |

---

## 1. Novelty check (run 25 Oct — result and limits)

### 1.1 What exists

| Work | What it does | Why it is not this |
|---|---|---|
| **MemFreezing** | Injects fake nodes/edges to lock TGN node memories into a frozen state | **Structural injection**, not timing. Closest collision — **read in full before committing (day 1)** |
| **HIA** (arXiv 2509.25418) | Ranks high-impact nodes, perturbs structure with low temporal violation | Structural, white-box |
| **TGA, TD-PGD, T-SPEAR** | Evasion on temporal graphs via edge/snapshot perturbation | Structural, white-box |
| **PRES, BADGNN, AnchorRM, StreamTGN** | Batch staleness as a training/serving **efficiency** problem | Never adversarial |
| **BlazingAML** | Notes laundering has "temporal fuzziness" — flexible timing order | **Supports our premise**, does no attack analysis |
| Alert-optimisation work | Grid-searches graph time windows (0–90 days) | Window as a hyperparameter, not an attack surface |
| GARG-AML | Lists transaction timing as future work | Confirms timing is under-modelled |

### 1.2 What appears open
- Framing a detector's **update/refresh interval** as an exploitable parameter.
- A **timing-only** adversary: changes nothing but *when* real transactions are submitted. No injected accounts, no changed amounts, no changed structure.
- **Randomised refresh boundaries** as a countermeasure.

### 1.3 The distinction the paper rests on
> Every prior temporal-GNN attack **modifies the graph**. Ours modifies **only the submission times of transactions that were going to happen anyway.** A launderer cannot inject fake accounts into a bank's ledger, but they have complete control over timing. This is a strictly weaker and far more realistic adversary.

### 1.4 Honest limits of this check
- Three queries only; one search engine; some results flagged as possibly dated.
- MemFreezing and HIA were seen through search excerpts, not full texts.
- **Day-1 task:** read MemFreezing and AnchorRM in full; re-run the search with "sliding-window graph detection", "adversarial evasion transaction monitoring", "model update lag adversarial".
- **If MemFreezing or another paper already covers timing-only evasion against the refresh cycle:** drop E-ADV, fall back to the measurement study, and record the decision in `DECISIONS.md`.

### 1.5 Research ethics
Synthetic data only (IBM simulator). The study demonstrates a weakness in a *detector design* and publishes defences alongside it. No real institution, chain or account is involved, and nothing here provides operational capability against a deployed system. State this explicitly in the report.

---

## 2. Inputs and missing information

**Based on:** the Phase 3 brief (7 components); Phase 2 Revision 2; the lab PC (RTX 4000 Ada 20 GB), MacBook Pro M1 Pro 32 GB, HP 15s 20 GB. The Quadro T1000 is excluded.

**Still needed from you:**

| # | Missing | Assumption | Impact |
|---|---|---|---|
| 1 | **Phase 2 class feedback** | Unknown; report reserves a section | Required by the brief; cannot be written for you |
| 2 | Lab access: days, weekends, screen-lock survival, `pip` proxy | 2–3 h on ~14 weekdays; no admin; nothing unattended | How much moves to Kaggle |
| 3 | Kaggle quota in **your** accounts | ~30 GPU-h/week/account, 12-h sessions (unofficial) | Main training capacity |
| 4 | Submission format, page limit, viva? | LaTeX report + repo (+ demo/slides) | Week-3 effort |
| 5 | Can Revision 2 be resubmitted? | Attached to Phase 3; changes listed in §3 | Avoids confusing the grader |

---

## 3. Changes from Phase 2 (use directly in the report)

| # | Change | Phase 2 location | Justification |
|---|---|---|---|
| C1 | RQs reordered: freshness is **RQ1**, multi-edge value **RQ3** | §9 | Freshness is the defensible, networking-specific contribution |
| C2 | New **RQ2** (IBC) and ablation **A7** | §9, Steps 4, 5, 10 | Measurement plus a tested remedy |
| C3 | Prior work added: FraudGT results (76.41 F1), SALT-GNN, PRES, BADGNN, AnchorRM, StreamTGN | §5.1.5–5.1.7, §6, §8 | FraudGT sets the real bar; the staleness line overlaps our claims |
| C4 | Multi-GNN's own formatter and split; boundaries recorded | Step 3 | Comparability (60/20/20 vs FraudGT's 64/19/17) |
| C5 | 3 seeds; 8-config grid instead of 20-trial Optuna | Steps 7a, 9 | Compute; block bootstrap carries the uncertainty |
| C6 | MEGA-GNN dropped; FraudGT added as published reference | Step 6 | No second offline reproduction in 21 days |
| C7 | D2 / LI-Small moved to winter | Steps 1–2, §13 | Compute realism |
| C8 | Live load generator → service-time profile + queue simulation + one live validation | Step 7c | Same quantities, a fraction of the GPU time |
| C9 | B1 uses our own causal features; GFP only on Linux | Step 6 | GFP unsupported on Windows |
| C10 | New metrics: class-conditional staleness, exposure ρ, end-to-end delay, recovered fraction | Step 8 | Needed for the class-asymmetry hypothesis and IBC |
| C11 | Two training regimes: fine (200 events), block (10-min units) | Step 5 | IBC needs block-like units in training; block training is faster |
| C12 | 12-week plan → 21-day plan, gates G0–G2, 2–3 GPU-h/day | §13 | Actual constraints |
| **C13** | **Adversarial timing study (E-ADV)** | **new** | **Converts a measurement into a security finding; near-zero marginal GPU cost** |
| **C14** | **Verification regime: reference implementations, property tests, provenance rule** | **Step 12 / tests** | **Agent-assisted code requires independent checking; also strengthens the methods section** |
| C15 | `SPEC.md` refinements: amounts summed in USD with train-only rates (D-2); every history input, including the port index, uses only earlier units, with IBC the sole exception (D-3); B4 keeps DMT-GNN's counterparty set (D-4); exposure ρ capped at 1 (D-6) | Steps 4, 6, 8 | Phase 2 left these unspecified; without D-3 the IBC on/off comparison would be contaminated |

---

## 4. Research questions and pre-registered criteria

Write these into `DECISIONS.md` **before computing anything**. Do not revise after seeing results.

### 4.1 Questions
- **RQ1 (primary):** How do minority F1, PR-AUC and detection delay change as updates coarsen (*b*, *Δ*, *R*)? Is the loss larger for illicit than benign transfers, and for some typologies more than others? What latency and sustainable throughput does each granularity give under load *λ*?
- **RQ2:** Does IBC recover the staleness penalty, and at what latency cost?
- **RQ3 (secondary):** Does DMT-GNN beat merged edges (B4/A1) and TGN (B3)? In which multiplicity and degree strata?
- **E-ADV (headline):** Can a timing-only adversary suppress detection by exploiting the refresh interval, and do IBC or randomised boundaries defend against it?

### 4.2 Hypotheses and thresholds

| ID | Hypothesis | Test | Supported if |
|---|---|---|---|
| **G0** | Same-unit history matters for illicit transfers | Exposure ρ at Δ=10 min, by class (CPU, data only) | Mean illicit ρ ≥ 0.10 **or** ≥ 1.5× benign. Else RQ2 → 1 exploratory config |
| H1 | Coarser units reduce quality | ΔPR-AUC(Δ=60 min), TGN and DMT-GNN | 95% bootstrap CI > 0 |
| H2 | The penalty is class-asymmetric | Illicit vs benign mean score drop | CI of the difference > 0 |
| H3 | IBC recovers the penalty | Recovered fraction at Δ=10 min | ≥ 0.5, CI > 0 |
| H4 | Per-transfer aggregation helps | DMT-GNN − B4 on PR-AUC, stratified | CI > 0 **and** gain concentrated in high-multiplicity strata |
| **H5** | **Timing-packing suppresses detection** | Attack success rate, C1 vs C2 (not vs C0) | CI > 0 |
| **H6** | **Vulnerability is typology-specific** | Per-typology attack success | Fan-out / scatter–gather > cycles, ordered as predicted |
| **H7** | **Defences reduce residual risk** | Attack success under D1, D2 | Residual < 0.5 × undefended, CI > 0 |

A flat curve, a null H5, or a negative H3/H4 are all **reportable results**. The report states which hypotheses were pre-registered and which held.

---

## 5. The adversarial timing study (E-ADV)

### 5.1 Threat model
- **Adversary controls:** only the submission time of transfers belonging to laundering patterns they operate.
- **Adversary cannot:** change amounts, counterparties, pattern structure, or transfer count; inject accounts; access model weights or gradients.
- **Budget:** each transfer shifts by at most *W* ∈ {1 h, 6 h}.
- **Causal constraint:** funds must arrive at an account before being forwarded from it. Timing shifts must preserve the within-pattern funds-flow precedence. **This is a hard constraint and is unit-tested.**
- **Knowledge variants:**
  - **A-full:** knows Δ and the boundary phase (worst case).
  - **A-partial:** knows Δ but not the phase (realistic).

### 5.2 Conditions
| ID | Condition | Purpose |
|---|---|---|
| C0 | Original timestamps | Baseline |
| C1 | **Packed** — shift within *W* to maximise same-unit co-occurrence | The attack |
| C2 | **Random shift**, same magnitude distribution as C1 | **The control.** Separates *packing* from mere perturbation — without this the study is worthless |
| C3 | **Anti-packed** — spread across units | Upper bound on detectability |

### 5.3 Measures
- **Attack success rate:** fraction of laundering patterns flipping detected → undetected (threshold τ fixed on validation, never re-tuned).
- ΔPR-AUC and Δrecall restricted to attacked patterns.
- Δ detection delay.
- Breakdown **by typology** (H6) and by *W*, Δ, and knowledge variant.
- All reported **C1 vs C2**, not C1 vs C0.

### 5.4 Defences
| ID | Defence | Cost |
|---|---|---|
| D1 | Intra-block context on | 17 readout inputs, no memory change |
| D2 | **Randomised refresh boundary** — jitter the unit boundary uniformly in [0, Δ) | None at inference; adds expected Δ/2 jitter to delay |
| D3 | Overlapping / sliding units (optional) | Extra inference passes |

Report residual attack success and the latency cost of each.

### 5.5 Cost
CPU perturbation + **inference only on existing checkpoints**. No retraining. **~3–6 GPU-h total.**

---

## 6. Verification regime (non-negotiable)

Agent-assisted code that is subtly wrong is worse than slow correct code, because it produces plausible numbers. This section is also a genuine methods-section strength.

### 6.1 Tier 1 — Independent reference implementations
For each item below, write an obviously-correct brute-force version (O(n²) is fine) and assert equality with the fast version on ≥ 100 random small synthetic streams.

**Rule: the reference is written from the written spec, in a separate session, without sight of the optimised implementation**, so that bugs do not correlate.

| Component | Reference |
|---|---|
| Intra-block context | Per-event linear scan over the unit |
| Causal rolling features | Per-event scan over history |
| Neighbour sampling | Brute-force "last S events with t_e < t" |
| Detection delay | Per-pattern scan |
| Exposure ρ | Brute-force set construction |
| Merged aggregates | Group-by from scratch |

### 6.2 Tier 2 — Property tests (invariants)
- No sampled or context event has `t_e >= t` (**strict** inequality).
- Equal-timestamp events never contribute to each other's features.
- Memory is byte-identical before and after scoring within a unit.
- IBC at the finest granularity is **all zero**, and IBC-on scores there equal IBC-off scores exactly.
- Merged mode yields exactly one edge per (pair, direction).
- `max(t_train) < min(t_val) < min(t_test)`.
- **Resume == no-resume:** a run killed and resumed twice matches an uninterrupted run's validation PR-AUC within nondeterminism tolerance.
- **E-ADV:** perturbed streams preserve funds-flow precedence, amounts, counterparties, pattern membership and labels; only timestamps differ.

### 6.3 Tier 3 — Sanity / behavioural tests
- **Shuffled labels → collapse:** PR-AUC ≈ prevalence. If a shuffled-label run scores well, something leaks.
- **Reversed-time training → degradation.** If it doesn't, temporal information isn't being used.
- Memory off + S=0 ≈ feature-only performance.
- Attack condition C3 (anti-packed) should not *improve* detection beyond C0 by more than noise.

### 6.4 Tier 4 — Numerical and statistical
- Metric implementations checked against `sklearn` on random inputs.
- **Queue simulator validated against the M/M/1 closed form** for exponential service times before being driven by measured service times.
- Bootstrap CI coverage checked on synthetic data with a known effect size.
- Live-run latency percentiles compared against the simulation; any discrepancy reported, not hidden.

### 6.5 Tier 5 — Provenance of every number
- **No hand-typed numbers in the report.** `make tables` regenerates every table from logs.
- Each run folder carries `config.yaml`, `git_hash.txt`, `device.txt`, `train_log.csv`, `scores_<granularity>.parquet`.
- **Two-person check:** B independently regenerates A's headline numbers from raw logs before the freeze.

### 6.6 The gate
> **No full training run starts until Tiers 1–3 pass.** A test failure stops the pipeline. This is the single most important rule in the plan.

---

## 7. Compute strategy under 2–3 GPU-hours per day

### 7.1 Capacity
| Resource | Availability | Use for | Not for |
|---|---|---|---|
| Lab PC (RTX 4000 Ada) | 2–3 h × ~14 weekdays = **28–42 h**, interruptible | Pilot timing; service-time profiling; live validation; block runs; sweeps; **E-ADV inference** | Anything needing > 2 h uninterrupted |
| Kaggle (×2 accounts) | plan **40–60 h** total | Fine-regime training; Multi-GNN (Linux); finest-granularity sweeps | Latency measurement (different hardware) |
| molab (marimo) | Free; optional RTX Pro 6000 Blackwell 96 GB (public preview); ≤ 12 h; idle > 90 min shutdown; small persistent storage; **0 h planned until acceptance tests pass** | Extra GPU capacity; consistent-GPU fallback for latency profiling | Anything with credentials in notebook code (notebooks are public by default) |
| MacBook M1 Pro 32 GB (**Saif**) | Continuous, overnight | Model and runner development on CPU (1% subsets); cloud launches; heavy overnight CPU jobs if the HP runs short of RAM | Full GNN training |
| HP 15s 20 GB, Windows (**Shahoud**) | Continuous, overnight | All HI-Small precompute; XGBoost; evaluation watcher; queue simulation; the Windows lab-environment pack; report | Training |

**Need 37–89 GPU-h vs 68–102 available** (lab + Kaggle; molab is upside). Accuracy is device-independent; every latency number comes from one GPU only (the lab PC, or molab as the fallback, named in the report).

### 7.2 What is worth doing (ordered by value)
1. **Precompute everything data-dependent on CPU** — sorted stream, neighbour indices per granularity, causal features, IBC, merged aggregates, exposure ρ, **and the E-ADV perturbed streams**. Computed once, reused by every model × seed × config.
2. **Data resident on the GPU** (~1–3 GB) — no per-step host↔device copies.
3. **Checkpoint every ≤ 10 min**, `--time-budget` for graceful stop, atomic writes. A lost session costs ≤ 10 min.
4. **Job queue + asynchronous evaluation** — GPU jobs write `scores.parquet` only; a laptop watcher computes metrics, bootstraps and figures.
5. **Cheap tuning** — 8 configs, 25% window, ≤ 3 epochs, drop the worse half after epoch 1.
6. **Block-regime training** — far fewer sequential steps per epoch.
7. **Inference-only sweeps** from saved checkpoints; the expensive finest-unit replay runs once per checkpoint.
8. **Queue simulation** instead of live load tests.
9. **TF32 matmul** — free.

**Only if G1 shows a gain:** run packing (2–4 concurrent, keep only if aggregate events/s ≥ 1.3×); BF16 autocast on attention/MLP with GRU memory and loss in FP32; fine units of 500 instead of 200.

**Not worth it:** `torch.compile` / CUDA graphs (Windows support, compile time, variable unit sizes); multi-GPU; Bayesian search at ~8 trials; reduced-precision recurrent memory; DataLoader workers.

### 7.3 molab — verdict
molab is marimo's free hosted notebook service: 4 CPUs + 32 GB RAM per notebook, an optional RTX Pro 6000 Blackwell (96 GB) in public preview, ≤ 12-h sessions, shutdown after > 90 min idle ([docs](https://docs.marimo.io/guides/molab/)).

- **Advantage:** potentially the strongest free GPU available to us, and a *consistent* GPU model — the best fallback for latency profiling if the lab PC is unavailable.
- **Unknowns:** whether a running job survives a closed browser ("idle" is undefined); the usage quota ("reasonable usage" is unpublished); availability during preview.
- **Constraints:** small persistent storage (rebuild derived files from the raw CSV per session); notebooks are **public by default** (no tokens in code).
- **Recommendation:** adopt as **additional** GPU capacity only after the three acceptance tests in `ROADMAP.md` S-02 pass. Kaggle stays primary for unattended runs; Colab is attended overflow.

(Data-engineering note, unchanged: causal aggregates are computed once with DuckDB or polars and stored as Parquet. No OLAP cube server is used.)

### 7.4 Working in parallel
Four streams run concurrently each day: Kaggle unattended, one lab session, HP precompute/evaluation, MacBook model development. **Saif** = models, runner, cloud GPU queue, lab sessions (MacBook). **Shahoud** = precompute, harness, simulator, statistics, report (HP). Swap reviews daily; **pair-review all causal code**.

---

## 8. Technical architecture

### 8.1 Stack
| Layer | Choice |
|---|---|
| Env | Miniforge user-level + pip; pinned `environment.yml` |
| DL / graph | PyTorch (CUDA wheel) + PyG (pure-Python core): `TGNMemory`, `LastNeighborLoader` |
| Precompute | polars / DuckDB + numba |
| Multi-GNN | Official repo on **Kaggle (Linux)**; 5-working-day time box |
| Tabular | XGBoost (CPU) |
| Simulation | numpy/heapq discrete-event (or SimPy) |
| Report | LaTeX in the Phase 2 style; reuse `references.bib` |

### 8.2 Repository layout
```
configs/      model × regime × seed YAMLs; jobs_lab.csv, jobs_kaggle.csv
data/         raw/ (git-ignored), processed/ (Parquet), checksums.txt, DATA_CARD.md
src/prep/     format, split, causal_features, neighbours, ibc, exposure, merged
src/adv/      perturb.py (C0–C3), precedence.py, attack_metrics.py
src/models/   tgn.py, dmt_gnn.py (flags: merged, shared_direction, use_memory, use_ibc), heads.py
src/train/    runner.py (resume, time budget), train_loop.py, early_stop.py
src/stream/   replay.py, profile_service.py, queue_sim.py, live_validate.py, jitter.py
src/eval/     metrics.py, delay.py, bootstrap.py, strata.py, watch_eval.py
tests/        tier1_reference/, tier2_properties/, tier3_sanity/, tier4_numerical/
results/      runs/<run_id>/, tables/, figures/
report/       Phase 3 LaTeX
DECISIONS.md  pre-registered criteria; every change with its reason and date
```

### 8.3 The runner
`runner.py --config X --time-budget 140m` — resumes from the latest checkpoint, checkpoints every 10 min and at epoch end, exits cleanly 5 min before the budget, prints `DONE` or `RESUME`. `run_queue.py jobs_lab.csv` picks the next unfinished job, so every lab session starts with one command.

---

## 9. Day-by-day (26 Oct – 15 Nov)

> **Superseded for scheduling by `ROADMAP.md` §9** (25 Oct). The university-PC access date is unknown, so the roadmap moves all lab work to "from the day access starts", adds molab and Colab, and assigns every task to Saif or Shahoud. The table below is kept for reference only.

**Build principle:** all code — including E-ADV — exists and passes tests by **30 Oct**. The G2 decision is then purely about GPU hours and analysis time, never "can we build it".

### Week 1 — code, verification, precompute, pilot, G0/G1
| Date | Lab session | Kaggle | Mac / HP | Output |
|---|---|---|---|---|
| Sun 25 | — | Accounts; attach IBM dataset; check quota | Download HI-Small + patterns; repo skeleton; **`DECISIONS.md` with all pre-registered criteria** | Plan frozen |
| Mon 26 | **S1:** env check (`nvidia-smi`, torch CUDA, bf16, PyG); install; copy data | — | Multi-GNN formatter → Parquet; **data card**; **read MemFreezing + AnchorRM in full**; re-run novelty searches | M0; **E-ADV go/no-go** |
| Tue 27 | **S2:** TGN edge classifier; kill-and-resume test | **K1:** Multi-GNN env + seed 42 | **Tier 1 reference implementations**; causal features; neighbour indices | Runner works |
| Wed 28 | **S3:** full-epoch timing TGN (fine/block); run-packing test | K1 continues | **Tier 2–3 tests pass**; IBC; merged; **exposure ρ → G0**; B1 continuous | **G0 decided**; first B1 number |
| Thu 29 | **S4:** DMT-GNN timing; BF16 vs FP32; IBC overhead | Upload precomputed files as a private Kaggle dataset | **E-ADV perturbation code + precedence tests**; generate C0–C3 streams | E-ADV ready |
| Fri 30 | **S5:** block regime: TGN ± IBC seed 42 | **K2:** tuning grids, successive halving | **G1:** fill budget table; fix seeds/packing/precision; **Tier 4 tests (M/M/1 validation)** | **G1 decided** |
| Sat 31 – Sun 1 | (bonus) | **K3/K4:** TGN fine ×3; DMT-GNN fine seed 42 | Replay harness; delay computation; `watch_eval.py` | — |

### Week 2 — main runs, profiling, G2, E-ADV
| Date | Lab session | Kaggle | Mac / HP | Output |
|---|---|---|---|---|
| Mon 2 | **S6:** block: DMT-GNN ± IBC seed 42 | **K5:** DMT-GNN fine seeds 101, 202 | Auto-evaluate finished runs; queue simulator | — |
| Tue 3 | **S7:** **service-time profile** (TGN, DMT-GNN) | **K6:** merged (B4) ×3 | B1 periodic (R = 10 min, 1 h, 24 h) | Service curves |
| Wed 4 | **S8:** profiles for DMT+IBC, B4, B1-CPU; remaining block seeds | **K7:** A2 ×3; **K1b:** Multi-GNN seed 101 | First staleness curves (RQ1) | — |
| Thu 5 | **S9:** granularity sweeps on block checkpoints | **K8:** A3 ×3 | **G2:** ≥ 80% of core runs evaluated? | **G2 decided** |
| Fri 6 | **S10:** **E-ADV inference** (C0–C3 × Δ × W, undefended) | **K9:** finest-unit sweeps | Bootstrap tables v1; figures v1 | Attack results |
| Sat 7 | **S11:** **E-ADV defences** (D1 IBC, D2 jitter) | Reruns | Attack analysis by typology (H6) | Defence results |
| Sun 8 | (bonus) | Remaining sweeps | Statistics; stratified + per-typology analysis | **M4: core complete** |

**If G2 fails:** skip S10/S11 entirely. The core report is complete without E-ADV; record the decision in `DECISIONS.md`.

### Week 3 — validation, analysis, report
| Date | Lab session | Mac / HP | Output |
|---|---|---|---|
| Mon 9 | **S12:** **live validation** of queue simulation (2 loads × 1 model) | Final tables and figures from logs | Validated latency |
| Tue 10 | **S13:** reruns / fixes only | Report: results + discussion | — |
| Wed 11 | **S14:** demo (optional) or buffer | Report: full draft; comparison vs published numbers | Draft v1 |
| Thu 12 | Buffer | **Tier 5 two-person provenance check**; README; repro commands | Numbers verified |
| Fri 13 | — | **Freeze**: code tag, report v2 | Frozen |
| Sat 14 – Sun 15 | — | Proofreading; submission | **Submitted** |

**Start drafting the report on 27 Oct** — sections 1–4 don't depend on results.

---

## 10. Run matrix and budget

| Task | Runs | Where | h/run | GPU-h |
|---|---|---|---|---|
| Pilot timing | 1 epoch/model | Lab | — | 1–2 |
| Tuning grids | 2 × 8 configs, 25% window | Kaggle/Lab | 0.1–0.4 | 2–6 |
| Fine runs: TGN, DMT-GNN, B4, A2, A3 | 5 × 3 seeds | Kaggle | 0.7–2.5 | 11–38 |
| Block runs ± IBC | 4 × 3 seeds | Lab | 0.2–0.6 | 2.5–7 |
| Multi-GNN (B2) | 1 config × 2 seeds | Kaggle | 4–8 | 8–16 |
| Granularity sweeps | 18 ckpts × 7; 9 × 2 | Lab/Kaggle | — | 8–12 |
| Profiling + live validation | 4 models | Lab | — | 1.5–2 |
| **E-ADV (inference only)** | **4 conditions × 3 Δ × 2 W × 2 defences** | **Lab** | — | **3–6** |
| B1, features, evaluation, simulation | — | CPU | — | 0 |
| **Total** | | | | **37–89** |

Per-run figures are **assumptions** until G1. The only published anchor is Multi-GNN's full model at 12,958 s ≈ 3.6 h on an RTX 3090.

**Reduction ladder (in order) if G1 projects an overrun:**
1. A2 → 2 seeds; ablation sweeps at 2 granularities only.
2. Fine units of 500 (report as part of the freshness analysis).
3. Multi-GNN → 1 seed, or published values only.
4. Drop A2 (A1, A3, A7 still satisfy the brief's three-ablation requirement).
5. E-ADV → one Δ and one *W* only.

**Never cut:** the verification tiers, 3 seeds on DMT-GNN/TGN/B4, the finest-unit reference, the IBC on/off comparison (unless G0 failed), or the C2 random-shift control (an E-ADV without C2 is not worth running).

---

## 11. Experiments, metrics, figures

### 11.1 Protocol
HI-Small, Multi-GNN formatter and split, full imbalance, edge labels. Chronological units; memory reset each epoch; early stopping on validation PR-AUC (patience 2); τ fixed on validation at the model's training granularity.

**Granularities:** finest (equal-timestamp groups); *b* ∈ {10, 100, 1000}; *Δ* ∈ {1, 10, 60} min; *R* ∈ {10 min, 1 h, 24 h} for B1.
**Regimes:** fine (200 events) and block (Δ_train = 10 min), each evaluated at every granularity.
**Load:** Poisson *λ* ∈ {10², 10³, 10⁴, 10⁵}/s plus block arrivals (12 s, 10 min); FIFO queue, dynamic batching (≤ 1024 events or 5 ms).

### 11.2 Metrics
- **Detection:** minority F1, PR-AUC (primary), P@100/day, per-typology recall.
- **Timeliness:** detection delay in data time and end-to-end; median and % within 10 min / 1 h / 24 h.
- **Staleness:** ΔPR-AUC(g); class-conditional score drop; recovered fraction (IBC); exposure ρ.
- **Systems:** service time vs unit size; p50/p95/p99 latency; sustainable *λ*; GPU memory.
- **Attack:** success rate (C1 vs C2), ΔPR-AUC and Δdelay on attacked patterns, by typology, *W*, Δ and knowledge variant; residual risk under D1/D2.
- **Statistics:** two-level paired moving-block bootstrap (2000 resamples, 1-h blocks, ≥ 30 blocks); Holm within each family (RQ2, RQ3, E-ADV); per-seed values always shown.

### 11.3 Ablations
Core: A1 no multigraph (= B4), A2 no direction, A3 no memory, **A7 no IBC**. Optional: A4 single-stage, A5 BCE, A6 S ∈ {5,10,20}.

### 11.4 Figures (all script-generated from logs)
1. **Staleness curve** — PR-AUC vs granularity, per model, with CIs (RQ1).
2. **Class-conditional score drop** + **exposure ρ by class and typology** (mechanism).
3. **Detection-delay CDF** per granularity, data-time vs end-to-end.
4. **Service time vs unit size**; **latency CDF**; **throughput vs offered load** (simulated + live points).
5. **Recovered fraction with IBC** per granularity (RQ2).
6. **Attack success rate vs Δ and W**, C1 against the C2 control (E-ADV).
7. **Attack success by typology** (H6) + **residual risk under D1/D2** (H7).
8. PR curves; F1/PR-AUC bars with CIs (RQ3 + baselines).
9. Gain vs pair multiplicity and recipient degree (RQ3 mechanism).
10. **Comparison with published results**, labelled "published, not re-run" (brief component 5).
11. Training curves; per-typology recall heatmap; architecture diagram.

---

## 12. Risks

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | No gain (H3/H4 negative) | M | M | RQ1 stands alone; report negatives with stratified analysis |
| R2 | Compute overrun | H | H | G1 measurements; reduction ladder; inference-only sweeps |
| R3 | Multi-GNN not reproducible | M | M | Kaggle/Linux, 5-day box; else published numbers marked "not reproduced" |
| R4 | Lab access worse than planned | H | H | 10-min checkpoints; long runs on Kaggle; if unusable, profile on Kaggle GPU and state the device |
| R5 | Kaggle quota lower | M | M | Two accounts; reduction ladder |
| R6 | Novelty overlap (MemFreezing, AnchorRM, new 2026 work) | M | H | **Day-1 full-text reading + repeat search**; drop E-ADV if taken; RQ1 unaffected |
| R7 | Negative G0 | M | M | RQ2 → exploratory; RQ1 still reported |
| R8 | Leakage bug | L–M | H | **Tier 1–3 gate**; shuffled-label run must collapse |
| R9 | Attack is an artefact of perturbation, not packing | M | H | **The C2 control exists precisely for this.** Report C1 vs C2 only |
| R10 | Timestamp ties | L | M | Equal-timestamp events form one unit, never see each other (tested) |
| R11 | Queue saturation | L | L | Measured outcome (sustainable *λ*) |
| R12 | Agent-written code subtly wrong | M | H | Tier 1 independent references; pair review of causal code |
| R13 | Can't defend own code in viva | M | H | Budget week-3 time to read the codebase; `DECISIONS.md` as the narrative |
| R14 | Report crunch | M | H | Draft from 27 Oct; all tables script-generated |

---

## 13. Definition of done (15 Nov)

**Must have (P0) — the 25% depends on these:**
- Tested implementation: DMT-GNN (± IBC), TGN (± IBC), merged, XGBoost (continuous + periodic). Multi-GNN reproduced or documented as published-only.
- **All verification tiers passing, with the test report included as an appendix.**
- HI-Small results: 3 seeds with CIs; ablations A1, A3, A7 (+ A2 unless cut); staleness curves by class; detection delay; validated latency and throughput.
- Figures 1–5 and 8–11 generated from logs; comparison table vs published FraudGT and Multi-GNN with caveats.
- Phase 3 report covering all 7 brief components, including "Changes from Phase 2" (§3) and the response to class feedback.
- `DECISIONS.md` showing pre-registration and every deviation.

**Should have (P1):**
- **E-ADV results + figures 6–7** (if G2 passed).
- LiveAML demo; slides.

**Winter extension (publication path):**
- LI-Small and HI-Medium; Ethereum window.
- Seeds 3 → 5.
- FraudGT under the streaming protocol (the single strongest addition).
- A third model family.
- ρ as a quantitative predictor of the measured penalty across datasets.
- Venue selection with checked calls for papers (Financial Cryptography, ACM AFT, IEEE ICBC, ICAIF).
