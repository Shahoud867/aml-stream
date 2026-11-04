# Five-Notebook Methodology: Decision Horizons in Graph-Based AML

**Status, 1 November 2026: the five notebooks, the merge notebook and the FraudGT audit job are written (§8). None has run on Kaggle, so the pilot is the next step.** C3b, the `blend` attack, sealed scoring and the FraudGT audit are approved and logged in `DECISIONS.md` D-012.

Background: `PHASE3_PLAN.md` (research design), `SPEC.md` (definitions), `DECISIONS.md` (frozen hypotheses).

---

## 1. Summary

Five independent Kaggle notebooks run with no live dependency on each other. A small sixth step then merges the results.

| Notebook | Job | Answers | Hardware |
|---|---|---|---|
| **NB1** | Official Multi-GIN+EU on HI-Small | RQ2, C1, C3 on a real GNN | GPU |
| **NB2** | Official models, breadth: Multi-PNA+EU, LI-Small, FraudGT | RQ2 | GPU |
| **NB3** | TD-GNN across horizons, ablations and attacks | RQ1, C6, C7 | GPU |
| **NB4** | Boosted-tree confirmation on unseen datasets, leakage checks, alert policy | RQ1, RQ3, C1–C5 | CPU |
| **NB5** | Timing attacks, logical-clock detector, feature ablations, GFP | RQ4, C6, C8 | CPU |

Three things differ from what you asked for, and each is explained below:
1. **Only three notebooks need a GPU.** Forcing GPUs on the other two would waste quota and speed nothing up (§4).
2. **The two CPU notebooks leave GPU quota unused,** which can absorb overflow from NB1–NB3 (§6).
3. **"All five finished" does not mean finished research.** Gated work stays gated, and a merge step is needed (§6, §7).

---

## 2. What is wrong with the current plan

You asked me not to simply approve it. These are the weaknesses, most serious first. Several are in my own earlier work.

### Scientific

| # | Weakness | Why it matters | Fix |
|---|---|---|---|
| **W1** | **C3 is close to true by construction.** Any detector that waits longer than an attempt lasts has stoppable value near 0, whatever its quality. A reviewer will say "you measured arithmetic" | This is the finding I emphasised most, and as framed it is weak | **Add C3b, which can fail:** at equal alerts, a short wait beats strict past on stoppable value. Make the interior optimum the claim, not the decline. HI-Small: +0.148, CI [+0.095, +0.205] |
| **W2** | **The result depends on AMLworld's schedule.** Attempts last days in this simulator, which is why waiting is cheap here. Fast crypto laundering could reverse it | The practical advice may not transfer | Report the curve against attempt speed. Rescale time as a sensitivity analysis. SAML-D gives an independent schedule |
| **W3** | **No real transaction-level labels exist in this plan.** Ethereum labels are per account, so it can support horizon curves but not stoppable value | The largest limit on impact, and no notebook can fix it | State it plainly. Do not claim real-world effect sizes |
| **W4** | **Fixed attacks prove little about a defence.** Burst and stretch are heuristics. Testing a defence only against attacks not designed for it is a well-known weakness in adversarial ML, and logical-clock evidence has an obvious counter | C8 could pass falsely, and reviewers would reject it | **Add an adaptive `blend` attack:** move transfers into the account's busiest periods, so background events push siblings out of the last-k window, using the same budget and order rules |
| **W5** | **Replay at fixed weights is not a fair horizon test.** A Multi-GNN trained on a whole-window view and then shown less will degrade partly from mismatch | It could overstate the look-ahead effect | Keep replay as a diagnostic. **Retrain at cadences** for the headline claim (NB1) |
| **W6** | **HI-Small is contaminated, by me.** I chose the negative-sampling rate and added features after seeing its test scores | Its numbers are optimistic | Keep it discovery-only, as already frozen, and say so in the paper |
| **W7** | **Seeing results while runs continue could lead to quiet tuning.** Reruns after a look at test scores would invalidate confirmation | Parallel work makes this more likely | **Sealed scoring (§5.5):** confirmatory notebooks write test scores and do not display test metrics. Only the merge step computes them |
| **W8** | **The official-model claim is one model family.** FraudGT holds the best published HI-Small score (76.41%), and the plan audits only Multi-GNN | "Published results are inflated" would rest on the weaker model | Add a time-boxed FraudGT region audit (NB2) |
| **W9** | **Thresholds differ between comparisons.** Official Multi-GNN uses argmax; ours are tuned on validation | An unfair comparison either way | Report both for every model |
| **W10** | **Stoppable value ignores 38% of positives.** On HI-Small they belong to no listed attempt | The metric covers only part of the laundering | Report event metrics alongside it, and analyse those transfers separately |
| **W11** | **SAML-D has no attempt IDs, as far as I can tell.** Attempts would have to be built from connected illicit edges | Its stoppable value would be approximate | Label it approximate. Inspect the file in the pilot |

### Engineering

| # | Weakness | Fix |
|---|---|---|
| **W12** | The official-model runner has never run. The port of the official formatter has never been compared with the original | Pilot gate: reproduce the published F1 within its spread. A job compares the port with the official formatter on Kaggle |
| **W13** | **The official training loop cannot resume.** It saves only the best model, and evaluates the test set every epoch, which is wasteful | A thin runner calls the official model, loader and evaluation code, adds epoch checkpoints, and scores test only when validation improves. Selection is unchanged, and a pilot run verifies equivalence |
| **W14** | `04_gnn.py` does not use the GPU at all | Move the model to the GPU. The sampler stays on CPU |
| **W15** | `prepare.py` will not handle 32M rows: its Python-object steps are too heavy | Rewrite those steps with polars before any Medium run |
| **W16** | The equal-alert step hard-codes HI-Small's two-day core | Scale it to each dataset's core length |
| **W17** | Logical-clock features, SAML-D and Ethereum loaders, the `blend` attack, GFP adapter and rolling folds do not exist | All must be built and tested before the freeze (§8) |

### Checked and found small
Official evaluation silently drops seed edges that the sampler did not return on `Small_HI`, because the add-back code runs only for `Small_J` and `Small_Q`. I suspected this could matter. On HI-Small, only **0.15%** of test edges go to an account with in-degree above 100, which roughly bounds the effect at 100 neighbours. It is probably minor, and NB1 will measure coverage instead of assuming so.

---

## 3. Execution strategy

```
Phase 0  FREEZE     Build the missing code, run tests, tag one code version
Phase 1  PILOT      Each notebook runs in pilot mode (under 1 hour) and must pass its gates
Phase 2  PRODUCTION All five run at once, chained across sessions
Phase 3  MERGE      Anyone runs the merge notebook on the five published outputs
```

**Why a pilot first:** five notebooks burning quota on a bug is the most expensive failure available. Pilots use synthetic data and HI-Small only, so the confirmatory data stays unseen.

**The one thing to run first:** the leakage check at all horizons (NB4, stage 0, about an hour on CPU). A failure would invalidate positive-horizon results in NB3, NB4 and NB5, so it should pass before quota is spent. This is the only ordering I recommend, and it can run alongside the pilots.

**Priority tiers:** every notebook has P0, P1 and P2 jobs. **P0 is what 15 November needs.** P1 and P2 can finish later without blocking anything.

---

## 4. Why this division

- **The split follows scientific independence, not GPU load.** Each notebook could fail completely and the others' results would stay valid.
- **The official-model work comes first and gets two notebooks,** because it is the most important evidence and the largest GPU load.
- **Attacks live with the detector they attack.** NB3 attacks its own GNN and NB5 trains its own trees, so neither waits for the other.
- **No CPU job gets a GPU.** The feature engine runs on numba and the boosted trees on CPU. Switching LightGBM to a GPU build would also change a frozen detector.

---

## 5. The notebooks

Hour figures are **estimates until the pilot measures them.** The only published anchor is about 3.6 h on an RTX 3090 for GIN with ports, ego IDs and reverse message passing, and a T4 is slower.

### NB1: Official Multi-GIN+EU on HI-Small (GPU)

**Objective:** does the published result depend on the tail and on look-ahead, in the published model itself?

| Tier | Job |
|---|---|
| **P0** | Train 3 seeds using official settings (100 epochs by default) |
| **P0** | Score validation and test. Report F1 at argmax and at a validation threshold, PR-AUC, and core, tail and coverage metrics |
| **P0** | Replay at 24 h, 6 h and 1 h cadences; stoppable value per cadence |
| **P1** | **Retrain at 24 h and 1 h cadences, then score at the same cadence**, so the training view matches the deployed one (W5) |
| **P1** | Inference with training-set normalisation, to measure the effect of per-snapshot `z_norm` |

- **Inputs:** AMLworld HI-Small, code package, official Multi-GNN at the pinned commit.
- **Outputs:** per-edge scores keyed by raw row, audit metrics, per-epoch logs, best and latest checkpoints.
- **Checkpoints:** every epoch (model, optimiser, RNG, epoch, best validation F1). Replay writes chunks and resumes by bucket.
- **Estimate:** 30–65 GPU hours, so probably two weekly quotas.
- **Depends on:** nothing live. It writes the scores the merge step needs for C7.

### NB2: Official models, breadth (GPU)

**Objective:** does NB1's result hold across model and dataset?

| Tier | Job |
|---|---|
| **P0** | Multi-PNA+EU on HI-Small, 2 seeds, with NB1's metrics and replay |
| **P0** | Multi-GIN+EU on LI-Small, 2 seeds |
| **P1** | Multi-PNA+EU on LI-Small |
| **P1** | **FraudGT region audit on HI-Small**, 1 seed, at most 20 training hours. Written from reading its code, and not yet run (`DECISIONS.md` D-014) |
| **P2** | Multi-GIN+EU on HI-Medium: only if the pilot shows it fits |

- **Checkpoints:** same runner as NB1.
- **Estimate:** 30–60 GPU hours; HI-Medium extra, and possibly impractical.
- **Failure mode:** if FraudGT does not run, the notebook records that and carries on.

### NB3: TD-GNN across horizons (GPU)

**Objective:** a learned model whose view can be restricted exactly, for fair per-horizon curves and equal-view comparison.

| Tier | Job |
|---|---|
| **P0** | H in {strict, same minute, 1 h, 24 h, whole window}, 3 seeds, on HI-Small and LI-Small |
| **P0** | Burst, stretch and `blend` with matched controls at W = 6 h on its own checkpoints |
| **P1** | Full horizon grid; other attack budgets |
| **P1** | Ablations A6 (no direction separation), A7 (one hop), A8 (no count term) |
| **P1** | Service-time profile on the session's GPU, run last and alone, with the device named |

- **Inputs:** AMLworld; prepared data built locally by the notebook.
- **Checkpoints:** every epoch; inference in resumable chunks.
- **Size:** 90 jobs (P0 41, P1 49). **Estimate:** 20–40 GPU hours. **The least reliable figure here,** since no full run has finished. The GPU matters less than usual, because the sampler is CPU-bound and may be the bottleneck.
- **Not included:** node memory and TGN (§7).

### NB4: Boosted-tree confirmation (CPU)

**Objective:** test the frozen hypotheses on unseen data.

| Tier | Job |
|---|---|
| **P0, stage 0** | Shuffled-label leakage check at all horizons on HI-Small. **Report immediately to the team** |
| **P0** | Finish the HI-Small documented sweep |
| **P0** | LI-Small: full grid, 3 seeds, both protocols, **sealed** |
| **P0** | Adaptive policy on all seeds |
| **P1** | SAML-D and HI-Medium, **sealed** |
| **P1** | LI-Medium, **sealed** |
| **P2** | Rolling-origin folds on HI-Small; Ethereum horizon curves. Its file format has not been confirmed |

- **Checkpoints:** each job is atomic; a finished job is never repeated. Features are recomputed, not saved.
- **Size:** 297 jobs (P0 118, P1 144, P2 35). **Estimate:** 60–90 CPU hours, revised up now that the job list is exact. Medium is the bottleneck, at roughly 6 times the size of Small.
- **Memory:** HI-Medium preparation peaks at 12.1 GB (measured), and modelling will need more. Kaggle's RAM is unconfirmed for that. If it does not fit, it moves to Saif's MacBook or molab with the same code.

### NB5: Timing robustness and attribution (CPU)

**Objective:** RQ4, the main novelty, plus the feature evidence behind RQ1.

| Tier | Job |
|---|---|
| **P0** | Wall-clock, logical-clock and combined detectors on HI-Small, clean data |
| **P0** | Burst, stretch, `blend` and controls at W in {1, 6, 24} h, H in {strict, same minute, 1 h, 24 h}, 3 seeds |
| **P0** | Ablations A1–A5 and A10 |
| **P1** | LI-Small attack, **sealed**; GFP + XGBoost baseline |
| **P1** | HI-Medium attack at W = 6 h, **sealed** |
| **P1** | CPU service profile |
| **P2** | Per-attempt attacks instead of all at once |

- **Checkpoints:** per (detector, H, seed, W, condition). Each condition is saved as it finishes, and the trained model is kept, so a restart never retrains.
- **Size:** 176 jobs (P0 108, P1 68). **Estimate:** 40–70 CPU hours. Only attacked rows are rescored, so false alerts elsewhere under attack are not measured.
- **Independence:** trains its own detectors, using NB4's frozen configuration but not its outputs.

---

## 5.5 Shared engineering standard

**One codebase, five thin notebooks.** The research code lives in one versioned package, uploaded as a private Kaggle dataset (`aml-code-vN`). Each notebook only orchestrates. Five copies of the feature engine would drift apart, and that would undermine the comparisons.

**Notebook structure (same in all five):**
1. Configuration: notebook ID, mode, priority tiers, time budget.
2. Environment: pinned install, GPU check, hardware recorded.
3. Preflight: code hash, raw-data row counts and hashes, tests, resume state.
4. Data: deterministic preparation, built locally.
5. Job queue: run jobs in priority order until the budget ends.
6. Verification: check every artefact.
7. Export and summary.

**Job states:** `TODO → RUNNING → DONE | INTERRUPTED | FAILED`. A failed job is retried twice, then recorded, and the queue moves on.

**Output layout:**
```
/kaggle/working/aml_run/
  RUN_MANIFEST.json     notebook, code hash, environment, data hashes, spec version
  ledger.jsonl          append-only: job started, resumed, finished, failed
  summary.md, summary.json
  jobs/<job_id>/
    config.json  status.json  log.csv  metrics.json
    scores_val.parquet  scores_test.parquet     (event_id, score)
    ckpt/latest.pt  ckpt/previous.pt  ckpt/best.pt
    DONE                                        (sha256 of every artefact)
```
- **Job ID:** `{nb}-{model}-{dataset}-{protocol}-H{h}-{variant}-s{seed}`.
- **Universal key:** `event_id`, the raw CSV row. Multi-GNN's `EdgeID` is the same row number, so every score from every notebook can be joined.

**Reliability rules:**

| Risk | Rule |
|---|---|
| Session limit | Kaggle's documented run limit is 12 h, and forum reports say output upload after an overrun is best effort. **Stop cleanly at 10.5 h**, and do not start a job that will not fit |
| Hard kill | Mirror checkpoints hourly to a private Kaggle dataset on the same account, with the API token in **Kaggle Secrets, never in code**. A mirror failure does not stop the run |
| Resume | Newest valid checkpoint found across working directory, mirror and attached earlier output |
| Corruption | Write to a temporary file, rename, verify hash. Fall back to `previous.pt` if `latest.pt` fails |
| Wrong code | Resume is refused if the code hash or config hash differs |
| Storage (20 GB output limit, per Kaggle docs) | Keep 2 latest plus best. Features are never exported |
| GPU out of memory | Inference halves its batch and logs it. **Training fails loudly**, because a silent batch change alters the experiment |
| NaN loss | Stop the job, mark it failed, and keep the last good checkpoint |
| False completion | `DONE` is written last, after row counts and hashes are verified |

**Sealed scoring:** for confirmatory datasets, notebooks display only validation metrics and health checks (row counts, score range, NaN count). Test metrics are computed only in the merge step, once. Resume on GPU is equivalent within tolerance, not bit-identical, and the pilot measures this.

**Pilot gates, per notebook:**
- Environment and tests pass.
- Resume after a forced kill gives results within tolerance.
- Timing is measured, replacing the estimates above.
- Quota use is checked, particularly whether T4 x2 costs double.
- NB1 only: the official formatter matches the port, and one seed reproduces the published F1 within its spread.

---

## 6. Parallel execution and integration

**What is shared:** raw Kaggle datasets and the code package. Nothing else. No notebook reads another's output or session.

**Preparation without ordering:** each notebook prepares data locally. Preparation is deterministic, and the merge step compares hashes across notebooks and reports any mismatch.

**Overflow:** NB4 and NB5 use CPU, so those two accounts have unused GPU quota. Job lists can be split by an explicit `SHARD` setting, so someone can take NB1's second seed, for example. Assignment is written down, so nobody duplicates work and no live coordination is needed.

**Sharing results:** after each session chain ends, the `aml_run` folder is published as a **private Kaggle dataset shared with the team**, named `aml-out-nbX-vN`.

**Merge notebook (CPU, minutes; anyone can run it):**
1. Attach the five output datasets.
2. Run the verification checklist below.
3. Unseal test scores and compute verdicts for each hypothesis, once.
4. Run `08_analyze.py` to generate all tables and figures.
5. Run the queue simulation from the NB3 and NB5 service profiles.
6. Write `MERGE_REPORT.md`, listing what is complete, partial, failed or missing.

The merge can run early with partial outputs, and marks anything missing. **It must not unseal a confirmatory dataset until its jobs are finished.**

### Final verification checklist

- [ ] All five manifests show the **same code hash** and spec version.
- [ ] Raw-data hashes match `SPEC.md` in every manifest.
- [ ] Prepared-data hashes agree wherever notebooks share a dataset.
- [ ] Every job in the frozen list is `DONE`, or recorded as failed with a reason.
- [ ] Every `DONE` file's hashes match its artefacts.
- [ ] Score files have the expected row counts and no NaNs, and `event_id` joins with no missing rows.
- [ ] Split boundaries match across notebooks, including Multi-GNN's own.
- [ ] Shuffled-label checks collapse to prevalence at every horizon.
- [ ] Strict-past metrics are identical with and without tail days, for the same seed.
- [ ] Attacked streams change only timestamps, within budget and in order.
- [ ] The ledger shows each confirmatory test set scored under one configuration only.
- [ ] Official reproduction is within published spread, or the gap is reported.

---

## 7. What these notebooks will not complete

| Item | Why it stays out |
|---|---|
| **DMT-GNN and TGN** | Gated on results by design. Building them regardless would spend weeks on the plan's weakest contribution. If the gate passes, it means one more notebook |
| **University GPU latency** | Optional confirmation. Kaggle's GPU gives a named figure |
| **Class feedback, instructor approval, paper writing** | Not computational |

So all five finishing completes the **core experimental programme, not the whole project.**

---

## 8. Before Phase 1: code still to build

Built on 1 November 2026, in `C:\dev\aml-stream`. **All 301 automated tests pass.** "Tested" means the code ran here on real or synthetic data. "Not run" means it compiles but has never executed, and the pilot must prove it.

| # | Item | Status | Evidence or gap |
|---|---|---|---|
| 1 | Shared runner: queue, checkpoints, ledger, time budget, mirroring, verification | **Tested** | Resume, corrupted checkpoints, tampered files and blocked config changes are all tested. Real Kaggle dataset mirroring is **not run**: only its failure handling is tested |
| 2 | Official Multi-GNN runner with cadence retraining (`mgnn.py`; replaces `multignn_audit.py`) | **Not run** | Official code was not executed here. **The largest remaining risk** |
| 3 | GPU support in TD-GNN, plus A6–A8 | **Tested on CPU** | Training, resume, attacks and all three ablations ran on a small fixture. The GPU path itself is **not run**, and no real-data run has finished |
| 4 | `prepare.py` rewrite | **Tested** | Identical output to the old version on HI-Small. HI-Medium: 102 s, 12.1 GB peak, 31,898,238 rows, 2,756 attempts, 0 unmatched lines |
| 5 | SAML-D loader | **Tested on synthetic rows** | Built from the real header and sample rows, read from the file itself. Drops the leaky `Laundering_type`. The full file has not been loaded |
| 5b | Ethereum loader | **Not run** | Field names are from memory, and the loader fails loudly if they differ. Refuses to unpickle anything beyond graph containers |
| 6 | Logical-clock features with property tests | **Tested** | Matches a brute-force reference. Unchanged for a dedicated account under `stretch` and `blend`, and wall-clock features do change under the same stretch |
| 7 | `blend` attack; TD-GNN attack path | **Tested** | Order, budget, grid and control invariants. On an account with no other traffic it reduces to a rigid shift, as predicted |
| 8 | Sealed scoring and merge | **Tested**; merge notebook **not run** | Sealed jobs write no test metrics. Incomplete datasets stay sealed. Verdict paths for C1–C6 and C8 run on a fixture |
| 9 | GFP adapter | **Not run** | Written from its documentation only. `snapml` is Linux-only |
| 10 | Rolling-origin folds | **Tested** | **A9 is dropped as a separate job:** with no re-scoring the policy reduces to a fixed horizon, so C5's contrast already measures it |
| 11 | Equal-alert scaling; C3b analysis | **Tested** | C3b selects its horizon on validation only |

**Regression check:** the refactored detector reproduces the earlier HI-Small run at 1 h, seed 0, exactly: every score is identical, with F1 0.7308, core F1 0.6216 and core PR-AUC 0.5686.

**Added after the table above, also on 1 November:**

| # | Item | Status | Evidence or gap |
|---|---|---|---|
| 12 | Five notebooks and merge notebook (`scripts/build_notebooks.py`) | **Cells tested; not run on Kaggle** | NB4's real cells ran end to end here on a small fixture in production mode, with Kaggle paths substituted. Every notebook compiles and carries the current code hash. The install cells have **not run** |
| 13 | Pilot mode: forced kill, resume, gates, timing projection | **Tested on CPU** | A real process was killed with no cleanup and resumed, and its scores matched. Failing to reach the kill point fails the gate. GPU pilots are **not run** |
| 14 | Two-GPU workers and chain-aware sharding | **Tested on CPU** | Two processes split the jobs with no overlap, and dependent jobs stay together. Real GPUs are **not run** |
| 15 | FraudGT audit (`fraudgt.py`, `fraudgt_driver.py`) | **Mostly not run** | Its code was read and not executed. Tested: the parent's handling of success, crash, stop and running out of time against a stand-in child; adding back unsampled edges; mapping edges to rows. **Training and scoring inside the real FraudGT have never run** |
| 17 | Colab support: platform switch, downloader, Drive output (`DECISIONS.md` D-015) | **Tested here; not run on Colab** | The downloader resumed a silently truncated download from a local test server, and NB4's real cells ran end to end with Drive paths. Mounting Drive and downloading from Colab's network are **not run** |
| 16 | Official models in the merge step (`official_audit`) | **Tested on synthetic scores** | Region metrics, official-style F1 and fallback to the training job's validation scores |

Official Multi-GNN and FraudGT code has not been executed on the development machine, and will first run on Kaggle. So the pilots of NB1 and NB2 may fail on the first try, and that is what they are for.

---

## 9. Bottlenecks

| Bottleneck | Effect | Mitigation |
|---|---|---|
| **Building §8** | The real schedule risk, larger than compute | Freeze before launch; P2 items can slip |
| **Official Multi-GNN runtime** | 100 epochs with heavy Python preprocessing | Skip test evaluation on non-improving epochs; measure in pilot; use overflow quota |
| **Python port computation** | Multi-GNN's `add_ports` is pure Python and slow on large graphs | Cache per dataset per session; another reason HI-Medium is P2 |
| **Sampler speed in TD-GNN** | The GPU may sit idle | Measure in pilot. Larger batches help only if the sampler keeps up |
| **Kaggle memory on Medium** | May not fit | polars rewrite; fall back to MacBook or molab |
| **Queueing at busy times** | Delays | Priority tiers; nothing depends on simultaneous start |

### What I could not confirm about Kaggle
| Item | Status |
|---|---|
| 12 h run limit, 20 GB output | From Kaggle's docs (search excerpts; the page itself could not be read) |
| Overrun output is best effort | Forum reports, not policy |
| About 30 GPU hours per week | Widely reported, not firmly documented |
| T4 x2 or P100, 16 GB each | Multiple sources |
| Whether T4 x2 uses quota at double rate | **Unknown; test in pilot.** If not, two seeds could share a session |
| CPU cores and RAM in current sessions | **Unknown; recorded at preflight** |

---

## 10. Is this strong enough for a reputable conference?

**My honest judgement, not a calculation:**

| If the results show | Likely outcome |
|---|---|
| The official model's score drops materially without the tail and at fair cadences, on several datasets | **Credible Paper A** at KDD Datasets & Benchmarks or ICAIF |
| Plus C3b: a short wait beats strict past across simulators | **Stronger Paper A**, with a practical finding |
| Official models barely change | The tail finding survives as a short note, and the horizon claim weakens to tree models |
| Logical-clock evidence survives `blend` at acceptable cost | **Credible Paper B** at Financial Cryptography; the most novel result |
| It fails against `blend` | A negative result, likely a workshop paper |

**What these notebooks cannot supply:**
- **Real transaction-level evidence.** This is the main reason a top main track is unlikely.
- **A guarantee.** Reviews are noisy, and someone may publish first.
- **Independent judgement.** Nobody outside this project has reviewed the design.

**Timeline:** P0 by 15 November is tight, because §8 comes first. P1 and P2 may run later, which the instructor's clarification allows.

---

## 11. Decisions for the team

1. ~~Add C3b, the `blend` attack, sealed scoring and the FraudGT audit to `DECISIONS.md`?~~ **Approved and logged on 1 Nov 2026 (D-012).**
2. ~~Approve building §8?~~ **Approved and built on 1 Nov 2026.**
