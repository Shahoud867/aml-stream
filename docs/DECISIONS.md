# DECISIONS.md: Research decisions and pre-registration

An append-only log. Each entry says what was known at the time. Measured numbers live in `C:\dev\aml-stream\results\` and are regenerated from logs.

---

## D-001 (2026-11-01): HI-Small is the discovery set; other datasets are confirmatory

**What had been seen when this was written:** on HI-Small only, the tail audit, the day-level volume audit of all four AMLworld files (descriptive counts, no modelling), and the gradient-boosted horizon sweep up to H = 24 h. **No model has been run on LI-Small, HI-Medium, LI-Medium or SAML-D.** An LI-Small sweep had been launched in the background, and its output had not been read.

**Why this matters:** the hypotheses below came from looking at HI-Small, so HI-Small cannot confirm them. They count as confirmed only on data not yet modelled. Changes after this point need a new dated entry and must be labelled exploratory.

### Frozen protocol
- **Views:** H in {0 (strict past), 1 s (same minute visible), 60 s, 10 min, 1 h, 6 h, 24 h, 72 h, infinity (whole split graph)}. Retrain at every H. Train and validation views are clipped at their split ends.
- **Splits:** Multi-GNN's whole-day rule. **Released** = all days in the file. **Documented** = events after the volume drop are removed before evaluation.
- **Detector:** the LightGBM configuration and feature set in `scripts/03_sweep.py` at this date, negative rate 0.25, seeds {0, 1, 2}. No tuning per dataset.
- **Thresholds:** F1 threshold and alert-budget thresholds set on validation only.
- **Attempt metric:** value share = share of an attempt's USD in transfers strictly after its first alert (alert time = t + H). Undetected = 0. Budgets B in {100, 500, 2000} alerts/day.

### Hypotheses (one-sided unless stated; 95% CI, attempt-level paired bootstrap for attempt metrics)
| ID | Claim | Confirmed on a dataset if |
|---|---|---|
| **C1** | The tail inflates headline F1 | Released test F1 minus core F1 > 5 points at H = infinity, for that dataset's detector |
| **C2** | Look-ahead is worth a large share of core performance | Core PR-AUC(H = infinity) minus core PR-AUC(H = 0) > 0.05, for every seed |
| **C3** | F1 and stoppable value disagree on the best horizon | At B = 500: mean value share at the best H <= 1 h exceeds that at H = 24 h, CI > 0, **while** core PR-AUC at 24 h is at least that at the same H |
| **C4** | Real-time performance is an interval, not a point | Core PR-AUC(same minute) minus core PR-AUC(strict) > 0.02 for every seed |
| **C5** | Adaptive alerting beats the best fixed horizon | Value-share difference at B = 500, CI > 0 (selected on validation; evaluated once on test) |
| **C6** | Bursts evade strict-past detectors; same-minute visibility blunts this | Detection loss under burst minus matched-displacement control > 0, CI > 0 at H = 0, and at most half that size at the same-minute view |
| **C7** | Equal views change the GNN vs boosted-tree comparison (two-sided) | Reported either way: GNN minus GBT core PR-AUC at H = 0 and at H = infinity, with CIs |

A failed hypothesis is a result and will be reported as such.

### Known limits of this pre-registration
- C1 and C2 thresholds were chosen after seeing HI-Small effect sizes. They are deliberately well below those sizes.
- C6 and C7 have no HI-Small estimate yet when frozen, so they are closer to genuine predictions.
- LI-Small has few attempts, so its attempt-level CIs will be wide; HI-Medium is the main confirmatory set for C3, C5 and C6.

---

## D-002 (2026-11-01): Retire "staleness penalty" as the primary question

**Finding.** The old H1 ("coarser update units reduce quality") assumes a unit-based detector may not use its own unit. That rule comes from TGN's training semantics, not from causality: an alert raised at unit close may legally use every event in that unit. A coarse detector therefore has *more* information than a real-time one. The sign of H1 is an artefact of forbidding this.

**Decision.** The primary axis is decision horizon H. Update units become one way to implement a horizon.

## D-003 (2026-11-01): Retire intra-block context (IBC) as a contribution

**Finding.** AnchorRM (arXiv:2610.27558) reconstructs event states from "causally preceding same-batch interactions", read in full today. The learned version of the idea is taken. Separately, a unit-close detector can use the *whole* unit, so "earlier in unit only" is needlessly weak.

**Decision.** IBC is dropped as a claimed contribution. RQ2 is replaced. Tie-aware views cover its useful part.

## D-004 (2026-11-01): DMT-GNN becomes a gated secondary question

**Finding.** Two-stage multi-edge aggregation already exists offline (MEGA, DIAM, Multi-GNN). A memory-based variant is an incremental modelling claim, and no evidence yet shows it matters more than the protocol effects found today.

**Decision.** Implement a direction-aware temporal GNN without memory now, because the horizon study needs a learned model. Add memory and per-counterparty stages after 15 Nov only if the GNN results show a modelling gap worth closing.

## D-005 (2026-11-01): Redesign the timing attack

**Finding.** Randomised refresh boundaries (old defence D2) cannot defend against tight bursts: a burst of span s is split by a uniformly random boundary with probability only min(1, s/Delta). And the old SPEC rule that equal timestamps never see each other makes same-minute packing the *strongest* attack against the old IBC.

**Novelty check.** Full-text reads today: MemFreezing (ICML 2025) injects fake nodes or edges; Heijmans et al. (arXiv:2610.31718, 22 Oct 2026) rewire receivers and cite timing only as motivation. Neither alters timestamps. This space could close quickly: Heijmans et al. list timestamp perturbation among possible extensions.

**Decision.** Attack families are burst and stretch, each with a matched-displacement control. The primary defence is view design, not boundary jitter.

## D-006 (2026-11-01): Tooling
- LightGBM 4.7.0 crashes on trivial data in this environment (access violation in `LGBM_DatasetSetField`), with NumPy 2.5.4 and 2.3.5 alike. Pinned to 4.6.0.
- The Kaggle endpoint truncated a download silently at 815k rows while returning exit code 0. All raw files now have verified line counts and SHA-256 hashes.

## D-007 (2026-11-01): Running third-party Multi-GNN locally needs a decision from the team
The harness in `scripts/multignn_audit.py` was written but **has never been executed**. Local execution of the cloned repository was blocked by the session's permission policy, and that restriction was respected. The options are for the team to run it on Kaggle, or to allow local execution.

## D-008 (2026-11-01): Add timing-invariant detection and hypothesis C8

**What had been seen:** no logical-clock detector exists, and no attack has been run, so this is a genuine prediction. No data has informed it.

**Reasoning.** Every attack and defence we found treats wall-clock time as given. Timing is the launderer's one free variable, so evidence that survives re-timing is the principled defence. Lamport's logical clocks show that ordering does not need physical time.

**C8 (frozen).** On each dataset where the attack runs, for the stretch attack at W = 6 h and B = 500:
- Detection loss beyond the matched control, for the logical-clock detector, is at most half that of the wall-clock detector, with the difference's CI above 0.
- **And** its clean core PR-AUC is at least 0.7 times the wall-clock detector's at the same H.

The 0.7 threshold is a judgement call, set before any measurement. A failure on either part is reported as such.

## D-009 (2026-11-01): Prior work found today narrows our claims

Reading in full changed what we can say:
- **Nguyen et al. (AISTATS 2022)** show that future transactions improve posterior card-fraud detection, on real data. "Look-ahead helps" is not ours to claim. Ours is the graph setting, an explicit time horizon, equal views, and the cost side.
- **FRAUDability and Shen & Kurshan (ICAIF 2020)** use money-denominated measures and capacity-aware alerting. Stoppable value is not presented as a new idea.
- **Daniluk & Dabrowski (2023)** show a trivial baseline beating TGB. This supports the publishability of the audit, and it is on a different benchmark.

## D-010 (2026-11-01): Compute plan

**Correction from the user:** experiments do not run on Shahoud's laptop as a matter of course. Resources are Kaggle GPUs, molab, Colab, the university GPU, and a friend's GPU and MacBook (`ROADMAP.md` §3). Every job is a script that runs anywhere.

Experiments started locally were **stopped on 1 Nov at the user's request**. As a result:
- The documented-window sweep is complete only to 1 h.
- The adaptive policy has one seed.
- The leakage check is complete only at H = 0.
- **No model ran on LI-Small**: D-001 said a sweep had been launched, but it never reached modelling. All confirmatory datasets remain unseen by any model.
- The attack and TD-GNN produced no results.

## D-011 (2026-11-01): Equal-alert analysis added

Thresholds set on validation gave unequal alert rates on test (944 against 575 per day at a nominal 500), which favoured strict past. The equal-alert analysis was added after seeing this, so it is exploratory on HI-Small and is now required on every dataset. It strengthened the finding: strict past 0.43, 10 min 0.58.

## D-012 (2026-11-01): Add C3b, the `blend` attack, sealed scoring and the FraudGT audit

**Approved by the team on 1 Nov 2026, before any model ran on a confirmatory dataset.** Reasons are in `NOTEBOOK_METHODOLOGY.md` §2 (W1, W4, W7, W8). None of this code has been written or run.

**What had been seen:** HI-Small results only. The C3b estimate below comes from HI-Small, with the best horizon picked on test, so it is exploratory and cannot confirm C3b. No attack has been run, so `blend` has no results at all.

### C3b: a short wait beats strict past (frozen)
C3 is close to true by construction, because any detector that waits longer than an attempt lasts has stoppable value near 0. **C3 stays as a descriptive result. C3b is the primary hypothesis about timing.**

On each confirmatory dataset (LI-Small, HI-Medium, LI-Medium, SAML-D), using LightGBM on the Released protocol:
- **Alerts:** equal at each horizon, using the top `B x D` scores, with `D` the test core's length in calendar days. Primary `B` = 500.
- **Candidates:** H in {1 s, 1 min, 10 min, 1 h}.
- **Selecting H\*:** the candidate with the highest mean stoppable value **on validation**, under the same equal-alert rule. Test data must not influence this.
- **Confirmed on a dataset if:** SV(H\*) minus SV(strict past) has a 95% paired attempt-bootstrap CI above 0.
- **Also reported:** the transfer-count analogue, and B in {100, 2000} with Holm correction. A result at these secondary budgets cannot rescue a failure at B = 500.
- **SAML-D:** attempts are built from connected illicit edges, so its result is labelled approximate.

HI-Small (exploratory): +0.148, CI [+0.095, +0.205].

### `blend`: an attack designed against logical-clock evidence (frozen)
Testing a defence only against attacks not designed for it would not be credible. `blend` stretches an attempt in **logical** time rather than wall-clock time, placing as many of the attempt's accounts' other events as possible between its transfers.

For attempt `A` with transfers sorted in canonical order, and budget `W`:
1. `c(t)` = number of non-attempt events up to `t` that involve any account in `A`.
2. Target for transfer `i` of `n`: `c_i = c(t_1 - W) + ((i - 1)/(n - 1)) * (c(t_n + W) - c(t_1 - W))`.
3. `t'_i` = the earliest minute with `c(t) >= c_i`, clipped to `[t_i - W, t_i + W]` and to the test range.
4. Apply a running maximum, so order is preserved.

**Control:** `shift_blend`, a rigid shift by the same mean displacement with random sign per attempt. The same invariants as the other attacks apply: only times change, the budget holds, times stay on the minute grid, and the map is non-decreasing.

**Known limits:**
- It uses the realised stream, including events a real launderer could not foresee, so it is an upper bound on this attack's strength rather than a realistic one.
- It uses one clock per attempt, not one per account.
- On an account with no other events it reduces to a rigid shift, as the logical-clock argument predicts.

### C8 amended
C8 (D-008) now requires its loss criterion to hold for **both `stretch` and `blend`**, each against its own control, at W = 6 h and B = 500. The clean-data criterion is unchanged. Passing only `stretch` is reported as a failure of C8.

### Sealed scoring
- **Applies to:** LI-Small, HI-Medium, LI-Medium and SAML-D, in every notebook. HI-Small is exempt, because it is already discovery data.
- **Notebooks may display:** validation metrics, and health checks on test scores (row count, score range, NaN count).
- **Notebooks must not display or log:** test metrics, or anything derived from test labels.
- **Unsealing:** once per dataset in the merge step, after that dataset's jobs finish, and recorded in the ledger.
- **After unsealing:** any rerun or change on that dataset is labelled exploratory.
- **If a bug is found after unsealing:** a fix is allowed, and both the original and corrected results are reported, with the reason.

### FraudGT audit
Region metrics (full test, core, tail) on HI-Small, 1 seed, with its published configuration. It is NB2 priority P1, time-boxed to 3 working days.
- **It has no confirmation threshold.** It is descriptive evidence for RQ2.
- **If it does not run or does not reproduce,** that is recorded, and published values stay labelled as not re-run.
- **Status:** only FraudGT's dataset class has been read. Its code has not been run.

## D-013 (2026-11-01): Shared notebook code built; implementation details frozen before any confirmatory run

No model has run on a confirmatory dataset. HI-Medium was prepared once to measure memory (102 s, 12.1 GB peak), which showed only row, account and attempt counts: 31,898,238 rows and 2,756 attempts. The output was deleted.

**Frozen detector:** `gbt_sweep` in `src/amlh/jobs_gbt.py` replaces `scripts/03_sweep.py` (now in `scripts/legacy/`) as the frozen configuration D-001 refers to. On HI-Small at 1 h, seed 0, the two give identical scores, so results already reported stand.

**Logical-clock family:** frozen as defined in `SPEC.md` §5. Chosen before any logical-clock result exists: k in {8, 32, 128}, forward k = 32, no hour of day, and pair and motif blocks over all visible history. C8 compares this family with the wall-clock detector.

**A9 dropped.** A policy with no re-scoring has one score per transfer, so it reduces to a fixed horizon. C5's contrast already measures the value of re-scoring.

**Attack scope:** only attacked rows are rescored on a re-timed stream. Detection of attacked attempts is measured, and false alerts on other transfers under attack are not.

**Replaced:** `scripts/multignn_audit.py` (D-007) is replaced by `src/amlh/mgnn.py`, which has also never been executed. Its pilot gate is to reproduce the published HI-Small F1 within its spread.

**Fixed along the way:**
- A data card with no documented span silently treated tail days as core, which would have inflated core results. `load_stream` now uses the known AMLworld spans, and a test covers it. No reported number was affected, because the reported runs used the correct split.
- Tied scores could leave nothing to threshold in the alert policy. A test covers it.
- The earlier queue simulator's last-event bug was already fixed and tested.

## D-014 (2026-11-01): Notebooks and FraudGT audit job written; pilot gates and cadence scoring fixed before any run

**Written, none of it run on Kaggle:** `notebooks/nb1.ipynb` to `nb5.ipynb`, `merge.ipynb`, `src/amlh/fraudgt.py` and `fraudgt_driver.py`.

**Cadence models are now scored at their own cadence.** Retraining at a cadence and then scoring on the whole test graph would have reversed the mismatch W5 was meant to remove. NB1 now has a replay for each of its 4 retrained models. This adds 4 P1 jobs, so NB1 has 24 jobs, and it fixes a design error before any result existed.

**Replay now covers validation as well as test,** so a cadence's threshold is tuned on the view it is used with. Stock-weight replay stays a diagnostic.

**The merge step now reports the official models.** It had no path for their scores, so Gate G1 could not have been read from it. `official_audit` gives official-style F1, F1 at a validation threshold, core and tail metrics, stoppable value and sampler coverage per job.

**FraudGT audit as frozen:**
- Configuration `AML-Small-HI-Multi-SparseNodeGT.yaml`, seed 42, one run, at most 20 training hours. A shorter run is reported as truncated, and its learning-rate schedule will not have finished.
- It has reverse message passing, ports and ego IDs, so it is the closest match to the paper's Multi-FraudGT (76.13%). **The repository does not state this mapping.** It is an inference, and the comparison is labelled approximate.
- It stays a descriptive result, with no threshold, as in D-012.

**Candidate finding, from reading the code and the split sizes. Nothing has been run.** FraudGT's test loader stops after `val.iter_per_epoch` shuffled batches. The published repository's HI-Small FraudGT configurations set 256 batches of 2,048, or at most 524,288 of the 863,900 test edges (60.7%), and a new random subset each time. The same repository's Multi-PNA+EU configuration sets 128, or at most 30.3%. Validation is at most 54.3%.
- **What this does not show:** whether the published table used these settings or a different evaluation run. The audit reports both the subset evaluation and a full pass, so the size of any gap will be measured. Until then this must not be stated as a flaw in the published result.

**Pilot gates, as built:**
- Pilots use HI-Small only. NB4 also prepares HI-Medium and SAML-D to check memory, and reports only their size.
- The forced kill ends a process with no cleanup, then resumes it in a new process. Boosted-tree jobs must match within 1e-5. GPU jobs need a correlation of at least 0.95, a floor chosen before any measurement, because their neighbour sampling is not deterministic. The pilot records the real value.
- Pilot output has its own folder. Resume and merge both ignore it.

**Not in the code package:** official repositories. Each is cloned at its pinned commit in the session, and the commit is checked.

## D-015 (2026-11-01): Notebooks also run on Google Colab

**Why:** Kaggle needs a phone-verified account for GPU and internet access, and that was not possible for every account that will run a notebook. Colab does not need it.

**What changed:** every notebook detects its platform. On Colab it mounts Drive, downloads the raw files it needs (resuming, and checking each file is whole), and writes its output to Drive, so a disconnect resumes by running all cells again. Kaggle behaviour is unchanged. No scientific setting changed, and the job list and thresholds are the same, so results from either platform are comparable. The recorded environment names the hardware and not the platform's account.

**Colab limits, set before any run:**
- **P0 only for NB4 and NB5 by default.** Their P1 jobs load Medium files. Preparing one peaked at 12.1 GB of RAM here, and a free Colab runtime has about 12 GB. Running out of memory kills the whole session, so the default avoids it. A larger runtime can add `'P1'`.
- **The pilot skips the Medium memory check below 24 GB of RAM,** reports the skip as a check to do by hand, and does not count it as a pass.
- **Time budget 5 hours by default,** because Colab sessions are shorter and less predictable than Kaggle's 12 hours. The figure is a guess and should be changed once real sessions are seen.
- **Free GPU time is not published,** so no hours can be promised on Colab. Compute estimates remain unmeasured until the pilots run.

**Not tested:** Drive mounting, the download from Colab's network (it may be blocked or rate-limited, and a Drive `aml/data` copy is the fallback), PyG wheels for Colab's torch, and Drive's behaviour under many small checkpoint writes. All of it first runs in a Colab pilot.

## D-016 (2026-11-01): First Colab pilot found two faults; both fixed before any result existed

**What happened:** NB4's pilot stopped before running a single job, with `unable to find column "Date"` while preparing SAML-D. The cause was that the download endpoint delivered `SAML-D.csv` as a zip archive, while the four AMLworld files arrived as plain CSV. The downloader did not check for this.

**Fixes, in `src/amlh/fetch.py`, `session.py` and `jobs_gbt.py`:**
- A file that arrives as a zip is unpacked when whole, and resumed when cut. A zip is never accepted as the CSV.
- A dataset touched only by a pilot's memory check is now prepared inside that check. A bad file therefore fails one job, and the other pilot jobs still run. Before, it crashed the whole session before any job started.

**What this shows about the evidence:** the AMLworld files passed their row counts, so the plain-CSV path is the one confirmed on Colab. SAML-D's preparation has still never run on the real file, so its first real run is NB4's pilot.

**Process note:** any code change gives a new code version, and every notebook refuses to run a different one. So each fix means re-uploading the zip and the notebooks. Pilot results under an older version are valid as pilots, but production must use one version everywhere.

## D-017 (2026-11-01): First pilots of NB2 and NB5 found four faults; one changes how the official model is evaluated

**Faults found by the first Colab pilots, all in my code and all fixed:**
1. **Multi-GNN neighbour counts arrived as strings.** The official parser gives `--num_neighs` no type, so values passed on the command line are strings, and the sampler rejected them. The official default is a list of ints. Converted after parsing.
2. **FraudGT could not find its raw file.** Its loader looks in `<dataset dir>/AML/` and reads the formatted file back from there. Both are now placed there.
3. **The GFP baseline needs `snapml`, which did not install** on Colab's Python 3.13. The cause is probably that no build exists for that Python (unconfirmed). GFP is now an optional pilot gate: its failure is reported and does not fail the pilot. It cannot run on Colab at all unless that changes, so its comparison row may end as a recorded failure.
4. **A bad optional file crashed a session** (D-016).

**A finding about the official code, verified by search of its source, not by running it.** Multi-GNN's Python files contain no call to `model.eval()` or `model.train()`. Its models use batch-norm and a final dropout of 0.5, so its published evaluation runs in PyTorch's default training mode, with batch statistics and active dropout. My runner had switched to evaluation mode, which would have scored a differently-behaving model than the one behind the published F1.

**Decision, taken before any official-model result exists:**
- **Primary (`eval_mode: official`, the default):** model selection and scoring follow the official code, with no mode switch. This is the run that the reproduction gate and Gate G1 compare with the published F1.
- **Secondary:** a second scoring pass in evaluation mode, written as `scores_*_eval`, which depends only on the weights. The merge report shows it as a separate row marked `[eval mode]`, with no published reference attached.
- **Why both:** in training mode a score depends on chance (dropout) and on which edges share its batch, including later edges. That is part of the published pipeline, so it belongs in the audit, but it makes horizon analysis noisy.
- **Not measured:** how much the two modes differ. Nothing may be claimed about it until the pilots and production runs show it.
- **FraudGT is unaffected:** its own evaluation loop calls `model.eval()`.

**Pilot rule adjusted to match:** the forced-kill resume check compares the evaluation-mode scores when a job writes both, because dropout makes two runs of the official-mode scores differ by chance.

## D-018 (2026-11-01): NB1's first pilot found two library-version faults; fixed before any result existed

**Faults, both caused by Colab running Python 3.13 with a recent PyTorch:**
1. **PyG 2.6.1 cannot trace `to_hetero` on current torch.** Its tracer reads `st.List` from `torch.fx._symbolic_trace`, and that module no longer exports `List`. The tagged source of PyG 2.7.0 imports `List` from `typing` instead, so the pin is now 2.7.0. That the rest of Multi-GNN and FraudGT work under 2.7.0 is **not verified**, and a different version from the one the official results used is a recorded difference.
2. **The official formatter needs `datatable`, which has no build for Python 3.13.** A small stand-in (`src/amlh/datatable_shim.py`) now replaces only the file reading, cell indexing, one-column stable sort and writing. The official script's own logic still runs. The check records whether the real package or the stand-in was used, and the pilot report shows it. The stand-in's behaviour matches datatable's only as far as I know datatable's documented behaviour, so a match with the port shows agreement with the official logic, not with datatable itself.

**Pilot reports now contain each failed job's error text and traceback,** so a failure no longer needs an extra cell to read.

**Version differences from the original experiments, to be stated in any paper:** PyG 2.7.0 and the platform's current torch, against whatever the Multi-GNN authors used. Their repository does not pin versions.

## D-019 (2026-11-01): Official-model jobs run in their own processes; FraudGT was killed for a reason not yet confirmed

**What was seen:** in NB2's pilot, FraudGT exited with -9 during its preprocessing ("Processing..."), on a machine with 12.7 GB of RAM. Exit -9 is the system's kill signal. **The cause is not confirmed.** Running out of memory is the likeliest, because FraudGT's preprocessing builds millions of Python objects (per-node neighbour lists and a dictionary of node pairs for each of three splits), but nothing here measured it.

**A contributing factor that is my design, not the official code's:** the notebook's own process ran the Multi-GNN job first and kept its memory. Python does not return freed objects' memory to the system, so a failed job still left the notebook large when FraudGT started.

**Change, made before any official-model result exists:** every official Multi-GNN job (`mgnn_train`, `mgnn_replay`, `mgnn_train_norm`, `mgnn_format_check`) now runs in its own process (`src/amlh/isolate.py`), and FraudGT already did. Memory, GPU memory and the official code's change of working directory end with each job. Checkpoints, logs, stop checks and the pilot's forced kill behave as before, and the notebook's process stays small. Every job's metrics now record its child's peak memory where the platform reports it. A kill signal is reported with the machine's RAM and a plain statement that lack of memory is the likely cause.

**What this does not settle:** if FraudGT is killed again with a small parent, it needs more memory than a 12.7 GB machine has on HI-Small, and the FraudGT audit then needs a larger runtime or ends as a recorded failure. FraudGT's own memory use has not been measured.

