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

## D-020 (2026-11-04): NB1's first real GPU pilot found an abstract-method crash and a severe per-epoch cost; the crash is fixed, the cost is given more time and a TF32 speed-up, neither frozen hyperparameter touched

**Machine:** one Windows RTX 3080 (10 GB), the first PC this project has actually run on. `memory.md` records that "the GPU code path has never run on a GPU" before this session; this entry is that first run.

**Fault 1, crash, fixed:** `train_util.AddEgoIds` (official code) defines `__call__` but not `forward`. PyG 2.7.0's `BaseTransform` (the version D-018 pinned) declares `forward` as `@abstractmethod`, which older PyG did not, so the class can no longer be instantiated. Fix in `src/amlh/mgnn.py` `_official()`: clears `AddEgoIds.__abstractmethods__` after import, so the official `__call__` still runs unchanged. This is a second, later instance of the same kind of fault D-018 already recorded (a newer PyG breaking older official code at the API surface, not the algorithm).

**Finding 2, severe per-epoch cost, not a bug:** one HI-Small Multi-GIN epoch (`batch_size=8192`, `num_neighs=[100,100]`, `reverse_mp`+`emlps`+`ego`, the official frozen hyperparameters) took **4,659 s (77.7 min)**, training and one validation pass together, measured end to end by the pilot's own job log. A follow-up timing of 8 isolated batches put **training alone at ~7.1 s/batch, ~397 batches/epoch, ~47 min/epoch**, which together with a smaller validation set (965,524 vs 3,248,921 train edges) accounts for nearly all of the 77.7 min. `py-spy` confirmed the process was inside the model's forward pass the whole time, not stalled, and `nvidia-smi` showed 100% GPU utilisation throughout.

**This contradicts `memory.md`'s only anchor** (~3.6 GPU-hours for 100 epochs on an RTX 3090, ~2.2 min/epoch) by roughly 35x. The official code sets no `num_workers` anywhere (checked `third_party/Multi-GNN` directly), so sampling is always single-process on any platform; this is not a Windows artefact. The discrepancy is **unexplained** — the anchor may describe a different configuration or may not have been a measurement taken on this codebase. Treat the 3.6 h/100-epoch figure as unverified until a 3090 or equivalent is measured directly.

**Consequence for the compute plan:** at ~78 min/epoch, one 100-epoch production job is roughly **130 GPU-hours**, not the 30-65 GPU-hours `memory.md` estimated for all of NB1's P0 jobs combined. The 10-way split and `--until` time budgets in `gpu_session/` were sized against the old estimate and are now very likely wrong for any RTX-3080-class machine. **Not yet decided:** whether to shorten epochs (`--n-epochs`, already supported and already requiring its own logged deviation), seek faster hardware, or accept the longer wall-clock time.

**Change made, kept inside the official algorithm's tolerance:** `torch.backends.cuda.matmul.allow_tf32 = True` and `cudnn.allow_tf32 = True`, set once per `mgnn_train` call when running on CUDA. This is a precision trade available on Ampere GPUs, not a change to the frozen hyperparameters, the model, or the official code's own logic; TF32 rounding error is far smaller than the run-to-run spread the pilot already tolerates (its resume-check accepts correlation ≥ 0.95 between two non-deterministic runs). **Not yet measured:** how much wall-clock time this actually saves, or whether it shifts `official_style_test_f1` outside its spread; the next pilot run measures both.

**Format-check gate, now fixed:** `mgnn_format_check` compared every column for exact equality, so the ~28% of rows differing at ~1e-16 relative in the two amount columns (float formatting noise, e.g. `10.299999999999999` vs `10.300000`) failed the gate even though they are not a porting error. `Amount Sent` and `Amount Received` now get a tolerance (`rtol=1e-9, atol=1e-6`) — about seven orders of magnitude looser than the noise actually seen, but far tighter than any real port bug (wrong column, wrong currency conversion, wrong sign) would produce. Every other column (ids, timestamp, currencies, payment format, the label) still requires exact equality, unchanged. Verified directly against the real HI-Small data after the change: `match: True`, all ten columns report zero differences.

**`gpu_session/run_gpu.py` budget bug, fixed:** pilot mode hardcoded `budget_hours=min(budget, 1.0)`, silently discarding any `--hours`/`--until` the caller passed. A pilot asked for 6 hours still stopped at ~1h. Fixed to honour an explicit `--hours`/`--until` in full, keeping the old 1h default only when neither flag is given. This file is outside `src/amlh` and `scripts/`, so the fix does not change the code hash the notebooks check.

**Second pilot, after both fixes:** every individual pilot job finished and passed its own verification, including the format-check (now `match: True`). The kill-and-resume correlation check did not pass: the forced-kill-and-resume run's evaluation-mode scores correlated 0.898 (test) and 0.879 (val) with the uninterrupted run's, below the >=0.95 floor D-014 set before any measurement. The resume mechanism itself is not in question — the resumed process correctly continued from epoch 1 rather than restarting, confirmed in its log. **Not yet confirmed, only a hypothesis:** at only 2 pilot epochs the model is far from converged, so GPU scatter/reduce non-determinism (inherent to message passing, independent of TF32) may move an immature model's predictions by more than it would move a converged one. Deferred — see below.

**Measured job costs, used for the production projection:** `pilot-nb1-mgnn-gin` (2 epochs, train+val): 10,797.8 s total (epoch0 3,902 s, epoch1 4,154 s, loss 0.0244->0.0163, val_f1 0.255->0.297). `pilot-nb1-trainnorm`: 230.1 s. `pilot-nb1-format-check`: 10.0 s. Extrapolating to the frozen 100-epoch, 3-seed, 3-replay-window P0 plan projects **~338 GPU-hours for P0 alone** (and ~451 for P1) on this one RTX 3080 — about 12x `memory.md`'s 30-65 GPU-hour estimate for all of NB1, confirming D-020's concern with a measured number instead of a single-epoch extrapolation.

## D-021 (2026-11-05): NB1 production scope cut to a single seed, 4 epochs, to fit a same-day deadline — the resume-correlation question deferred, not resolved

**Why:** a 3pm deadline on 2026-11-05 was set with ~5.5 GPU-hours available on the one RTX 3080. D-020's measured cost (~65-78 min/epoch) makes the full frozen P0 plan (~338 GPU-hours) and even one full 100-epoch seed (~110-130 GPU-hours) impossible in that window on this hardware. No amount of further debugging changes this; it is a scope decision, not a bug fix.

**Decision, made under time pressure, approved by the user:** run only `nb1-mgnn-gin-HI-Small-s1` (one seed of three) with `--n-epochs 4` (not the frozen 100) via `run_gpu.py --mode production --tiers P0 --only "nb1-format-check|nb1-mgnn-gin-HI-Small-s1$" --n-epochs 4 --until 14:30`. The other two seeds and all nine replay jobs are deferred, not run. This is a deviation from the frozen plan on two axes (epoch count, seed count) purely to produce one real, GPU-trained, verified result by the deadline — it is not a claim about the published hypotheses, which need the full protocol.

**What this run can and cannot show:** it can show the pipeline (format-check, training, checkpointing) producing a genuine trained model end to end on real hardware. It cannot stand in for C1-C8 or any confirmatory claim, and must not be reported as such. The kill-and-resume correlation question above is explicitly deferred past this run, not answered by it.

**Status:** started 2026-11-05 ~09:30, stopped by the user at ~11:30 after epoch 1 of 4 (loss 0.0248, val_f1 0.187) once an equivalent run completed on another machine (their "PC 9"). Not re-run here.

## D-022 (2026-11-05): NB2's PNA model overflows this RTX 3080's 10 GB at the frozen batch_size; batch_size lowered as a logged deviation

**What was seen:** `04_NB2_pna_HI_s1_PILOT` had previously only been diagnosed, never run, on this machine; another machine ("PC 18") attempted it and reported hitting a GPU memory limit. Reproduced directly: building the PNA model and running 5 training batches at the frozen `batch_size=8192`, `num_neighs=[100,100]` used **13.7 GB allocated / up to 25.75 GB reserved** by PyTorch's caching allocator — far past the card's 10 GB. On this machine it did not crash; Windows silently paged the overflow into system RAM instead (NVIDIA's WDDM shared-GPU-memory fallback), which did not error but made training catastrophically slow: 5 batches took 304 s (~61 s/batch) versus Multi-GIN's ~7.1 s/batch (D-020) at the same batch size. PC 18 apparently has a configuration that raises a hard CUDA OOM instead of paging, which is the "hit GPU memory limit, stopped" it reported — same underlying cause, different failure mode.

**Why PNA costs more than GIN at the same batch_size/num_neighs:** PNA (Principal Neighbourhood Aggregation) concatenates multiple aggregators (mean, min, max, std) with degree-based scalers per message-passing step, multiplying intermediate activation memory, where GIN's aggregation is a single sum. This is architectural, not a bug in this project's code.

**Fix, measured on the same cached data:**
| batch_size | peak allocated | peak reserved | s/batch | projected min/epoch (train only) |
|---|---|---|---|---|
| 8192 (frozen) | 13.70 GB | up to 25.75 GB | ~61 | ~440 |
| 4096 | 8.45 GB | 15.67 GB | 18.95 | 250.8 |
| 2048 | 5.03 GB | 11.02 GB (still over 10 GB) | 0.87 | 22.9 |
| 1024 | 2.86 GB | 6.17 GB | 0.30 | 16.0 |

**`pilot-nb2-fraudgt`, two further faults found and fixed after the environment gaps above:**
1. **Same abstract-`forward` fault as D-020's `AddEgoIds`, in FraudGT's own code:** `fraudGT.sampler.custom_sampler.AddEgoIdsForLinkNeighbor` also defines only `__call__`, not `forward`, so PyG 2.7.0 cannot instantiate it either. Fixed identically in `fraudgt_driver.py`: `AddEgoIdsForLinkNeighbor.__abstractmethods__ = frozenset()` right after import.
2. **DataLoader workers crashed silently on Windows:** `RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly`, with FraudGT's official default of `num_workers=2`. Multi-GNN's own code avoids this class of fault entirely by never using worker processes. Fixed in `fraudgt.py`'s `overrides()`: `num_workers` now defaults to 0 on Windows (`sys.platform == "win32"`) and keeps the official default of 2 elsewhere (Kaggle/Colab are Linux, unaffected). This is a platform-specific reliability default, not a change to any frozen hyperparameter or the published experiment.

**A process note on diagnosing these:** the runner caps retries (`runner.py`'s `run_job`: once `attempts >= retries` and `status == "FAILED"`, it returns the cached failure without re-executing). Several apparent re-failures of `pilot-nb2-fraudgt` during this session were this cache, not fresh evidence — confirmed by a brief diagnostic print showing the job function was not being called at all. Deleting the job's folder under `aml_pilot/jobs/` forces a genuine retry; this is worth knowing before concluding a fix didn't work.

**Retested at `batch_size=512` (smaller, for more headroom against the variable-batch-size spike seen below):** training completed in 3,721 s for 2 epochs (vs. 7,646 s at 1024), and the kill-and-resume correlation check **passed**: 0.951 (test), 0.962 (val), both above the 0.95 floor — the first time this gate has cleared on this machine. This suggests the earlier NB1 resume-correlation failure (D-020/D-021, 0.88-0.90) may be sensitive to batch size: a smaller batch means more gradient-noise averaging per epoch (more, noisier steps) or simply less GPU scatter-op non-determinism per step; this is still a hypothesis, not confirmed, and NB1's own resume-correlation question is still open.

**A live measurement of the actual spill, not just inference from timing:** Windows' own GPU memory counters (`Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`), read while a `batch_size=1024` job was mid-epoch, showed **~10.04 GB dedicated (essentially the full 10.24 GB card) plus ~1.87 GB spilled into shared system memory** — confirming the slowdown at 1024 was a genuine, partial VRAM-to-RAM spill, not measurement noise. `nvidia-smi` cannot see this; only the Windows per-process/per-adapter counters show the shared-memory component.

**`pilot-nb2-fraudgt` separately failed, environment-only, fixed:** first local exercise of FraudGT on this machine surfaced a chain of missing Python packages never caught before. A shallow `import fraudGT` check only exercises the package's `__init__` chain (`yacs`, `ogb`, `gdown`, `matplotlib`), not the full set of submodules `fraudgt_driver.py` actually imports (`fraudGT.logger` alone needed `torchmetrics`); confirmed by reproducing the exact subprocess invocation directly (`sys.executable -m amlh.fraudgt_driver <spec>`) outside the multi-hour pilot, which is far cheaper to iterate on than re-running the whole pilot per missing package. All five (`yacs`, `ogb`, `gdown`, `matplotlib`, `torchmetrics`) now installed into `.venv-gpu`; no code change. After this, the same direct reproduction gets past every import and fails only on a deliberately-bogus dummy config path, confirming the dependency chain is now complete.

**Clean full pass achieved:** after the `AddEgoIdsForLinkNeighbor` fix and the Windows `num_workers=0` fix, a from-scratch `pilot-nb2-fraudgt` run completed in 120.1 s, and every individual pilot job (format-check, training, replay, FraudGT) now passes. The resume-correlation gate's genuine pass (0.951/0.962, real forced kill at exit 137) was already demonstrated in an earlier run in this same session and is not re-demonstrated in every subsequent report only because the `-killed` job folder stays DONE across reruns in the same `--work` folder — a report showing `killed_exit_code: 0` after that point is this caching artifact, not a new failure, as noted above.

**Decision:** `batch_size=512` is used for PNA jobs on this machine (not 1024 — still confirmed to spill) — the smallest tested value, chosen for a comfortable margin under the 10 GB limit (2048 was still technically over physical VRAM in this test and cannot be trusted not to page intermittently on a longer run). This is a deviation from the frozen `batch_size=8192`, exactly like D-021's `--n-epochs` deviation, and is machine-specific: a GPU with enough memory should still use the frozen value. Implemented as a new `--batch-size` CLI override in `gpu_session/run_gpu.py` (production mode) and a matching override path in pilot mode, mirroring the existing `--n-epochs` override exactly. Not yet measured: whether a smaller batch_size changes `official_style_test_f1` outside the pilot's existing tolerance (smaller batches change gradient noise, which is a real training-dynamics effect, not just a performance one) — the next pilot run on this machine checks this against the resume-correlation and format-check gates same as any other run.

## D-023 (2026-11-05): Multi-PNA batch size reduced from 8192 to 1024 on 10 GB GPUs, HI-Small

*Consolidation note (D-024): this entry was written on branch PC-18 as **D-020**. PC-17 had independently written a different D-020 (2026-11-04) on its branch, so this one was renumbered to D-023 when the branches were merged. References to "D-020" inside the PC-18 memory file (`memory_pc18.md`) mean this entry. Its statement that "the code hash is unchanged" holds only for the PC-18 branch on its own (code hash 9686ce184d525c0a); see D-024 for the merged code lineage.*


**What was seen:** `nb2-mgnn-pna-HI-Small-s1` (shard 04) and `nb2-mgnn-pna-HI-Small-s2` (shard 05) both failed with a genuine `torch.OutOfMemoryError` on an RTX 3080 (10 GB), at the official batch size of 8192 and neighbour counts [100, 100]. This was confirmed directly (not inferred): PyTorch had ~8.8 GiB allocated and asked for ~1 GiB more with the card full, both times at the same point (the PNA message step, `torch.cat([x_i, x_j, edge_attr])`). Capping PyTorch's memory fraction (which fixed an unrelated Windows memory-spill problem on Multi-GIN the same day) made no difference here, because this is not spill or fragmentation — the model's batch genuinely does not fit.

**Change:** on 10 GB cards, Multi-PNA's `batch_size` is reduced from the official 8192 to **1024** via `gpu_session/run_gpu.py --batch-size 1024` (a new flag, mirroring the existing `--n-epochs` override; `src/amlh/mgnn.py` and the frozen notebooks are unchanged, so the code hash is unchanged). Neighbour counts [100, 100] are kept as official.

**Why this is a deviation that must be reported as such, not silently absorbed:** a smaller batch changes gradient noise and batch normalisation statistics, so Multi-PNA/HI-Small results from a 10 GB card are not directly comparable to a run at the official 8192 on a larger card, and must be labelled with the batch size used whenever reported. Multi-GIN did not need this change (it fits at 8192 once the memory-spill problem is fixed; see the same day's gpu_session fix) and is not affected.

**Not yet done:** nobody has checked whether 1024 is still large enough for PNA's degree-scaling statistics to be meaningful on this dataset, or compared a 1024-batch run against an 8192-batch run on a card that can hold both, to see how much the smaller batch costs. Treat shard 04/05 results run this way as provisional until that comparison exists.

## D-024 (2026-11-06): Consolidation of the four PCs' work: code lineage, TF32, PC-9 shard 06, and what was found missing

**What was merged** (branch `consolidated/all-pcs`, history of every source branch preserved, nothing force-pushed or reset): `origin/pc9` (which already contains `origin/main`), `origin/PC-17`, `origin/PC-18`, `origin/pc8`, and the maintainer's uncommitted planning-session work (kept on `backup/local-wip-2026-11-06`). `pc8` has no commits of its own: PC-8's work existed only as an untracked folder and is now `results/shard03_NB1_gin_s2`. PC-18's full output existed only in an untracked zip. Backups of every ref and every untracked file were taken first (bundle plus sha256 lists, outside the repository).

**Data-integrity findings (all verified by recomputing hashes, not read from summaries):**
- Each job's `DONE` marker holds the sha256 of every file it lists. Git on these PCs uses `core.autocrlf=true`, which rewrote CRLF files to LF on commit (PC-17: 11 files, PC-18: 84) and back on checkout, so `verify_job` failed on committed results although nothing was corrupted. `results/**` is now `-text` (`.gitattributes`). Original bytes were restored only where the original source existed, or where converting back to CRLF reproduced the recorded hash exactly.
- Several PCs committed only the small files. PC-9's branch holds 25 of the 123 files its seven `DONE` markers list (98 stayed on PC-9: every `scores_*.parquet`, every `log.jsonl`, replay state and 68 leftover bucket files). PC-18's branch lacked all 275 score and attack parquet files (512 MB); they came from its zip. PC-9's missing files were **not recoverable** from any source available here (`results/shard02_NB1_gin_s1/INCOMPLETE.md`, `results/shard06_NB2_gin_LI_s1/INCOMPLETE.md`).
- After consolidation 108 job folders verify completely (PC-8: 4, PC-18: 90 + 4 pilot, PC-17: 10), 7 are incomplete (PC-9), 0 are corrupted (`gpu_session/consolidation/verify_results.py`).
- `D-020` had been used twice (PC-17 and PC-18). PC-17's D-020, D-021, D-022 are unchanged; PC-18's entry is now **D-023**, with a note. `memory_pc18.md` keeps PC-18's text unchanged with a header note.

**PC-9 shard 06 is a deviation that was not logged.** `nb2-mgnn-gin-LI-Small-s1` trained about 84 of 100 epochs (best validation F1 0.2663 at epoch 68); the epoch counter inside `ckpt/latest.bin` was then edited so the job would finish, so `metrics.json` `epoch: 100` is not an epoch count. The stop was decided on LI-Small validation F1 only, which respects D-012's letter but not D-001's no-per-dataset-tuning spirit. Treat it as 'early-stopped, validation-informed, single seed', never as a full-protocol seed.

**Code lineage.** Every finished result (shards 02, 03, 06, 08, 09, 10) and PC-18's unfinished shard 05 carry code hash `9686ce184d525c0a` and ran on Windows, torch 2.5.1+cu121, PyG 2.7.0, with environment hooks that the hash does not cover (`gpu_session/compat/sitecustomize.py` on PC-9, PC-18's `fix_pyg.py` site-packages hook, PC-8's unrecorded). The merged tree has hash `39a6ee5169e533f6`. Its differences from `9686…`: (1) `AddEgoIds` abstract-flag fix inside `src/` (PC-17) plus an equivalent adapter (`ego_transform`, with test); (2) the format-check gate tolerates float-printing noise in the two amount columns, exact equality elsewhere (PC-17's D-020 change, **kept as written; it relaxes a gate and still needs the maintainer's ratification**); (3) FraudGT: `num_workers` defaults to 0 on Windows and its `AddEgoIdsForLinkNeighbor` gets the same fix; (4) TF32. Three PC-17 intermediate hashes (`110bf6ef…`, `d417f99d…`, `68ab7cec…`) match no commit and cannot be reproduced; a PC-17 job's recorded hash is the session-start hash, not necessarily the code that ran.

**TF32 is off by default.** PC-17 enabled TF32 unconditionally in `mgnn_train`. Merging that as written would have silently changed numerics for every future run while all finished results used full precision, and its accuracy effect was never measured. It is now opt-in (`AMLH_TF32=1`) and recorded as `tf32` in each job's metrics. PC-17's speed-up claim stands as stated in D-020 (unmeasured against a controlled baseline).

**What this entry does not settle (needs the maintainer):**
1. Multi-PNA batch size. PC-17 (D-022) uses 512 and says 1024 still spills; PC-18 (D-023) uses 1024 under a different memory-limit regime; the official value is 8192, which does not fit 10 GB. Shards 04 and 05 are both unfinished and must not be pooled unless they use the same batch size, and neither can be compared with the published number without 'batch X, not 8192' in the label.
2. Whether the remaining shards run under `9686…` (comparable with everything finished) or `39a6…`; a job with a checkpoint cannot resume under a different hash.
3. The PC-9 originals (the raw `aml_out` folders on that machine, believed to be `E:\aml_out\02_NB1_gin_s1\aml_run` and `E:\aml_out\06_NB2_gin_LI_s1\aml_run`, inferred from `memory_pc9.md`) must be copied off and imported before shards 02 and 06 can enter the merge step.
4. Not in any decision until now: PC-17's NB2 epoch caps (40 for PNA, 30 for Multi-GIN LI-Small) and its finding that Multi-GIN on LI-Small stayed at validation F1 0.0 at batch sizes 512 and 4096 (only in `memory_pc17.md`).

## D-025 (2026-11-07): RTX PRO 6000 Blackwell needs a newer torch/PyG combo than the RTX 3080 fleet; install_gnn_stack() now tries both

**What was found, live, on a real Molab Blackwell session** (pairing the `01_gnn_baselines` Molab notebook directly, via the `marimo-pair` skill): `install_gnn_stack()`'s single hard-coded fallback, `torch==2.5.1+cu121`, raised `RuntimeError: no matching torch-scatter and torch-sparse wheels, even for torch 2.5.1`. Confirmed directly: the card is an RTX PRO 6000 Blackwell Server Edition, compute capability `(12, 0)` (sm_120). torch 2.5.1+cu121 reported it plainly: "NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90" -- confirmed by one of the project's own tests (`test_tdgnn_trains_resumes_and_attacks`) genuinely failing on that card for exactly this reason.

**Diagnosis, not guessed:** read PyG's wheel index (`data.pyg.org/whl/`) directly. It publishes `torch-scatter`/`torch-sparse` builds up to and including `torch-2.9.1+cu130`, and nothing newer (consistent with the PyG project moving away from these compiled C++ extensions in favour of native PyTorch ops in later releases). `torch-2.9.1+cu130` is therefore the newest real, installable combination with matching wheels.

**Verified end to end, live, before committing to it** (not assumed from the wheel index alone): installed `torch==2.9.1+cu130`, then `torch-scatter`/`torch-sparse` for that exact combo, then `torch-geometric==2.7.0` (the project's existing pin, confirmed it still imports against this newer torch). Ran, on that session's actual CUDA device: a raw matmul (finite result, zero compatibility warnings), `torch_scatter.scatter_add` on CUDA, and a `GINEConv` forward pass (the exact layer type the official Multi-GNN model uses) on CUDA with a finite output. All passed.

**Fix:** `install_gnn_stack()` (`src/amlh/session.py`) now takes a `FALLBACK_TORCH` list, `[("2.5.1", "cu121"), ("2.9.1", "cu130")]`, tried in order, stopping at the first combination with real wheels. `2.5.1+cu121` stays first, so nothing changes for the RTX 3080 fleet or any of its already-finished results. Two regression tests added (`tests/test_colab.py`): the fallback loop tries combinations in order and stops at the first with wheels, and it raises naming every combination tried when none have wheels. `requirements-mgnn.txt`'s header now explains it is a static pin for manual setup (`gpu_session/setup_windows.ps1`) and does not reflect what `install_gnn_stack()` actually does.

**What this does not settle:** whether `2.9.1+cu130` is the best choice if a still-newer card appears later (the fix is a short, explicit list to extend, not a general solver); whether non-GINEConv parts of the official Multi-GNN/FraudGT code behave identically under 2.9.1 (only GINEConv was exercised live); and the live session's own torch install was corrupted twice by an unrelated cause before this succeeded (an earlier `pip install` inside a `marimo-pair` scratchpad call got cut off mid-write when the SSE reporting connection timed out, leaving a partial `torch` package) -- worth knowing if a future live-paired install silently leaves a broken package: check for `pip`'s own "Ignoring invalid distribution ~X" warnings, and retry with the install detached via a real background process (`subprocess.Popen` writing to a log file, polled separately) rather than one blocking call.

## D-026 (2026-11-07): notebook 01's run cell never passed `official_root`, so every live `mgnn_train`/FraudGT job failed this session (not a D-025 regression)

**What was found, live, on the same Molab Blackwell session as D-025** (paired again via `marimo-pair`): after D-025 landed and this session's torch/PyG install succeeded (`torch 2.9.1+cu130`, confirmed in the session's own printed install info), every job that actually tried to launch `mgnn_train` or FraudGT still failed with exit 1. NB1 showed only its already-`DONE` jobs from prior sessions as complete; NB2 was 0 of 15 verified. Read a failed job's `status.json` directly (`nb1-mgnn-gin-HI-Small-s1`) rather than guessing from the truncated `RuntimeError` text the summary cell prints:

```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/aml_official/Multi-GNN'
```

Not a hardware/compatibility issue at all — a path mismatch between two cells in `01_gnn_baselines.ipynb`. The "Install dependencies" cell calls `setup_environment('NB2', official_root=BASE / 'official')`, which clones Multi-GNN and FraudGT into the persistent `/home/marimo/aml_molab/official/` (confirmed present and correctly populated on the live filesystem). The "Run NB1, NB2, and NB3's remaining work" cell calls `amlh.session.run_notebook(...)` without an `official_root` argument, so it silently used `run_notebook`'s own default, `/tmp/aml_official` -- confirmed absent on the live filesystem (`/tmp` does not survive a session/container restart the way `BASE` does, while the job ledger and checkpoints under `BASE/out_nb1` do, so the notebook's resume logic correctly picked up prior progress but then tried to launch training against a path that no longer existed).

**Fix:** pass `official_root=BASE / 'official'` to that `run_notebook` call, matching what the install cell already uses. Patched the equivalent cell live in the running session (via `marimo-pair`'s `cm.edit_cell`) so it can resume without a fresh clone, and fixed `gpu_session/molab/build_molab_notebooks.py` (the source of truth) plus regenerated `01_gnn_baselines.ipynb`. Notebooks 02 and 03 (NB4/NB5) never call `setup_environment` with `OFFICIAL_NEEDED` code (`OFFICIAL_NEEDED = {"NB1": ["Multi-GNN"], "NB2": ["Multi-GNN", "FraudGT"]}`), so they were never exposed to this. No `src/amlh/` change, so the code hash and every other notebook/dist artifact are untouched.

**What this does not settle:** whether any `DONE` job this session (or a prior one) was silently affected by a similar restart-wiped-`/tmp` gap before this fix existed -- every job inspected so far either came from `PRIOR_RESULTS` on disk or failed loudly, but this was not exhaustively checked across every shard in `results/`.

## D-027 (2026-11-07): `nvidia-cusparselt-cu13`'s pip install was corrupted (metadata present, `.so` file missing) on a fresh Blackwell Molab sandbox, exactly the D-025 gotcha predicted

**What was found, live, on a brand new Molab sandbox** (a different sandbox URL than D-025/D-026's, after a kernel restart actually recycled the whole container, not just the Python process -- `BASE/official` existed correctly this time, confirming D-026's fix works, but every `mgnn_train`/FraudGT job still failed): read `nb1-mgnn-gin-HI-Small-s1`'s `status.json` directly rather than assuming it was a repeat of D-026.

```
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

Raised from `import torch` itself, inside the isolated child subprocess (`amlh/isolate.py`'s `child_main`) -- not from any `amlh` code. The notebook's "Install dependencies" cell never actually imports torch to verify the install; it only runs `install_gnn_stack()` (pip installs plus a version string) and `accelerators()` (GPU detection via `nvidia-smi`), so a broken torch install is invisible until the first real job tries to use it.

**Diagnosed, not guessed:** `/tmp/uv-venv/bin/python -m pip list` showed `nvidia-cusparselt-cu13 0.8.0` as installed, and `find /` located a real `libcusparseLt.so.0` on disk -- but at `/usr/local/lib/python3.13/site-packages/nvidia/cusparselt/lib/`, a *different* Python environment than `/tmp/uv-venv` (the one torch 2.9.1+cu130 and the isolated jobs actually use). Checked `/tmp/uv-venv/lib/python3.13/site-packages/nvidia/` directly: no `cusparselt` directory at all, despite pip's own metadata claiming it was installed. This is exactly the failure mode D-025 already flagged as a risk ("an earlier pip install ... got cut off mid-write ... leaving a partial package") -- pip's dist-info record survived, the ~170 MB binary payload did not.

**Fix, verified end to end before reporting it:** `/tmp/uv-venv/bin/python -m pip install --force-reinstall --no-deps nvidia-cusparselt-cu13==0.8.0` (full 169.9 MB wheel re-downloaded and installed cleanly). Confirmed the `.so` now exists in the right venv, then ran `import torch; torch.cuda.is_available(); (torch.randn(4,4,device="cuda") @ ...).sum()` directly via that venv's interpreter -- succeeded, real CUDA result. The 14 jobs that had failed only on this `ImportError` were reset (`attempts=0`, `status=TODO`) so they retry under the fixed install; their reset `reason` names this entry.

**What this does not settle:** whether this is a one-off corrupted download (this specific sandbox, this specific package) or a systemic gap -- `install_gnn_stack()` still has no post-install verification step, so the same silent-corruption failure mode could recur with a different package on a future sandbox, and would again only surface inside a real job's isolated subprocess rather than at install time. Adding an actual `import torch` + CUDA smoke test to `install_gnn_stack()` (or `setup_environment()`) would catch this class of failure immediately instead of burning a job's retry budget on it, but that code change was not made here -- only the live data fixed.
