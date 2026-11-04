# Roadmap: Decision Horizons in Graph-Based AML

**Updated:** 1 November 2026. **Progress milestone:** Sunday 15 November 2026. **Long-term target:** a conference paper.

| Document | Answers |
|---|---|
| `PHASE3_PLAN.md` | What we are doing and why |
| `DECISIONS.md` | Frozen hypotheses and the reason for each decision |
| `SPEC.md` | Exact definitions |
| `paper/cn_phase2.tex` (Revision 3) | Literature review and full methodology |
| This file | Who does what, when, and on which machine |

---

## 1. Where things stand

**Done:**
- All four AMLworld datasets downloaded; line counts and hashes verified.
- Shared code for all five notebooks: runner, horizon view, wall-clock and logical-clock features, GNN, alert policy, attacks, sealed scoring and merge. 301 tests pass. See `NOTEBOOK_METHODOLOGY.md` §8 for exactly what has and has not run.
- The five Kaggle notebooks, merge notebook, pilot mode and FraudGT audit job, in `notebooks/`. **Written and tested here, but never run on Kaggle.** The run guide is `notebooks/README.md`.
- HI-Small main horizon sweep (9 horizons, 3 seeds) and the tail audit.
- Literature review updated to Revision 3.

**Partial:** documented-window sweep (to 1 h); adaptive policy (1 seed); leakage check (H = 0 only).

**Not run:** any pilot, official Multi-GNN, FraudGT, TD-GNN on real data, timing attack, GFP, ablations, all confirmatory datasets.

Code, documents and the paper source are in this repository (`C:\dev\aml-stream` on the HP); the data is not, because it is 6.8 GB. **The next step is to upload the code package and run the pilots (task 1b).** The package holds everything the notebooks need, so GitHub does not block them, though Saif still needs the repo for development.

---

## 2. Tasks

Owners are suggestions; swap them as suits you. Any job can run on any suitable machine.

| # | Task | Owner | Done when | By |
|---|---|---|---|---|
| 1 | Create a private GitHub repo and push the code (exclude `data/`, `.venv*`, `results/**/scores_*`) | Shahoud | Saif can clone it | Mon 2 Nov |
| 1b | Put `dist/aml-code-<hash>.zip` where the notebooks can find it (a Drive folder `aml` on Colab, or a private dataset on Kaggle), import the notebooks and **run all five pilots** (`notebooks/README.md`). On Kaggle, note whether T4 x2 counts double | Both | Five `PILOT_REPORT.md` files, or the failures to fix | Tue 3 Nov |
| 2 | Finish the **leakage check** at all horizons (NB4, stage 0). If any shuffled run scores above prevalence, stop and investigate before anything else | Shahoud | All runs at prevalence | Tue 3 Nov |
| 3 | **Tell the instructor** about the direction change; ask for Phase 2 feedback and the report format | Both | Reply in `DECISIONS.md` | Tue 3 Nov |
| 4 | **Run official Multi-GNN** (GIN + EU, reverse MP, ports, ego IDs) in NB1, using `amlh/mgnn.py`. That code has **never been executed**, so allow a day to debug it. Time box: 3 days | Saif | `audit.json`, or a note saying why not | Thu 5 Nov |
| 5 | Same for Multi-PNA + EU, and a second seed if quota allows | Saif | As above | Sat 7 Nov |
| 6 | **Gate G1:** does the official GNN show the tail and horizon effects? Record it either way | Both | Entry in `DECISIONS.md` | Sat 7 Nov |
| 7 | LI-Small sweep and policy (NB4) at frozen settings | Shahoud | Verdicts C1–C5 on LI-Small | Thu 5 Nov |
| 8 | TD-GNN at 5 horizons, 3 seeds (NB3) | Saif | C7 answered | Sat 7 Nov |
| 9 | ~~Build logical-clock features and add them to the test suite~~ **Done 1 Nov** | — | Tests pass | — |
| 10 | Timing attack (NB5) on wall-clock and logical-clock detectors | Shahoud | Verdicts C6, C8 on HI-Small | Mon 9 Nov |
| 11 | HI-Medium and LI-Medium sweeps (NB4). `prepare.py` rewrite **done 1 Nov**: 102 s, 12.1 GB peak | Saif | Verdicts on Medium datasets | Mon 9 Nov |
| 12 | Get SAML-D, write its loader, run the sweep | Shahoud | Verdicts C1–C4 on SAML-D | Tue 10 Nov |
| 13 | GFP + XGBoost through `snapml`, same split | Saif | A comparison row, or a recorded failure | Tue 10 Nov |
| 14 | Ablations A1–A10 | Both | `tab_ablation.tex` | Tue 10 Nov |
| 15 | Service-time profile on one named GPU (NB3) and CPU (NB5); queue simulation in the merge step | Saif | Latency table | Wed 11 Nov |
| 16 | Phase 3 progress report: all 7 brief components, each marked complete or preliminary | Shahoud | PDF | Thu 12 Nov |
| 17 | Each person explains every module to the other, and regenerates the other's headline tables from raw logs | Both | Numbers match | Fri 13 Nov |
| 18 | **Submit the progress milestone** | Both | Submitted | Sat 14 Nov |

---

## 3. Where each job runs

No single machine is a dependency. Every job is a script that takes a dataset and writes a results folder. **Laptops are for development and analysis.**

| Resource | Use it for | Why |
|---|---|---|
| **Kaggle GPU** | Multi-GNN (tasks 4, 5); TD-GNN (task 8) | Runs unattended; hosts the dataset directly. Needs a phone-verified account |
| **Kaggle CPU** | Leakage check, LI-Small, attacks, ablations, GFP | Linux, which GFP needs; no GPU quota used |
| **molab** (32 GB RAM; optional 96 GB GPU) | Medium datasets; extra GPU capacity | Memory; only after the 3 acceptance tests below |
| **University GPU** (RTX 4000 Ada, 2–3 h/day) | Latency profile; TD-GNN overflow; Multi-GNN fallback | One consistent device for timing |
| **Saif's MacBook** (32 GB) | Medium sweeps overnight; development | Memory, unattended CPU |
| **Colab** | The five notebooks, where a Kaggle account cannot be verified (`DECISIONS.md` D-015). P0 jobs only on a free runtime | Output goes to Drive, so a disconnect resumes. About 12 GB of RAM, so no Medium datasets. Free GPU time is not published |
| **Friend's GPU** | **0 hours planned until its specification is confirmed** | The earlier plan excluded a Quadro T1000; see §6 |
| **Shahoud's HP** | Development, analysis, writing | Not for long experiments |

**molab acceptance tests:** GPU attaches; a 2-hour job survives a closed browser; data load time is acceptable. Notebooks are public by default, so never put tokens in the code.

### Compute budget

| Job | GPU hours | Basis |
|---|---|---|
| Multi-GNN: GIN, PNA, second seed | 12–24 | **Estimate:** 4–8 h per run. Only anchor: about 3.6 h on an RTX 3090 (published) |
| TD-GNN: 4 horizons × 3 seeds | 12–24 | **Estimate:** no full run has finished |
| GPU latency profile | 2–3 | **Estimate** |
| **Core total** | **26–51** | |
| TGN comparison, if Gate G2 passes | 10–20 | **Estimate** |

CPU jobs need no GPU. Measured on an i5-1235U, one LightGBM run at HI-Small scale took about 1–2 minutes including features. Medium datasets are about 6 times larger.

**Capacity:** Kaggle is commonly reported at about 30 GPU hours per week per account. That figure is unofficial, so check it in your accounts. Two accounts over two weeks should cover the core total without relying on anything else.

---

## 4. What 15 November must show

| Brief component | Plan | Status now |
|---|---|---|
| 1. Implementation, feedback, changes documented | Revision 3 records the changes | Implementation exists; **class feedback missing** |
| 2. Dataset and preprocessing | Data card, split reproduction, audit | Done for HI-Small |
| 3. Baselines | LightGBM done; TD-GNN, Multi-GNN, GFP pending | Partial |
| 4. Graphs | Generated from logs | Partial |
| 5. Comparison with published work | Table with views stated | Needs Multi-GNN |
| 6. Ablations | A1–A10 | Not run |
| 7. Discussion | Follows the verdicts | Not started |

**Success means:** a verdict on each frozen hypothesis on at least one confirmatory dataset, and the Multi-GNN audit run or a documented reason why not. It does not mean a finished paper.

---

## 5. After 15 November

| When | Work | Gate |
|---|---|---|
| Late Nov | **Gate G2:** is there a modelling gap at small horizons worth closing? | Build DMT-GNN only if yes |
| Dec | Rolling-origin folds; per-attempt attacks; real Ethereum window; hop-aware stoppable value | Effects hold across folds |
| Dec | Repeat the literature search; contact the AMLworld maintainers | Your decision |
| Jan | Draft Paper A | |
| Feb 2027 | Submit to KDD Datasets & Benchmarks if its second-cycle call opens (**not yet published**) | Otherwise ICAIF 2027 (call not yet published) |
| Sep 2027 | Paper B to FC 2028 (dates not published) | Must hold beyond one simulator |

---

## 6. Needed from you

| # | What | Why |
|---|---|---|
| 1 | **Friend's GPU model and memory** | If it is the Quadro T1000 excluded earlier, tell me whether that has changed |
| 2 | Instructor's response to the direction change | The topic moved from DMT-GNN to decision horizons |
| 3 | Phase 2 class feedback | The brief requires it, and I cannot write it for you |
| 4 | University PC access date | Needed for the latency numbers only |
| 5 | Whether to post a preprint | The timing-attack space may close soon |

---

## 7. Working rules

- Every reported number comes from a log through `scripts/08_analyze.py`. Nothing is typed by hand.
- HI-Small is the discovery set. A claim counts as confirmed only on data not modelled when the hypotheses were frozen.
- A failed hypothesis is a result, and it gets reported.
- Check downloads by line count and hash: Kaggle truncated one file while reporting success.
- Commit to git only when you decide to.
