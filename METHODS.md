# Code map

Definitions live in `SPEC.md` in the project documents. If code and that file disagree, it is a bug.

| Module | Role |
|---|---|
| `src/amlh/prepare.py` | Raw files to event tables; AMLworld and SAML-D |
| `src/amlh/datasets.py` | Loads one dataset under one protocol or fold |
| `src/amlh/features.py` | Horizon view and wall-clock features |
| `src/amlh/logical.py` | Logical-clock features |
| `src/amlh/design.py` | Feature families for the boosted detector |
| `src/amlh/sampler.py`, `gnn.py`, `jobs_gnn.py` | TD-GNN: view-restricted sampler, model, jobs |
| `src/amlh/jobs_gbt.py` | Boosted detector: sweeps, alert policy, attacks, service time |
| `src/amlh/perturb.py` | Timing attacks and their controls |
| `src/amlh/mgnn.py`, `mgnn_format.py` | Official Multi-GNN runner. **Never executed** |
| `src/amlh/fraudgt.py`, `fraudgt_driver.py` | FraudGT audit: job, and the child process that runs the published code. **Never executed** |
| `src/amlh/isolate.py` | Runs a job in its own process so its memory is returned when it ends |
| `src/amlh/datatable_shim.py` | Stand-in for the few datatable calls the official formatter makes |
| `src/amlh/fetch.py` | Downloads raw files on Colab and checks each is whole |
| `src/amlh/extras.py` | GFP and Ethereum adapters. **Never executed** |
| `src/amlh/evaluate.py`, `metrics.py` | Metrics, stoppable value, C3b |
| `src/amlh/runner.py`, `session.py`, `plans.py` | Job queue and checkpoints; notebook setup, pilot and GPU workers; the frozen job list and pilot jobs |
| `src/amlh/sealing.py`, `merge.py` | Sealed scoring; verification, unsealing, verdicts |
| `src/amlh/queue_sim.py` | Queue simulator, checked against M/M/1 |
| `scripts/make_code_dataset.py` | Packs the code for a private Kaggle dataset and prints its hash |
| `scripts/build_notebooks.py` | Writes `notebooks/nb1.ipynb` to `nb5.ipynb` and `merge.ipynb`, each carrying the expected code hash |
| `notebooks/README.md` | How to run them |
| `scripts/02b_audit_all.py`, `08_analyze.py` | Tail audit; tables and figures |
| `scripts/legacy/` | Superseded experiment scripts, kept because earlier results came from them |

Run the tests with `python -m pytest tests`. The official Multi-GNN modules are never imported by the tests.
