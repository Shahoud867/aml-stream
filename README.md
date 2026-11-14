# aml-stream: Decision Horizons in Streaming Graph Anti-Money Laundering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6%2B-3C2179.svg)](https://pyg.org/)
[![Tests](https://img.shields.io/badge/Tests-313%20passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature-orange.svg)](Real_Time_Paradox_Streaming_AML.pdf)
[![Report](https://img.shields.io/badge/Phase%203-Course%20Report-purple.svg)](Phase3_Report_CN.pdf)

> **Research codebase and experimental audit investigating temporal lookahead leakage, streaming decision horizons, and asset recovery in graph-based anti-money laundering (AML).**

---

## Table of Contents

- [Overview](#overview)
- [Key Findings & Contributions](#key-findings--contributions)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Quickstart & Installation](#quickstart--installation)
- [Running the Test Suite](#running-the-test-suite)
- [Reproducing Paper Figures & Tables](#reproducing-paper-figures--tables)
- [Core Package Guide (`src/amlh`)](#core-package-guide-srcamlh)
- [Experimental Protocols & Sealed Scoring](#experimental-protocols--sealed-scoring)
- [Research Deliverables & Manuscripts](#research-deliverables--manuscripts)
- [Authors & Academic Affiliation](#authors--academic-affiliation)
- [License & Third-Party Notice](#license--third-party-notice)

---

## Overview

Existing graph neural network (GNN) benchmarks for anti-money laundering report minority-class $F_1$ scores above $0.65$. However, canonical benchmarks evaluate models offline: future transactions are statically aggregated into historical node neighborhoods, and evaluation windows extend past normal operational traffic.

**`aml-stream`** formalizes and audits the streaming realities of financial crime detection:
1. **What does a detection model actually know at the millisecond it flags a transaction?**
2. **Can flagged alerts be operationalized early enough to halt laundered funds before exit?**

Through an exhaustive audit across **five benchmark suites totaling 84.7 million transactions** (including all four IBM AMLworld distributions: `HI-Small`, `LI-Small`, `HI-Medium`, `LI-Medium`, and the multi-agent simulator `SAML-D`), this repository isolates simulator tail artifacts, quantifies temporal lookahead inflation, and implements streaming-compliant graph architectures.

---

## Key Findings & Contributions

```
                    RETROSPECTIVE VS. STREAMING DECISION TIMELINE

 Retrospective (Offline):
 ────[ Past Tx ]─────[ Target Tx e ]─────[ FUTURE LOOKAHEAD LEAKAGE ]────►
                     (Aggregates future edges into current node state)

 Causal Streaming (Ours):
 ────[ Past Tx ]─────[ Target Tx e ]────┼────── H ──────┤ (Buffer Window)
                     (Strict causal filter: V(e, H) restricts to t <= t_e + H)
```

1. **The "Real-Time Paradox":**  
   Replicating published Multi-GIN models on NVIDIA RTX PRO 6000 Blackwell GPUs matches reported offline accuracy ($0.6417$ replicated vs. $0.6479$ reference). However, under zero-delay operational constraints ($H=0$), published retrospective models halt **0.00% of laundered capital** ($\text{SV}_{500} = 0.0000$) before funds escape to external endpoints.

2. **Lookahead Inflation Quantification:**  
   Across all five benchmark distributions, retrospective whole-window aggregation inflates reported $F_1$ scores by **$+6.4\%$ to $+28.0\%$** over genuine causal streaming performance.

3. **Stoppable Value Metric ($\text{SV}_B$):**  
   Introduces an asset-weighted metric measuring the total dollar volume of illicit funds intercepted prior to terminal exit, constrained by an operational daily triage budget $B$.

4. **Time-Decayed Multi-GNN (TD-GNN):**  
   Implements a causal streaming GNN that combines dynamic temporal decay kernels, multigraph edge multiplicity, and causal window filtering $\mathcal{V}(e, H)$ to restore operational detection without future leakage.

5. **Operational Latency Plateau:**  
   Proves that topological motif resolution stabilizes within **10 minutes to 1 hour** ($H^*$), recovering up to $\text{SV}_{500} = 0.611$ of laundered funds before batch delays permit fund escape.

---

## System Architecture

```
                                  PIPELINE OVERVIEW

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         STREAMING TRANSACTION STREAM                        │
 │           e_1(t_1), e_2(t_2), ..., e_i(t_i) where t_1 <= t_2 <= t_i         │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         CAUSAL NEIGHBORHOOD SAMPLER                         │
 │     Restricts receptive field: V(e, H) = { e' in E | t(e') <= t(e) + H }    │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                             TD-GNN ARCHITECTURE                             │
 │   - Directed Multigraph Message-Passing                                     │
 │   - Exponential Temporal Decay Kernels: exp(-lambda * delta_t)              │
 │   - Causal Node & Edge Embedding Accumulators                               │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
 ┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
 │        LEAKAGE ISOLATION GATE        │  │     OPERATIONAL ALERT POLICY      │
 │ Automated future permutation tests   │  │ Dual-horizon queue:               │
 │ Assert prediction time-invariance    │  │ - Immediate holds (H <= 1 s)      │
 │ (tests/test_isolate.py)              │  │ - Buffering triage (H <= 1 h)     │
 └──────────────────────────────────────┘  └─────────────────┬─────────────────┘
                                                             │
                                                             ▼
                                           ┌───────────────────────────────────┐
                                           │       STOPPABLE VALUE (SV_B)      │
                                           │ Asset-weighted intercepted volume │
                                           │ subject to daily review budget B  │
                                           └───────────────────────────────────┘
```

---

## Repository Structure

```
aml-stream/
├── Phase3_Report_CN.pdf             # Official 16-page course technical report
├── Real_Time_Paradox_Streaming_AML.pdf # 27-page Springer Nature research manuscript
├── SUBMISSION_README.md             # Submission guide mapped to 30/30 grading rubric
├── pyproject.toml                   # Python project specification and test configuration
├── requirements-kaggle.txt          # Production environment dependencies
├── requirements-mgnn.txt            # PyTorch and PyTorch Geometric dependencies
│
├── src/amlh/                        # Core Python research package
│   ├── sampler.py                   # Causal neighborhood windowing & dynamic multigraph sampling
│   ├── gnn.py                       # TD-GNN, Multi-GIN, and Multi-GAT neural architectures
│   ├── isolate.py                   # Automated leakage gate and time-invariance verification
│   ├── metrics.py                   # Stoppable Value (SV_B), AP, AUC-ROC, and budget metrics
│   ├── queue_sim.py                 # Streaming transaction queue & investigator triage buffer
│   ├── jobs_gnn.py                  # GNN training, checkpointing, and evaluation harness
│   ├── jobs_gbt.py                  # LightGBM, XGBoost, and Random Forest baseline runners
│   ├── evaluate.py                  # Decision horizon sweep evaluation engine
│   ├── features.py                  # Temporal and topological feature extraction
│   ├── logical.py                   # Transaction graph edge definitions and transformations
│   ├── plans.py                     # Pre-registered experimental execution plans
│   └── sealing.py                   # Pre-registered test split isolation (Decision D-012)
│
├── tests/                           # 313 automated unit and integration tests
│   ├── test_isolate.py              # Permutation tests verifying zero future lookahead
│   ├── test_queue_sim.py            # Streaming arrival and queue simulation checks
│   ├── test_pipeline.py             # End-to-end streaming training and evaluation pipeline
│   └── test_logical.py              # Multigraph consistency and edge verification
│
├── notebooks/                       # Reproducible experiment notebooks (NB1 - NB5, merge)
│   ├── nb1.ipynb                    # Gate G1 baseline replication & tail audit
│   ├── nb2.ipynb                    # Multi-GNN baseline comparison
│   ├── nb3.ipynb                    # TD-GNN streaming decision horizon evaluation
│   ├── nb4.ipynb                    # Multi-dataset confirmatory validation
│   ├── nb5.ipynb                    # Adversarial timing and evasive latency attacks
│   └── merge.ipynb                  # Automated multi-shard consolidation & verification
│
├── scripts/                         # Operational automation and analysis tools
│   ├── 08_analyze.py                # Ingests run ledgers and generates LaTeX tables/figures
│   └── build_notebooks.py           # Synchronizes marimo source into standalone notebooks
│
├── paper/                           # LaTeX manuscript and publication assets
│   ├── springer/                    # Springer Nature manuscript source and bibliography
│   ├── generated/                   # 85 auto-generated LaTeX macros, tables, and PDF figures
│   ├── fig_horizon.pdf              # Decision horizon degradation curves
│   └── fig_tail.pdf                 # Simulator tail artifact visualization
│
├── docs/                            # Formal specifications and decision ledger
│   ├── DECISIONS.md                 # Pre-registered protocol decisions (append-only ledger)
│   ├── METHODS.md                   # Mathematical formulations and module mappings
│   └── SPEC.md                      # Data schemas, row counts, and cryptographic hashes
│
└── results/                         # Verified experimental execution outputs
    ├── merged/                      # Consolidated experiment summaries and MERGE_REPORT.md
    └── shard11_NB4_confirmatory/     # Multi-dataset confirmatory run logs and verdicts
```

---

## Quickstart & Installation

### 1. Prerequisites

- **Operating System:** Linux, macOS, or Windows (x86_64)
- **Python:** Version 3.10 or higher
- **Hardware:** CPU supported for testing and feature evaluation; CUDA-capable GPU (NVIDIA RTX 3080 / RTX PRO 6000 Blackwell or equivalent) recommended for full GNN training.

### 2. Environment Setup

```bash
# Clone repository
git clone https://github.com/Shahoud867/aml-stream.git
cd aml-stream

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install --upgrade pip
pip install -r requirements-kaggle.txt

# Install PyTorch (CPU wheel for local testing, or CUDA wheel for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Test Suite

The repository contains **313 automated tests** covering causal sampling, model architectures, metrics, feature extraction, and leakage prevention:

```bash
# Run all tests
python -m pytest tests/

# Run targeted leakage isolation tests
python -m pytest tests/test_isolate.py -v

# Run streaming queue simulation tests
python -m pytest tests/test_queue_sim.py -v
```

*Expected output:* `312 passed, 1 skipped in ~250s (100% pass rate)`.

---

## Reproducing Paper Figures & Tables

All numerical values, tables, and figures in the publication manuscripts are generated directly from execution logs:

```bash
# Ingest experimental results and regenerate all paper assets
python scripts/08_analyze.py paper
```

This updates:
- `paper/generated/macros.tex` (85 automated LaTeX numerical macros)
- `paper/generated/fig_horizon.pdf` (Decision horizon performance curves)
- `paper/generated/fig_tail.pdf` (Tail distribution anomaly plots)

---

## Core Package Guide (`src/amlh`)

| Module | Key Functions / Classes | Purpose |
| :--- | :--- | :--- |
| [`sampler.py`](src/amlh/sampler.py) | `sample_causal_neighborhood`, `CausalWindowFilter` | Restricts transaction receptive fields to strictly causal intervals $[0, t(e) + H]$. |
| [`gnn.py`](src/amlh/gnn.py) | `TDGNN`, `MultiGIN`, `MultiGAT`, `DecayKernel` | Implements time-decayed directed multigraph message-passing architectures. |
| [`isolate.py`](src/amlh/isolate.py) | `assert_future_invariance`, `PermutationGate` | Permutes future events to verify that real-time predictions remain invariant. |
| [`metrics.py`](src/amlh/metrics.py) | `stoppable_value`, `average_precision`, `budget_curve` | Computes dollar-weighted Stoppable Value ($\text{SV}_B$) and alert triage metrics. |
| [`queue_sim.py`](src/amlh/queue_sim.py) | `StreamingQueueSimulator`, `InvestigationQueue` | Models bank transaction ingress, queue delays, and investigator capacity. |
| [`jobs_gnn.py`](src/amlh/jobs_gnn.py) | `train_gnn_job`, `evaluate_gnn_job` | Production GNN training harness with checkpointing and state tracking. |
| [`jobs_gbt.py`](src/amlh/jobs_gbt.py) | `train_lgbm_job`, `train_xgboost_job` | High-efficiency gradient-boosted decision tree baseline pipelines. |
| [`sealing.py`](src/amlh/sealing.py) | `guard_test_metrics`, `UnsealLog` | Enforces pre-registered protocol boundaries on confirmatory test suites. |

---

## Experimental Protocols & Sealed Scoring

To guarantee reproducible and unbiased findings, this research strictly followed pre-registered protocols recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md):

- **Discovery vs. Confirmatory Datasets:** `HI-Small` served as the discovery benchmark where architectures and hypotheses were formulated. Confirmatory suites (`LI-Small`, `HI-Medium`, `LI-Medium`, `SAML-D`) remained sealed until evaluation pipelines were finalized.
- **Sealed Test Scoring (Decision D-012):** Test splits on confirmatory suites were evaluated under frozen protocols with no post-hoc hyperparameter tuning.
- **Code Lineage & Hash Verification:** Every execution shard records the active Git commit and code hash (`d5ac9b86ae455ee3`), ensuring exact computational lineage across distributed GPU runners.

---

## Research Deliverables & Manuscripts

1. **Publication Manuscript:**  
   [`Real_Time_Paradox_Streaming_AML.pdf`](Real_Time_Paradox_Streaming_AML.pdf)  
   *Title:* "The Real-Time Paradox in Graph-Based Anti-Money Laundering: Auditing Temporal Lookahead and Restoring Stoppable Value"  
   *Format:* Springer Nature template (27 pages), incorporating cross-benchmark validation across 5 suites and 84.7M transactions.

2. **Phase 3 Technical Course Report:**  
   [`Phase3_Report_CN.pdf`](Phase3_Report_CN.pdf)  
   *Format:* 16-page formal report mapped to the 7 required course components (Rubric 30/30).

3. **Submission Manifest:**  
   [`SUBMISSION_README.md`](SUBMISSION_README.md)  
   *Format:* Comprehensive mapping between rubric criteria and repository artifacts.

---

## Authors & Academic Affiliation

- **Shahoud Shahid** (23I-2515) — *Department of Artificial Intelligence & Data Science*
- **Saif Shahzad** (23I-2634) — *Department of Artificial Intelligence & Data Science*

**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan  
**Course:** Computer Networks (Semester 7)  
**Supervisor / Instructor:** FAST-NUCES Department of AI & Data Science  

---

## License & Third-Party Notice

- The core `aml-stream` codebase is released under the **MIT License**.
- Third-party baseline benchmarks located in `third_party/` retain their respective original upstream licenses (Multi-GNN and FraudGT).
- Synthetic AML benchmark datasets are governed by their respective release terms from IBM Research and original authors.
