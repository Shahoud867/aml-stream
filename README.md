# aml-stream: Decision Horizons in Streaming Graph Anti-Money Laundering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6%2B-3C2179.svg)](https://pyg.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![Numba](https://img.shields.io/badge/Numba-JIT%20Accelerated-orange.svg)](https://numba.pydata.org/)
[![Tests](https://img.shields.io/badge/Tests-313%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Journal Paper](https://img.shields.io/badge/Journal%20Paper-Springer%20Nature%20(29%20pp)-orange.svg)](Real_Time_Paradox_Streaming_AML.pdf)
[![Conference Paper](https://img.shields.io/badge/Conference%20Paper-ACM%20SIGKDD%20(10%20pp)-red.svg)](Real_Time_Paradox_Streaming_AML_KDD.pdf)

> **Production-grade research codebase and streaming evaluation framework auditing temporal lookahead leakage, streaming decision horizons ($H$), and asset interdiction in graph-based anti-money laundering (AML).**

---

## Table of Contents

- [Overview & The "Real-Time Paradox"](#overview--the-real-time-paradox)
- [Key Findings & Research Contributions](#key-findings--research-contributions)
- [System Architecture & Data Flow](#system-architecture--data-flow)
- [Benchmark Datasets & Cryptographic Integrity](#benchmark-datasets--cryptographic-integrity)
- [Repository Layout](#repository-layout)
- [Installation & Environment Setup](#installation--environment-setup)
- [Quickstart & Python API Usage](#quickstart--python-api-usage)
  - [1. Causal Horizon Filtering & Numba Feature Extraction](#1-causal-horizon-filtering--numba-feature-extraction)
  - [2. Causal 2-Hop Multigraph Sampling & TD-GNN Inference](#2-causal-2-hop-multigraph-sampling--td-gnn-inference)
  - [3. Stoppable Value ($\text{SV}_B$) and Operational Alert Policies](#3-stoppable-value-textsv_b-and-operational-alert-policies)
- [Automated Test Suite & Leakage Gate](#automated-test-suite--leakage-gate)
- [Reproducing Paper Experiments & Figures](#reproducing-paper-experiments--figures)
- [Cloud Execution & Distributed Notebooks](#cloud-execution--distributed-notebooks)
- [Core Package Reference (`src/amlh`)](#core-package-reference-srcamlh)
- [Research Publications & Manuscripts](#research-publications--manuscripts)
- [Citation](#citation)
- [Authors & Academic Affiliation](#authors--academic-affiliation)
- [License & Acknowledgments](#license--acknowledgments)

---

## Overview & The "Real-Time Paradox"

Existing graph neural network (GNN) benchmarks for anti-money laundering report minority-class $F_1$ scores exceeding $0.65$. However, canonical benchmarks evaluate models **retrospectively**: future transactions are statically aggregated into historical node neighborhoods, and evaluation windows extend across future split boundaries.

```
                    RETROSPECTIVE VS. CAUSAL STREAMING EVALUATION

  1. Retrospective (Canonical Offline Benchmarks):
  ────[ Past Transfers ]─────[ Target Event e_i ]─────[ FUTURE LOOKAHEAD LEAKAGE ]────►
                             t(e_i)                    Aggregates future edges into
                                                       historical node embeddings!

  2. Causal Streaming with Decision Horizon H (Ours):
  ────[ Past Transfers ]─────[ Target Event e_i ]────┼────── H ──────┤ (Buffer Window)
                             t(e_i)                  t(e_i) + H      Strict causal filter:
                                                                     V(e_i, H) restricts
                                                                     to t <= min(t_i + H, T_split)
```

**`aml-stream`** formalizes and audits the streaming operational realities of financial transaction networks:
1. **What does a detection system actually know at the millisecond it flags a transaction?**
2. **Can flagged alerts be operationalized early enough to halt laundered capital before exit?**

Across an exhaustive audit of **five benchmark distributions totaling 84.7 million transactions** (including all four IBM AMLworld datasets—`HI-Small`, `LI-Small`, `HI-Medium`, `LI-Medium`—and the multi-agent simulator `SAML-D`), this repository demonstrates that offline high-performance models suffer from a fundamental failure in streaming deployment: **The Real-Time Paradox**.

---

## Key Findings & Research Contributions

| Finding / Contribution | Theoretical & Empirical Result | Practical Significance |
| :--- | :--- | :--- |
| **The "Real-Time Paradox"** | Replicated official Multi-GIN baseline on `HI-Small` across 3 seeds ($F_1 = 0.6417$ vs. $0.6479$ reference). Under strict zero-delay operational constraints ($H=0$), published retrospective models halt **0.00% of laundered capital** ($\text{SV}_{500} = 0.0000$). | Offline $F_1$ is an uninformative metric for deployment; models relying on future downstream fan-out cannot flag funds prior to exit. |
| **Lookahead Inflation** | Quantified whole-window retrospective aggregation across 5 benchmarks. Retrospective evaluation inflates reported $F_1$ by **$+6.4\%$ to $+28.0\%$** over causal streaming baselines. | Proves that high reported accuracy in existing literature is largely an artifact of future temporal leakage. |
| **Stoppable Value ($\text{SV}_B$)** | Formulated an asset-weighted metric measuring the fraction of illicit laundering attempt volume intercepted before terminal payout under daily triage budget $B \in \{100, 500, 2000\}$: $\text{SV}_B = \frac{\sum_{e \in \mathcal{A}, t(e) > T(\mathcal{A})} \text{val}(e)}{\sum_{e \in \mathcal{A}} \text{val}(e)}$. | Aligns ML evaluation directly with AML financial asset recovery rather than unweighted event classification. |
| **Operational Latency Plateau** | Evaluated decision horizon grid $H \in \{0, 1\text{ s}, 1\text{ m}, 10\text{ m}, 1\text{ h}, 6\text{ h}, 24\text{ h}, 72\text{ h}, \infty\}$. Laundering motif resolution stabilizes within **10 minutes to 1 hour** ($H^*$), recovering up to $\text{SV}_{500} = 0.611$. | Demonstrates that financial institutions need not wait 24+ hours for batch graph updates; 10–60 minutes of buffering captures optimal graph context. |
| **Adaptive Multi-Horizon Policy $\pi$** | Designed a sequential multi-horizon threshold cascade across checkpoints $\{1\text{ s}, 10\text{ m}, 1\text{ h}, 6\text{ h}, 24\text{ h}\}$ calibrated via a budget simplex $\boldsymbol{\alpha}$ on validation data. Outperforms any static horizon ($+0.031$ SV gain at budget 100, $p < 0.05$). | Balances immediate payment interdiction (under EU Instant Payments Regulation limits) with deferred investigation for complex rings. |
| **Low-Latency Streaming GNN** | Time-Decayed Multigraph GNN (TD-GNN) achieves **$7.13\text{ ms}$ median inference latency** ($7.24\text{ ms}$ p95, $140\text{ ev/s}$) at batch $n=1$, and **$84,884\text{ ev/s}$** at batch $n=10,000$ on NVIDIA RTX 3080. | Satisfies banking sub-second SLA requirements without compromising causal multigraph structural representations. |

---

## System Architecture & Data Flow

The complete end-to-end streaming architecture is partitioned into four strictly causal stages:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STAGE 1: INGESTION & CHRONOLOGICAL SPLIT GATE                          │
│                                                                                                        │
│   Multigraph Stream e_i = (u_i, v_i, t_i, x_i, y_i) ──► Cryptographic Verification (SHA-256)          │
│                                                                  │                                     │
│                                                                  ▼                                     │
│   Chronological Day Cuts (60/20/20) ───────────────────► Split Gate: Train -> Val -> Test Core        │
│                                                          (Tail events t_i >= T_doc isolated)           │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    STAGE 2: CAUSAL VIEW ENGINE V(e_i, H)                               │
│                                                                                                        │
│   Sorted CSR Adjacency Index:                                                                          │
│   - Forward / Backward Adjacency Pointers (out.ptr, in.ptr, pair.ptr)                                  │
│   - Cumulative USD Prefix Sums (out.cum, in.cum) for O(1) Window Aggregation                           │
│   - Binary Search (np.searchsorted) for Exact Temporal Range Slicing                                   │
│                                                                                                        │
│   Causal Window Definition:                                                                            │
│     * H = 0 (Strict Past):  V(e_i, 0) = { e_j != e_i | t_j < t_i }                                     │
│     * H > 0 (Buffered):     V(e_i, H) = { e_j != e_i | t_j <= min(t_i + H, T_split) }                 │
└───────────────────────────────────┬───────────────────────────────────┬────────────────────────────────┘
                                    │                                   │
                                    ▼                                   ▼
┌───────────────────────────────────────────────────┐   ┌────────────────────────────────────────────────┐
│      STAGE 3a: STREAMING TD-GNN (PyTorch / PyG)   │   │   STAGE 3b: NUMBA CSR FEATURE ENGINE + LightGBM│
│                                                   │   │                                                │
│  1. Causal 2-Hop Neighborhood Sampler             │   │  1. Numba CSR Prefix-Sum Window Aggregation    │
│     Hop 1: s_p = 8 past, s_f = 8 forward          │   │     Past Windows: W in {6 h, 24 h, 72 h}       │
│     Hop 2: s_2p = 3 past, s_2f = 3 forward        │   │     Forward Window: [t_i, min(t_i+H, T_split)] │
│                                                   │   │                                                │
│  2. EdgeEncoder: Continuous (log USD, delta-t,    │   │  2. 126 Handcrafted Topological Features:      │
│     I[delta-t > 0], same-curr, self-loop)         │   │     * Base attributes (17 dims)                │
│     + Categorical Embeddings (cur_p, cur_r, fmt)  │   │     * Past window stats (72 dims)              │
│                                                   │   │     * Forward window stats (24 dims)           │
│  3. Direction-Aware Aggregation & Pooling:        │   │     * Pairwise reciprocity (8 dims)            │
│     Hop 2: phi_2(64 -> 64) -> Masked Pool (129-d) │   │     * Topological motifs (5 dims: 2-paths,     │
│     Hop 1: phi_1,{in,out}(322 -> 64) -> h in R^516│   │       3-cycles open/close, sibling convergence)│
│                                                   │   │                                                │
│  4. MLP Readout Head:                             │   │  3. LightGBM (126 features, num_leaves 63,     │
│     [z_seed || h] in R^580 -> 128 -> 64 -> 1      │   │     neg_rate 0.25, early-stopping AP)          │
└───────────────────────────────────┬───────────────┘   └───────────────────────┬────────────────────────┘
                                    │                                           │
                                    └───────────────────┬───────────────────────┘
                                                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 4: ADAPTIVE POLICY & ASSET INTERDICTION                             │
│                                                                                                        │
│   Multi-Horizon Checkpoints: H_j in {1 s, 10 min, 1 h, 6 h, 24 h}                                      │
│   Decision Policy pi: Fire alert if score s_j(e_i) >= tau_j (allocated daily budget alpha_j)           │
│   Immediate Hold (1 s) vs. Deferred Review (10 m - 24 h)                                               │
│                                                                                                        │
│   Asset Recovery Evaluation:                                                                           │
│   Interdiction Time T(A_k) = min_{e in A_k, flagged} (t_e + H)                                         │
│   Stoppable Value (SV_B) Intercepted Volume under Daily Review Budget B in {100, 500, 2000} Alerts/Day     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> A full vector publication architecture diagram is available in [paper/methodology_architecture.pdf](paper/methodology_architecture.pdf) (compiled from [paper/methodology_architecture.tex](paper/methodology_architecture.tex)).

---

## Benchmark Datasets & Cryptographic Integrity

All experiments are conducted across five benchmark distributions representing **84,727,458 financial transactions**:

| Dataset Suite | Total Transactions | Accounts | Illicit Attempts | Documented Days | Core Test Days | Raw File SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`HI-Small`** | 5,078,345 | 605,716 | 370 patterns (3,209 tx) | Day 0–9 (10 d) | Days 8–9 (2 d) | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| **`LI-Small`** | 6,924,049 | 823,283 | — | Day 0–9 (10 d) | Days 8–9 (2 d) | `f7a9940339c78b5d1476071505b5867c0f5319640b9c4279d13f531bf965319d` |
| **`HI-Medium`** | 31,898,238 | 3,747,511 | — | Day 0–15 (16 d) | Days 13–15 (3 d)| `3126afb8155e7c8815d62bc5370549a7b5ae6bf7dfe872b3cd7813e66a3d7ff5` |
| **`LI-Medium`** | 31,251,483 | 3,702,311 | — | Day 0–15 (16 d) | Days 13–15 (3 d)| `0fc89584453c97472b1bfde7ca746e21d7167140c22a6c4875272b197a4a3910` |
| **`SAML-D`** | 9,575,343 | 98,061 | Multi-Agent Simulation | Full span | Whole-day cuts | Cleaned; synthetic temporal multi-agent banking network |

### Pre-Registered Protocols & Sealed Scoring
To guarantee scientific reproducibility without post-hoc cherry-picking:
1. **Discovery vs. Confirmatory Protocol:** `HI-Small` served as the single discovery dataset on which feature sets, hyperparameters, and causal hypotheses were developed. Confirmatory suites (`LI-Small`, `HI-Medium`, `LI-Medium`, `SAML-D`) remained cryptographically sealed until pipelines were frozen.
2. **Sealed Scoring Gate ([Decision D-012](docs/DECISIONS.md)):** Confirmatory test splits were scored exactly once using `src/amlh/sealing.py`; unsealing events were logged with SHA-256 digests.
3. **Simulator Tail Artifact Isolation:** Analysis identified that synthetic transactions past day $T_{\text{doc}}$ contain severe structural anomalies (e.g. laundering prevalence dropping to zero or patterns collapsing). Evaluation strictly isolates the validated **Core** region from the **Tail** artifact.

---

## Repository Layout

```
aml-stream/
├── Real_Time_Paradox_Streaming_AML_KDD.pdf # [CONFERENCE] 10-page ACM SIGKDD camera-ready paper
├── Real_Time_Paradox_Streaming_AML.pdf     # [JOURNAL] 29-page Springer Nature research manuscript
├── Phase3_Report_CN.pdf                    # [COURSE REPORT] 16-page technical report (Rubric 30/30)
├── SUBMISSION_README.md                    # Course evaluation manifest & rubric crosswalk
├── pyproject.toml                          # Project build configuration and pytest options
├── requirements-kaggle.txt                 # Core dependencies (Polars, Numba, LightGBM 4.6.0, XGBoost)
├── requirements-mgnn.txt                   # GPU & Graph Neural Network stack (PyTorch 2.5.1, PyG 2.7.0)
│
├── src/amlh/                               # Core Python package
│   ├── features.py                         # Numba-accelerated streaming CSR feature engine (126 features)
│   ├── sampler.py                          # Causal 2-hop multigraph neighborhood sampler
│   ├── gnn.py                              # Time-Decayed Multigraph GNN (TD-GNN) architecture
│   ├── metrics.py                          # Stoppable Value (SV_B), PR-AUC, minority F1, budget curves
│   ├── isolate.py                          # Automated future permutation leakage gate
│   ├── queue_sim.py                        # Streaming queue arrival simulator & investigator buffer
│   ├── jobs_gbt.py                         # LightGBM training harness & adaptive policy calibration
│   ├── jobs_gnn.py                         # PyTorch GNN training harness with horizon-specific views
│   ├── evaluate.py                         # Multi-horizon sweep evaluation & metric aggregator
│   ├── logical.py                          # Logical-clock transaction features (order-preserving retiming)
│   ├── perturb.py                          # Adversarial timing and evasion attack generators
│   ├── sealing.py                          # Cryptographic test split isolation gate (Decision D-012)
│   ├── plans.py                            # Pre-registered experimental execution plans
│   ├── session.py                          # Distributed runner environment & GPU capability checks
│   ├── prepare.py                          # Raw dataset ingestion, hashing, and chronological day splits
│   └── mgnn.py                             # Multi-GNN baseline replication & cadence evaluation
│
├── tests/                                  # 313 automated unit, integration, and leakage tests
│   ├── test_isolate.py                     # Permutation tests verifying zero future lookahead
│   ├── test_features_reference.py          # Numba feature engine equivalence against reference tests
│   ├── test_pipeline.py                    # End-to-end streaming training & evaluation assertions
│   ├── test_queue_sim.py                   # Queue latency & M/M/1 queuing theory sanity checks
│   ├── test_perturb.py                     # Evasion attacks & temporal perturbation invariant checks
│   └── test_session_and_verdicts.py        # Experimental hypothesis confirmatory checks (C1–C5)
│
├── notebooks/                              # Modular, reproducible Marimo & Jupyter notebooks
│   ├── nb1.ipynb                           # NB1: Multi-GIN baseline replication & tail audit (GPU)
│   ├── nb2.ipynb                           # NB2: Multi-PNA baseline comparison & FraudGT audit (GPU)
│   ├── nb3.ipynb                           # NB3: Streaming TD-GNN decision horizon evaluation (GPU)
│   ├── nb4.ipynb                           # NB4: Multi-dataset confirmatory validation (CPU)
│   ├── nb5.ipynb                           # NB5: Adversarial timing & evasion attacks (CPU)
│   ├── merge.ipynb                         # Multi-shard consolidation & automated hypothesis verdicts
│   └── README.md                           # Detailed execution instructions for Kaggle and Colab
│
├── scripts/                                # Analysis and automation scripts
│   ├── 08_analyze.py                       # Ingests run ledgers; generates LaTeX tables, macros, and figures
│   ├── build_notebooks.py                  # Synchronizes Python sources into standalone notebooks
│   ├── make_code_dataset.py                # Packages source code for distributed cloud execution
│   └── polish_paper_figures.py             # Formats publication vector figures and plots
│
├── paper/                                  # LaTeX publication source files
│   ├── kdd/                                # ACM SIGKDD conference paper source (main.tex)
│   ├── springer/                           # Springer Nature journal paper source
│   ├── methodology_architecture.tex        # TikZ / PGF source for the 4-stage pipeline architecture
│   ├── methodology_architecture.pdf        # Standalone compiled architecture figure
│   └── generated/                          # 85 automated LaTeX macros, tables, and PDF figures
│
├── docs/                                   # Formal specifications and decision ledger
│   ├── DECISIONS.md                        # Pre-registered protocol decisions (append-only ledger)
│   ├── METHODS.md                          # Mathematical formulations and module mappings
│   └── SPEC.md                             # Data schemas, row counts, and cryptographic hashes
│
└── results/                                # Verified experimental execution ledgers & results
    ├── merged/                             # Consolidated cross-benchmark experiment outputs
    └── shard11_NB4_confirmatory/           # Sealed confirmatory run outputs and hypothesis logs
```

---

## Installation & Environment Setup

### 1. Prerequisites
- **Operating System:** Linux, macOS, or Windows (x86_64)
- **Python:** Version 3.10, 3.11, or 3.12
- **Hardware:**
  - *Feature extraction & GBT baselines:* Any modern multi-core CPU (8+ GB RAM; 32+ GB recommended for Medium datasets).
  - *GNN training & inference:* CUDA-capable GPU (tested on NVIDIA RTX 3080, T4, and RTX PRO 6000 Blackwell).

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shahoud867/aml-stream.git
cd aml-stream

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Dependency Installation

**Option A: Core Stack (CPU / Gradient Boosted Trees / Feature Engine)**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
# Or via pinned requirements:
pip install -r requirements-kaggle.txt
```

**Option B: Full Stack (GNN / PyTorch / PyG GPU Acceleration)**
```bash
# Install PyTorch with CUDA support (example: CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric extensions
pip install torch-geometric==2.7.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Install remaining dependencies
pip install -r requirements-mgnn.txt
```

---

## Quickstart & Python API Usage

The core Python package `src/amlh` is modular, allowing users to extract causal features, sample multigraph neighborhoods, or evaluate asset-weighted metrics directly.

### 1. Causal Horizon Filtering & Numba Feature Extraction

Extract 126 streaming features using sorted CSR arrays without lookahead leakage:

```python
import numpy as np
from amlh.features import compute_features_at_h, feature_names

# event_id, src, dst, timestamp (seconds), amount, currencies, format
# Canonical chronological multigraph arrays
t = np.array([100, 150, 200, 350, 400], dtype=np.int64)
src = np.array([0, 1, 0, 2, 1], dtype=np.int64)
dst = np.array([1, 2, 2, 0, 3], dtype=np.int64)
amt_usd = np.array([1200.0, 50000.0, 150.0, 48000.0, 12000.0], dtype=np.float32)

# Set decision horizon H (e.g., 3600 seconds = 1 hour)
H_seconds = 3600
end_split_sec = 86400

# Compute feature matrix (N_events x 126)
# Window aggregation over {6h, 24h, 72h} past and [t, min(t+H, T_split)] forward
names = feature_names()
print(f"Total features: {len(names)}")  # 126 features
```

### 2. Causal 2-Hop Multigraph Sampling & TD-GNN Inference

Sample causal neighborhoods where second-hop nodes obey the seed event's horizon view $\mathcal{V}(e, H)$:

```python
import torch
from amlh.sampler import sample_batch
from amlh.gnn import TemporalDirectedGNN

# Initialize direction-aware GNN (d = 64 hidden dimensions)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalDirectedGNN(n_cur=20, n_fmt=10, d=64).to(device)

# Seed events: (seed_e, hop1_e, hop1_mask, hop2_e, hop2_mask)
# Predictions are invariant to future perturbations outside V(e, H)
model.eval()
with torch.no_grad():
    # Forward pass outputs streaming risk scores s_H(e)
    # scores = model(seed_e, hop1_e, hop1_mask, hop2_e, hop2_mask)
    pass
```

### 3. Stoppable Value ($\text{SV}_B$) and Operational Alert Policies

Compute asset interdiction subject to a daily investigator budget:

```python
import numpy as np
from amlh.metrics import attempt_outcomes, threshold_for_budget

# Validation scores to calibrate daily alert budget (e.g. 500 alerts / day)
val_scores = np.random.uniform(0, 1, size=100000)
val_days = 2.0
tau_500 = threshold_for_budget(val_scores, val_days=val_days, alerts_per_day=500.0)

# Scored test attempts: evaluate interdiction volume
att_ids = np.array([1, 1, 1, 2, 2])
ev_t = np.array([1000, 3600, 7200, 1500, 2000], dtype=np.int64)
ev_usd = np.array([10000.0, 9500.0, 9000.0, 500.0, 450.0], dtype=np.float64)
test_scores = np.array([0.85, 0.40, 0.10, 0.90, 0.80], dtype=np.float64)

# Horizon H = 600 s (10 min triage buffer)
outcomes = attempt_outcomes(att_ids, ev_t, ev_usd, test_scores, tau=tau_500, H=600)
print(f"Stoppable Value (SV): {outcomes['sv']:.4f}")
print(f"Stopped Attempts: {outcomes['stopped_attempts']} / {outcomes['total_attempts']}")
```

---

## Automated Test Suite & Leakage Gate

The repository includes **313 automated tests** enforcing mathematical invariants, zero-lookahead causality, queuing simulation limits, and metric stability:

```bash
# Run the complete test suite
python -m pytest tests/
```

### Targeted Test Suites

```bash
# 1. Leakage Isolation Gate: permutes future events and asserts score invariance
python -m pytest tests/test_isolate.py -v

# 2. Feature Reference Check: asserts Numba engine matches slow reference formulas
python -m pytest tests/test_features_reference.py -v

# 3. Queue Simulator: verifies M/M/1 queuing theory and triage delay bounds
python -m pytest tests/test_queue_sim.py -v

# 4. End-to-End Pipeline & Split Integrity
python -m pytest tests/test_pipeline.py -v
```

> [!NOTE]
> All 313 unit and integration tests execute in approximately 3–4 minutes with a 100% pass rate.

---

## Reproducing Paper Experiments & Figures

All empirical tables, LaTeX macros, and vector plots published in the manuscripts are generated directly from execution ledgers using automated analysis tools:

```bash
# Ingest execution logs from results/ and regenerate all paper assets
python scripts/08_analyze.py paper
```

This single command updates:
- `paper/generated/macros.tex` — 85 automated numerical LaTeX macros ensuring text-data consistency.
- `paper/generated/fig_horizon.pdf` — Stoppable value vs. decision horizon curves.
- `paper/generated/fig_tail.pdf` — Empirical distribution anomaly plots isolating simulator tail artifacts.
- `paper/generated/tab_*.tex` — Complete LaTeX tables reporting hypothesis verdicts and benchmark contrasts.

---

## Cloud Execution & Distributed Notebooks

Experiments are sharded into independent, stateless notebooks designed to run seamlessly on **Kaggle** or **Google Colab**:

| Notebook | Focus & Methodology | Primary Hardware |
| :--- | :--- | :--- |
| [`notebooks/nb1.ipynb`](notebooks/nb1.ipynb) | Multi-GIN baseline replication, whole-window audit, and tail artifact isolation | GPU (T4 / P100 / RTX) |
| [`notebooks/nb2.ipynb`](notebooks/nb2.ipynb) | Multi-PNA multigraph baseline, LI-Small evaluation, and FraudGT comparison | GPU (T4 / P100 / RTX) |
| [`notebooks/nb3.ipynb`](notebooks/nb3.ipynb) | Streaming TD-GNN across decision horizon grid $H \in \{0, 1\text{ s}, \dots, \infty\}$ | GPU (T4 / P100 / RTX) |
| [`notebooks/nb4.ipynb`](notebooks/nb4.ipynb) | Multi-dataset confirmatory validation (LightGBM on sealed benchmarks) | CPU (Multi-core) |
| [`notebooks/nb5.ipynb`](notebooks/nb5.ipynb) | Adversarial timing attacks, logical-clock evaluation, and evasion resistance | CPU (Multi-core) |
| [`notebooks/merge.ipynb`](notebooks/merge.ipynb) | Consolidates distributed shards, validates checksums, and computes verdicts | CPU (Fast) |

For complete instructions on mounting storage, managing API credentials, running pilot passes, and merging shard outputs, refer to [`notebooks/README.md`](notebooks/README.md).

---

## Core Package Reference (`src/amlh`)

| Module | Core Functionality & Exported Classes |
| :--- | :--- |
| [`features.py`](src/amlh/features.py) | High-performance Numba CSR window scan computing 126 features across past $(6\text{ h}, 24\text{ h}, 72\text{ h})$ and bounded forward intervals. |
| [`sampler.py`](src/amlh/sampler.py) | Causal multigraph sampler producing 2-hop neighborhood tensors constrained to seed event horizon view $\mathcal{V}(e, H)$. |
| [`gnn.py`](src/amlh/gnn.py) | Direction-separated `TemporalDirectedGNN`, `EdgeEncoder`, and masked pooling modules ($\mu, \max, \log(1+c)$). |
| [`metrics.py`](src/amlh/metrics.py) | Metric library for Stoppable Value ($\text{SV}_B$), budget-constrained thresholding, PR-AUC, and paired bootstrap statistics. |
| [`isolate.py`](src/amlh/isolate.py) | Automated future permutation test suite verifying temporal invariance and preventing data leakage. |
| [`jobs_gbt.py`](src/amlh/jobs_gbt.py) | Training pipelines for LightGBM with subsampling (`neg_rate = 0.25`), and calibration of multi-horizon adaptive policies $\pi$. |
| [`jobs_gnn.py`](src/amlh/jobs_gnn.py) | PyTorch training harness with horizon-specific causal views, cosine annealing, and evaluation checkpoints. |
| [`queue_sim.py`](src/amlh/queue_sim.py) | Discrete-event queuing simulator modeling investigator triage queues, batch latency, and system sojourn times. |
| [`logical.py`](src/amlh/logical.py) | Order-preserving logical clock features robust against adversarial burst and stretch retiming attacks. |
| [`perturb.py`](src/amlh/perturb.py) | Adversarial temporal perturbation engine generating realistic transaction retiming and evasion patterns. |
| [`sealing.py`](src/amlh/sealing.py) | Cryptographic enforcement engine isolating test splits on confirmatory suites until final verification. |
| [`prepare.py`](src/amlh/prepare.py) | Raw CSV ingestion, cryptographic SHA-256 validation, and whole-day chronological split partitioning. |
| [`mgnn.py`](src/amlh/mgnn.py) | Upstream Multi-GNN baseline training, checkpointing, and evaluation adapter. |

---

## Research Publications & Manuscripts

This repository accompanies two research manuscripts detailing the theoretical framework, algorithmic proofs, and empirical validation:

1. **Conference Paper (ACM SIGKDD Format, 10 pages):**  
   [`Real_Time_Paradox_Streaming_AML_KDD.pdf`](Real_Time_Paradox_Streaming_AML_KDD.pdf)  
   *Title:* "The Real-Time Paradox in Graph-Based Anti-Money Laundering: Auditing Temporal Lookahead and Restoring Stoppable Value"  
   *Venue:* ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)  
   *LaTeX Source:* [`paper/kdd/main.tex`](paper/kdd/main.tex)

2. **Journal Article (Springer Nature Format, 29 pages):**  
   [`Real_Time_Paradox_Streaming_AML.pdf`](Real_Time_Paradox_Streaming_AML.pdf)  
   *Title:* "The Real-Time Paradox in Graph-Based Anti-Money Laundering: Auditing Temporal Lookahead and Restoring Stoppable Value"  
   *LaTeX Source:* [`paper/springer/Real_Time_Paradox_Streaming_AML.tex`](paper/springer/Real_Time_Paradox_Streaming_AML.tex)

3. **Academic Course Technical Report (16 pages):**  
   [`Phase3_Report_CN.pdf`](Phase3_Report_CN.pdf)  
   *Course:* Computer Networks (Semester 7), Department of AI & Data Science, FAST-NUCES Islamabad.  
   *Evaluation Crosswalk:* [`SUBMISSION_README.md`](SUBMISSION_README.md)

---

## Citation

If you find this work, codebase, or the Stoppable Value metric useful in your research, please cite our manuscripts:

```bibtex
@inproceedings{shahid2026realtimeparadox,
  author    = {Shahid, Shahoud and Shahzad, Saif},
  title     = {The Real-Time Paradox in Graph-Based Anti-Money Laundering: Auditing Temporal Lookahead and Restoring Stoppable Value},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  pages     = {1--10}
}

@article{shahid2026streamingaml_journal,
  author    = {Shahid, Shahoud and Shahzad, Saif},
  title     = {The Real-Time Paradox in Graph-Based Anti-Money Laundering: Auditing Temporal Lookahead and Restoring Stoppable Value},
  journal   = {Springer Nature Computing Research},
  year      = {2026},
  volume    = {1},
  pages     = {1--29}
}
```

---

## Authors & Academic Affiliation

- **Shahoud Shahid** (23I-2515) — *Department of Artificial Intelligence & Data Science*
- **Saif Shahzad** (23I-2634) — *Department of Artificial Intelligence & Data Science*

**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan  
**Department:** Department of Artificial Intelligence & Data Science  
**Project Repository:** [https://github.com/Shahoud867/aml-stream](https://github.com/Shahoud867/aml-stream)

---

## License & Acknowledgments

- **Codebase License:** The core `aml-stream` codebase and custom modules are open-sourced under the [MIT License](LICENSE).
- **Third-Party Baselines:** Upstream baseline code in `third_party/Multi-GNN` and `third_party/FraudGT` retains original licensing terms from their respective authors.
- **Datasets:** Synthetic AML transaction datasets are released by IBM Research under their respective distribution licenses.
