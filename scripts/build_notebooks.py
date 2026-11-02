"""Writes notebooks/nb1.ipynb to nb5.ipynb and merge.ipynb. They only orchestrate: the logic lives in amlh, which is tested.

Each notebook runs on Kaggle or on Colab and detects which. Rebuild after any code change, because each notebook carries
the code hash it expects:
    python scripts/build_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.runner import code_hash  # noqa: E402

CODE_HASH = code_hash(ROOT)
AML = "`ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (holds all four AMLworld datasets)"
SAML = "`berkanoztas/synthetic-transaction-monitoring-dataset-aml` (`SAML-D.csv`)"

NOTEBOOKS = {
    "NB1": {
        "title": "Official Multi-GIN+EU on HI-Small",
        "goal": "Does the published result depend on the tail and on look-ahead, in the published model itself?",
        "hardware": "GPU T4 x2", "colab_hardware": "T4 GPU", "tiers": ("P0", "P1"), "colab_tiers": ("P0", "P1"), "attach": [AML],
        "estimate": "30 to 65 GPU hours (an estimate until the pilot measures it)",
        "jobs": ["**P0:** formatter check; 3 seeds of 100 epochs; replay at 24 h, 6 h and 1 h cadences",
                 "**P1:** retraining at 24 h and 1 h cadences, scored at the same cadence; training-set normalisation"],
    },
    "NB2": {
        "title": "Official models, breadth",
        "goal": "Does NB1's result hold across model and dataset?",
        "hardware": "GPU T4 x2", "colab_hardware": "T4 GPU", "tiers": ("P0", "P1"), "colab_tiers": ("P0", "P1"), "attach": [AML],
        "estimate": "30 to 60 GPU hours (an estimate until the pilot measures it)",
        "jobs": ["**P0:** Multi-PNA+EU on HI-Small and Multi-GIN+EU on LI-Small, 2 seeds each, with replay",
                 "**P1:** Multi-PNA+EU on LI-Small; FraudGT region audit on HI-Small, at most 20 training hours",
                 "**P2 (only if the pilot shows it fits):** Multi-GIN+EU on HI-Medium"],
    },
    "NB3": {
        "title": "TD-GNN across horizons",
        "goal": "A learned model whose view can be restricted exactly, for fair per-horizon curves.",
        "hardware": "GPU T4 x2", "colab_hardware": "T4 GPU", "tiers": ("P0", "P1"), "colab_tiers": ("P0", "P1"), "attach": [AML],
        "estimate": "20 to 40 GPU hours (the least reliable estimate: no full run has finished)",
        "jobs": ["**P0:** leakage check; 5 horizons and 3 seeds on HI-Small and LI-Small; timing attack at 3 horizons",
                 "**P1:** remaining horizons; three ablations; attack on LI-Small; GPU service time"],
    },
    "NB4": {
        "title": "Boosted-tree confirmation",
        "goal": "Test the frozen hypotheses on unseen data. **The leakage check runs first:** if a shuffled run scores above prevalence, stop and investigate.",
        "hardware": "None (CPU)", "colab_hardware": "None (CPU), High-RAM if offered", "tiers": ("P0", "P1"), "colab_tiers": ("P0",), "attach": [AML, SAML],
        "estimate": "60 to 90 CPU hours (an estimate until the pilot measures it)",
        "jobs": ["**P0:** leakage check; 9 horizons, 2 protocols and 3 seeds on HI-Small and LI-Small; alert policy",
                 "**P1:** the same on SAML-D, HI-Medium and LI-Medium (the Medium files need more than 12 GB of RAM)",
                 "**P2:** rolling-origin folds; Ethereum (needs its own input)"],
    },
    "NB5": {
        "title": "Timing robustness and attribution",
        "goal": "RQ4, the main novelty: can timing alone defeat a wall-clock detector, and do logical-clock features resist it?",
        "hardware": "None (CPU)", "colab_hardware": "None (CPU), High-RAM if offered", "tiers": ("P0", "P1"), "colab_tiers": ("P0",), "attach": [AML],
        "estimate": "40 to 70 CPU hours (an estimate until the pilot measures it)",
        "jobs": ["**P0:** logical and combined detectors; attacks at 4 horizons on 3 families; feature ablations on HI-Small",
                 "**P1:** the same on LI-Small and HI-Medium; GFP baseline; CPU service time"],
    },
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


PATHS = """
# Where things live. Kaggle attaches datasets under /kaggle/input. Colab has none, so it downloads the raw files and keeps its output in Google Drive,
# which means a disconnected session resumes from where it stopped.
import os, sys
from pathlib import Path

PLATFORM = os.environ.get('AMLH_PLATFORM') or ('colab' if 'google.colab' in sys.modules else 'kaggle')
CONTENT = Path('/content')
DRIVE = CONTENT / 'drive' / 'MyDrive'
if PLATFORM == 'colab':
    if not DRIVE.is_dir():
        from google.colab import drive
        drive.mount(str(CONTENT / 'drive'))
    HOME = DRIVE / 'aml'                       # put the code package (aml-code-*.zip) here
    INPUT = CONTENT / 'aml_data'               # raw files, downloaded at the start of each session
    CODE_SEARCH = [HOME, CONTENT]
    OUTPUTS = HOME / 'collect'                 # the merge notebook reads every notebook's output from here
    WORKING = HOME / (f"out_{NOTEBOOK.lower()}" + (f"_shard{SHARD[0]}of{SHARD[1]}" if SHARD else ''))
    RESTORE = []                               # the output already stays in Drive
    WORKING.mkdir(parents=True, exist_ok=True)
else:
    INPUT = Path('/kaggle/input')
    CODE_SEARCH = [INPUT]
    OUTPUTS = INPUT
    WORKING = Path('/kaggle/working')
    RESTORE = None                             # attached earlier outputs are found under /kaggle/input
print(PLATFORM, 'input', INPUT, 'output', WORKING)
"""

BOOTSTRAP = """
# Unpacks the code package outside the output folder, so it is not exported. Kaggle may unzip an uploaded zip, so both forms are handled.
import glob, shutil, subprocess, sys, time, zipfile

CODE = Path('/tmp/aml_code')
shutil.rmtree(CODE, ignore_errors=True)
if PLATFORM == 'colab':
    found = {p.name: p for root in CODE_SEARCH for p in sorted(Path(root).glob('aml-code-*.zip'))}  # the same file in two places counts once
    zips, folders = [str(p) for p in found.values()], []
else:
    zips = glob.glob(str(INPUT / '**' / 'aml-code-*.zip'), recursive=True)
    folders = [p.parents[2] for p in INPUT.glob('**/src/amlh/runner.py')]
assert len(zips) + len(folders) == 1, f'attach or upload exactly one code package; found {zips + [str(f) for f in folders]}'
if zips:
    zipfile.ZipFile(zips[0]).extractall(CODE)
else:
    shutil.copytree(folders[0], CODE)
sys.path.insert(0, str(CODE / 'src'))

from amlh.runner import code_hash
assert code_hash(CODE) == EXPECTED_CODE_HASH, f'this notebook expects code {EXPECTED_CODE_HASH}, but the package is {code_hash(CODE)}'
if INSTALL:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE / 'requirements-kaggle.txt')], check=True)
print('code version', EXPECTED_CODE_HASH)
"""


def notebook(name: str, spec: dict) -> dict:
    attach = "\n".join(f"- {x}" for x in spec["attach"]) + "\n- The code package (a private dataset holding `aml-code-*.zip`)"
    jobs = "\n".join(f"- {x}" for x in spec["jobs"])
    tiers, colab_tiers = repr(spec["tiers"]), repr(spec["colab_tiers"])
    gpu_note = "Keep `GPUS = 1`. On Kaggle, a pilot shows whether T4 x2 costs double; if it does not, `GPUS = 2` runs two jobs at once." if spec["hardware"].startswith("GPU") else "This notebook needs no GPU, so it spends no GPU quota."
    colab_p1 = "" if spec["colab_tiers"] == spec["tiers"] else "\n**On Colab only P0 runs by default,** because the P1 jobs load Medium files and a free runtime has about 12 GB of RAM. Add `'P1'` to `TIERS` only on a runtime with 32 GB or more."
    return {"cells": [
        md(f"""
# {name}: {spec['title']}

**Goal:** {spec['goal']}

{jobs}

**Time:** {spec['estimate']}.

## How to run

### On Kaggle
**Settings (right-hand panel):** Accelerator **{spec['hardware']}**; Internet **on**.

**Attach as inputs:**
{attach}

### On Colab
**Runtime, Change runtime type:** **{spec['colab_hardware']}**. Keep this tab open: an idle or closed tab ends the session.

**One-time setup in Google Drive:** make a folder `aml` and put `aml-code-*.zip` in it. The notebook mounts Drive, downloads the raw files it needs, and keeps its output in `aml/out_{name.lower()}`. If a session disconnects, use **Runtime, Run all** again: finished jobs are not repeated, and interrupted ones continue from their checkpoints.

If the download is blocked, put the raw files in `aml/data` in Drive and the notebook copies them from there.{colab_p1}

### Either platform
**Step 1, pilot (under 2 hours).** Leave `MODE = 'pilot'` and run all cells. Read `aml_pilot/PILOT_REPORT.md` at the end. Go on only if it says PASS.

**Step 2, production.** Set `MODE = 'production'` and run all cells. The run stops cleanly at the time budget.

**Step 3, continue.** If the last cell says jobs remain, run again. On Kaggle, first attach this notebook's output as an input (**Add Input, Notebook Output**).

**Step 4, hand over.** On Kaggle, publish the final output's `aml_run` folder as a private dataset named `aml-out-{name.lower()}-v1`. On Colab, share the `out_{name.lower()}` folder.

{gpu_note}

**Rules:** do not edit the code, change a job's settings or copy in a token. A job whose code or settings changed refuses to resume, and that is deliberate. Test metrics for sealed datasets are never shown here: only the merge step computes them.
"""),
        code(f"""
# Settings. These are the only values to change.
import os, sys
NOTEBOOK = '{name}'
MODE = 'pilot'            # 'pilot' first; 'production' only after PILOT_REPORT.md says PASS
_COLAB = bool(os.environ.get('AMLH_PLATFORM') == 'colab' or 'google.colab' in sys.modules)
TIERS = {colab_tiers} if _COLAB else {tiers}   # P0 runs first. P0 is what the 15 November milestone needs
BUDGET_HOURS = 5.0 if _COLAB else 10.5   # stops cleanly. Colab sessions are shorter and less predictable than Kaggle's 12 h
GPUS = 1                  # number of GPU worker processes
SHARD = None              # (i, n) runs every n-th job chain, for taking overflow on another machine
MIRROR_SLUG = None        # Kaggle only: 'account/private-dataset' for hourly copies; needs Kaggle Secrets KAGGLE_USERNAME and KAGGLE_KEY
RUN_TESTS = True          # a failing test stops the notebook before it spends quota
INSTALL = True
EXPECTED_CODE_HASH = '{CODE_HASH}'
"""),
        code(PATHS),
        code(BOOTSTRAP),
        code("""
# Colab has no attached datasets, so this fetches the raw files (and checks each is whole). Kaggle already has them.
if PLATFORM == 'colab':
    from amlh.fetch import ensure_raw
    from amlh.session import datasets_of, pilot_specs
    if MODE == 'pilot':
        names = sorted({j.cfg['dataset'] for j in pilot_specs(NOTEBOOK)[0] if 'dataset' in j.cfg})
    else:
        names = datasets_of(NOTEBOOK, TIERS)
    print('raw files needed:', names)
    if MODE == 'pilot':   # a file that cannot be fetched is skipped and reported by the pilot, not fatal
        for n in names:
            try:
                ensure_raw([n], INPUT, backup=HOME / 'data')
            except Exception as exc:
                print('could not fetch', n, '-', exc)
    else:
        ensure_raw(names, INPUT, backup=HOME / 'data')
"""),
        code("""
from amlh.session import setup_environment, run_preflight_tests
print(setup_environment(NOTEBOOK) if INSTALL else 'install skipped')
if RUN_TESTS:
    run_preflight_tests(CODE)
"""),
        code("""
# Kaggle only. Tokens stay in Kaggle Secrets. They are never written to a file or printed.
if MIRROR_SLUG and PLATFORM == 'kaggle':
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
"""),
        code("""
from amlh.session import print_summary, run_notebook, run_pilot, run_workers

start = time.time()
if MODE == 'pilot':
    report = run_pilot(NOTEBOOK, input_root=INPUT, working=WORKING)
    assert report['pass'], 'pilot failed: read aml_pilot/PILOT_REPORT.md before production'
elif GPUS > 1:
    run_workers(NOTEBOOK, GPUS, tiers=TIERS, budget_hours=BUDGET_HOURS, input_root=INPUT, working=WORKING, mirror_slug=MIRROR_SLUG, restore_roots=RESTORE)
else:
    run_notebook(NOTEBOOK, tiers=TIERS, budget_hours=BUDGET_HOURS, input_root=INPUT, working=WORKING, shard=SHARD, mirror_slug=MIRROR_SLUG, restore_roots=RESTORE)
print(f'session used {(time.time() - start) / 3600:.2f} h')
print_summary(WORKING)
"""),
        md("## Final verification\n\nPrints a pass or fail for each check, and what to do next."),
        code("""
from amlh.session import final_checklist
if MODE == 'production':
    final_checklist(NOTEBOOK, TIERS, WORKING, SHARD, platform=PLATFORM)
"""),
    ], "metadata": META, "nbformat": 4, "nbformat_minor": 5}


def merge_notebook() -> dict:
    return {"cells": [
        md("""
# Merge: verify the five notebooks and compute verdicts

Runs on CPU in minutes. This is the only place sealed test metrics are computed. A sealed dataset stays sealed until its planned sweep jobs have all finished, and every unsealing is logged. **Do not rerun on a changed configuration and report the new numbers as confirmation:** a rerun under different code is labelled exploratory.

**On Kaggle, attach as inputs:** the code package, each notebook's output dataset (`aml-out-nbX-vN`), and the raw datasets.

**On Colab:** put the code package in Drive `aml`, and copy every notebook's output folder (`out_nb1` and so on) into Drive `aml/collect`. The raw files are downloaded for you. A free runtime has about 12 GB of RAM, so by default only HI-Small and LI-Small are merged there. Add the Medium datasets to `DATASETS` only on a larger runtime.
"""),
        code(f"""
import os, sys
NOTEBOOK = 'merge'
SHARD = None
_COLAB = bool(os.environ.get('AMLH_PLATFORM') == 'colab' or 'google.colab' in sys.modules)
DATASETS = ['HI-Small', 'LI-Small'] if _COLAB else ['HI-Small', 'LI-Small', 'SAML-D', 'HI-Medium', 'LI-Medium']
EXPECTED_CODE_HASH = '{CODE_HASH}'
INSTALL = True
"""),
        code(PATHS),
        code(BOOTSTRAP),
        code("""
if PLATFORM == 'colab':
    from amlh.fetch import ensure_raw
    ensure_raw(DATASETS, INPUT, backup=HOME / 'data')
"""),
        code("""
from amlh.jobs_gbt import Paths
from amlh.merge import merge
from amlh.session import find_raw, verify_raw

available = [d for d in DATASETS if list(INPUT.rglob(f'{d}_Trans.csv')) or (d == 'SAML-D' and list(INPUT.rglob('SAML-D.csv')))]
Paths.raw = find_raw(available, INPUT)
Paths.cache = Path('/tmp/aml_cache')
Paths.cache.mkdir(parents=True, exist_ok=True)
for d in available:
    print(verify_raw(d, Paths.raw[d]))
"""),
        code("""
OUT = WORKING / 'merged'
report = merge([OUTPUTS], OUT, available, analysis_root=CODE)
print((OUT / 'MERGE_REPORT.md').read_text())
"""),
        md("""
## Tables and figures

`08_analyze.py` regenerates every table, figure and in-text number from the exported results. It currently covers the Small datasets. Its output is copied to `merged/report`.
"""),
        code("""
os.makedirs(CODE / 'data', exist_ok=True)
link = CODE / 'data' / 'processed'
if not link.exists():
    os.symlink(Paths.cache, link)
subprocess.run([sys.executable, str(CODE / 'scripts' / '02b_audit_all.py')], check=False)
result = subprocess.run([sys.executable, str(CODE / 'scripts' / '08_analyze.py'), str(OUT)], capture_output=True, text=True)
print(result.stdout[-1500:], result.stderr[-1500:])
if (CODE / 'results' / 'report').exists():
    shutil.copytree(CODE / 'results' / 'report', OUT / 'report', dirs_exist_ok=True)
"""),
    ], "metadata": META, "nbformat": 4, "nbformat_minor": 5}


META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}

if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, spec in NOTEBOOKS.items():
        (out / f"{name.lower()}.ipynb").write_text(json.dumps(notebook(name, spec), indent=1))
    (out / "merge.ipynb").write_text(json.dumps(merge_notebook(), indent=1))
    print("code version", CODE_HASH, "wrote", ", ".join(sorted(p.name for p in out.glob("*.ipynb"))))
