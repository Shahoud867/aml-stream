"""The datatable stand-in, run on a small script that makes the same kinds of calls as the official formatter (the official script itself
is never run here), and the pilot report's failure section."""
import importlib.util
from pathlib import Path

import pytest

import amlh.session as sess
from amlh.mgnn import run_official_formatter
from amlh.runner import JobSpec
from test_notebooks import SWEEP, _common, _small_input, restore_globals  # noqa: F401 (autouse fixture)

SCRIPT = '''
import datatable as dt
from datatable import f, join, sort
import os, sys
inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"
raw = dt.fread(inPath, columns=dt.str32)
with open(outPath, "w") as w:
    w.write("EdgeID,x,Timestamp,y\\n")
    for i in range(raw.nrows):
        w.write("%d,%s,%d,%s\\n" % (i, raw[i, "a"] + raw[i, 3], int(raw[i, "ts"]), raw[i, 4]))
formatted = dt.fread(outPath)
formatted = formatted[:, :, sort(2)]
formatted.to_csv(outPath)
'''


def test_the_stand_in_reads_indexes_sorts_stably_and_writes(tmp_path):
    if importlib.util.find_spec("datatable") is not None:
        pytest.skip("the real datatable is installed, so the stand-in would not be used")
    (tmp_path / "script.py").write_text(SCRIPT)
    data = tmp_path / "data"
    data.mkdir()
    # Two columns are both called "a", as the real file has two "Account" columns, and equal timestamps must keep their order.
    (data / "in.csv").write_text("ts,a,b,a,v\n5,p,x,q,v0\n3,r,x,s,v1\n3,t,x,u,v2\n1,w,x,y,v3\n")
    code, stderr, used = run_official_formatter(tmp_path / "script.py", data / "in.csv", install=False)
    assert (code, used) == (0, "shim"), stderr
    rows = (data / "formatted_transactions.csv").read_text().splitlines()
    assert rows[0] == "EdgeID,x,Timestamp,y"
    assert [r.split(",")[0] for r in rows[1:]] == ["3", "1", "2", "0"], "sorted by timestamp, ties in original order"
    assert rows[1].split(",")[1] == "wy"  # raw[i, "a"] is the first column named a, and raw[i, 3] is the second one, read by position


def test_a_failing_official_script_reports_its_error(tmp_path):
    (tmp_path / "script.py").write_text("raise ValueError('boom from the script')\n")
    (tmp_path / "in.csv").write_text("a\n1\n")
    code, stderr, _ = run_official_formatter(tmp_path / "script.py", tmp_path / "in.csv", install=False)
    assert code != 0 and "boom from the script" in stderr


def test_the_pilot_report_carries_the_failure_text(small_data, tmp_path):
    bad = JobSpec("pilot-required", "gbt_sweep", {**SWEEP, "H": "not-a-horizon"})
    report = sess.run_pilot("NB4", specs=[JobSpec("pilot-sweep", "gbt_sweep", SWEEP), bad], kill_point=("pilot-sweep", 1), budget_hours=1.0,
                            **_common(_small_input(small_data, tmp_path), tmp_path))
    assert not report["pass"]
    assert [f["job"] for f in report["failures"]] == ["pilot-required"] and "not-a-horizon" in report["failures"][0]["reason"]
    text = Path(tmp_path / "w/aml_pilot/PILOT_REPORT.md").read_text()
    assert "## Failed or unfinished jobs" in text and "### pilot-required" in text and "Traceback" in text
