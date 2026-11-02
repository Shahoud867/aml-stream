"""The GFP job needs snapml, which may not exist for the platform's Python. Its failure is reported but must not fail the pilot."""
import pytest

import amlh.session as sess
from amlh.runner import JobSpec
from test_notebooks import SWEEP, _common, _small_input, restore_globals  # noqa: F401 (autouse fixture)


def test_an_optional_job_that_fails_is_reported_but_does_not_fail_the_pilot(small_data, tmp_path):
    try:
        import snapml  # noqa: F401
        pytest.skip("snapml is installed here, so the job would succeed")
    except ImportError:
        pass
    gfp = JobSpec("pilot-nb5-gfp", "gfp_baseline", {"dataset": "HI-Small", "protocol": "released", "batch": 2048, "seed": 0})
    report = sess.run_pilot("NB5", specs=[JobSpec("pilot-sweep", "gbt_sweep", SWEEP), gfp], kill_point=("pilot-sweep", 1), budget_hours=1.0,
                            **_common(_small_input(small_data, tmp_path), tmp_path))
    assert report["pass"]
    assert any("Optional job" in g[0] and "pilot-nb5-gfp" in g[0] and g[1] is None and "snapml" in g[2] for g in report["gates"])


def test_a_required_job_that_fails_still_fails_the_pilot(small_data, tmp_path):
    bad = JobSpec("pilot-required", "gbt_sweep", {**SWEEP, "dataset": "HI-Small", "H": "not-a-horizon"})
    report = sess.run_pilot("NB4", specs=[JobSpec("pilot-sweep", "gbt_sweep", SWEEP), bad], kill_point=("pilot-sweep", 1), budget_hours=1.0,
                            **_common(_small_input(small_data, tmp_path), tmp_path))
    assert not report["pass"] and "pilot-required" in report["gates"][0][2]
