import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[2] / "examples/evaluation_demo.py"


def test_synthetic_demo_expected_values():
    result = runpy.run_path(str(DEMO))["run_demo"]()
    assert result["synthetic_teaching_example"] is True
    assert (result["TP"], result["FP"], result["eligible_seizures"]) == (1, 1, 2)
    assert result["event_sensitivity"] == 0.5
    assert result["background_seconds"] == 3580.0
    assert result["FA_per_24h"] == pytest.approx(24 * 3600 / 3580)
    assert [(event["start_s"], event["end_s"]) for event in result["alarm_events"]] == [(100, 110), (500, 510)]
    assert len(result["matched_pairs"]) == 1


def test_demo_runs_outside_repository(tmp_path):
    completed = subprocess.run([sys.executable, str(DEMO)], cwd=tmp_path,
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["TP"] == 1
