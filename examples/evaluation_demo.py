"""Synthetic evaluation tutorial, not paper results or full router reproduction."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.evaluation.metrics import event_metrics


def run_demo() -> dict:
    monitoring_s = 3600.0
    times = np.arange(0.0, monitoring_s, 2.0)
    scores = np.full(times.shape, 0.1)
    scores[((times >= 100) & (times < 110)) | ((times >= 500) & (times < 510))] = 0.9
    truths = [(100.0, 110.0), (300.0, 310.0)]
    events = eventize(times, scores, threshold=0.5, ema_alpha=1.0,
                      min_duration_s=4.0, merge_gap_s=10.0, refractory_s=30.0)
    matches = match_events(events, truths, minimum_overlap_s=1.0)
    background_s = monitoring_s - sum(end - start for start, end in truths)
    metrics = event_metrics(events, truths, nonseizure_hours=background_s / 3600.0)
    return {
        "synthetic_teaching_example": True,
        "monitoring_seconds": monitoring_s,
        "background_seconds": background_s,
        "alarm_events": [asdict(event) for event in events],
        "matched_pairs": [asdict(pair) for pair in matches.pairs],
        "TP": metrics.true_positive_events,
        "FP": metrics.false_alarm_events,
        "eligible_seizures": metrics.truth_events,
        "event_sensitivity": metrics.event_sensitivity,
        "FA_per_24h": metrics.fa_per_24h,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
