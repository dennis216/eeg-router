"""Patient-level bootstrap confidence intervals for the frozen BSN H1 results."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVENT_PATH = ROOT / "outputs/events/bsn_pilot_v1_test_events.parquet"
OUT_PATH = ROOT / "outputs/reports/bsn_pilot_v1_h1_bootstrap.csv"
N_BOOTSTRAP = 2000

_SPEC = importlib.util.spec_from_file_location("h1_impl", ROOT / "scripts/19_build_bsn_h1.py")
assert _SPEC and _SPEC.loader
_H1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_H1)
PAIRS = _H1.PAIRS
SEEDS = _H1.SEEDS
_event_index = _H1._event_index
_patient_f1_values = _H1._patient_f1_values


def main() -> None:
    events = pd.read_parquet(EVENT_PATH)
    split = json.loads((ROOT / "manifests/splits/split_17.json").read_text())
    patients = tuple(split["test"])
    rng = np.random.default_rng(20260805)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for left_model, right_model in PAIRS:
            left = events[(events.seed == seed) & (events.model == left_model) & (events.event_type == "fp")]
            right = events[(events.seed == seed) & (events.model == right_model) & (events.event_type == "fp")]
            values = _patient_f1_values(_event_index(left), _event_index(right), patients)
            vector = np.asarray([values[patient] for patient in patients], dtype=float)
            sampled_medians = np.empty(N_BOOTSTRAP, dtype=float)
            for index in range(N_BOOTSTRAP):
                sampled = rng.integers(0, len(vector), size=len(vector))
                sampled_medians[index] = float(np.median(vector[sampled]))
            rows.append(
                {
                    "seed": seed,
                    "left_model": left_model,
                    "right_model": right_model,
                    "observed_patient_median_f1": float(np.median(vector)),
                    "patient_f1_values": json.dumps(dict(zip(patients, vector.tolist(), strict=True)), sort_keys=True),
                    "bootstrap_median_q025": float(np.quantile(sampled_medians, 0.025)),
                    "bootstrap_median_q975": float(np.quantile(sampled_medians, 0.975)),
                    "bootstrap_replicates": N_BOOTSTRAP,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(OUT_PATH, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
