"""Descriptive H3 model-state association for frozen H2 HMM v2 states.

Patient-level bootstrap compares model-specific state occupancy within each
fold and partition. It uses no test events, does not refit HMMs, and does not
select K or tune any downstream model.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h2-expanded-attribute-hmm-v2"
B = 2000
SEED = 20260823

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    d = pd.read_parquet(OUT / "state_sequences.parquet")
    d = d[d.partition.isin(["train", "validation"])].copy()
    occ_cols = sorted([c for c in d.columns if c.startswith("state_") and c.endswith("_occupancy")],
                      key=lambda x: int(x.split("_")[1]))
    rng = np.random.default_rng(SEED)
    rows = []
    for (fold, partition), block in d.groupby(["fold", "partition"], sort=True):
        models = sorted(block.model.unique())
        # Patient-level means keep long records from dominating model association.
        patient = block.groupby(["patient", "model"], sort=True)[occ_cols].mean().reset_index()
        pivot = patient.pivot(index="patient", columns="model", values=occ_cols)
        for state_idx, col in enumerate(occ_cols):
            for a, b in combinations(models, 2):
                if (col, a) not in pivot.columns or (col, b) not in pivot.columns:
                    continue
                pair = pivot[[ (col, a), (col, b) ]].dropna()
                n = len(pair)
                if n < 2:
                    continue
                diff = pair.iloc[:, 0].to_numpy(float) - pair.iloc[:, 1].to_numpy(float)
                boot = np.empty(B, dtype=float)
                for i in range(B):
                    idx = rng.integers(0, n, size=n)
                    boot[i] = diff[idx].mean()
                rows.append({
                    "fold": int(fold), "partition": partition, "state": state_idx,
                    "model_a": a, "model_b": b, "patients_common": n,
                    "mean_occupancy_a": float(pair.iloc[:, 0].mean()),
                    "mean_occupancy_b": float(pair.iloc[:, 1].mean()),
                    "mean_difference_a_minus_b": float(diff.mean()),
                    "bootstrap_median_difference": float(np.median(boot)),
                    "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_sign_fraction": float(np.mean(np.sign(boot) == np.sign(diff.mean()))),
                })
        print(f"H3_ASSOCIATION_DONE fold={fold} partition={partition}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "h3_dynamic_model_state_association.csv", index=False)
    manifest = {
        "analysis_id": "h3-dynamic-model-state-association-v1",
        "status": "complete",
        "input_analysis_id": "h2-expanded-attribute-hmm-v2",
        "input_manifest_sha256": sha256(OUT / "manifest.json"),
        "resampling_unit": "patient",
        "bootstrap_replicates": B,
        "seed": SEED,
        "partitions_used": ["train", "validation"],
        "test_used": False,
        "hmm_refit": False,
        "k_selection": False,
        "interpretation": "descriptive model-state association; state labels are fold-local",
        "output": "h3_dynamic_model_state_association.csv",
    }
    (OUT / "h3_dynamic_association_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

