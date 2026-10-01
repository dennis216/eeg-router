"""Build corrected H6 v2 labels without overwriting the initial audit.

The initial labels-only audit is retained.  This corrected namespace follows
the stated rule that a seizure horizon with all three experts correct still
has a multi-expert soft policy target; only all-safe background has no policy
ranking loss.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/root/b_false_alarm_atlas")
NS = "h6-causal-policy-router-v2-corrected"
SOURCE_OUT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NS
MODELS = ("singlem", "cbramod", "tcn_gat")

def load_builder():
    path = ROOT / "scripts/161_build_h6_causal_policy_labels.py"
    spec = importlib.util.spec_from_file_location("h6_label_builder_v2_corrected", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def corrected_target(row: pd.Series):
    has = bool(int(row.future_has_seizure))
    correct = np.asarray([int(row[f"correct_{m}"]) for m in MODELS], dtype=int)
    n = int(correct.sum())
    mask = str(row.correct_mask)
    if has:
        group = "unroutable_seizure" if n == 0 else ("seizure_all_correct" if n == 3 else "seizure_selective")
    else:
        group = "background_all_fp" if mask == "000" else ("background_all_safe" if mask == "111" else "background_selective")
    reject = group == "background_all_fp"
    unroutable = group == "unroutable_seizure"
    eligible = group in {"seizure_all_correct", "seizure_selective", "background_selective"}
    q = np.zeros(3, dtype=float)
    if eligible:
        ix = np.flatnonzero(correct == 1)
        outcome = np.asarray([1 - min(max(float(row[f"delay_s_{m}"]), 0), 60.0) / 60.0 for m in MODELS])[ix] if has else np.ones(len(ix))
        conf = np.asarray([float(row[f"confidence_{m}"]) for m in MODELS])[ix]
        def sm(x, tau):
            z = x / tau; z -= z.max(); e = np.exp(np.clip(z, -60, 60)); return e / e.sum()
        q[ix] = .90 * sm(outcome, .25) + .10 * sm(conf, .25)
        q /= q.sum()
    return group, reject, unroutable, eligible, q

def main():
    builder = load_builder()
    builder.NS = NS
    builder.OUT = OUT
    builder.target = corrected_target
    # Reuse the builder's immutable source v1 label tables; Eval is not read.
    builder.main()
    manifest_path = OUT / "label_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["correction"] = "seizure_all_correct horizons are policy-eligible under the multiple-correct rule"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10))
    manifest_path.with_suffix(".sha256").write_text(builder.sha256(manifest_path) + chr(10))

if __name__ == "__main__":
    main()
