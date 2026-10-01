"""Run the immutable v2 oracle on the corrected label namespace."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path("/root/b_false_alarm_atlas")
NS = "h6-causal-policy-router-v2-corrected"
path = ROOT / "scripts/162_compute_h6_causal_policy_oracle.py"
spec = importlib.util.spec_from_file_location("h6_oracle_corrected_wrapper", path)
if spec is None or spec.loader is None:
    raise ImportError(path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.LABEL_NS = NS
mod.NS = NS
mod.OUT = ROOT / "outputs/reports" / NS / "oracle"
mod.main()
