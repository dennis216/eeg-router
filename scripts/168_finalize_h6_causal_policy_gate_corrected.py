"""Finalize the corrected H6 v2 oracle gate without overwriting the first audit."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path("/root/b_false_alarm_atlas")
path = ROOT / "scripts/163_finalize_h6_causal_policy_gate.py"
spec = importlib.util.spec_from_file_location("h6_gate_corrected_wrapper", path)
if spec is None or spec.loader is None:
    raise ImportError(path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.NS = "h6-causal-policy-router-v2-corrected"
mod.OUT = ROOT / "outputs/reports" / mod.NS
mod.main()
