"""Independent reweighted retry after the v2 Dev safety gate failed.

This does not alter the original v2 smoke/formal artifacts.  It keeps the
same corrected labels, raw-only architecture, fixed four-stratum sampler and
frozen experts, adding only causal per-window robust scaling and a capped
3x policy-loss weight on seizure horizons.
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
MODULE_PATH = ROOT / "scripts/170_smoke_h6_causal_policy_router_v2.py"
spec = importlib.util.spec_from_file_location("h6_causal_reweighted_base", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    mod.OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted/training_formal_train_only"
    mod.POLICY_SEIZURE_WEIGHT = 3.0
    mod.NORMALIZE_INPUT = True
    args = Namespace(
        updates=5000,
        per_stratum=16,
        seed=3407,
        lr=1e-4,
        device=None,
        checkpoint_every=250,
        formal=True,
    )
    mod.run(args)


if __name__ == "__main__":
    main()
