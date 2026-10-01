"""Launch the formal Train-only phase after the v2 oracle and smoke gates.

The implementation is shared with the audited 25-update smoke so the model,
losses, data strata and raw-EEG contract cannot silently diverge.  This phase
does not read TUSZ Eval and does not select thresholds; it only fits the
router on frozen-expert Train labels.  The output lives in a new namespace and
refuses to overwrite an existing run.
"""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
MODULE_PATH = ROOT / "scripts/170_smoke_h6_causal_policy_router_v2.py"
spec = importlib.util.spec_from_file_location("h6_causal_smoke", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    # A separate formal namespace is mandatory; the smoke artifacts remain
    # immutable and are referenced by their hashes in the final report.
    mod.OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/training_formal_train_only"
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
