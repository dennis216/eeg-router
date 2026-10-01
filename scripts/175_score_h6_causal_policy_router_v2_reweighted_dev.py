"""Dev-only scorer for the reweighted H6-v2 retry."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
MODULE_PATH = ROOT / "scripts/172_score_h6_causal_policy_router_v2_dev.py"
spec = importlib.util.spec_from_file_location("h6_reweighted_dev_scorer", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    mod.LABEL_NS = "h6-causal-policy-router-v2-reweighted"
    # Labels remain the corrected, frozen v2 labels; only the trained
    # checkpoint and output namespace change.
    mod.LABEL_PATH = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/labels/dev_causal_policy.parquet"
    mod.LABEL_MANIFEST = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/label_manifest.json"
    mod.CKPT = ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted/training_formal_train_only/checkpoint.pt"
    mod.OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2-reweighted/scoring_dev_formal_train_only"
    mod.NORMALIZE_INPUT = True
    mod.main()


if __name__ == "__main__":
    main()
