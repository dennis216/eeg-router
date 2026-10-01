import numpy as np
import pandas as pd

from bfa.same_window_moe import (
    build_policy_targets,
    false_positive_overlap,
    threshold_normalize,
)


def test_threshold_normalize_maps_each_threshold_to_half():
    p = np.array([0.0, 0.2, 0.6, 1.0])
    t = np.array([0.2, 0.2, 0.6, 0.6])
    q = threshold_normalize(p, t)
    assert np.allclose(q, [0.0, 0.5, 0.5, 1.0])


def test_reject_only_background_all_fp_and_all_missed_seizure_is_kept():
    labels = np.array([0.0, 1.0, 1.0])
    probabilities = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3], [0.9, 0.2, 0.8]])
    thresholds = np.array([0.5, 0.5, 0.5])
    result = build_policy_targets(labels, probabilities, thresholds)
    assert result["hard_action"][0] == 3
    assert result["outcome_group"][1] == "seizure_all_missed"
    assert result["hard_action"][1] == 2
    assert np.all(result["target"][labels == 1, 3] == 0)
    assert np.allclose(result["target"].sum(axis=1), 1.0)


def test_false_positive_overlap():
    frame = pd.DataFrame({
        "label": [0, 0, 0, 1],
        "positive_singlem": [1, 1, 0, 1],
        "positive_cbramod": [1, 0, 1, 1],
        "positive_tcn_gat": [1, 0, 0, 1],
    })
    result = false_positive_overlap(frame)
    assert result["fp_iou_singlem_cbramod"] == 1 / 3
    assert result["all_three_fp_fraction_of_union"] == 1 / 3
    assert result["only_singlem_fp_fraction_of_union"] == 1 / 3
