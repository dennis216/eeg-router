import pandas as pd

from bfa.training.sampler import PatientBalancedSampler


def _index() -> pd.DataFrame:
    rows = []
    for patient in ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"):
        for label, count in ((0, 80), (1, 20)):
            rows.extend({"patient_id": patient, "label": label} for _ in range(count))
    return pd.DataFrame(rows)


def test_sampler_targets_positive_fraction_without_patient_domination() -> None:
    index = _index()
    sampler = PatientBalancedSampler(
        index, batch_size=64, positive_fraction=0.30, seed=17, epoch_size=6400
    )
    sampled = index.iloc[list(sampler)]
    assert abs(sampled.label.mean() - 0.30) < 0.03
    assert sampled.patient_id.value_counts(normalize=True).max() < 0.15


def test_sampler_state_resumes_exactly() -> None:
    index = _index()
    sampler = PatientBalancedSampler(index, batch_size=8, seed=17, epoch_size=64)
    iterator = iter(sampler)
    _ = [next(iterator) for _ in range(19)]
    state = sampler.state_dict()
    expected = [next(iterator) for _ in range(20)]

    resumed = PatientBalancedSampler(index, batch_size=8, seed=999, epoch_size=64)
    resumed.load_state_dict(state)
    actual_iterator = iter(resumed)
    assert [next(actual_iterator) for _ in range(20)] == expected
