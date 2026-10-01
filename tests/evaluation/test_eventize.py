import numpy as np

from bfa.evaluation.eventize import Event, causal_ema, eventize


def test_causal_ema_does_not_use_future_values() -> None:
    values = np.array([0.0, 1.0, 0.0, 1.0])
    prefix = causal_ema(values[:3], alpha=1 / 3)
    full = causal_ema(values, alpha=1 / 3)
    np.testing.assert_allclose(prefix, full[:3])


def test_eventize_merge_and_refractory() -> None:
    times = np.arange(0, 30, 2.0)
    probabilities = np.array([0, 0, 0.9, 0.9, 0, 0, 0.8, 0.8, 0, 0, 0, 0.9, 0.9, 0, 0])
    events = eventize(
        times,
        probabilities,
        threshold=0.5,
        ema_alpha=1.0,
        min_duration_s=4,
        merge_gap_s=10,
        refractory_s=30,
    )
    assert events == [Event(start_s=4.0, end_s=26.0, peak_probability=0.9)]
