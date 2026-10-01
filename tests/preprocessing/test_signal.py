import numpy as np
import pytest

from bfa.preprocessing.channels import CHANNELS, MissingCanonicalChannel, canonical_order
from bfa.preprocessing.signal import dominant_frequency, model_view


def test_channel_order_is_canonical_and_missing_raises() -> None:
    shuffled = list(reversed(CHANNELS))
    assert [shuffled[index] for index in canonical_order(shuffled)] == list(CHANNELS)
    with pytest.raises(MissingCanonicalChannel):
        canonical_order(shuffled[:-1])
    duplicated = [*CHANNELS[:-2], "T8-P8-0", CHANNELS[-1], "T8-P8-1"]
    order = canonical_order(duplicated)
    assert duplicated[order[14]] == "T8-P8-0"


def test_model_views_preserve_ten_hz_frequency() -> None:
    sampling_hz = 256.0
    time = np.arange(0, 10, 1 / sampling_hz)
    signal = 50 * np.sin(2 * np.pi * 10 * time)
    data = np.tile(signal, (16, 1)).astype(np.float32)
    for model in ("singlem", "cbramod", "tcn_gat"):
        view = model_view(data, sampling_hz, model)
        target_hz = {"singlem": 128, "cbramod": 200, "tcn_gat": 256}[model]
        assert abs(dominant_frequency(view[0], target_hz) - 10.0) < 0.2
        assert np.isfinite(view).all()
