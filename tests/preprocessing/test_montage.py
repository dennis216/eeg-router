import numpy as np

from bfa.preprocessing.channels import CHANNELS
from bfa.preprocessing.signal import derive_bipolar_uv


def test_common_reference_channels_are_derived_as_bipolar_uv() -> None:
    electrodes = sorted({name for channel in CHANNELS for name in channel.split("-")})
    channel_names = [f"{name}-CS2" for name in electrodes]
    data_v = np.stack(
        [np.full(8, index * 1e-6, dtype=float) for index in range(len(electrodes))]
    )
    output = derive_bipolar_uv(channel_names, data_v)
    fp1 = electrodes.index("FP1")
    f7 = electrodes.index("F7")
    assert output.shape == (16, 8)
    assert np.allclose(output[0], fp1 - f7)


def test_zero_one_channel_typo_maps_to_o1() -> None:
    electrodes = sorted({name for channel in CHANNELS for name in channel.split("-")})
    channel_names = ["01" if name == "O1" else name for name in electrodes]
    data_v = np.zeros((len(electrodes), 4), dtype=float)
    assert derive_bipolar_uv(channel_names, data_v).shape == (16, 4)
