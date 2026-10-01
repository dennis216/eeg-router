import numpy as np

from bfa.preprocessing.channels import CHANNELS
from bfa.preprocessing.tuh import derive_tuh_bipolar_uv, normalize_tuh_electrode


def test_tuh_legacy_electrode_mapping() -> None:
    assert normalize_tuh_electrode("EEG T3-REF") == "T7"
    assert normalize_tuh_electrode("EEG T4-LE") == "T8"
    assert normalize_tuh_electrode("EEG T5-REF") == "P7"
    assert normalize_tuh_electrode("EEG T6-LE") == "P8"


def test_tuh_reference_channels_derive_frozen_bipolar_order() -> None:
    modern = sorted({x for pair in CHANNELS for x in pair.split("-")})
    inverse = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    names = [f"EEG {inverse.get(x, x)}-REF" for x in modern]
    data = np.arange(len(names), dtype=float)[:, None]
    output = derive_tuh_bipolar_uv(names, data)
    index = {name: i for i, name in enumerate(modern)}
    expected = np.array([(index[a] - index[b]) * 1e6 for a, b in (p.split("-") for p in CHANNELS)])
    np.testing.assert_allclose(output[:, 0], expected)
