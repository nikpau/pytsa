import pickle
from pathlib import Path

import numpy as np

from pytsa.data.geometry import __path__ as geometry_path
from pytsa.data.quantiles import __path__ as quantiles_path

QUANTILE_FAMILIES = {"dquants", "trquants", "squants", "diffquants", "tquants"}


def assert_quantile_arrays(data: dict | np.ndarray | np.number) -> None:
    if isinstance(data, np.ndarray):
        assert data.shape == (1001,)
        return

    if isinstance(data, np.number):
        return

    assert data
    for value in data.values():
        assert_quantile_arrays(value)


def test_quantile_data_is_packaged_with_expected_shapes() -> None:
    files = list(Path(quantiles_path[0]).rglob("*.pkl"))

    assert len(files) >= 2
    for file in files:
        with file.open("rb") as data_file:
            data = pickle.load(data_file)
        assert set(data) == QUANTILE_FAMILIES
        for family in data.values():
            assert_quantile_arrays(family)


def test_geometry_data_is_packaged() -> None:
    assert len(list(Path(geometry_path[0]).glob("*.json"))) >= 5