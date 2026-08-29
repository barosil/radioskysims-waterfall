"""\
Módulo: tests/test_operations.py

Testes das operações puras (slice, decimate, interpolate, rebin, normalize_dims).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall.domain.operations import (
    decimate,
    interpolate,
    normalize_dims,
    rebin,
    slice_by_value,
    slice_data,
)


def _da(n_time: int = 10, n_freq: int = 8) -> xr.DataArray:
    rng = np.random.default_rng(0)
    time = np.arange(n_time, dtype=float)
    freq = np.linspace(100, 200, n_freq)
    return xr.DataArray(
        rng.normal(size=(n_time, n_freq)),
        dims=("time", "freq"),
        coords={"time": time, "freq": freq},
    )


def test_normalize_dims_renames_and_orders() -> None:
    da = _da().rename({"time": "t", "freq": "f"})
    out = normalize_dims(da, time_name="t", freq_name="f")
    assert out.dims == ("time", "freq")


def test_normalize_dims_transposes() -> None:
    da = _da().transpose("freq", "time")
    out = normalize_dims(da)
    assert out.dims == ("time", "freq")


def test_normalize_dims_raises_on_1d() -> None:
    with pytest.raises(ValueError):
        normalize_dims(xr.DataArray(np.arange(5), dims="time"))


def test_slice_data() -> None:
    da = _da()
    out = slice_data(da, time=slice(2, 6), freq=slice(0, 4))
    assert out.shape == (4, 4)
    scalar = slice_data(da, time=3)
    assert scalar.shape == (8,)


def test_slice_by_value() -> None:
    da = _da()
    out = slice_by_value(da, freq=slice(120.0, 160.0))
    assert out.sizes["freq"] >= 1
    assert float(out["freq"].min()) >= 120.0


def test_decimate_shape_and_value() -> None:
    da = _da()
    out = decimate(da, time_factor=2, freq_factor=2, method="mean")
    assert out.shape == (5, 4)
    # média por blocos: valor esperado no bloco [0:2, 0:2]
    expected = da.isel(time=slice(0, 2), freq=slice(0, 2)).mean()
    assert out.isel(time=0, freq=0) == pytest.approx(expected.item())


def test_decimate_median() -> None:
    da = _da()
    out = decimate(da, time_factor=2, method="median")
    assert out.shape == (5, 8)


def test_decimate_invalid_method() -> None:
    with pytest.raises(ValueError):
        decimate(_da(), time_factor=2, method="bogus")


def test_interpolate_upsample() -> None:
    da = _da()
    new_freq = np.linspace(100, 200, 17)
    out = interpolate(da, freq=new_freq)
    assert out.sizes["freq"] == 17
    # interpolação linear passa pelos pontos originais (100 e 200 estão em ambas as grades)
    orig = da.sel(freq=100.0, method="nearest")
    interp = out.sel(freq=100.0, method="nearest")
    assert np.allclose(interp.values, orig.values, atol=1e-6)
    orig2 = da.sel(freq=200.0, method="nearest")
    interp2 = out.sel(freq=200.0, method="nearest")
    assert np.allclose(interp2.values, orig2.values, atol=1e-6)


def test_rebin() -> None:
    da = _da()
    edges = np.array([100, 150, 200])
    out = rebin(da, freq_edges=edges)
    assert out.sizes["freq"] == 2
    assert out["freq"].values[0] == pytest.approx(125.0)
    assert out["freq"].values[1] == pytest.approx(175.0)


def test_rebin_time() -> None:
    da = _da()
    out = rebin(da, time_edges=np.array([0, 5, 10]))
    assert out.sizes["time"] == 2
