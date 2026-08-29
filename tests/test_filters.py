"""\
Módulo: tests/test_filters.py

Testes dos filtros temporal/espectral/espacial.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall.domain.filters import (
    filter_data,
    filter_spatial,
    filter_spectral,
    filter_temporal,
)


@pytest.fixture
def da() -> xr.DataArray:
    rng = np.random.default_rng(0)
    time = np.arange(30, dtype=float)
    freq = np.arange(20, dtype=float)
    # fundo com gradiente espectral + ruído
    data = np.outer(np.ones(30), np.linspace(10, 20, 20)) + rng.normal(0, 0.1, (30, 20))
    return xr.DataArray(data, dims=("time", "freq"), coords={"time": time, "freq": freq})


def test_filter_spectral_median_running_shape(da: xr.DataArray) -> None:
    out = filter_spectral(da, method="median", window=5)
    assert out.shape == da.shape


def test_filter_spectral_median_running_smooths(da: xr.DataArray) -> None:
    # sinal suave ao longo de freq: mediana móvel preserva o gradiente
    out = filter_spectral(da, method="median", window=5, mode="subtract")
    # resíduo deve ter ruído reduzido (std menor que o original)
    assert float(out.std()) < float(da.std())


def test_filter_temporal_global_mean_removes_level(da: xr.DataArray) -> None:
    out = filter_temporal(da, method="mean", window=None, mode="subtract")
    # média temporal de cada canal ~ 0
    assert np.allclose(out.mean("time").values, 0.0, atol=1e-8)


def test_filter_spatial_shape(da: xr.DataArray) -> None:
    out = filter_spatial(da, method="median", window=(3, 3))
    assert out.shape == da.shape


def test_filter_spatial_global(da: xr.DataArray) -> None:
    out = filter_spatial(da, method="mean", window=None, mode="subtract")
    assert abs(float(out.mean())) < 1e-8


def test_filter_zscore_scale(da: xr.DataArray) -> None:
    out = filter_temporal(da, method="mean", window=None, mode="zscore")
    assert np.allclose(out.std("time").values, 1.0, atol=1e-6)


def test_filter_return_mode(da: xr.DataArray) -> None:
    prof = filter_temporal(da, method="mean", window=None, mode="return")
    assert prof.dims == ("freq",)
    assert np.allclose(prof.values, da.mean("time").values)


def test_filter_mask_honored(da: xr.DataArray) -> None:
    mask = xr.full_like(da, True, dtype=bool)
    mask[{"time": slice(0, 5)}] = False
    out = filter_temporal(da, method="mean", window=None, mask=mask, mode="subtract")
    # pixel mascarado vira NaN
    assert np.isnan(out.isel(time=0).values).all()


def test_filter_arpls_requires_no_window(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        filter_spectral(da, method="arpls", window=5)


def test_filter_arpls_global(da: xr.DataArray) -> None:
    out = filter_spectral(da, method="arpls", window=None, mode="subtract")
    assert out.shape == da.shape


def test_filter_spatial_arpls_unsupported(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        filter_spatial(da, method="arpls")


def test_filter_data_dispatch(da: xr.DataArray) -> None:
    assert filter_data(da, axis="temporal", method="median", window=3).shape == da.shape
    assert filter_data(da, axis="spectral", method="median", window=3).shape == da.shape
    assert filter_data(da, axis="spatial", method="median", window=(3, 3)).shape == da.shape
    with pytest.raises(ValueError):
        filter_data(da, axis="bogus")


def test_filter_invalid_method(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        filter_temporal(da, method="bogus")
