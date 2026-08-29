"""\
Módulo: tests/test_baseline_bandpass.py

Testes de baseline (median/mad/arpls) e bandpass.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall.domain.bandpass import get_bandpass
from radioskysims_waterfall.domain.baseline import baseline_1d, baseline_arpls, get_baseline


@pytest.fixture
def da() -> xr.DataArray:
    rng = np.random.default_rng(1)
    time = np.arange(40, dtype=float)
    freq = np.linspace(300, 400, 32)
    # banda de interesse em torno de 340-360 MHz com sinal forte
    band_mask = (freq >= 340) & (freq <= 360)
    data = rng.normal(5, 0.5, size=(len(time), len(freq)))
    data[:, band_mask] += 20.0
    return xr.DataArray(data, dims=("time", "freq"), coords={"time": time, "freq": freq})


# ---------------------------------------------------------------------------
# Baseline 1D
# ---------------------------------------------------------------------------


def test_baseline_arpls_length() -> None:
    y = np.linspace(0, 10, 100) + 2.0 * np.exp(-((np.arange(100) - 50) ** 2) / 100)
    bl = baseline_arpls(y)
    assert bl.shape == y.shape
    assert np.all(np.isfinite(bl))


def test_baseline_1d_global_median() -> None:
    y = np.array([1.0, 2.0, 3.0, 100.0, 4.0])
    bl = baseline_1d(y, method="median")
    assert np.allclose(bl, np.median(y))


def test_baseline_1d_running_median() -> None:
    y = np.arange(20, dtype=float)
    bl = baseline_1d(y, method="median", window=5)
    assert bl.shape == y.shape
    assert np.all(np.isfinite(bl))


def test_baseline_1d_mad() -> None:
    y = np.arange(20, dtype=float)
    bl = baseline_1d(y, method="mad", window=None)
    assert np.allclose(bl, np.median(y))


def test_baseline_1d_arpls_rejects_window() -> None:
    with pytest.raises(ValueError):
        baseline_1d(np.arange(10, dtype=float), method="arpls", window=5)


def test_baseline_1d_unknown_method() -> None:
    with pytest.raises(ValueError):
        baseline_1d(np.arange(10, dtype=float), method="bogus")


# ---------------------------------------------------------------------------
# Baseline 2D (xarray)
# ---------------------------------------------------------------------------


def test_get_baseline_median_global_shape(da: xr.DataArray) -> None:
    bl = get_baseline(da, method="median", window=None, dim="freq")
    assert bl.shape == da.shape
    # global ao longo de freq: constante por linha
    assert np.allclose(bl.isel(time=0).values, bl.isel(time=0).values[0], atol=1e-8)


def test_get_baseline_median_running(da: xr.DataArray) -> None:
    bl = get_baseline(da, method="median", window=7, dim="freq")
    assert bl.shape == da.shape


def test_get_baseline_mad(da: xr.DataArray) -> None:
    bl = get_baseline(da, method="mad", window=7, dim="freq")
    assert bl.shape == da.shape


def test_get_baseline_arpls(da: xr.DataArray) -> None:
    bl = get_baseline(da, method="arpls", dim="freq")
    assert bl.shape == da.shape
    assert np.all(np.isfinite(bl.values))


def test_get_baseline_unknown_method(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        get_baseline(da, method="bogus")


# ---------------------------------------------------------------------------
# Bandpass
# ---------------------------------------------------------------------------


def test_bandpass_keeps_band_of_interest(da: xr.DataArray) -> None:
    band = get_bandpass(da, threshold=3.0)
    assert band.sizes["freq"] > 0
    # canais mantidos estão dentro/perto da banda de interesse (340-360)
    kept_freq = band["freq"].values
    assert (kept_freq >= 335).all()
    assert (kept_freq <= 365).all()


def test_bandpass_returns_slices(da: xr.DataArray) -> None:
    slices = get_bandpass(da, threshold=3.0, return_slices=True)
    assert isinstance(slices, list)
    assert len(slices) >= 1
    # cada slice tem start/stop válidos
    n = da.sizes["freq"]
    for s in slices:
        assert 0 <= s.start < s.stop <= n


def test_bandpass_returns_mask(da: xr.DataArray) -> None:
    mask = get_bandpass(da, threshold=3.0, return_mask=True)
    assert mask.dtype == bool
    assert mask.sizes["freq"] == da.sizes["freq"]


def test_bandpass_slices_and_mask_exclusive(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        get_bandpass(da, return_slices=True, return_mask=True)


def test_bandpass_invalid_method(da: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        get_bandpass(da, method="bogus")
