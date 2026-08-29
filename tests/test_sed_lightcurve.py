"""\
Módulo: tests/test_sed_lightcurve.py

Testes de SED e LightCurve.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np

from radioskysims_waterfall import SED, LightCurve


def test_sed_values_are_time_mean(wf) -> None:
    sed = wf.sed()
    expected = wf.data.mean("time")
    assert np.allclose(sed.values.values, expected.values)
    assert sed.n_samples == wf.n_time
    assert sed.freq.shape == (wf.n_freq,)


def test_sed_interval(wf) -> None:
    sed = wf.sed(interval=slice(0, 10))
    expected = wf.data.isel(time=slice(0, 10)).mean("time")
    assert np.allclose(sed.values.values, expected.values)
    assert sed.n_samples == 10


def test_sed_channels(wf) -> None:
    sed = wf.sed(channels=slice(0, 16))
    assert len(sed) == 16
    assert sed.freq.shape == (16,)


def test_sed_std_is_positive(wf) -> None:
    sed = wf.sed()
    assert float(sed.std.min()) >= 0.0


def test_sed_getitem(wf) -> None:
    sed = wf.sed()
    sub = sed[0:8]
    assert isinstance(sub, SED)
    assert len(sub) == 8


def test_sed_to_dataarray(wf) -> None:
    sed = wf.sed()
    da = sed.to_dataarray()
    assert da.dims == ("freq",)


def test_lightcurve_values_are_freq_mean(wf) -> None:
    lc = wf.lightcurve()
    expected = wf.data.mean("freq")
    assert np.allclose(lc.values.values, expected.values)
    assert lc.n_channels == wf.n_freq


def test_lightcurve_frequencies(wf) -> None:
    lc = wf.lightcurve(frequencies=slice(8, 24))
    assert lc.n_channels == 16
    assert len(lc) == wf.n_time


def test_lightcurve_interval(wf) -> None:
    lc = wf.lightcurve(interval=slice(0, 20))
    assert len(lc) == 20


def test_lightcurve_getitem(wf) -> None:
    lc = wf.lightcurve()
    sub = lc[0:5]
    assert isinstance(sub, LightCurve)
    assert len(sub) == 5


def test_lightcurve_to_dataarray(wf) -> None:
    assert wf.lightcurve().to_dataarray().dims == ("time",)


def test_sed_lightcurve_repr(wf) -> None:
    assert "SED" in repr(wf.sed())
    assert "LightCurve" in repr(wf.lightcurve())
