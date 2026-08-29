"""\
Módulo: tests/test_models.py

Testes do value object Waterfall (construção, metadados, chainable,
normalização, persistência de metadados).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall import Waterfall


def test_construct_from_tuple(wf) -> None:
    assert wf.shape == (50, 64)
    assert wf.data.dims == ("time", "freq")
    assert wf.meta.name == "wf"


def test_construct_from_dataarray(make_wf) -> None:
    wf = make_wf()
    da = wf.data
    wf2 = Waterfall(da)
    assert wf2.shape == wf.shape


def test_construct_from_dataset() -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"flux": (("time", "freq"), rng.normal(size=(10, 8)))},
        coords={"time": np.arange(10, dtype=float), "freq": np.arange(8, dtype=float)},
    )
    wf = Waterfall(ds, data_name="flux")
    assert wf.shape == (10, 8)


def test_construct_from_ndarray_with_coords() -> None:
    data = np.zeros((5, 4))
    wf = Waterfall(data, time=np.arange(5), freq=np.arange(4))
    assert wf.shape == (5, 4)


def test_construct_from_ndarray_default_coords() -> None:
    wf = Waterfall(np.zeros((5, 4)))
    assert wf.shape == (5, 4)


def test_construct_invalid_ndim() -> None:
    with pytest.raises(ValueError):
        Waterfall(np.zeros(5))


def test_meta_defaults() -> None:
    wf = Waterfall((np.zeros((3, 3)), np.arange(3), np.arange(3)))
    assert wf.meta.freq_unit == "Hz"
    assert wf.meta.time_unit == "s"


def test_properties(wf) -> None:
    assert wf.n_time == 50
    assert wf.n_freq == 64
    assert wf.dtype == np.dtype("float64")
    assert not wf.is_lazy
    assert wf.time.shape == (50,)
    assert wf.freq.shape == (64,)


def test_chainable_ops_return_new(wf) -> None:
    out = wf.slice(freq=slice(0, 32)).decimate(time_factor=2).normalize(method="minmax")
    assert isinstance(out, Waterfall)
    assert out is not wf
    assert out.shape == (25, 32)


def test_slice_and_repr(wf) -> None:
    assert "Waterfall" in repr(wf)
    sliced = wf.slice(time=slice(0, 5))
    assert sliced.n_time == 5


def test_mask_chain(wf) -> None:
    from radioskysims_waterfall.domain.masks import from_threshold

    m = from_threshold(wf.data, 0.0, "gt")
    wf_m = wf.with_mask(m)
    assert wf_m.mask is not None
    applied = wf_m.apply_mask()
    assert np.isnan(applied.values[~m.values]).all()
    cleared = wf_m.clear_mask()
    assert cleared.mask is None


def test_normalize_minmax(wf) -> None:
    out = wf.normalize(method="minmax")
    assert float(out.values.min()) == pytest.approx(0.0)
    assert float(out.values.max()) == pytest.approx(1.0)


def test_normalize_zscore(wf) -> None:
    out = wf.normalize(method="zscore")
    assert float(out.values.mean()) == pytest.approx(0.0, abs=1e-8)
    assert float(out.values.std()) == pytest.approx(1.0, abs=1e-6)


def test_normalize_db(wf) -> None:
    out = wf.normalize(method="db", ref=1.0)
    assert out.shape == wf.shape


def test_estimate_memory(wf) -> None:
    assert wf.estimate_memory() == 50 * 64 * 8


def test_to_dataset_embeds_meta(wf) -> None:
    ds = wf.to_dataset()
    assert ds.attrs["name"] == "wf"


def test_copy(wf) -> None:
    c = wf.copy()
    assert c is not wf
    assert c.shape == wf.shape


def test_compute_returns_waterfall(wf) -> None:
    assert wf.compute().shape == wf.shape


def test_getitem_len(wf) -> None:
    assert len(wf) == 50
