"""\
Módulo: tests/test_masks_events.py

Testes de máscaras e eventos.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall.domain.events import (
    Event,
    aggregate,
    to_counts,
    to_density,
    to_indices,
    to_mask,
)
from radioskysims_waterfall.domain.masks import (
    apply,
    combine,
    count,
    fraction,
    from_range,
    from_threshold,
    invert,
)


@pytest.fixture
def da() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(size=(10, 8)),
        dims=("time", "freq"),
        coords={"time": np.arange(10, dtype=float), "freq": np.arange(8, dtype=float)},
    )


def test_from_threshold(da: xr.DataArray) -> None:
    mask = from_threshold(da, 0.0, comparison="gt")
    assert mask.dtype == bool
    assert mask.sum() == (da.values > 0).sum()


def test_from_range(da: xr.DataArray) -> None:
    mask = from_range(da, vmin=-1.0, vmax=1.0)
    assert mask.sum() == ((da.values >= -1.0) & (da.values <= 1.0)).sum()


def test_combine_and_or(da: xr.DataArray) -> None:
    m1 = from_threshold(da, 0.0, "gt")
    m2 = from_threshold(da, 1.0, "gt")
    assert combine(m1, m2, op="and").sum() == (m1 & m2).sum()
    assert combine(m1, m2, op="or").sum() == (m1 | m2).sum()


def test_invert_fraction_count(da: xr.DataArray) -> None:
    m = from_threshold(da, 0.0, "gt")
    assert invert(m).sum() == (~m).sum()
    assert 0.0 <= fraction(m) <= 1.0
    assert count(m) == int(m.sum())


def test_apply_mask(da: xr.DataArray) -> None:
    m = from_threshold(da, 0.0, "gt")
    out = apply(da, m, fill_value=np.nan)
    assert np.isnan(out.values[~m.values]).all()
    assert not np.isnan(out.values[m.values]).any()


def test_apply_none_mask_returns_data(da: xr.DataArray) -> None:
    assert apply(da, None) is da


def test_events_to_indices() -> None:
    time = np.arange(10, dtype=float)
    freq = np.linspace(300, 400, 8)
    evs = [Event(time=2.0, freq=300.0), Event(time=7.0, freq=400.0)]
    t_idx, f_idx = to_indices(evs, time, freq)
    assert t_idx.tolist() == [2, 7]
    assert f_idx.tolist() == [0, 7]


def test_events_mask_counts() -> None:
    time = np.arange(10, dtype=float)
    freq = np.linspace(300, 400, 8)
    evs = [Event(time=2.0, freq=312.5), Event(time=2.0, freq=312.5)]
    mask = to_mask(evs, time, freq)
    counts = to_counts(evs, time, freq)
    assert int(mask.sum()) == 1
    assert int(counts.sum()) == 2


def test_events_density() -> None:
    time = np.arange(10, dtype=float)
    freq = np.arange(8, dtype=float)
    evs = [Event(time=5.0, freq=4.0)]
    d = to_density(evs, time, freq, time_radius=1, freq_radius=1)
    assert int(d.sum()) == 9  # 3x3 vizinhança


def test_events_aggregate_modes() -> None:
    time = np.arange(10, dtype=float)
    freq = np.arange(8, dtype=float)
    evs = [Event(time=1.0, freq=2.0)]
    assert aggregate(evs, time, freq, mode="mask").dtype == bool
    assert aggregate(evs, time, freq, mode="count").dtype == np.int64


def test_events_duck_dataframe_like() -> None:
    """Aceita objeto DataFrame-like sem importar pandas."""

    class FakeDF:
        def to_dict(self, orient: str) -> list[dict[str, float]]:
            return [{"time": 3.0, "freq": 5.0}]

    time = np.arange(10, dtype=float)
    freq = np.arange(8, dtype=float)
    t_idx, f_idx = to_indices(FakeDF(), time, freq)
    assert t_idx.tolist() == [3]
    assert f_idx.tolist() == [5]


def test_events_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        to_indices(42, np.arange(5), np.arange(5))
