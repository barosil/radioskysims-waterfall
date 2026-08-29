"""\
Módulo: tests/test_loaders_storage.py

Testes de loaders (netcdf/hdf5/zarr/fits/xarray) e storage com versionamento.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from radioskysims_waterfall import Waterfall, create_loader, load_waterfall


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def wf(make_wf) -> Waterfall:
    return make_wf(n_time=40, n_freq=32)


def test_load_xarray_object(wf) -> None:
    out = load_waterfall(wf.data)
    assert out.shape == wf.shape


def test_infer_and_load_netcdf(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.nc"
    wf.save(p)
    loaded = load_waterfall(p)
    assert loaded.shape == wf.shape
    assert np.allclose(loaded.values, wf.values)


def test_netcdf_name_mapping(tmp_dir: Path) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 8))
    ds = xr.Dataset(
        {"flux": (("t", "f"), data)},
        coords={"t": np.arange(10, dtype=float), "f": np.arange(8, dtype=float)},
    )
    p = tmp_dir / "named.nc"
    ds.to_netcdf(p)
    wf = load_waterfall(p, time_name="t", freq_name="f", data_name="flux")
    assert wf.data.dims == ("time", "freq")
    assert np.allclose(wf.values, data)


def test_load_netcdf_lazy_chunks(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "lazy.nc"
    wf.save(p)
    loaded = load_waterfall(p, chunks={"time": 16})
    assert loaded.is_lazy


def test_hdf5_roundtrip(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.h5"
    wf.save(p, format="hdf5")
    loaded = load_waterfall(p)
    assert loaded.shape == wf.shape
    assert np.allclose(loaded.values, wf.values)


def test_zarr_roundtrip(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.zarr"
    wf.save(p, format="zarr")
    loaded = load_waterfall(p)
    assert loaded.shape == wf.shape
    assert np.allclose(loaded.values, wf.values)


def test_fits_roundtrip(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.fits"
    wf.save(p, format="fits")
    loaded = load_waterfall(p)
    assert loaded.shape == wf.shape


def test_unknown_extension_raises(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.unknown"
    wf.save(p, format="netcdf")
    with pytest.raises(ValueError):
        load_waterfall(p)


def test_missing_file_raises(tmp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_waterfall(tmp_dir / "nao_existe.nc")


def test_create_loader_unknown_format() -> None:
    with pytest.raises(ValueError):
        create_loader("bogus")


def test_save_versions(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.nc"
    wf.save(p)
    wf.save(p)
    wf.save(p)
    versions = sorted(x.name for x in tmp_dir.glob("wf*"))
    assert "wf.nc" in versions
    assert "wf.v1.nc" in versions
    assert "wf.v2.nc" in versions


def test_save_no_version_overwrites(wf, tmp_dir: Path) -> None:
    p = tmp_dir / "wf.nc"
    wf.save(p, version=False)
    wf.save(p, version=False)
    versions = [x.name for x in tmp_dir.glob("wf*")]
    assert versions == ["wf.nc"]


def test_save_unsupported_format_raises(wf, tmp_dir: Path) -> None:
    with pytest.raises(ValueError):
        wf.save(tmp_dir / "wf.bogus", format="bogus")


def test_save_missing_dir_raises(wf, tmp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        wf.save(tmp_dir / "sub" / "wf.nc", format="netcdf")
