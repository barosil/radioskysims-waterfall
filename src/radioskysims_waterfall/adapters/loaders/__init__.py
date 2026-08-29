"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/__init__.py

Descrição:
    Loaders de waterfalls por formato, com factory por extensão/formato.

    Formatos suportados: xarray (objeto), netcdf, hdf5, zarr, fits.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from radioskysims_waterfall.adapters.loaders.fits_loader import FitsLoader
from radioskysims_waterfall.adapters.loaders.hdf5_loader import HDF5Loader
from radioskysims_waterfall.adapters.loaders.netcdf_loader import NetCDFLoader
from radioskysims_waterfall.adapters.loaders.xarray_loader import XArrayLoader
from radioskysims_waterfall.adapters.loaders.zarr_loader import ZarrLoader
from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort

if TYPE_CHECKING:
    from radioskysims_waterfall.domain.models import Waterfall

_LOADERS: dict[str, type[WaterfallLoaderPort]] = {
    "xarray": XArrayLoader,
    "netcdf": NetCDFLoader,
    "hdf5": HDF5Loader,
    "zarr": ZarrLoader,
    "fits": FitsLoader,
}

_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".netcdf": "netcdf",
    ".cdf": "netcdf",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".hdf": "hdf5",
    ".zarr": "zarr",
    ".fits": "fits",
    ".fit": "fits",
    ".fts": "fits",
    ".fits.gz": "fits",
    ".fit.gz": "fits",
}


def create_loader(format: str) -> WaterfallLoaderPort:
    """Cria um loader para o formato dado.

    Args:
        format: "xarray", "netcdf", "hdf5", "zarr" ou "fits".

    Returns:
        Instância do loader correspondente.

    Raises:
        ValueError: Se o formato não for suportado.
    """
    key = str(format).strip().lower()
    cls = _LOADERS.get(key)
    if cls is None:
        msg = f"Formato não suportado: {format!r} (use {sorted(_LOADERS)})"
        raise ValueError(msg)
    return cls()


def infer_format(source: str | Path) -> str | None:
    """Infere o formato a partir da extensão do caminho."""
    s = str(source)
    lower = s.lower()
    for ext, fmt in sorted(_EXTENSION_TO_FORMAT.items(), key=lambda kv: -len(kv[0])):
        if lower.endswith(ext):
            return fmt
    return None


def load_waterfall(source: str | Path | Any, format: str | None = None, **kwargs: Any) -> Waterfall:
    """Carrega um waterfall a partir de um arquivo ou objeto.

    Args:
        source: Caminho de arquivo, diretório zarr, ``xr.DataArray`` ou
            ``xr.Dataset``.
        format: Formato explícito. None → inferido da extensão; para objetos
            xarray usa o loader "xarray".
        **kwargs: Repassados ao loader (ex.: ``freq_name``, ``time_name``,
            ``data_name``, ``chunks``, ``meta``).

    Returns:
        Waterfall carregado (lazy quando o formato permite).

    Raises:
        FileNotFoundError: Se o caminho não existir.
        ValueError: Se o formato não puder ser determinado/suportado.
    """
    from xarray import DataArray, Dataset

    if isinstance(source, DataArray | Dataset):
        return create_loader("xarray").load(source, **kwargs)

    path = Path(source)
    if not path.exists():
        msg = f"Arquivo não encontrado: {path}"
        raise FileNotFoundError(msg)

    fmt = format or infer_format(path)
    if fmt is None:
        msg = f"Não foi possível inferir o formato de {path.name}; passe format='netcdf'|'hdf5'|'zarr'|'fits'"
        raise ValueError(msg)
    return create_loader(fmt).load(path, **kwargs)


__all__ = [
    "FitsLoader",
    "HDF5Loader",
    "NetCDFLoader",
    "WaterfallLoaderPort",
    "XArrayLoader",
    "ZarrLoader",
    "create_loader",
    "infer_format",
    "load_waterfall",
]
