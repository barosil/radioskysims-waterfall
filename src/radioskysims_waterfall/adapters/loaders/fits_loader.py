"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/fits_loader.py

Descrição:
    Loader de waterfalls a partir de arquivos FITS. Tenta o backend "fits"
    do xarray; em caso de falha, lê o HDU primário com astropy.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.domain.models import Waterfall


class FitsLoader(WaterfallLoaderPort):
    """Carrega um Waterfall de um arquivo FITS."""

    format_name = "fits"

    def load(
        self,
        source: str | Path,
        *,
        data_name: str | None = None,
        time_name: str = "time",
        freq_name: str = "freq",
        hdu: int = 0,
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Waterfall:
        """Carrega um Waterfall de um arquivo FITS.

        Args:
            source: Caminho do arquivo.
            data_name: Variável de dados (para o backend xarray).
            time_name: Nome da dimensão temporal (xarray).
            freq_name: Nome da dimensão de frequência (xarray).
            hdu: HDU a usar no fallback astropy (default 0).
            meta: Metadados (dict).
            **kwargs: Repassados a ``xr.open_dataset(engine='fits')``.

        Returns:
            Waterfall.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se não houver dados 2D legíveis.
        """
        path = Path(source)
        if not path.exists():
            msg = f"Arquivo não encontrado: {path}"
            raise FileNotFoundError(msg)

        try:
            import xarray as xr

            ds = xr.open_dataset(path, engine="fits", **kwargs)
            return Waterfall(ds, data_name=data_name, time_name=time_name, freq_name=freq_name, meta=meta)
        except Exception as exc:
            return self._load_astropy(path, hdu=hdu, meta=meta, xarray_error=exc)

    def _load_astropy(
        self, path: Path, *, hdu: int, meta: dict[str, Any] | None, xarray_error: Exception
    ) -> Waterfall:
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover
            msg = "Backend FITS indisponível (instale astropy): xarray falhou e astropy ausente"
            raise RuntimeError(msg) from exc

        with fits.open(path) as hdul:
            if len(hdul) <= hdu:
                msg = f"HDU {hdu} inexistente em {path}"
                raise ValueError(msg) from xarray_error
            data = hdul[hdu].data

        arr = np.asarray(data)
        if arr.ndim != 2:
            msg = f"HDU {hdu} não contém dados 2D (shape={arr.shape})"
            raise ValueError(msg) from xarray_error

        time = np.arange(arr.shape[0])
        freq = np.arange(arr.shape[1])
        return Waterfall((arr, time, freq), meta=meta)


__all__ = ["FitsLoader"]
