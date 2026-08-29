"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/zarr_loader.py

Descrição:
    Loader de waterfalls a partir de stores Zarr (lazy via xarray).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.domain.models import Waterfall


class ZarrLoader(WaterfallLoaderPort):
    """Carrega um Waterfall de um store Zarr usando ``xr.open_zarr``."""

    format_name = "zarr"

    def load(
        self,
        source: str | Path,
        *,
        chunks: dict[str, int] | str | None = None,
        data_name: str | None = None,
        time_name: str = "time",
        freq_name: str = "freq",
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Waterfall:
        """Carrega um Waterfall de um diretório Zarr.

        Args:
            source: Caminho do store Zarr.
            chunks: Chunks dask (dict por dim ou "auto").
            data_name: Variável de dados (quando várias).
            time_name: Nome da dimensão temporal no store.
            freq_name: Nome da dimensão de frequência no store.
            meta: Metadados (dict).
            **kwargs: Repassados a ``xr.open_zarr``.

        Returns:
            Waterfall (lazy/dask).

        Raises:
            FileNotFoundError: Se o store não existir.
        """
        path = Path(source)
        if not path.exists():
            msg = f"Store Zarr não encontrado: {path}"
            raise FileNotFoundError(msg)

        ds = xr.open_zarr(path, chunks=chunks, **kwargs)
        return Waterfall(
            ds,
            data_name=data_name,
            time_name=time_name,
            freq_name=freq_name,
            meta=meta,
        )


__all__ = ["ZarrLoader"]
