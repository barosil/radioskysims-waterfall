"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/netcdf_loader.py

Descrição:
    Loader de waterfalls a partir de arquivos NetCDF (lazy por padrão via
    xarray; opcionalmente com chunks dask).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.domain.models import Waterfall


class NetCDFLoader(WaterfallLoaderPort):
    """Carrega um Waterfall de um arquivo NetCDF usando xarray."""

    format_name = "netcdf"

    def load(
        self,
        source: str | Path,
        *,
        engine: str | None = None,
        chunks: dict[str, int] | str | None = None,
        data_name: str | None = None,
        time_name: str = "time",
        freq_name: str = "freq",
        decode_times: bool = True,
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Waterfall:
        """Carrega um Waterfall de um arquivo NetCDF.

        Args:
            source: Caminho do arquivo.
            engine: Backend xarray ("netcdf4", "h5netcdf", "scipy").
                None → "h5netcdf".
            chunks: Chunks dask (dict por dim, ou "auto") para lazy/dask.
                None → xarray lazy (leitura preguiçosa do backend).
            data_name: Variável de dados (quando o arquivo tem várias).
            time_name: Nome da dimensão temporal no arquivo.
            freq_name: Nome da dimensão de frequência no arquivo.
            decode_times: Decodificar coordenadas de tempo.
            meta: Metadados (dict).
            **kwargs: Repassados a ``xr.open_dataset``.

        Returns:
            Waterfall (lazy quando o backend suporta).

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        path = Path(source)
        if not path.exists():
            msg = f"Arquivo não encontrado: {path}"
            raise FileNotFoundError(msg)

        ds = xr.open_dataset(
            path,
            engine=engine or "h5netcdf",
            chunks=chunks,
            decode_times=decode_times,
            **kwargs,
        )
        return Waterfall(
            ds,
            data_name=data_name,
            time_name=time_name,
            freq_name=freq_name,
            meta=meta,
        )


__all__ = ["NetCDFLoader"]
