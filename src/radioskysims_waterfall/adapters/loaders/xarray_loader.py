"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/xarray_loader.py

Descrição:
    Loader de waterfalls a partir de objetos xarray (DataArray/Dataset)
    já em memória ou lazy.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.domain.models import Waterfall


class XArrayLoader(WaterfallLoaderPort):
    """Constrói um Waterfall a partir de ``xr.DataArray`` ou ``xr.Dataset``."""

    format_name = "xarray"

    def load(
        self,
        source: Any,
        *,
        data_name: str | None = None,
        time_name: str = "time",
        freq_name: str = "freq",
        meta: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Waterfall:
        """Carrega um Waterfall de um objeto xarray.

        Args:
            source: ``xr.DataArray`` ou ``xr.Dataset``.
            data_name: Nome da variável de dados (quando Dataset).
            time_name: Nome da dimensão temporal na entrada.
            freq_name: Nome da dimensão de frequência na entrada.
            meta: Metadados (dict).
            **kwargs: Ignorados (compatibilidade).

        Returns:
            Waterfall.
        """
        return Waterfall(
            source,
            data_name=data_name,
            time_name=time_name,
            freq_name=freq_name,
            meta=meta,
        )


__all__ = ["XArrayLoader"]
