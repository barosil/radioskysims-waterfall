"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/ports/loader.py

Descrição:
    Port `WaterfallLoaderPort`: contrato para carregar (preferencialmente
    lazy) um waterfall de diferentes fontes/formatos (xarray, netcdf, hdf5,
    zarr, fits) com opções de mapeamento de nomes de variáveis/dimensões.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from radioskysims_waterfall.domain.models import Waterfall


class WaterfallLoaderPort(ABC):
    """Contrato para carregamento de waterfalls.

    Cada adapter concreto implementa ``load`` para um formato/fonte. A
    assinatura é estável e previsível: ``load(source, **kwargs)``.
    """

    format_name: str = "generic"

    @abstractmethod
    def load(self, source: str | Path, **kwargs: Any) -> Waterfall:
        """Carrega um Waterfall da fonte.

        Args:
            source: Caminho do arquivo ou fonte (depende do adapter).
            **kwargs: Opções específicas do formato, incluindo o mapeamento
                de nomes: ``freq_name``, ``time_name``, ``data_name``,
                ``meta``, ``chunks`` (lazy/dask), etc.

        Returns:
            Waterfall com dados (idealmente lazy/dask).

        Raises:
            FileNotFoundError: Se a fonte não existir.
            ValueError: Se o formato/conteúdo for inválido.
        """
        raise NotImplementedError


__all__ = ["WaterfallLoaderPort"]
