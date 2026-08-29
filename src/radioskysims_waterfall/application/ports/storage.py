"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/ports/storage.py

Descrição:
    Port `WaterfallStoragePort`: contrato para persistir um waterfall em
    disco, com suporte opcional a controle de versão de arquivos e
    integridade (escrita atômica).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from radioskysims_waterfall.domain.models import Waterfall


class WaterfallStoragePort(ABC):
    """Contrato para persistência de waterfalls."""

    @abstractmethod
    def save(
        self,
        waterfall: Waterfall,
        path: str | Path,
        *,
        format: str | None = None,
        version: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Persiste o waterfall.

        Args:
            waterfall: Waterfall a salvar.
            path: Caminho de destino.
            format: Formato ("netcdf", "zarr", "hdf5", "fits"). None → infere
                da extensão.
            version: Se True, preserva versões anteriores do arquivo.
            **kwargs: Opções específicas do formato.

        Returns:
            Caminho efetivamente escrito.

        Raises:
            ValueError: Se o formato for inválido/não suportado.
        """
        raise NotImplementedError


__all__ = ["WaterfallStoragePort"]
