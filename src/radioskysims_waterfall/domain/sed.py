"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/sed.py

Descrição:
    SED (Spectral Energy Distribution) de um waterfall: distribuição do
    sinal por canal de frequência, calculada como média (e dispersão) ao
    longo de um intervalo temporal. Suporta seleção de canais e intervalo.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM, normalize_dims, slice_data


class SED:
    """Espectro médio (SED) de um waterfall ao longo de um intervalo de tempo.

    Attributes:
        values: Perfil médio por canal (xr.DataArray 1D sobre ``freq``).
        std: Desvio padrão por canal (incerteza da média no intervalo).
        median: Mediana por canal.
        n_samples: Número de amostras temporais usadas.
    """

    def __init__(
        self,
        data: xr.DataArray,
        channels: int | slice | list[int] | None = None,
        interval: int | slice | list[int] | None = None,
        mask: xr.DataArray | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Constrói a SED a partir de um waterfall.

        Args:
            data: DataArray 2D (time, freq).
            channels: Seleção de canais (índices). None → todos.
            interval: Intervalo temporal (índices). None → todo.
            mask: Máscara booleana opcional (pixels False → NaN).
            name: Nome da SED (para plots/labels).
        """
        from radioskysims_waterfall.domain.masks import apply as mask_apply

        data = normalize_dims(data)
        if mask is not None:
            data = mask_apply(data, mask)

        selected = data
        if interval is not None:
            selected = slice_data(selected, time=interval)
        if channels is not None:
            selected = slice_data(selected, freq=channels)

        self._data = selected
        self.name = name or getattr(data, "name", "SED")
        self.values: xr.DataArray = selected.mean(TIME_DIM, skipna=True)
        self.std: xr.DataArray = selected.std(TIME_DIM, skipna=True)
        self.median: xr.DataArray = selected.median(TIME_DIM, skipna=True)
        self.n_samples = int(selected.sizes[TIME_DIM])

    # ------------------------------------------------------------------
    @property
    def freq(self) -> xr.DataArray:
        """Coordenada de frequência do perfil."""
        return self.values[FREQ_DIM]

    @property
    def shape(self) -> tuple[int]:
        return self.values.shape

    def to_dataarray(self) -> xr.DataArray:
        """Perfil SED como DataArray (freq)."""
        return self.values

    def __getitem__(self, key: int | slice | list[int]) -> SED:
        """Sub-seleção de canais da SED."""
        return SED(self._data, channels=key, name=self.name)

    def __len__(self) -> int:
        return int(self.values.sizes[FREQ_DIM])

    def __repr__(self) -> str:
        return f"SED(name={self.name!r}, n_channels={len(self)}, n_samples={self.n_samples})"

    def plot(self, **kwargs: Any):
        """Plota a SED (curva). Ver ``plot_sed`` em application/visualization."""
        from radioskysims_waterfall.application.visualization.waterfall_plot import plot_sed

        return plot_sed(self, **kwargs)


__all__ = ["SED"]
