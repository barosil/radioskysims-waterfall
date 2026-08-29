"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/lightcurve.py

Descrição:
    Curva de luz de um waterfall: variação do sinal no tempo, calculada como
    média (e dispersão) sobre uma faixa de frequências. Suporta seleção de
    faixa espectral e intervalo temporal.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM, normalize_dims, slice_data


class LightCurve:
    """Curva de luz média de um waterfall ao longo de uma faixa de frequência.

    Attributes:
        values: Perfil médio por instante (xr.DataArray 1D sobre ``time``).
        std: Desvio padrão por instante.
        median: Mediana por instante.
        n_channels: Número de canais usados.
    """

    def __init__(
        self,
        data: xr.DataArray,
        frequencies: int | slice | list[int] | None = None,
        interval: int | slice | list[int] | None = None,
        mask: xr.DataArray | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Constrói a curva de luz a partir de um waterfall.

        Args:
            data: DataArray 2D (time, freq).
            frequencies: Seleção de canais (índices). None → todos.
            interval: Intervalo temporal (índices). None → todo.
            mask: Máscara booleana opcional (pixels False → NaN).
            name: Nome da curva (para plots/labels).
        """
        from radioskysims_waterfall.domain.masks import apply as mask_apply

        data = normalize_dims(data)
        if mask is not None:
            data = mask_apply(data, mask)

        selected = data
        if frequencies is not None:
            selected = slice_data(selected, freq=frequencies)
        if interval is not None:
            selected = slice_data(selected, time=interval)

        self._data = selected
        self.name = name or getattr(data, "name", "LightCurve")
        self.values: xr.DataArray = selected.mean(FREQ_DIM, skipna=True)
        self.std: xr.DataArray = selected.std(FREQ_DIM, skipna=True)
        self.median: xr.DataArray = selected.median(FREQ_DIM, skipna=True)
        self.n_channels = int(selected.sizes[FREQ_DIM])

    # ------------------------------------------------------------------
    @property
    def time(self) -> xr.DataArray:
        """Coordenada temporal da curva."""
        return self.values[TIME_DIM]

    @property
    def shape(self) -> tuple[int]:
        return self.values.shape

    def to_dataarray(self) -> xr.DataArray:
        """Curva de luz como DataArray (time)."""
        return self.values

    def __getitem__(self, key: int | slice | list[int]) -> LightCurve:
        """Sub-seleção temporal da curva de luz."""
        return LightCurve(self._data, interval=key, name=self.name)

    def __len__(self) -> int:
        return int(self.values.sizes[TIME_DIM])

    def __repr__(self) -> str:
        return f"LightCurve(name={self.name!r}, n_points={len(self)}, n_channels={self.n_channels})"

    def plot(self, **kwargs: Any):
        """Plota a curva de luz. Ver ``plot_lightcurve`` em application/visualization."""
        from radioskysims_waterfall.application.visualization.waterfall_plot import plot_lightcurve

        return plot_lightcurve(self, **kwargs)


__all__ = ["LightCurve"]
