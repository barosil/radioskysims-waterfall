"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/filters.py

Descrição:
    Filtros para waterfalls com assinatura estável e previsível:

        filter_temporal / filter_spectral / filter_spatial

    Métodos (``method``): "mean", "median", "mad" (MAD), "arpls".
    Janela (``window``): ``None`` → estatística global (eixo inteiro);
    ``int`` → estatística móvel (running) para filtros 1D; tupla (t, f)
    para o filtro espacial.
    Modo (``mode``): "subtract" (padrão), "divide", "zscore" (padronização
    robusta) ou "return" (devolve a estatística/perfil).

    Máscaras podem ser informadas via ``mask`` (pixels False ignorados no
    cálculo da estatística).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from radioskysims_waterfall.domain.baseline import get_baseline
from radioskysims_waterfall.domain.masks import apply as mask_apply
from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM, normalize_dims

FilterMethod = Literal["mean", "median", "mad", "arpls"]
FilterMode = Literal["subtract", "divide", "zscore", "return"]

_EPS = 1e-12


def _rolling_stat(
    data: xr.DataArray,
    dims: dict[str, int],
    method: FilterMethod,
    skipna: bool,
) -> xr.DataArray:
    roll = data.rolling(dims, center=True, min_periods=1)
    if method == "mean":
        return roll.mean(skipna=skipna)
    if method in {"median", "mad"}:
        return roll.median(skipna=skipna)
    msg = "ArPLS não suporta janela móvel; use window=None"
    raise ValueError(msg)


def _global_stat(
    data: xr.DataArray,
    dim: str | None,
    method: FilterMethod,
    skipna: bool,
) -> xr.DataArray:
    if method == "mean":
        return data.mean(dim, skipna=skipna) if dim else data.mean(skipna=skipna)
    if method in {"median", "mad"}:
        return data.median(dim, skipna=skipna) if dim else data.median(skipna=skipna)
    msg = "ArPLS não é uma estatística global redutora; use get_baseline(method='arpls')"
    raise ValueError(msg)


def _scale_stat(
    data: xr.DataArray,
    center: xr.DataArray,
    dim: str | None,
    method: FilterMethod,
    window: int | None,
    skipna: bool,
) -> xr.DataArray:
    """Escala para o modo 'zscore': std (mean) ou MAD (median/mad)."""
    if method == "mean":
        if window is None:
            return data.std(dim, skipna=skipna) if dim else data.std(skipna=skipna)
        return data.rolling({dim: window}, center=True, min_periods=1).std(skipna=skipna)
    dev = np.abs(data - center)
    if window is None:
        return dev.median(dim, skipna=skipna) if dim else dev.median(skipna=skipna)
    return dev.rolling({dim: window}, center=True, min_periods=1).median(skipna=skipna)


def _combine(
    data: xr.DataArray,
    profile: xr.DataArray,
    scale: xr.DataArray | None,
    method: FilterMethod,
    mode: FilterMode,
) -> xr.DataArray:
    if mode == "return":
        return profile
    if mode == "subtract":
        return data - profile
    if mode == "divide":
        return data / (profile + _EPS)
    if mode == "zscore":
        sc = scale if scale is not None else data
        return (data - profile) / (sc + _EPS)
    msg = f"Modo de filtro desconhecido: {mode!r}"
    raise ValueError(msg)


def _filter_1d(
    data: xr.DataArray,
    dim: str,
    method: FilterMethod,
    window: int | None,
    mode: FilterMode,
    skipna: bool,
    mask: xr.DataArray | None,
    kwargs: dict[str, Any],
) -> xr.DataArray:
    data = normalize_dims(data)
    work = mask_apply(data, mask)  # pixels inválidos → NaN

    if method == "arpls":
        if window is not None:
            msg = "ArPLS é global (não usa janela); use window=None"
            raise ValueError(msg)
        profile = get_baseline(work, method="arpls", dim=dim, skipna=skipna, **kwargs)
        scale = _scale_stat(work, profile, None, "median", None, skipna)
        out = _combine(work, profile, scale, method, mode)
        return out.transpose(*data.dims) if set(out.dims) == set(data.dims) else out

    if window is None:
        profile = _global_stat(work, dim, method, skipna)
        # broadcasting para subtração ao longo do eixo filtrado
        scale = _scale_stat(work, profile, dim, method, None, skipna)
    else:
        profile = _rolling_stat(work, {dim: window}, method, skipna)
        scale = _scale_stat(work, profile, dim, method, window, skipna)

    out = _combine(work, profile, scale, method, mode)
    return out.transpose(*data.dims) if set(out.dims) == set(data.dims) else out


def filter_temporal(
    data: xr.DataArray,
    method: FilterMethod = "median",
    window: int | None = None,
    mode: FilterMode = "subtract",
    skipna: bool = True,
    mask: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Filtro ao longo da dimensão temporal (por canal de frequência).

    Args:
        data: DataArray 2D (time, freq).
        method: "mean", "median", "mad" ou "arpls".
        window: Janela móvel em amostras (None → global).
        mode: "subtract" (padrão), "divide", "zscore" ou "return".
        skipna: Ignorar NaN na estatística.
        mask: Máscara booleana opcional (True = válido).
        **kwargs: Repassados ao método (ex.: lam/ratio para arpls).

    Returns:
        DataArray filtrado.
    """
    return _filter_1d(data, TIME_DIM, method, window, mode, skipna, mask, kwargs)


def filter_spectral(
    data: xr.DataArray,
    method: FilterMethod = "median",
    window: int | None = None,
    mode: FilterMode = "subtract",
    skipna: bool = True,
    mask: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Filtro ao longo da dimensão espectral (por instante de tempo)."""
    return _filter_1d(data, FREQ_DIM, method, window, mode, skipna, mask, kwargs)


def filter_spatial(
    data: xr.DataArray,
    method: FilterMethod = "median",
    window: tuple[int, int] | int | None = None,
    mode: FilterMode = "subtract",
    skipna: bool = True,
    mask: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Filtro espacial 2D (tempo × frequência) com janela retangular.

    Args:
        data: DataArray 2D (time, freq).
        method: "mean", "median" ou "mad" (ArPLS não suporta 2D).
        window: Tupla ``(janela_tempo, janela_freq)``. Se int, janela
            quadrada. None → estatística global (escalar).
        mode: "subtract", "divide", "zscore" ou "return".
        skipna: Ignorar NaN.
        mask: Máscara booleana opcional.
        **kwargs: Reservado (sem efeito em 2D por ora).

    Returns:
        DataArray filtrado.
    """
    if method == "arpls":
        msg = "Filtro espacial não suporta ArPLS; use filter_spectral/temporal"
        raise ValueError(msg)

    data = normalize_dims(data)
    work = mask_apply(data, mask)

    if window is None:
        profile = _global_stat(work, None, method, skipna)
        scale = _scale_stat(work, profile, None, method, None, skipna)
        out = _combine(work, profile, scale, method, mode)
        return out.transpose(*data.dims) if set(out.dims) == set(data.dims) else out

    if isinstance(window, int):
        window = (window, window)
    w_t, w_f = int(window[0]), int(window[1])
    if w_t < 1 or w_f < 1:
        msg = f"Janela espacial deve ser >= 1: {window}"
        raise ValueError(msg)

    profile = _rolling_stat(work, {TIME_DIM: w_t, FREQ_DIM: w_f}, method, skipna)
    if mode == "zscore":
        dev = np.abs(work - profile)
        scale = dev.rolling({TIME_DIM: w_t, FREQ_DIM: w_f}, center=True, min_periods=1).median(skipna=skipna)
    else:
        scale = None
    out = _combine(work, profile, scale, method, mode)
    return out.transpose(*data.dims) if set(out.dims) == set(data.dims) else out


def filter_data(
    data: xr.DataArray,
    axis: Literal["temporal", "spectral", "spatial"],
    method: FilterMethod = "median",
    window: int | tuple[int, int] | None = None,
    mode: FilterMode = "subtract",
    skipna: bool = True,
    mask: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Despacho genérico de filtro por eixo.

    Args:
        data: DataArray 2D.
        axis: "temporal", "spectral" ou "spatial".
        method: Método de filtro.
        window: Janela (int para 1D, tupla para espacial, None → global).
        mode: Modo de aplicação.
        skipna: Ignorar NaN.
        mask: Máscara opcional.
        **kwargs: Repassados ao filtro.

    Returns:
        DataArray filtrado.
    """
    if axis == "temporal":
        return filter_temporal(
            data, method=method, window=window, mode=mode, skipna=skipna, mask=mask, **kwargs
        )
    if axis == "spectral":
        return filter_spectral(
            data, method=method, window=window, mode=mode, skipna=skipna, mask=mask, **kwargs
        )
    if axis == "spatial":
        return filter_spatial(
            data, method=method, window=window, mode=mode, skipna=skipna, mask=mask, **kwargs
        )
    msg = f"Eixo desconhecido: {axis!r}"
    raise ValueError(msg)


__all__ = [
    "FilterMethod",
    "FilterMode",
    "filter_data",
    "filter_spatial",
    "filter_spectral",
    "filter_temporal",
]
