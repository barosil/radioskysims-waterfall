"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/operations.py

Descrição:
    Operações puras (sem estado) sobre arrays xarray 2D (tempo × frequência)
    usadas pelo value object `Waterfall`:
    - normalização de dimensões
    - slicing por índice (isel)
    - decimation por fator inteiro (coarsen)
    - interpolação / resampling para nova grade de coordenadas
    - rebinning por arestas arbitrárias (groupby_bins)
    - materialização de arrays lazy (compute)

    Todas as funções devolvem novos objetos (sem mutação), preservando a
    semântica imutável/chainable do `Waterfall`.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

# Nomes canônicos das dimensões do waterfall
TIME_DIM = "time"
FREQ_DIM = "freq"

_REDUCTION_METHODS = {
    "mean",
    "median",
    "min",
    "max",
    "sum",
    "std",
    "var",
    "first",
    "last",
}


def normalize_dims(
    data: xr.DataArray,
    *,
    time_name: str = TIME_DIM,
    freq_name: str = FREQ_DIM,
) -> xr.DataArray:
    """Renomeia as dimensões para os nomes canônicos e garante ordem (time, freq).

    Args:
        data: DataArray com duas dimensões (tempo, frequência).
        time_name: Nome atual da dimensão temporal.
        freq_name: Nome atual da dimensão de frequência.

    Returns:
        DataArray com dimensões ``("time", "freq")`` (nessa ordem).

    Raises:
        ValueError: Se o array não tiver exatamente duas dimensões ou se os
            nomes canônicos já existirem mas não corresponderem às dimensões.
    """
    if data.ndim != 2:
        msg = f"Waterfall requer um array 2D (time, freq); recebido ndim={data.ndim}"
        raise ValueError(msg)

    rename = {}
    dims = list(data.dims)

    # Resolve conflito quando o array já usa os nomes canônicos
    if TIME_DIM in dims and FREQ_DIM in dims:
        # já está no formato canônico (ordem não importa; transposição abaixo)
        pass
    elif time_name in dims and freq_name in dims:
        rename = {time_name: TIME_DIM, freq_name: FREQ_DIM}
    elif TIME_DIM in dims and freq_name in dims:
        rename = {freq_name: FREQ_DIM}
    elif time_name in dims and FREQ_DIM in dims:
        rename = {time_name: TIME_DIM}
    else:
        # Sem nomes confiáveis: adota posições [tempo, frequência]
        if data.ndim != 2:
            msg = f"Não foi possível mapear dimensões {dims} para (time, freq)"
            raise ValueError(msg)
        rename = {dims[0]: TIME_DIM, dims[1]: FREQ_DIM}

    out = data
    if rename:
        out = out.rename(rename)
    return out.transpose(TIME_DIM, FREQ_DIM)


def slice_data(
    data: xr.DataArray,
    *,
    time: int | slice | Sequence[int] | None = None,
    freq: int | slice | Sequence[int] | None = None,
) -> xr.DataArray:
    """Fatia por índice (posicional) ao longo das dimensões time/freq.

    Aceita inteiros (escalar), ``slice`` ou sequências de índices. Usa
    ``isel`` (sem rótulos), compatível com arrays lazy (dask).

    Args:
        data: DataArray 2D (time, freq).
        time: Índices/slice temporal (posicional).
        freq: Índices/slice de frequência (posicional).

    Returns:
        DataArray fatiado.
    """
    data = normalize_dims(data)
    selectors: dict[str, Any] = {}
    if time is not None:
        selectors[TIME_DIM] = _as_selector(time)
    if freq is not None:
        selectors[FREQ_DIM] = _as_selector(freq)
    return data.isel(selectors) if selectors else data


def slice_by_value(
    data: xr.DataArray,
    *,
    time: Any = None,
    freq: Any = None,
) -> xr.DataArray:
    """Fatia por rótulo (valor da coordenada) usando ``sel``.

    ``time``/``freq`` podem ser escalares, ``slice(valor_inicial, valor_final)``
    ou listas de rótulos.
    """
    data = normalize_dims(data)
    selectors: dict[str, Any] = {}
    if time is not None:
        selectors[TIME_DIM] = time
    if freq is not None:
        selectors[FREQ_DIM] = freq
    return data.sel(selectors) if selectors else data


def _as_selector(value: int | slice | Sequence[int]) -> int | slice | list[int]:
    if isinstance(value, slice):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    return list(value)


def decimate(
    data: xr.DataArray,
    *,
    time_factor: int = 1,
    freq_factor: int = 1,
    method: str = "mean",
    skipna: bool = False,
) -> xr.DataArray:
    """Decimação por fator inteiro usando redução por blocos (``coarsen``).

    Cada bloco de ``time_factor`` amostras temporais e ``freq_factor`` canais
    é reduzido pela estatística ``method`` (mean/median/min/max/sum/std/...).
    Restos (boundary) são descartados.

    Args:
        data: DataArray 2D (time, freq).
        time_factor: Fator de decimação temporal (>= 1).
        freq_factor: Fator de decimação espectral (>= 1).
        method: Redução por bloco ("mean", "median", "min", "max", "sum",
            "std", "var", "first", "last").
        skipna: Ignorar NaN dentro dos blocos.

    Returns:
        DataArray decimado.
    """
    data = normalize_dims(data)
    if time_factor < 1 or freq_factor < 1:
        msg = "Fatores de decimação devem ser >= 1"
        raise ValueError(msg)
    if time_factor == 1 and freq_factor == 1:
        return data
    if method not in _REDUCTION_METHODS:
        msg = f"Método de redução desconhecido: {method!r}"
        raise ValueError(msg)

    coarsen_kwargs: dict[str, Any] = {}
    if time_factor > 1:
        coarsen_kwargs[TIME_DIM] = time_factor
    if freq_factor > 1:
        coarsen_kwargs[FREQ_DIM] = freq_factor

    coarsened = data.coarsen(coarsen_kwargs, boundary="trim")
    reducer = getattr(coarsened, method)
    return reducer(skipna=skipna) if method not in {"first", "last"} else reducer()


def interpolate(
    data: xr.DataArray,
    *,
    time: np.ndarray | None = None,
    freq: np.ndarray | None = None,
    method: str = "linear",
    fill_value: Any = np.nan,
    assume_sorted: bool = True,
) -> xr.DataArray:
    """Interpola o waterfall para uma nova grade de coordenadas.

    Pode subamostrar (downsample) ou superamostrar (upsample). Pontos fora da
    grade original são preenchidos com ``fill_value`` (padrão NaN).

    Args:
        data: DataArray 2D (time, freq).
        time: Novas coordenadas temporais (array). Se None, mantém.
        freq: Novas coordenadas de frequência (array). Se None, mantém.
        method: Método de interpolação ("linear", "nearest", "cubic", "slinear", ...).
        fill_value: Valor para pontos fora do domínio.
        assume_sorted: Se as novas coordenadas já estão ordenadas.

    Returns:
        DataArray interpolado.
    """
    data = normalize_dims(data)
    targets: dict[str, Any] = {}
    if time is not None:
        targets[TIME_DIM] = np.asarray(time)
    if freq is not None:
        targets[FREQ_DIM] = np.asarray(freq)
    if not targets:
        return data
    return data.interp(
        targets,
        method=method,
        kwargs={"fill_value": fill_value},
        assume_sorted=assume_sorted,
    )


def resample_data(
    data: xr.DataArray,
    *,
    time: np.ndarray | None = None,
    freq: np.ndarray | None = None,
    method: str = "linear",
    fill_value: Any = np.nan,
    assume_sorted: bool = True,
) -> xr.DataArray:
    """Resampling para nova grade de coordenadas (alias de ``interpolate``)."""
    return interpolate(
        data,
        time=time,
        freq=freq,
        method=method,
        fill_value=fill_value,
        assume_sorted=assume_sorted,
    )


def rebin(
    data: xr.DataArray,
    *,
    time_edges: np.ndarray | None = None,
    freq_edges: np.ndarray | None = None,
    method: str = "mean",
    skipna: bool = False,
) -> xr.DataArray:
    """Rebin por arestas arbitrárias ao longo das dimensões.

    As coordenadas são agrupadas em bins definidos por ``*_edges`` e cada bin
    é reduzido pela estatística ``method``. A nova coordenada é o centro de
    cada bin.

    Args:
        data: DataArray 2D (time, freq).
        time_edges: Arestas dos bins temporais (array crescente).
        freq_edges: Arestas dos bins de frequência (array crescente).
        method: Redução ("mean", "median", "min", "max", "sum", ...).
        skipna: Ignorar NaN dentro dos bins.

    Returns:
        DataArray com as dimensões reduzidas aos centros dos bins.
    """
    data = normalize_dims(data)
    out = data
    for dim, edges_in in ((TIME_DIM, time_edges), (FREQ_DIM, freq_edges)):
        if edges_in is None:
            continue
        edges = np.asarray(edges_in, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            msg = f"Arestas de {dim} devem ser um array 1D com pelo menos 2 pontos"
            raise ValueError(msg)
        centers = 0.5 * (edges[:-1] + edges[1:])
        grouped = out.groupby_bins(
            dim,
            bins=edges,
            labels=centers,
            include_lowest=True,
        )
        reducer = getattr(grouped, method)
        out = reducer(skipna=skipna) if method not in {"first", "last"} else reducer()
        out = out.rename({f"{dim}_bins": dim})
        out = out.assign_coords({dim: out[dim].values})
    return out


def compute(data: xr.DataArray) -> xr.DataArray:
    """Materializa um array lazy (dask) em memória (numpy)."""
    if data.chunks is not None and hasattr(data.data, "compute"):
        return data.compute()
    return data


def estimate_nbytes(data: xr.DataArray) -> int:
    """Estima o tamanho em bytes da materialização de ``data``."""
    n = int(np.prod(data.shape)) if data.size else 0
    itemsize = int(np.dtype(data.dtype).itemsize)
    return n * itemsize


__all__ = [
    "FREQ_DIM",
    "TIME_DIM",
    "compute",
    "decimate",
    "estimate_nbytes",
    "interpolate",
    "normalize_dims",
    "rebin",
    "resample_data",
    "slice_by_value",
    "slice_data",
]
