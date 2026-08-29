"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/masks.py

Descrição:
    Utilidades de máscaras (booleanas) para waterfalls. Uma máscara tem as
    mesmas dimensões do waterfall (time × freq) e marca pixels válidos (True)
    ou inválidos (False). As funções são puras e devolvem novos arrays.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from radioskysims_waterfall.domain.operations import normalize_dims

Comparison = Literal["gt", "ge", "lt", "le", "eq", "ne"]


def _compare(values: np.ndarray, threshold: float, comparison: str) -> np.ndarray:
    ops = {
        "gt": np.greater,
        "ge": np.greater_equal,
        "lt": np.less,
        "le": np.less_equal,
        "eq": np.equal,
        "ne": np.not_equal,
    }
    if comparison not in ops:
        msg = f"Comparação desconhecida: {comparison!r} (use {sorted(ops)})"
        raise ValueError(msg)
    return ops[comparison](values, threshold)


def from_threshold(
    data: xr.DataArray,
    threshold: float,
    comparison: Comparison = "gt",
) -> xr.DataArray:
    """Cria uma máscara booleana a partir de um limiar.

    ``True`` onde ``data op threshold`` (padrão: maior que o limiar).

    Args:
        data: DataArray 2D (time, freq).
        threshold: Limiar de comparação.
        comparison: Operador ("gt", "ge", "lt", "le", "eq", "ne").

    Returns:
        Máscara booleana com as mesmas dimensões de ``data``.
    """
    data = normalize_dims(data)
    mask = xr.apply_ufunc(
        _compare,
        data,
        threshold,
        comparison,
        dask="parallelized",
        output_dtypes=[bool],
    )
    return mask


def from_range(
    data: xr.DataArray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> xr.DataArray:
    """Máscara de valores dentro de ``[vmin, vmax]`` (bounds inclusivos).

    ``None`` em um dos limites significa sem restrição naquele lado.
    """
    data = normalize_dims(data)
    mask = xr.ones_like(data, dtype=bool)
    if vmin is not None:
        mask = mask & (data >= vmin)
    if vmax is not None:
        mask = mask & (data <= vmax)
    return mask


def from_finite(data: xr.DataArray) -> xr.DataArray:
    """Máscara de valores finitos (não-NaN e não-inf)."""
    data = normalize_dims(data)
    return xr.apply_ufunc(
        np.isfinite,
        data,
        dask="parallelized",
        output_dtypes=[bool],
    )


def combine(
    *masks: xr.DataArray,
    op: Literal["and", "or", "xor"] = "and",
) -> xr.DataArray:
    """Combina múltiplas máscaras booleanas com um operador lógico."""
    if not masks:
        msg = "Pelo menos uma máscara é necessária"
        raise ValueError(msg)
    result = normalize_dims(masks[0]).astype(bool, copy=True)
    for other_mask in masks[1:]:
        m = normalize_dims(other_mask).astype(bool)
        if op == "and":
            result = result & m
        elif op == "or":
            result = result | m
        elif op == "xor":
            result = result ^ m
        else:
            msg = f"Operador desconhecido: {op!r}"
            raise ValueError(msg)
    return result


def invert(mask: xr.DataArray) -> xr.DataArray:
    """Inverte uma máscara booleana."""
    mask = normalize_dims(mask)
    return ~mask.astype(bool)


def fraction(mask: xr.DataArray) -> float:
    """Fração de pixels True em uma máscara (0..1)."""
    mask = normalize_dims(mask).astype(bool)
    return float(mask.sum().item() / mask.size)


def count(mask: xr.DataArray) -> int:
    """Número de pixels True em uma máscara."""
    mask = normalize_dims(mask).astype(bool)
    return int(mask.sum().item())


def as_bool(mask: xr.DataArray) -> xr.DataArray:
    """Garante que a máscara seja booleana."""
    return normalize_dims(mask).astype(bool, copy=False)


def apply(
    data: xr.DataArray,
    mask: xr.DataArray | None,
    *,
    fill_value: float = np.nan,
) -> xr.DataArray:
    """Aplica uma máscara ao dado: onde for False, substitui por ``fill_value``.

    Args:
        data: DataArray 2D.
        mask: Máscara booleana (True = válido). Se None, retorna ``data``.
        fill_value: Valor para pixels mascarados (padrão NaN).

    Returns:
        DataArray com os pixels inválidos preenchidos por ``fill_value``.
    """
    if mask is None:
        return data
    data = normalize_dims(data)
    mask = as_bool(mask)
    return data.where(mask, other=fill_value)


__all__ = [
    "apply",
    "as_bool",
    "combine",
    "count",
    "fraction",
    "from_finite",
    "from_range",
    "from_threshold",
    "invert",
]
