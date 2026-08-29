"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/normalize.py

Descrição:
    Normalizações de waterfalls com tratamento de NaN (skipna). Todas as
    funções são puras e devolvem novos DataArrays.

    Métodos: "minmax", "zscore", "robust", "percentile", "median", "max",
    "sum", "db".

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from radioskysims_waterfall.domain.operations import normalize_dims

NormalizeMethod = Literal["minmax", "zscore", "robust", "percentile", "median", "max", "sum", "db"]

_MAD_K = 1.4826
_EPS = 1e-12


def _mad(values: xr.DataArray) -> xr.DataArray:
    center = values.median(skipna=True)
    return np.abs(values - center).median(skipna=True)


def normalize(
    data: xr.DataArray,
    method: NormalizeMethod = "minmax",
    *,
    pmin: float | None = None,
    pmax: float | None = None,
    ref: float = 1.0,
    **kwargs: float,
) -> xr.DataArray:
    """Normaliza um waterfall.

    Args:
        data: DataArray 2D (time, freq).
        method: Estratégia de normalização (ver abaixo).
        pmin: Percentil inferior para "minmax"/"percentile" (None → min/0).
        pmax: Percentil superior para "minmax"/"percentile" (None → max/100).
        ref: Valor de referência para "db".
        **kwargs: Reservado.

    Métodos:
        - "minmax": (x - lo) / (hi - lo), com lo/hi = min/max ou percentis.
        - "zscore": (x - mean) / std.
        - "robust": (x - median) / (1.4826 * MAD).
        - "percentile": mapeia [pmin, pmax] → [0, 1].
        - "median": x / median.
        - "max": x / max(abs(x)).
        - "sum": x / sum(x).
        - "db": 10 * log10(x / ref).

    Returns:
        DataArray normalizado.
    """
    data = normalize_dims(data)
    x = data.astype(float)

    if method == "minmax":
        lo = x.quantile(pmin / 100.0, skipna=True) if pmin is not None else x.min(skipna=True)
        hi = x.quantile(pmax / 100.0, skipna=True) if pmax is not None else x.max(skipna=True)
        return (x - lo) / (hi - lo + _EPS)

    if method == "percentile":
        if pmin is None or pmax is None:
            msg = "percentile requer pmin e pmax"
            raise ValueError(msg)
        lo = x.quantile(pmin / 100.0, skipna=True)
        hi = x.quantile(pmax / 100.0, skipna=True)
        return (x - lo) / (hi - lo + _EPS)

    if method == "zscore":
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        return (x - mu) / (sd + _EPS)

    if method == "robust":
        center = x.median(skipna=True)
        scale = _mad(x)
        return (x - center) / (_MAD_K * scale + _EPS)

    if method == "median":
        return x / (x.median(skipna=True) + _EPS)

    if method == "max":
        return x / (np.abs(x).max(skipna=True) + _EPS)

    if method == "sum":
        return x / (x.sum(skipna=True) + _EPS)

    if method == "db":
        return 10.0 * np.log10(np.abs(x) / ref + _EPS)

    msg = f"Método de normalização desconhecido: {method!r}"
    raise ValueError(msg)


__all__ = ["NormalizeMethod", "normalize"]
