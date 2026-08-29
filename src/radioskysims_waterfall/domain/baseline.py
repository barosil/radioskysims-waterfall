"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/baseline.py

Descrição:
    Estimativa de baseline (linha de base) de perfis 1D e de waterfalls.

    Métodos suportados:
    - "median": mediana global ou móvel (running) com janela.
    - "mad": centro robusto (mediana) e dispersão MAD; baseline = mediana.
    - "arpls": Asymmetrically Reweighted Penalized Least Squares (Baek et al.
      2015) — baseline suave, global (não usa janela).

    As funções são puras; aceitam NaN via ``skipna`` e devolvem arrays do
    mesmo formato da entrada.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from radioskysims_waterfall.domain.operations import FREQ_DIM, normalize_dims

BaselineMethod = Literal["median", "mad", "arpls"]

_MAD_K = 1.4826  # fator de escala para MAD → desvio padrão (distribuição normal)


def baseline_arpls(
    y: np.ndarray,
    lam: float = 1e4,
    ratio: float = 0.05,
    itermax: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Baseline por ArPLS (1D), baseado em Baek et al. 2015.

    Copia a implementação de referência (vendored em ``src/arpls``) usando
    apenas numpy + scipy (sem numba), tornando o pacote autossuficiente.

    Args:
        y: Sinal 1D de entrada.
        lam: Suavidade (quanto maior, mais suave o baseline).
        ratio: Critério de parada (variação relativa dos pesos).
        itermax: Número máximo de iterações.
        tol: Tolerância numérica adicional para parada.

    Returns:
        Baseline 1D com o mesmo comprimento de ``y``.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.full_like(y, np.nanmedian(y) if n else 0.0)

    # D = diferença de segunda ordem (n-2, n) como matriz esparsa
    d0 = eye(n, format="csc")
    d = d0[1:] - d0[:-1]
    d = d[1:] - d[:-1]
    d = d.T

    w = np.ones(n)
    for _ in range(itermax):
        wd = diags(w, 0, shape=(n, n))
        z = spsolve(wd + lam * (d @ d.T), w * y)
        r = y - z
        neg = r[r < 0]
        if neg.size == 0:
            break
        m = float(np.mean(neg))
        s = float(np.std(neg))
        if s == 0.0:
            s = 1.0
        with np.errstate(over="ignore", divide="ignore"):
            wt = 1.0 / (1.0 + np.exp(2.0 * (r - (2.0 * s - m)) / s))
        delta = float(np.linalg.norm(w - wt) / max(np.linalg.norm(w), 1e-12))
        w = wt
        if delta < ratio or delta < tol:
            break
    return z


def _running_stat_nan(values: np.ndarray, window: int, reducer) -> np.ndarray:
    """Estatística móvel 1D com tratamento de NaN (janela centrada)."""
    n = values.size
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        seg = seg[~np.isnan(seg)]
        out[i] = reducer(seg) if seg.size else np.nan
    return out


def _running_median(values: np.ndarray, window: int, skipna: bool = True) -> np.ndarray:
    """Mediana móvel 1D (numpy/scipy), com tratamento de NaN."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if skipna and np.isnan(values).any():
        return _running_stat_nan(values, window, np.median)
    from scipy.ndimage import median_filter

    return median_filter(values, size=window, mode="nearest")


def _running_mean(values: np.ndarray, window: int, skipna: bool = True) -> np.ndarray:
    """Média móvel 1D (numpy/scipy), com tratamento de NaN."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if skipna and np.isnan(values).any():
        return _running_stat_nan(values, window, np.mean)
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(values, size=window, mode="nearest")


def baseline_1d(
    y: np.ndarray,
    method: BaselineMethod = "median",
    window: int | None = None,
    skipna: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Baseline 1D por método.

    Args:
        y: Sinal 1D.
        method: "median", "mad" ou "arpls".
        window: Janela da mediana móvel. None → global (todo o eixo).
        skipna: Ignorar NaN.
        **kwargs: Repassados ao método (ex.: lam/ratio/itermax do arpls).

    Returns:
        Baseline 1D (mesmo comprimento de ``y``).

    Raises:
        ValueError: Para método desconhecido ou arpls com ``window``.
    """
    y = np.asarray(y, dtype=float)

    if method == "arpls":
        if window is not None:
            msg = "ArPLS é global (não usa janela); passe window=None"
            raise ValueError(msg)
        yc = y if not skipna else np.nan_to_num(y, nan=np.nanmedian(y))
        return baseline_arpls(yc, **kwargs)

    if method == "median":
        if window is None:
            return np.full_like(y, np.nanmedian(y) if skipna else np.median(y))
        return _running_median(y, window, skipna=skipna)

    if method == "mad":
        center = baseline_1d(y, "median", window=window, skipna=skipna)
        # Baseline = centro robusto; escala disponível via mad_scale_1d()
        return center

    msg = f"Método de baseline desconhecido: {method!r}"
    raise ValueError(msg)


def mad_scale_1d(
    y: np.ndarray,
    window: int | None = None,
    skipna: bool = True,
) -> np.ndarray:
    """Dispersão MAD (escala robusta) de um sinal 1D, global ou móvel."""
    center = baseline_1d(y, "median", window=window, skipna=skipna)
    dev = np.abs(np.asarray(y, dtype=float) - center)
    return baseline_1d(dev, "median", window=window, skipna=skipna)


def get_baseline(
    data: xr.DataArray,
    method: BaselineMethod = "median",
    window: int | None = None,
    dim: Literal["freq", "time"] = FREQ_DIM,
    skipna: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Baseline de um waterfall ao longo de uma dimensão (freq ou time).

    Para ``dim="freq"``, aplica o baseline em cada linha temporal (perfil
    espectral por instante). Para ``dim="time"``, aplica em cada canal.
    Usa métodos nativos de xarray (rolling/median) para median/mad.

    Args:
        data: DataArray 2D (time, freq).
        method: Método de baseline ("median", "mad", "arpls").
        window: Janela móvel (None → global).
        dim: Dimensão ao longo da qual estimar o baseline.
        skipna: Ignorar NaN.
        **kwargs: Repassados ao método (ex.: lam/ratio do arpls).

    Returns:
        Baseline com as mesmas dimensões de ``data``.
    """
    data = normalize_dims(data)

    if method == "arpls":
        if window is not None:
            msg = "ArPLS é global (não usa janela); passe window=None"
            raise ValueError(msg)
        return _apply_arpls(data, dim=dim, skipna=skipna, **kwargs)

    if method in {"median", "mad"}:
        if window is None:
            center = data.median(dim, skipna=skipna)
            return xr.broadcast(center, data)[0]
        return data.rolling({dim: window}, center=True, min_periods=1).median(skipna=skipna)

    msg = f"Método de baseline desconhecido: {method!r}"
    raise ValueError(msg)


def _apply_arpls(
    data: xr.DataArray,
    *,
    dim: str,
    skipna: bool,
    **kwargs: Any,
) -> xr.DataArray:
    """Aplica ArPLS por fatia ao longo de ``dim`` (numpy/scipy, sem pandas)."""

    def _one(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if skipna and np.isnan(y).any():
            y = np.where(np.isnan(y), np.nanmedian(y), y)
        if y.ndim == 1:
            return baseline_arpls(y, **kwargs)
        # dask="parallelized" envia o bloco com a core dim como último eixo
        return np.apply_along_axis(lambda v: baseline_arpls(v, **kwargs), -1, y)

    return xr.apply_ufunc(
        _one,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[data.dtype],
    )


__all__ = [
    "_MAD_K",
    "BaselineMethod",
    "baseline_1d",
    "baseline_arpls",
    "get_baseline",
    "mad_scale_1d",
]
