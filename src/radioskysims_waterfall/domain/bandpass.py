"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/bandpass.py

Descrição:
    Estimativa de banda de passagem de um waterfall. Preserva o sinal de
    interesse: canais cuja mediana (por tempo) está acima de um limiar
    customizável (referência + ``threshold`` × ruído robusto MAD).

    Opções de retorno:
    - ``slices``: lista de ``slice`` (índices de frequência) da(s) banda(s).
    - ``mask``: máscara booleana espectral.
    - padrão: dados restritos à banda de interesse.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

from radioskysims_waterfall.domain.masks import apply as mask_apply
from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM, normalize_dims

BandpassMethod = Literal["median", "mean"]

_MAD_K = 1.4826


def _mad(values: np.ndarray) -> float:
    """MAD (mediana dos desvios absolutos) de um array."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    center = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - center)))


def get_bandpass(
    data: xr.DataArray,
    threshold: float = 3.0,
    percentile: float = 50.0,
    method: BandpassMethod = "median",
    mask: xr.DataArray | None = None,
    min_bandwidth: int = 1,
    return_slices: bool = False,
    return_mask: bool = False,
    **kwargs: Any,
) -> list[slice] | xr.DataArray:
    """Determina a banda de passagem de um waterfall.

    Perfil espectral ``p(f)`` = mediana (ou média) ao longo do tempo por
    canal. A referência é o percentil ``percentile`` do perfil e o ruído é a
    dispersão robusta MAD. Um canal é mantido se
    ``p(f) > ref + threshold * mad``.

    Args:
        data: DataArray 2D (time, freq).
        threshold: Limiar em unidades do ruído MAD (default 3.0).
        percentile: Percentil do perfil usado como referência (default 50).
        method: Redução temporal do perfil ("median" ou "mean").
        mask: Máscara booleana opcional (pixels False ignorados).
        min_bandwidth: Largura mínima (em canais) para manter uma banda
            (runs menores que isso são descartados).
        return_slices: Se True, retorna lista de ``slice`` (índices de
            frequência) das bandas contíguas.
        return_mask: Se True, retorna máscara booleana espectral.
        **kwargs: Reservado.

    Returns:
        - Se ``return_slices``: ``list[slice]``.
        - Se ``return_mask``: DataArray booleano (freq).
        - Senão: DataArray restrito à banda de interesse.

    Raises:
        ValueError: Se ``return_slices`` e ``return_mask`` forem ambos True.
    """
    if return_slices and return_mask:
        msg = "Use apenas um de return_slices / return_mask"
        raise ValueError(msg)

    data = normalize_dims(data)
    work = mask_apply(data, mask)

    if method == "median":
        profile = work.median(TIME_DIM, skipna=True)
    elif method == "mean":
        profile = work.mean(TIME_DIM, skipna=True)
    else:
        msg = f"Método desconhecido: {method!r}"
        raise ValueError(msg)

    profile_values = np.asarray(profile.values, dtype=float)
    ref = float(np.nanpercentile(profile_values, percentile)) if profile_values.size else 0.0
    noise = _mad(profile_values)
    if noise <= 0.0:
        noise = float(np.nanstd(profile_values)) or 1.0

    band_values = profile_values > (ref + threshold * noise)

    # Limpeza: remove runs menores que min_bandwidth
    if min_bandwidth > 1 and band_values.any():
        band_values = _filter_small_runs(band_values, min_bandwidth)

    band = xr.DataArray(
        band_values,
        dims=(FREQ_DIM,),
        coords={FREQ_DIM: data[FREQ_DIM].values},
    )

    if return_mask:
        return band
    if return_slices:
        return _contiguous_slices(band_values)

    # Restringe os dados à banda de interesse
    return work.where(band, drop=True)


def _contiguous_slices(band: np.ndarray) -> list[slice]:
    """Converte uma máscara 1D em lista de slices contíguos (índices)."""
    slices: list[slice] = []
    start = None
    for i, v in enumerate(band):
        if v and start is None:
            start = i
        elif not v and start is not None:
            slices.append(slice(start, i))
            start = None
    if start is not None:
        slices.append(slice(start, len(band)))
    return slices


def _filter_small_runs(band: np.ndarray, min_len: int) -> np.ndarray:
    """Descarta runs contíguos de True com comprimento < min_len."""
    out = band.copy()
    n = len(band)
    i = 0
    while i < n:
        if band[i]:
            j = i
            while j < n and band[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


__all__ = ["BandpassMethod", "get_bandpass"]
