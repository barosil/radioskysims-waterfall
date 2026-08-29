"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/events.py

Descrição:
    Suporte a eventos associados a um waterfall (por tempo e frequência).
    Um evento é um ponto (tempo, frequência) com rótulo opcional; eventos são
    agregados à grade do waterfall como máscara, contagem ou densidade.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM


class Event(BaseModel):
    """Evento pontual (tempo, frequência) associável a um waterfall.

    Attributes:
        time: Valor da coordenada temporal (mesma unidade da coordenada do
            waterfall; tipicamente segundos ou MJD).
        freq: Valor da coordenada de frequência (tipicamente Hz).
        label: Rótulo textual do evento (ex.: identificador do trânsito).
        kind: Tipo/categoria do evento (ex.: "satellite", "rfi", "pulse").
        extra: Metadados adicionais livres.
    """

    model_config = ConfigDict(extra="allow")

    time: float = Field(..., description="Coordenada temporal do evento")
    freq: float = Field(..., description="Coordenada de frequência do evento")
    label: str | None = None
    kind: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def _to_arrays(events: Sequence[Event] | Any) -> tuple[np.ndarray, np.ndarray]:
    """Extrai arrays de tempo/frequência dos eventos (sem dependência de pandas).

    Aceita:
    - Sequência de `Event`;
    - objeto DataFrame-like (duck-typing via ``to_dict``), ex.: pandas DataFrame;
    - dict com chaves "time"/"freq";
    - ndarray estruturado ou 2D com colunas (time, freq).
    """
    if isinstance(events, list | tuple) and events and isinstance(events[0], Event):
        times = np.array([e.time for e in events], dtype=float)
        freqs = np.array([e.freq for e in events], dtype=float)
        return times, freqs

    to_dict = getattr(events, "to_dict", None)
    if callable(to_dict):  # DataFrame-like (ex.: pandas.DataFrame)
        records = to_dict("records")
        times = np.array([r["time"] for r in records], dtype=float)
        freqs = np.array([r["freq"] for r in records], dtype=float)
        return times, freqs

    if isinstance(events, dict):
        times = np.asarray(events["time"], dtype=float)
        freqs = np.asarray(events["freq"], dtype=float)
        return times, freqs

    arr = np.asarray(events)
    if arr.dtype.names:  # ndarray estruturado
        times = arr["time"].astype(float)
        freqs = arr["freq"].astype(float)
        return times, freqs
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0].astype(float), arr[:, 1].astype(float)

    msg = "Eventos devem ser Sequence[Event], DataFrame-like, dict ou ndarray"
    raise TypeError(msg)


def _nearest_index(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Índice (posicional) do valor mais próximo em ``values`` para cada target."""
    if targets.size == 0:
        return np.array([], dtype=int)
    idx = np.searchsorted(values, targets)
    idx = np.clip(idx, 1, len(values) - 1)
    left = values[idx - 1]
    right = values[idx]
    idx -= targets - left < right - targets
    return idx.astype(int)


def to_indices(
    events: Sequence[Event] | Any,
    time: np.ndarray,
    freq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Converte coordenadas de eventos para índices posicionais na grade.

    Args:
        events: Eventos (Sequence[Event] ou DataFrame-like com colunas
            time/freq).
        time: Coordenadas temporais do waterfall (ordenadas).
        freq: Coordenadas de frequência do waterfall (ordenadas).

    Returns:
        Tupla ``(time_idx, freq_idx)`` com os índices posicionais.
    """
    time_vals, freq_vals = _to_arrays(events)
    if time_vals.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    t_idx = _nearest_index(np.asarray(time, dtype=float), time_vals)
    f_idx = _nearest_index(np.asarray(freq, dtype=float), freq_vals)
    return t_idx, f_idx


def to_mask(
    events: Sequence[Event] | Any,
    time: np.ndarray,
    freq: np.ndarray,
) -> xr.DataArray:
    """Máscara booleana (time × freq) marcando pixels que contêm eventos.

    Returns:
        DataArray booleano com dims (time, freq); True onde há evento.
    """
    t_idx, f_idx = to_indices(events, time, freq)
    mask = np.zeros((len(time), len(freq)), dtype=bool)
    if t_idx.size:
        mask[t_idx, f_idx] = True
    return xr.DataArray(mask, dims=(TIME_DIM, FREQ_DIM), coords={TIME_DIM: time, FREQ_DIM: freq})


def to_counts(
    events: Sequence[Event] | Any,
    time: np.ndarray,
    freq: np.ndarray,
) -> xr.DataArray:
    """Contagem de eventos por pixel da grade (time × freq)."""
    t_idx, f_idx = to_indices(events, time, freq)
    counts = np.zeros((len(time), len(freq)), dtype=np.int64)
    if t_idx.size:
        np.add.at(counts, (t_idx, f_idx), 1)
    return xr.DataArray(counts, dims=(TIME_DIM, FREQ_DIM), coords={TIME_DIM: time, FREQ_DIM: freq})


def to_density(
    events: Sequence[Event] | Any,
    time: np.ndarray,
    freq: np.ndarray,
    *,
    time_radius: int = 0,
    freq_radius: int = 0,
) -> xr.DataArray:
    """Densidade de eventos por pixel, com vizinhança (raios em pixels).

    Cada evento contribui 1 para todos os pixels dentro do retângulo
    ``[time_idx - time_radius, time_idx + time_radius]`` ×
    ``[freq_idx - freq_radius, freq_idx + freq_radius]``.

    Returns:
        DataArray float com a densidade (contagem com vizinhança).
    """
    t_idx, f_idx = to_indices(events, time, freq)
    counts = np.zeros((len(time), len(freq)), dtype=np.int64)
    n_time, n_freq = len(time), len(freq)
    for ti, fi in zip(t_idx, f_idx, strict=False):
        t0, t1 = max(0, ti - time_radius), min(n_time, ti + time_radius + 1)
        f0, f1 = max(0, fi - freq_radius), min(n_freq, fi + freq_radius + 1)
        counts[t0:t1, f0:f1] += 1
    return xr.DataArray(counts, dims=(TIME_DIM, FREQ_DIM), coords={TIME_DIM: time, FREQ_DIM: freq})


def aggregate(
    events: Sequence[Event] | Any,
    time: np.ndarray,
    freq: np.ndarray,
    *,
    mode: Literal["mask", "count", "density"] = "mask",
    time_radius: int = 0,
    freq_radius: int = 0,
) -> xr.DataArray:
    """Agrega eventos à grade do waterfall no modo escolhido.

    Args:
        events: Eventos (lista de `Event` ou DataFrame).
        time: Coordenadas temporais.
        freq: Coordenadas de frequência.
        mode: "mask" (bool), "count" (int) ou "density" (com vizinhança).
        time_radius: Raio temporal em pixels (apenas mode="density").
        freq_radius: Raio de frequência em pixels (apenas mode="density").

    Returns:
        DataArray agregado (time, freq).
    """
    if mode == "mask":
        return to_mask(events, time, freq)
    if mode == "count":
        return to_counts(events, time, freq)
    if mode == "density":
        return to_density(events, time, freq, time_radius=time_radius, freq_radius=freq_radius)
    msg = f"Modo desconhecido: {mode!r}"
    raise ValueError(msg)


__all__ = [
    "Event",
    "aggregate",
    "to_counts",
    "to_density",
    "to_indices",
    "to_mask",
]
