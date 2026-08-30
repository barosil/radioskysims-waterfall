"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/models.py

Descrição:
    Value object `Waterfall` (array 2D tempo × frequência + metadados) com
    métodos *chainable* para slicing, resampling, filtros, bandpass, baseline,
    SED, lightcurve, normalização, pipeline, persistência e visualização.

    O `Waterfall` encapsula um ``xr.DataArray`` com dimensões canônicas
    ("time", "freq") — preferencialmente lazy (dask) —, uma máscara booleana
    opcional, um dataset de eventos opcional e metadados ``WaterfallMetadata``.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from radioskysims_waterfall.domain import operations as ops
from radioskysims_waterfall.domain.operations import FREQ_DIM, TIME_DIM, normalize_dims

if TYPE_CHECKING:
    from collections.abc import Sequence

    from radioskysims_waterfall.domain.events import Event
    from radioskysims_waterfall.domain.lightcurve import LightCurve
    from radioskysims_waterfall.domain.sed import SED

# Sentinela para `_derive`: distingue "não informado" (mantém atual) de None
_KEEP = object()


class WaterfallMetadata(BaseModel):
    """Metadados de um waterfall.

    Attributes:
        name: Nome do waterfall.
        source: Origem dos dados (caminho/arquivo/fonte).
        instrument: Instrumento de medida.
        band: Banda de frequência.
        polarization: Polarização.
        calibration: Descrição da calibração aplicada.
        freq_unit: Unidade da coordenada de frequência (default "Hz").
        time_unit: Unidade da coordenada temporal (default "s").
        data_unit: Unidade dos valores (K, Jy, ...).
        version: Versão dos dados.
        created: Data de criação.
        extra: Metadados adicionais livres.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    source: str | None = None
    instrument: str | None = None
    band: str | None = None
    polarization: str | None = None
    calibration: str | None = None
    freq_unit: str = "Hz"
    time_unit: str = "s"
    data_unit: str | None = None
    version: str | None = None
    created: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Waterfall:
    """Waterfall (tempo × frequência) com metadados e operações chainable.

    O objeto é imutável por convenção: cada operação devolve um novo
    ``Waterfall`` (os dados subjacentes podem ser lazy/dask).

    Attributes:
        data: DataArray canônico com dims ("time", "freq").
        meta: Metadados (``WaterfallMetadata``).
        mask: Máscara booleana opcional (True = válido).
        events: Agregação de eventos na grade (bool/int) ou None.
    """

    __slots__ = ("_data", "_events", "_mask", "_meta")

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset | tuple | np.ndarray,
        *,
        meta: WaterfallMetadata | dict[str, Any] | None = None,
        mask: xr.DataArray | None = None,
        events: xr.DataArray | None = None,
        time: Any = None,
        freq: Any = None,
        time_name: str = TIME_DIM,
        freq_name: str = FREQ_DIM,
        data_name: str | None = None,
    ) -> None:
        """Constrói um Waterfall a partir de várias formas de entrada.

        Args:
            data: DataArray, Dataset (primeira variável de dados ou
                ``data_name``), tupla ``(values, time, freq)`` ou ndarray
                (com ``time``/``freq`` explícitos ou índices default).
            meta: Metadados (objeto ou dict).
            mask: Máscara booleana opcional (dims time, freq).
            events: Agregação de eventos opcional (bool/int na grade).
            time: Coordenadas temporais (quando ``data`` é ndarray).
            freq: Coordenadas de frequência (quando ``data`` é ndarray).
            time_name: Nome da dimensão temporal na entrada.
            freq_name: Nome da dimensão de frequência na entrada.
            data_name: Nome da variável de dados (quando ``data`` é Dataset).
        """
        arr = self._coerce_dataarray(
            data,
            time=time,
            freq=freq,
            time_name=time_name,
            freq_name=freq_name,
            data_name=data_name,
        )
        self._data: xr.DataArray = normalize_dims(arr, time_name=time_name, freq_name=freq_name)
        self._meta: WaterfallMetadata = (
            meta if isinstance(meta, WaterfallMetadata) else WaterfallMetadata(**(meta or {}))
        )
        if self._meta.name is None:
            self._meta = self._meta.model_copy(update={"name": getattr(self._data, "name", None)})
        self._mask: xr.DataArray | None = mask
        self._events: xr.DataArray | None = events

    # ------------------------------------------------------------------
    # Construção / coerção
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_dataarray(
        data: Any,
        *,
        time: Any,
        freq: Any,
        time_name: str,
        freq_name: str,
        data_name: str | None,
    ) -> xr.DataArray:
        _ = (time_name, freq_name)  # interface interna (reservado)
        if isinstance(data, xr.DataArray):
            return data
        if isinstance(data, xr.Dataset):
            var = data_name if data_name is not None else next(iter(data.data_vars))
            if var not in data:
                msg = f"Variável de dados {var!r} não encontrada no Dataset"
                raise KeyError(msg)
            return data[var]
        if isinstance(data, tuple) and len(data) == 3:
            values, t, f = data
            return xr.DataArray(
                values,
                dims=(TIME_DIM, FREQ_DIM),
                coords={TIME_DIM: t, FREQ_DIM: f},
            )
        arr = np.asarray(data)
        if arr.ndim != 2:
            msg = f"Waterfall requer array 2D (time, freq); recebido ndim={arr.ndim}"
            raise ValueError(msg)
        t = np.arange(arr.shape[0]) if time is None else np.asarray(time)
        f = np.arange(arr.shape[1]) if freq is None else np.asarray(freq)
        return xr.DataArray(
            arr,
            dims=(TIME_DIM, FREQ_DIM),
            coords={TIME_DIM: t, FREQ_DIM: f},
        )

    # ------------------------------------------------------------------
    # Acesso a dados
    # ------------------------------------------------------------------
    @property
    def data(self) -> xr.DataArray:
        """DataArray subjacente (dims time, freq)."""
        return self._data

    @property
    def values(self) -> np.ndarray:
        """Valores materializados (numpy)."""
        return self._data.values

    @property
    def masked_data(self) -> xr.DataArray:
        """Dados com a máscara aplicada (pixels inválidos → NaN)."""
        from radioskysims_waterfall.domain.masks import apply as mask_apply

        return mask_apply(self._data, self._mask)

    @property
    def time(self) -> xr.DataArray:
        """Coordenada temporal."""
        return self._data[TIME_DIM]

    @property
    def freq(self) -> xr.DataArray:
        """Coordenada de frequência."""
        return self._data[FREQ_DIM]

    @property
    def meta(self) -> WaterfallMetadata:
        return self._meta

    @property
    def mask(self) -> xr.DataArray | None:
        return self._mask

    @property
    def events(self) -> xr.DataArray | None:
        return self._events

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def n_time(self) -> int:
        return int(self._data.sizes[TIME_DIM])

    @property
    def n_freq(self) -> int:
        return int(self._data.sizes[FREQ_DIM])

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def is_lazy(self) -> bool:
        """True se os dados são lazy (dask)."""
        return self._data.chunks is not None and hasattr(self._data.data, "compute")

    @property
    def chunks(self) -> Any:
        return self._data.chunks

    # ------------------------------------------------------------------
    # Operações chainable (todas devolvem novo Waterfall)
    # ------------------------------------------------------------------
    def slice(
        self,
        time: int | slice | list[int] | None = None,
        freq: int | slice | list[int] | None = None,
    ) -> Waterfall:
        """Fatia por índice (posicional). Ver ``operations.slice_data``."""
        return self._derive(ops.slice_data(self._data, time=time, freq=freq))

    def slice_by_value(
        self,
        time: Any = None,
        freq: Any = None,
    ) -> Waterfall:
        """Fatia por rótulo (valor de coordenada)."""
        return self._derive(ops.slice_by_value(self._data, time=time, freq=freq))

    def decimate(
        self,
        time_factor: int = 1,
        freq_factor: int = 1,
        method: str = "mean",
        skipna: bool = False,
    ) -> Waterfall:
        """Decima por fator inteiro (redução por blocos)."""
        return self._derive(
            ops.decimate(
                self._data, time_factor=time_factor, freq_factor=freq_factor, method=method, skipna=skipna
            )
        )

    def interpolate(
        self,
        time: np.ndarray | None = None,
        freq: np.ndarray | None = None,
        method: str = "linear",
        fill_value: Any = np.nan,
    ) -> Waterfall:
        """Interpola para nova grade de coordenadas."""
        return self._derive(
            ops.interpolate(self._data, time=time, freq=freq, method=method, fill_value=fill_value)
        )

    def resample(
        self,
        time: np.ndarray | None = None,
        freq: np.ndarray | None = None,
        method: str = "linear",
        fill_value: Any = np.nan,
    ) -> Waterfall:
        """Resampling para nova grade (alias de ``interpolate``)."""
        return self.interpolate(time=time, freq=freq, method=method, fill_value=fill_value)

    def rebin(
        self,
        time_edges: np.ndarray | None = None,
        freq_edges: np.ndarray | None = None,
        method: str = "mean",
        skipna: bool = False,
    ) -> Waterfall:
        """Rebin por arestas arbitrárias."""
        return self._derive(
            ops.rebin(self._data, time_edges=time_edges, freq_edges=freq_edges, method=method, skipna=skipna)
        )

    # ------------------------------------------------------------------
    # Máscaras e eventos
    # ------------------------------------------------------------------
    def with_mask(self, mask: xr.DataArray | np.ndarray | None) -> Waterfall:
        """Retorna um novo Waterfall com a máscara fornecida (True = válido)."""
        if mask is None:
            return self.clear_mask()
        m = (
            mask
            if isinstance(mask, xr.DataArray)
            else xr.DataArray(np.asarray(mask, dtype=bool), dims=(TIME_DIM, FREQ_DIM))
        )
        return self._derive(self._data, mask=m)

    def apply_mask(self, fill_value: float = np.nan) -> Waterfall:
        """Aplica a máscara atual nos dados (pixels inválidos → fill_value)."""
        from radioskysims_waterfall.domain.masks import apply as mask_apply

        return self._derive(mask_apply(self._data, self._mask, fill_value=fill_value))

    def clear_mask(self) -> Waterfall:
        """Remove a máscara."""
        return self._derive(self._data, mask=None)

    def with_events(self, events: Sequence[Event] | Any | xr.DataArray) -> Waterfall:
        """Associa eventos ao waterfall.

        Aceita uma sequência de ``Event``, um DataFrame com colunas
        time/freq ou um array agregado na grade (xr.DataArray).
        """
        # Guarda a fonte bruta (ou array agregado) para agregar sob demanda
        return self._derive(self._data, events=events)

    def event_mask(self) -> xr.DataArray | None:
        """Máscara booleana (time × freq) dos eventos associados, se houver."""
        ev = self._events
        if ev is None:
            return None
        if isinstance(ev, xr.DataArray):
            return ev.astype(bool)
        from radioskysims_waterfall.domain.events import aggregate

        return aggregate(ev, self.time.values, self.freq.values, mode="mask")

    def event_counts(self) -> xr.DataArray | None:
        """Contagem de eventos por pixel (time × freq), se houver."""
        ev = self._events
        if ev is None:
            return None
        if isinstance(ev, xr.DataArray):
            return ev.astype(int)
        from radioskysims_waterfall.domain.events import aggregate

        return aggregate(ev, self.time.values, self.freq.values, mode="count")

    def event_density(
        self,
        time_radius: int = 0,
        freq_radius: int = 0,
    ) -> xr.DataArray | None:
        """Densidade de eventos com vizinhança (em pixels), se houver."""
        ev = self._events
        if ev is None or isinstance(ev, xr.DataArray):
            return self.event_counts()
        from radioskysims_waterfall.domain.events import aggregate

        return aggregate(
            ev,
            self.time.values,
            self.freq.values,
            mode="density",
            time_radius=time_radius,
            freq_radius=freq_radius,
        )

    # ------------------------------------------------------------------
    # Filtros, bandpass, baseline, normalização
    # ------------------------------------------------------------------
    def filter_temporal(
        self, method: str = "median", window: int | None = None, mode: str = "subtract", **kwargs: Any
    ) -> Waterfall:
        """Filtro ao longo do tempo. Ver ``domain.filters.filter_temporal``."""
        from radioskysims_waterfall.domain.filters import filter_temporal

        return self._derive(
            filter_temporal(self._data, method=method, window=window, mode=mode, mask=self._mask, **kwargs)
        )

    def filter_spectral(
        self, method: str = "median", window: int | None = None, mode: str = "subtract", **kwargs: Any
    ) -> Waterfall:
        """Filtro ao longo da frequência."""
        from radioskysims_waterfall.domain.filters import filter_spectral

        return self._derive(
            filter_spectral(self._data, method=method, window=window, mode=mode, mask=self._mask, **kwargs)
        )

    def filter_spatial(
        self,
        method: str = "median",
        window: tuple[int, int] | int | None = None,
        mode: str = "subtract",
        **kwargs: Any,
    ) -> Waterfall:
        """Filtro espacial 2D."""
        from radioskysims_waterfall.domain.filters import filter_spatial

        return self._derive(
            filter_spatial(self._data, method=method, window=window, mode=mode, mask=self._mask, **kwargs)
        )

    def get_bandpass(
        self,
        threshold: float = 3.0,
        percentile: float = 50.0,
        method: str = "median",
        min_bandwidth: int = 1,
        return_slices: bool = False,
        return_mask: bool = False,
        **kwargs: Any,
    ) -> Waterfall | list[slice] | xr.DataArray:
        """Determina a banda de passagem preservando o sinal de interesse.

        Por padrão retorna um ``Waterfall`` restrito à banda. Com
        ``return_slices=True`` retorna ``list[slice]``; com ``return_mask=True``
        retorna a máscara espectral.
        """
        from radioskysims_waterfall.domain.bandpass import get_bandpass

        result = get_bandpass(
            self._data,
            threshold=threshold,
            percentile=percentile,
            method=method,
            mask=self._mask,
            min_bandwidth=min_bandwidth,
            return_slices=return_slices,
            return_mask=return_mask,
            **kwargs,
        )
        if return_slices or return_mask:
            return result
        return self._derive(result)

    def get_baseline(
        self,
        method: str = "median",
        window: int | None = None,
        dim: str = FREQ_DIM,
        **kwargs: Any,
    ) -> Waterfall:
        """Estima o baseline ao longo de ``dim`` (padrão freq).

        Devolve um ``Waterfall`` cujos dados são o baseline (mesma forma).
        """
        from radioskysims_waterfall.domain.baseline import get_baseline

        bl = get_baseline(self._data, method=method, window=window, dim=dim, **kwargs)
        return self._derive(bl)

    def subtract_baseline(
        self, method: str = "median", window: int | None = None, dim: str = FREQ_DIM, **kwargs: Any
    ) -> Waterfall:
        """Subtrai o baseline dos dados."""
        from radioskysims_waterfall.domain.baseline import get_baseline

        bl = get_baseline(self._data, method=method, window=window, dim=dim, **kwargs)
        return self._derive(self._data - bl)

    def normalize(self, method: str = "minmax", **kwargs: Any) -> Waterfall:
        """Normaliza os dados. Ver ``domain.normalize.normalize``."""
        from radioskysims_waterfall.domain.normalize import normalize

        return self._derive(normalize(self._data, method=method, **kwargs))

    # ------------------------------------------------------------------
    # SED / LightCurve
    # ------------------------------------------------------------------
    def sed(
        self,
        channels: int | slice | list[int] | None = None,
        interval: int | slice | list[int] | None = None,
        name: str | None = None,
    ) -> SED:
        """Espectro médio (SED) sobre um intervalo temporal e canais."""
        from radioskysims_waterfall.domain.sed import SED

        return SED(self._data, channels=channels, interval=interval, mask=self._mask, name=name)

    def lightcurve(
        self,
        frequencies: int | slice | list[int] | None = None,
        interval: int | slice | list[int] | None = None,
        name: str | None = None,
    ) -> LightCurve:
        """Curva de luz média sobre uma faixa de frequências e intervalo."""
        from radioskysims_waterfall.domain.lightcurve import LightCurve

        return LightCurve(self._data, frequencies=frequencies, interval=interval, mask=self._mask, name=name)

    # ------------------------------------------------------------------
    # Pipeline / persistência / visualização
    # ------------------------------------------------------------------
    def pipeline(self, steps: Sequence[dict[str, Any]]) -> Waterfall:
        """Aplica uma sequência de passos declarativos ao waterfall.

        ``steps`` é uma lista de dicts ``{"name": ..., "kwargs": {...}}``
        (ou ``{"func": callable}``). Ver ``application.services.pipeline``.

        Returns:
            Waterfall resultante da aplicação de todos os passos.
        """
        from radioskysims_waterfall.application.services.pipeline import WaterfallPipeline

        return WaterfallPipeline(steps).run(self)

    def save(
        self, path: str | Path, format: str | None = None, *, version: bool = True, **kwargs: Any
    ) -> Path:
        """Persiste o waterfall com controle de versão opcional.

        Ver ``adapters.storage.file_storage.FileStorage``.
        """
        from radioskysims_waterfall.adapters.storage.file_storage import FileStorage

        return FileStorage().save(self, path, format=format, version=version, **kwargs)

    def plot(self, other: Waterfall | None = None, **kwargs: Any):
        """Plota o waterfall (holoviews + datashader), opcionalmente com um
        segundo waterfall lado a lado. Ver ``plot_waterfall``."""
        from radioskysims_waterfall.application.visualization.waterfall_plot import plot_waterfall

        return plot_waterfall(self, other=other, **kwargs)

    # ------------------------------------------------------------------
    # Materialização / utilidades
    # ------------------------------------------------------------------
    def compute(self) -> Waterfall:
        """Materializa dados lazy (dask) em memória."""
        return self._derive(ops.compute(self._data))

    def estimate_memory(self) -> int:
        """Bytes estimados da materialização dos dados."""
        return ops.estimate_nbytes(self._data)

    def to_dataarray(self) -> xr.DataArray:
        """DataArray canônico (dims time, freq)."""
        return self._data

    def to_dataset(self) -> xr.Dataset:
        """Dataset com os dados + metadados em attrs."""
        ds = self._data.to_dataset(name=self._data.name or "data")
        ds.attrs.update(self._meta.model_dump(exclude_none=True))
        return ds

    def copy(self) -> Waterfall:
        return Waterfall(
            self._data.copy(), meta=self._meta.model_copy(deep=True), mask=self._mask, events=self._events
        )

    def _derive(self, data: xr.DataArray, *, mask: Any = _KEEP, events: Any = _KEEP) -> Waterfall:
        """Cria um novo Waterfall preservando metadados (e máscara/eventos por padrão)."""
        return Waterfall(
            data,
            meta=self._meta,
            mask=self._mask if mask is _KEEP else mask,
            events=self._events if events is _KEEP else events,
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lazy = ", lazy" if self.is_lazy else ""
        name = f" {self._meta.name!r}" if self._meta.name else ""
        dtype = np.dtype(self.dtype).name
        return (
            f"Waterfall{name}(time={self.n_time}, freq={self.n_freq}, "
            f"dtype={dtype}{lazy}, mask={'yes' if self._mask is not None else 'no'})"
        )

    def __len__(self) -> int:
        return self.n_time


__all__ = ["Waterfall", "WaterfallMetadata"]
