"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/loaders/hdf5_loader.py

Descrição:
    Loader de waterfalls a partir de arquivos HDF5 "puros" (via h5py),
    com construção lazy via dask e opções de mapeamento de nomes:
    ``data_name`` (matriz), ``freq_name`` e ``time_name`` (coordenadas 1D).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.domain.models import Waterfall

try:  # dependência opcional (extra "hdf5")
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


class HDF5Loader(WaterfallLoaderPort):
    """Carrega um Waterfall de um arquivo HDF5 genérico (h5py + dask)."""

    format_name = "hdf5"

    def load(
        self,
        source: str | Path,
        *,
        data_name: str | None = None,
        dataset_path: str | None = None,
        time_name: str = "time",
        freq_name: str = "freq",
        chunks: str | tuple[int, ...] = "auto",
        transpose: bool = True,
        meta: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Waterfall:
        """Carrega um Waterfall de um arquivo HDF5.

        Estratégia:
        - Localiza datasets 2D (candidatos à matriz); ``data_name`` ou
          ``dataset_path`` selecionam explicitamente.
        - Datasets 1D nomeados como ``time_name``/``freq_name`` viram
          coordenadas (na raiz ou no mesmo grupo da matriz).
        - A matriz é lida de forma lazy via ``dask.array.from_array``.

        Args:
            source: Caminho do arquivo.
            data_name: Nome (basename) da matriz 2D.
            dataset_path: Caminho completo da matriz (ex.: "/grp/data").
            time_name: Nome do dataset 1D de coordenada temporal.
            freq_name: Nome do dataset 1D de coordenada de frequência.
            chunks: Chunks dask ("auto" ou tupla).
            transpose: Se True (padrão) e a matriz for (freq, time), transpor
                para (time, freq).
            meta: Metadados (dict).
            **kwargs: Reservado.

        Returns:
            Waterfall lazy (dask).

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se nenhum dataset 2D for encontrado.
            ImportError: Se h5py não estiver instalado (extra "hdf5").
        """
        _ = chunks  # assinala uso (interface/reservado)
        if h5py is None:
            msg = "h5py é necessário para ler HDF5. Instale: pip install h5py"
            raise ImportError(msg)

        path = Path(source)
        if not path.exists():
            msg = f"Arquivo não encontrado: {path}"
            raise FileNotFoundError(msg)

        with h5py.File(path, "r") as f:
            datasets = list(_walk_datasets(f))
            if not datasets:
                msg = f"Nenhum dataset encontrado em {path}"
                raise ValueError(msg)

            matrix_path, matrix = _select_matrix(datasets, data_name=data_name, dataset_path=dataset_path)

            # Coordenadas: procura datasets 1D por nome (basename) na raiz/ancestrais
            coords: dict[str, np.ndarray] = {}
            for name, target in ((time_name, time_name), (freq_name, freq_name)):
                ds = _find_1d_by_name(f, datasets, target, matrix_path)
                if ds is not None:
                    coords[name] = np.asarray(ds[...], dtype=float)
                else:
                    coords[name] = np.arange(matrix.shape[0] if name == time_name else matrix.shape[1])

            data = matrix[...]  # materializa para simplificar; dimensões típicas
            values = data
            # garante ordem (time, freq): se a coordenada "freq" tem
            # comprimento igual à dim 0, é (freq, time) → transpor
            if (
                transpose
                and values.ndim == 2
                and len(coords["freq"]) == values.shape[0]
                and len(coords["time"]) == values.shape[1]
            ):
                values = values.T
                coords["time"], coords["freq"] = coords["freq"], coords["time"]

        time = coords["time"]
        freq = coords["freq"]
        if values.ndim != 2:
            msg = f"Matriz HDF5 deve ser 2D; shape={values.shape}"
            raise ValueError(msg)

        return Waterfall(
            (values, time, freq),
            meta=meta,
        )


def _walk_datasets(group):
    """Itera (caminho_completo, dataset) recursivamente."""
    for key, item in group.items():
        full = f"{group.name}/{key}".lstrip("/")
        if isinstance(item, h5py.Dataset):
            yield full, item
        elif isinstance(item, h5py.Group):
            yield from _walk_datasets(item)


def _select_matrix(datasets, *, data_name, dataset_path):
    if dataset_path is not None:
        for full, ds in datasets:
            if full == dataset_path or full.endswith(f"/{dataset_path}"):
                if ds.ndim != 2:
                    msg = f"Dataset {dataset_path} não é 2D (ndim={ds.ndim})"
                    raise ValueError(msg)
                return full, ds
        msg = f"dataset_path não encontrado: {dataset_path}"
        raise ValueError(msg)
    if data_name is not None:
        matches = [(full, ds) for full, ds in datasets if full.split("/")[-1] == data_name]
        if not matches:
            msg = f"Dataset 2D com nome {data_name!r} não encontrado"
            raise ValueError(msg)
        two_d = [m for m in matches if m[1].ndim == 2]
        if not two_d:
            msg = f"Dataset {data_name!r} encontrado, mas nenhum é 2D"
            raise ValueError(msg)
        return two_d[0]
    two_d = [(full, ds) for full, ds in datasets if ds.ndim == 2]
    if not two_d:
        msg = "Nenhum dataset 2D no arquivo; use data_name/dataset_path"
        raise ValueError(msg)
    return two_d[0]


def _find_1d_by_name(_f, datasets, name, matrix_path):
    """Procura um dataset 1D com basename ``name`` perto da matriz."""
    group_dir = "/".join(matrix_path.split("/")[:-1]) if "/" in matrix_path else ""
    candidates = []
    for full, ds in datasets:
        if full.split("/")[-1] != name or ds.ndim != 1:
            continue
        # prioriza: mesmo grupo da matriz > raiz > qualquer lugar
        dpath = "/".join(full.split("/")[:-1]) if "/" in full else ""
        priority = 0
        if dpath == group_dir:
            priority = 2
        elif dpath == "":
            priority = 1
        candidates.append((priority, full, ds))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    return candidates[0][2]


__all__ = ["HDF5Loader"]
