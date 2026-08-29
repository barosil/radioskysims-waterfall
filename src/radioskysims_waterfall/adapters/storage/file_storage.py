"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/storage/file_storage.py

Descrição:
    Persistência de waterfalls em arquivo (netcdf, zarr, hdf5, fits) com:
    - escrita atômica (arquivo temporário + rename);
    - controle de versão de arquivos (backups versionados ``<nome>.v<n>``);
    - embutimento de metadados (attrs/header) e número de versão.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radioskysims_waterfall.application.ports.storage import WaterfallStoragePort

if TYPE_CHECKING:
    from radioskysims_waterfall.domain.models import Waterfall
    from radioskysims_waterfall.infrastructure.monitor import SystemMonitor

_FORMAT_EXTENSIONS = {
    "netcdf": {".nc", ".nc4", ".netcdf", ".cdf"},
    "zarr": {".zarr"},
    "hdf5": {".h5", ".hdf5", ".hdf"},
    "fits": {".fits", ".fit", ".fts"},
}

_VERSION_RE = re.compile(r"^(.+)\.v(\d+)(\.[^.]+)$")


class FileStorage(WaterfallStoragePort):
    """Persistência de waterfalls em arquivo, com versionamento opcional."""

    def save(
        self,
        waterfall: Waterfall,
        path: str | Path,
        *,
        format: str | None = None,
        version: bool = True,
        meta: dict[str, Any] | None = None,
        monitor: SystemMonitor | None = None,
        **kwargs: Any,
    ) -> Path:
        """Persiste o waterfall.

        Args:
            waterfall: Waterfall a salvar.
            path: Caminho de destino.
            format: Formato ("netcdf", "zarr", "hdf5", "fits"). None → infere
                da extensão.
            version: Se True (padrão), rotaciona versões anteriores para
                ``<nome>.v<n><ext>`` antes de gravar.
            meta: Metadados adicionais a embutir (merge sobre os do waterfall).
            monitor: Monitor opcional para checagem de espaço em disco.
            **kwargs: Repassados ao backend de escrita.

        Returns:
            Caminho efetivamente escrito.

        Raises:
            ValueError: Se o formato não for suportado/inferível.
            FileNotFoundError: Se o diretório pai não existir.
        """
        path = Path(path)
        fmt = (format or self._infer_format(path)).lower()
        if fmt not in _FORMAT_EXTENSIONS:
            msg = f"Formato não suportado: {fmt!r} (use {sorted(_FORMAT_EXTENSIONS)})"
            raise ValueError(msg)

        parent = path.parent
        if not parent.exists():
            msg = f"Diretório de destino não existe: {parent}"
            raise FileNotFoundError(msg)

        if monitor is not None:
            with monitor.ensure_resources(
                path=str(parent),
                required_disk_bytes=waterfall.estimate_memory(),
            ):
                pass

        if version:
            self._rotate_versions(path)

        return self._write(waterfall, path, fmt, meta=meta, **kwargs)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_format(path: Path) -> str | None:
        for fmt, exts in _FORMAT_EXTENSIONS.items():
            if path.suffix.lower() in exts:
                return fmt
        return None

    @classmethod
    def _rotate_versions(cls, path: Path) -> None:
        """Move o arquivo atual para ``<stem>.v<n><suffix>`` (n = próximo livre)."""
        if not path.exists():
            return
        stem, suffix = path.stem, path.suffix
        pattern = re.compile(rf"^{re.escape(stem)}\.v(\d+){re.escape(suffix)}$")
        nums = []
        for p in path.parent.glob(f"{stem}.v*{suffix}"):
            m = pattern.match(p.name)
            if m:
                nums.append(int(m.group(1)))
        nxt = (max(nums) if nums else 0) + 1
        target = path.with_name(f"{stem}.v{nxt}{suffix}")
        os.replace(path, target)

    def _write(
        self, waterfall: Waterfall, path: Path, fmt: str, *, meta: dict[str, Any] | None, **kwargs: Any
    ) -> Path:
        merged = dict(waterfall.meta.model_dump(exclude_none=True))
        if meta:
            merged.update(meta)
        merged.setdefault("saved_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

        if fmt == "netcdf":
            return self._write_netcdf(waterfall, path, merged, **kwargs)
        if fmt == "zarr":
            return self._write_zarr(waterfall, path, merged, **kwargs)
        if fmt == "hdf5":
            return self._write_hdf5(waterfall, path, merged, **kwargs)
        if fmt == "fits":
            return self._write_fits(waterfall, path, merged, **kwargs)
        msg = f"Formato não suportado: {fmt!r}"
        raise ValueError(msg)  # pragma: no cover

    @staticmethod
    def _write_netcdf(waterfall: Waterfall, path: Path, attrs: dict[str, Any], **kwargs: Any) -> Path:
        ds = waterfall.to_dataset()
        ds.attrs.update({k: str(v) for k, v in attrs.items() if v is not None})
        engine = kwargs.pop("engine", "h5netcdf")
        tmp = _temp_path(path)
        ds.to_netcdf(tmp, engine=engine, **kwargs)
        _atomic_replace(tmp, path)
        return path

    @staticmethod
    def _write_zarr(waterfall: Waterfall, path: Path, attrs: dict[str, Any], **kwargs: Any) -> Path:
        ds = waterfall.to_dataset()
        ds.attrs.update({k: str(v) for k, v in attrs.items() if v is not None})
        mode = kwargs.pop("mode", "w")
        # Store Zarr é diretório: usa caminho temporário único NÃO pré-criado
        tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.zarr"
        try:
            ds.to_zarr(tmp, mode=mode, **kwargs)
            _atomic_replace_dir(tmp, path)
        finally:
            if tmp.exists():
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)
        return path

    @staticmethod
    def _write_hdf5(waterfall: Waterfall, path: Path, attrs: dict[str, Any], **kwargs: Any) -> Path:
        import h5py

        data = waterfall.compute().data
        tmp = _temp_path(path)
        with h5py.File(tmp, "w") as f:
            f.create_dataset("data", data=data.values, **kwargs)
            f.create_dataset("time", data=data["time"].values)
            f.create_dataset("freq", data=data["freq"].values)
            for k, v in attrs.items():
                if v is not None:
                    f.attrs[k] = str(v)
        _atomic_replace(tmp, path)
        return path

    @staticmethod
    def _write_fits(waterfall: Waterfall, path: Path, attrs: dict[str, Any], **kwargs: Any) -> Path:
        from astropy.io import fits

        data = waterfall.compute().values
        hdu = fits.PrimaryHDU(data)
        for k, v in attrs.items():
            if v is not None:
                hdu.header.setdefault(k.upper()[:8], str(v))
        tmp = _temp_path(path)
        hdu.writeto(tmp, overwrite=True, **kwargs)
        _atomic_replace(tmp, path)
        return path


def _temp_path(path: Path) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    return Path(tmp_name)


def _atomic_replace(tmp: Path, target: Path) -> None:
    os.replace(tmp, target)


def _atomic_replace_dir(tmp: Path, target: Path) -> None:
    if target.exists():
        os.replace(target, _temp_path(target))  # preserva o antigo (backup temporário)
    os.replace(tmp, target)


__all__ = ["FileStorage"]
