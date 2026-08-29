"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/infrastructure/monitor.py

Descrição:
    Monitor de recursos do sistema (CPU, memória, disco) para execução
    segura de operações pesadas sobre waterfalls (materialização dask,
    filtros 2D, escrita em disco, etc.). Usa ``psutil``.

    Uso típico:
        >>> mon = SystemMonitor()
        >>> ok, reason = mon.check(required_nbytes=2**30)
        >>> with mon.ensure_resources(required_nbytes=2**30):
        ...     resultado = wf.compute()

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psutil

log = logging.getLogger("radioskysims_waterfall.monitor")

# Fração da memória disponível considerada segura por padrão
_DEFAULT_SAFE_FRACTION = 0.8
# Bytes mínimos de folga reservados para o sistema/sobrecarga
_SYSTEM_RESERVE_BYTES = 256 * 1024 * 1024


class ResourceError(RuntimeError):
    """Erro lançado quando recursos insuficientes para uma operação."""


class SystemMonitor:
    """Monitor de CPU/memória/disco via psutil."""

    def __init__(self, safe_fraction: float = _DEFAULT_SAFE_FRACTION) -> None:
        """Inicializa o monitor.

        Args:
            safe_fraction: Fração da memória disponível considerada segura
                (default 0.8).
        """
        if not 0.0 < safe_fraction <= 1.0:
            msg = "safe_fraction deve estar em (0, 1]"
            raise ValueError(msg)
        self.safe_fraction = safe_fraction

    # ------------------------------------------------------------------
    # Leitura de recursos
    # ------------------------------------------------------------------
    @staticmethod
    def cpu_count(logical: bool = True) -> int:
        """Número de CPUs (lógicos por padrão)."""
        return psutil.cpu_count(logical=logical) or 1

    @staticmethod
    def available_memory_bytes() -> int:
        """Memória disponível (bytes)."""
        return int(psutil.virtual_memory().available)

    @staticmethod
    def total_memory_bytes() -> int:
        return int(psutil.virtual_memory().total)

    @staticmethod
    def memory_percent() -> float:
        """Percentual de memória em uso (0..100)."""
        return float(psutil.virtual_memory().percent)

    @staticmethod
    def disk_free_bytes(path: str | Any = ".") -> int:
        """Espaço livre em disco no filesystem de ``path`` (bytes)."""
        return int(psutil.disk_usage(str(path)).free)

    @staticmethod
    def estimate_nbytes(shape: tuple[int, ...], dtype: Any = np.float64) -> int:
        """Estima bytes de um array com ``shape`` e ``dtype``."""
        n = int(np.prod(shape)) if len(shape) else 1
        return n * int(np.dtype(dtype).itemsize)

    # ------------------------------------------------------------------
    # Verificação
    # ------------------------------------------------------------------
    def safe_memory_bytes(self) -> int:
        """Memória disponível limitada pela fração segura."""
        return max(0, int(self.available_memory_bytes() * self.safe_fraction) - _SYSTEM_RESERVE_BYTES)

    def check(
        self,
        required_nbytes: int = 0,
        required_cpus: int = 1,
        *,
        path: str | None = None,
        required_disk_bytes: int = 0,
    ) -> tuple[bool, str]:
        """Verifica se há recursos para uma operação.

        Args:
            required_nbytes: Memória estimada da operação (bytes).
            required_cpus: CPUs necessárias (default 1).
            path: Caminho para verificação de disco (opcional).
            required_disk_bytes: Espaço em disco necessário (bytes).

        Returns:
            Tupla ``(ok, reason)``: ``ok`` False se recursos insuficientes.
        """
        reasons: list[str] = []
        if required_cpus > 0 and required_cpus > self.cpu_count():
            reasons.append(f"CPUs: precisa {required_cpus}, tem {self.cpu_count()}")
        if required_nbytes > 0:
            avail = self.available_memory_bytes()
            limit = self.safe_memory_bytes()
            if required_nbytes > limit:
                reasons.append(
                    f"Memória: precisa {required_nbytes / 2**30:.2f} GiB, "
                    f"disponível segura {limit / 2**30:.2f} GiB (total {avail / 2**30:.2f} GiB)"
                )
        if required_disk_bytes > 0:
            if path is None:
                path = "."
            free = self.disk_free_bytes(path)
            if required_disk_bytes > free:
                reasons.append(
                    f"Disco({path}): precisa {required_disk_bytes / 2**30:.2f} GiB, "
                    f"livre {free / 2**30:.2f} GiB"
                )
        if reasons:
            return False, "; ".join(reasons)
        return True, "ok"

    @contextmanager
    def ensure_resources(
        self,
        required_nbytes: int = 0,
        required_cpus: int = 1,
        *,
        path: str | None = None,
        required_disk_bytes: int = 0,
        raise_on_error: bool = True,
    ) -> Iterator[SystemMonitor]:
        """Context manager que garante recursos antes de uma operação.

        Args:
            required_nbytes: Memória estimada (bytes).
            required_cpus: CPUs necessárias.
            path: Caminho para checagem de disco.
            required_disk_bytes: Espaço em disco necessário.
            raise_on_error: Se True (padrão), lança ``ResourceError`` quando
                insuficiente; senão apenas registra warning.

        Raises:
            ResourceError: Se não houver recursos e ``raise_on_error=True``.
        """
        ok, reason = self.check(
            required_nbytes=required_nbytes,
            required_cpus=required_cpus,
            path=path,
            required_disk_bytes=required_disk_bytes,
        )
        if not ok:
            if raise_on_error:
                raise ResourceError(reason)
            log.warning("Recursos insuficientes (prosseguindo): %s", reason)
        else:
            log.debug("Recursos OK: %s", reason)
        yield self

    # ------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        """Relatório resumido dos recursos atuais."""
        vm = psutil.virtual_memory()
        return {
            "cpu_count": self.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_total_bytes": vm.total,
            "memory_available_bytes": vm.available,
            "memory_percent": vm.percent,
            "safe_memory_bytes": self.safe_memory_bytes(),
        }

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"SystemMonitor(cpus={r['cpu_count']}, mem={r['memory_percent']:.1f}% usado, "
            f"safe={r['safe_memory_bytes'] / 2**30:.2f} GiB)"
        )


__all__ = ["ResourceError", "SystemMonitor"]
