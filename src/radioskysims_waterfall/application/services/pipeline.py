"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/services/pipeline.py

Descrição:
    Pipeline declarativo e *chainable* para waterfalls. Um pipeline é uma
    lista ordenada de passos; cada passo resolve um método do ``Waterfall``
    (ou um callable) e é aplicado em sequência.

    Verificação segura de recursos: se um ``SystemMonitor`` for fornecido,
    operações potencialmente pesadas são verificadas quanto a CPU/memória
    disponíveis antes da execução.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from radioskysims_waterfall.domain.models import Waterfall
    from radioskysims_waterfall.infrastructure.monitor import SystemMonitor

Step = dict[str, Any]


class LoggerProtocol(Protocol):
    """Contrato mínimo de logging usado pela camada Application."""

    def info(self, event: str, **kwargs: object) -> None:
        """Registra um evento informativo estruturado."""
        ...


class WaterfallPipeline:
    """Pipeline de transformações aplicadas a um ``Waterfall``.

    Exemplo:
        >>> pipe = WaterfallPipeline()
        >>> pipe.add("filter_spectral", method="median", window=5)
        >>> pipe.add("normalize", method="minmax")
        >>> out = pipe.run(wf)
    """

    def __init__(
        self,
        steps: Sequence[Step] | None = None,
        *,
        monitor: SystemMonitor | None = None,
        safe: bool = False,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Inicializa o pipeline.

        Args:
            steps: Passos iniciais (lista de dicts com "name" e "kwargs",
                ou "func" e "kwargs").
            monitor: Monitor de recursos opcional (execução segura).
            safe: Se True, exige um monitor para verificação de recursos.
            logger: Logger estruturado opcional para eventos do pipeline.
        """
        self._steps: list[Step] = []
        self._monitor = monitor
        self._safe = safe
        self._logger = logger
        for step in steps or []:
            self.add(**step)

    # ------------------------------------------------------------------
    @property
    def steps(self) -> list[Step]:
        """Passos do pipeline (somente leitura)."""
        return list(self._steps)

    def add(
        self, name: str | None = None, *, func: Callable[[Waterfall], Waterfall] | None = None, **kwargs: Any
    ) -> WaterfallPipeline:
        """Adiciona um passo ao pipeline (chainable).

        Um passo é definido por ``name`` (nome de método do ``Waterfall``)
        ou ``func`` (callable que recebe o Waterfall e devolve Waterfall).

        Args:
            name: Nome do método do ``Waterfall`` (ex.: "filter_spectral").
            func: Callable alternativo ``f(waterfall) -> waterfall``.
            **kwargs: Argumentos do passo (repassados ao método/func; por
                exemplo ``method="median"`` para filtros).

        Returns:
            O próprio pipeline (para encadeamento).
        """
        if name is None and func is None:
            msg = "Pipeline.add requer 'name' ou 'func'"
            raise ValueError(msg)
        if name is not None and func is not None:
            msg = "Pipeline.add aceita 'name' OU 'func', não ambos"
            raise ValueError(msg)
        step: Step = {"name": name} if name is not None else {"func": func}
        step["kwargs"] = dict(kwargs)
        self._steps.append(step)
        return self

    def describe(self) -> list[str]:
        """Descrição textual dos passos."""
        out = []
        for i, step in enumerate(self._steps):
            if "name" in step:
                out.append(f"{i}: {step['name']}({step.get('kwargs', {})})")
            else:
                fn = step.get("func")
                out.append(f"{i}: func({getattr(fn, '__name__', fn)!r}, {step.get('kwargs', {})})")
        return out

    def run(self, waterfall: Waterfall, *, monitor: SystemMonitor | None = None) -> Waterfall:
        """Aplica todos os passos em sequência e devolve o Waterfall final.

        Args:
            waterfall: Waterfall de entrada.
            monitor: Monitor opcional; sobrepõe o do construtor se fornecido.

        Returns:
            Waterfall transformado.

        Raises:
            RuntimeError: Se ``safe=True`` e não houver monitor disponível.
            AttributeError: Se um passo referenciar método inexistente.
        """
        mon = monitor or self._monitor
        if self._safe and mon is None:
            msg = "Pipeline.safe=True requer um SystemMonitor (construtor ou run())"
            raise RuntimeError(msg)

        current = waterfall
        for i, step in enumerate(self._steps):
            kwargs = step.get("kwargs", {})
            if "name" in step:
                name = step["name"]
                fn = getattr(current, name, None)
                if fn is None or not callable(fn):
                    msg = f"Passo {i}: Waterfall não possui método {name!r}"
                    raise AttributeError(msg)
                label = name
                current = fn(**kwargs)
            else:
                fn = step["func"]
                label = getattr(fn, "__name__", "func")
                current = fn(current, **kwargs)
            if self._logger is not None:
                self._logger.info(
                    "waterfall_pipeline_step_completed",
                    step_index=i + 1,
                    step_count=len(self._steps),
                    step=label,
                )

            if mon is not None:
                mon.ensure_resources(required_nbytes=current.estimate_memory())
        return current

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"WaterfallPipeline(steps={len(self._steps)}): {self.describe()}"


__all__ = ["Step", "WaterfallPipeline"]
