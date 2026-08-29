"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/__init__.py

Descrição:
    Camada de aplicação do pacote radioskysims_waterfall: ports (contratos)
    e serviços de orquestração (pipeline).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.application.services.pipeline import WaterfallPipeline

__all__ = ["WaterfallPipeline"]
