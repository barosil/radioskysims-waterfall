"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/services/__init__.py

Descrição:
    Serviços de orquestração da camada de aplicação (pipeline de waterfall).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.application.services.pipeline import WaterfallPipeline

__all__ = ["WaterfallPipeline"]
