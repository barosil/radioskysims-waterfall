"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/__init__.py

Descrição:
    Pacote radioskysims_waterfall — ferramentas de gerenciamento de
    waterfalls (tempo x frequência) para radioastronomia, com arquitetura
    hexagonal (domain / application / ports / adapters).

    API pública de alto nível: `Waterfall`, `WaterfallMetadata`, `SED`,
    `LightCurve`, `WaterfallPipeline`, loaders e monitor de recursos.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.adapters import create_loader, load_waterfall
from radioskysims_waterfall.application import WaterfallPipeline
from radioskysims_waterfall.domain.lightcurve import LightCurve
from radioskysims_waterfall.domain.models import Waterfall, WaterfallMetadata
from radioskysims_waterfall.domain.sed import SED
from radioskysims_waterfall.infrastructure import SystemMonitor

__all__ = [
    "SED",
    "LightCurve",
    "SystemMonitor",
    "Waterfall",
    "WaterfallMetadata",
    "WaterfallPipeline",
    "create_loader",
    "load_waterfall",
]

__version__ = "0.1.0"
