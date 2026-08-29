"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/__init__.py

Descrição:
    Camada de domínio do pacote radioskysims_waterfall. Contém o value
    object `Waterfall`, operações puras (slice, resample, filtros, SED,
    lightcurve, baseline, bandpass, máscaras, eventos) e utilidades de
    unidades.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.domain.lightcurve import LightCurve
from radioskysims_waterfall.domain.models import Waterfall, WaterfallMetadata
from radioskysims_waterfall.domain.sed import SED

__all__ = ["SED", "LightCurve", "Waterfall", "WaterfallMetadata"]
