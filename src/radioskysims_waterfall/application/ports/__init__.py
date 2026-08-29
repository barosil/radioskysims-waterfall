"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/ports/__init__.py

Descrição:
    Ports (contratos) da camada de aplicação para carregamento e
    persistência de waterfalls.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.application.ports.loader import WaterfallLoaderPort
from radioskysims_waterfall.application.ports.storage import WaterfallStoragePort

__all__ = ["WaterfallLoaderPort", "WaterfallStoragePort"]
