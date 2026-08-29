"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/adapters/__init__.py

Descrição:
    Adapters do pacote radioskysims_waterfall: loaders de formatos de
    arquivo (netcdf/hdf5/zarr/fits/xarray) e storage (persistência com
    versionamento).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from radioskysims_waterfall.adapters.loaders import create_loader, load_waterfall

__all__ = ["create_loader", "load_waterfall"]
