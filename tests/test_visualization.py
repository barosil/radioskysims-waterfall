"""\
Módulo: tests/test_visualization.py

Testes da camada de visualização (holoviews + datashader).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import pytest

from radioskysims_waterfall.application.visualization import (
    plot_lightcurve,
    plot_sed,
    plot_waterfall,
)


def test_plot_waterfall_layout(wf) -> None:
    layout = plot_waterfall(wf, use_datashader=False)
    # waterfall + lightcurve + SED
    assert len(layout) == 3


def test_plot_waterfall_two_side_by_side(wf) -> None:
    layout = plot_waterfall(wf, other=wf.slice(freq=slice(0, 16)), use_datashader=False)
    assert len(layout) == 6


def test_plot_waterfall_without_extra_panels(wf) -> None:
    layout = plot_waterfall(wf, use_datashader=False, include_lightcurve=False, include_sed=False)
    assert len(layout) == 1


def test_plot_waterfall_datashader(wf) -> None:
    pytest.importorskip("datashader")
    layout = plot_waterfall(wf, use_datashader=True)
    assert len(layout) >= 1


def test_plot_sed(wf) -> None:
    plot = plot_sed(wf.sed())
    assert plot is not None


def test_plot_lightcurve(wf) -> None:
    plot = plot_lightcurve(wf.lightcurve())
    assert plot is not None


def test_sed_plot_method(wf) -> None:
    assert wf.sed().plot() is not None


def test_lightcurve_plot_method(wf) -> None:
    assert wf.lightcurve().plot() is not None
