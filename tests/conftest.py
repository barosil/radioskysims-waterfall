"""\
Módulo: tests/conftest.py

Fixtures compartilhadas dos testes do pacote radioskysims-waterfall.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest

from radioskysims_waterfall import Waterfall, WaterfallMetadata


@pytest.fixture
def make_wf():
    """Factory de Waterfall sintético (time × freq) com metadados."""

    def _make(n_time: int = 50, n_freq: int = 64, seed: int = 0, **meta: object) -> Waterfall:
        rng = np.random.default_rng(seed)
        data = rng.normal(10, 2, size=(n_time, n_freq)) + np.linspace(0, 5, n_freq)[None, :]
        time = np.arange(n_time, dtype=float)
        freq = np.linspace(300, 400, n_freq)
        return Waterfall(
            (data, time, freq),
            meta=WaterfallMetadata(name="wf", freq_unit="MHz", data_unit="K", **meta),
        )

    return _make


@pytest.fixture
def wf(make_wf) -> Waterfall:
    return make_wf()
