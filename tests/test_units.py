"""\
Módulo: tests/test_units.py

Testes das conversões de unidades do domínio.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import numpy as np
import pytest

from radioskysims_waterfall.domain.units import (
    convert,
    flux_from_jy,
    flux_to_jy,
    frequency_from_hz,
    frequency_from_mhz,
    frequency_to_hz,
    frequency_to_mhz,
    temperature_to_kelvin,
    time_to_seconds,
)


def test_frequency_hz_to_mhz() -> None:
    assert frequency_to_mhz(300.0e6, "Hz") == pytest.approx(300.0)
    assert frequency_to_hz(300.0, "MHz") == pytest.approx(300.0e6)


def test_frequency_roundtrip() -> None:
    for unit in ("Hz", "kHz", "MHz", "GHz"):
        val = frequency_from_hz(1.234e9, unit)
        assert frequency_to_hz(val, unit) == pytest.approx(1.234e9)


def test_frequency_from_mhz() -> None:
    assert frequency_from_mhz(300.0, "Hz") == pytest.approx(300.0e6)
    assert frequency_from_mhz(1.0, "GHz") == pytest.approx(1.0e-3)


def test_time_to_seconds() -> None:
    assert time_to_seconds(1.0, "min") == pytest.approx(60.0)
    assert time_to_seconds(1000.0, "ms") == pytest.approx(1.0)
    assert time_to_seconds(1.0, "h") == pytest.approx(3600.0)


def test_temperature() -> None:
    assert temperature_to_kelvin(1.0, "mK") == pytest.approx(1.0e-3)
    assert temperature_to_kelvin(1.0, "uK") == pytest.approx(1.0e-6)


def test_flux() -> None:
    assert flux_to_jy(1.0, "mJy") == pytest.approx(1.0e-3)
    assert flux_from_jy(1.0, "mJy") == pytest.approx(1.0e3)


def test_convert_array() -> None:
    vals = np.array([1.0, 2.0, 3.0])
    out = convert(vals, "MHz", "Hz", "frequency")
    assert np.allclose(out, vals * 1.0e6)


def test_unknown_unit_raises() -> None:
    with pytest.raises(ValueError):
        frequency_to_hz(1.0, "bogus")
    with pytest.raises(ValueError):
        convert(1.0, "Hz", "bogus", "frequency")
