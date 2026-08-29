"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/domain/units.py

Descrição:
    Conversões de unidades para dados de waterfall em radioastronomia:
    frequência (Hz/kHz/MHz/GHz), tempo (s/ms/us/ns/min/h/dia), temperatura
    (K/mK/uK) e fluxo (Jy/mJy/uJy).

    Convenções do projeto (ver AGENTS.md):
    - Frequência: MHz para o usuário; manipulações internas em Hz nos dados.
    - Tempo interno em MJD; intervalos padrão em segundos.
    - Ângulos em graus para o usuário, radianos internamente.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Registros de conversão (fator para a unidade base SI)
# ---------------------------------------------------------------------------

_FREQUENCY_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}

_TIME_TO_SECONDS = {
    "ns": 1.0e-9,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "ms": 1.0e-3,
    "s": 1.0,
    "sec": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "day": 86400.0,
    "d": 86400.0,
}

_TEMPERATURE_TO_KELVIN = {
    "k": 1.0,
    "mk": 1.0e-3,
    "uk": 1.0e-6,
    "µk": 1.0e-6,
}

_FLUX_TO_JY = {
    "jy": 1.0,
    "mjy": 1.0e-3,
    "ujy": 1.0e-6,
    "µjy": 1.0e-6,
}

# Aliases normalizados (minúsculas) por categoria
_UNIT_REGISTRIES: dict[str, dict[str, float]] = {
    "frequency": _FREQUENCY_TO_HZ,
    "time": _TIME_TO_SECONDS,
    "temperature": _TEMPERATURE_TO_KELVIN,
    "flux": _FLUX_TO_JY,
}


def _normalize_unit(unit: str) -> str:
    """Normaliza um símbolo de unidade para lookup (minúsculas, sem espaços)."""
    return str(unit).strip().lower()


def convert(
    value: float | np.ndarray,
    from_unit: str,
    to_unit: str,
    category: str,
) -> float | np.ndarray:
    """Converte `value` de `from_unit` para `to_unit` dentro de `category`.

    Args:
        value: Valor (escalar ou array) a converter.
        from_unit: Unidade de origem.
        to_unit: Unidade de destino.
        category: Categoria ("frequency", "time", "temperature", "flux").

    Returns:
        Valor convertido.

    Raises:
        ValueError: Se alguma unidade for desconhecida na categoria.
    """
    registry = _UNIT_REGISTRIES[category]
    f = _normalize_unit(from_unit)
    t = _normalize_unit(to_unit)
    if f not in registry:
        msg = f"Unidade de {category} desconhecida: {from_unit!r}"
        raise ValueError(msg)
    if t not in registry:
        msg = f"Unidade de {category} desconhecida: {to_unit!r}"
        raise ValueError(msg)
    return np.asarray(value) * (registry[f] / registry[t])


# ---------------------------------------------------------------------------
# Frequência
# ---------------------------------------------------------------------------


def frequency_to_hz(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte um valor de frequência para Hz."""
    return convert(value, unit, "Hz", "frequency")


def frequency_from_hz(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte um valor em Hz para a unidade `unit`."""
    return convert(value, "Hz", unit, "frequency")


def frequency_to_mhz(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte um valor de frequência para MHz (unidade de apresentação padrão)."""
    return convert(value, unit, "MHz", "frequency")


def frequency_from_mhz(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte um valor em MHz para a unidade `unit`."""
    return convert(value, "MHz", unit, "frequency")


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------


def time_to_seconds(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte um intervalo de tempo para segundos."""
    return convert(value, unit, "s", "time")


def time_from_seconds(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte segundos para a unidade `unit`."""
    return convert(value, "s", unit, "time")


# ---------------------------------------------------------------------------
# Temperatura
# ---------------------------------------------------------------------------


def temperature_to_kelvin(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte temperatura para Kelvin."""
    return convert(value, unit, "K", "temperature")


def temperature_from_kelvin(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte Kelvin para a unidade `unit`."""
    return convert(value, "K", unit, "temperature")


# ---------------------------------------------------------------------------
# Fluxo
# ---------------------------------------------------------------------------


def flux_to_jy(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte fluxo para Jy."""
    return convert(value, unit, "Jy", "flux")


def flux_from_jy(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    """Converte Jy para a unidade `unit`."""
    return convert(value, "Jy", unit, "flux")


# ---------------------------------------------------------------------------
# Helpers para coordenadas xarray
# ---------------------------------------------------------------------------


def coord_unit(coord: Any, default: str) -> str:
    """Obtém a unidade de uma coordenada xarray via attrs, com fallback."""
    attrs = getattr(coord, "attrs", {}) or {}
    return str(attrs.get("units", default))


def convert_coord(coord: Any, to_unit: str, category: str) -> Any:
    """Converte uma coordenada xarray (com attrs['units']) para `to_unit`.

    Atualiza ``attrs["units"]`` e devolve uma nova coordenada.
    """
    from_unit = coord_unit(coord, _default_unit(category))
    values = convert(coord.values, from_unit, to_unit, category)
    new_coord = coord.copy(data=values)
    new_coord.attrs["units"] = to_unit
    return new_coord


def _default_unit(category: str) -> str:
    return {
        "frequency": "Hz",
        "time": "s",
        "temperature": "K",
        "flux": "Jy",
    }[category]


__all__ = [
    "convert",
    "convert_coord",
    "coord_unit",
    "flux_from_jy",
    "flux_to_jy",
    "frequency_from_hz",
    "frequency_from_mhz",
    "frequency_to_hz",
    "frequency_to_mhz",
    "temperature_from_kelvin",
    "temperature_to_kelvin",
    "time_from_seconds",
    "time_to_seconds",
]
