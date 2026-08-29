"""\
Módulo: tests/test_monitor.py

Testes do SystemMonitor (execução segura de recursos).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import pytest

from radioskysims_waterfall.infrastructure.monitor import ResourceError, SystemMonitor


def test_monitor_reports() -> None:
    mon = SystemMonitor()
    r = mon.report()
    assert r["cpu_count"] >= 1
    assert r["memory_total_bytes"] > 0
    assert 0.0 <= r["memory_percent"] <= 100.0


def test_monitor_estimate_nbytes() -> None:
    assert SystemMonitor.estimate_nbytes((10, 10), "float64") == 800


def test_check_ok() -> None:
    mon = SystemMonitor()
    ok, reason = mon.check(required_nbytes=0, required_cpus=1)
    assert ok is True
    assert reason == "ok"


def test_check_insufficient_cpus() -> None:
    mon = SystemMonitor()
    ok, reason = mon.check(required_cpus=10**9)
    assert ok is False
    assert "CPUs" in reason


def test_ensure_resources_ok() -> None:
    mon = SystemMonitor()
    with mon.ensure_resources(required_nbytes=0):
        pass  # não deve levantar


def test_ensure_resources_raises() -> None:
    mon = SystemMonitor()
    with pytest.raises(ResourceError), mon.ensure_resources(required_nbytes=10**18):
        pass  # nunca executa


def test_ensure_resources_warns_not_raise() -> None:
    mon = SystemMonitor()
    with mon.ensure_resources(required_nbytes=10**18, raise_on_error=False):
        pass  # deve prosseguir com warning


def test_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError):
        SystemMonitor(safe_fraction=0.0)
    with pytest.raises(ValueError):
        SystemMonitor(safe_fraction=1.5)


def test_safe_memory_bounds() -> None:
    mon = SystemMonitor()
    assert 0 <= mon.safe_memory_bytes() <= mon.available_memory_bytes()
