"""\
Módulo: tests/test_pipeline.py

Testes do pipeline declarativo e da execução segura com monitor.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

import pytest

from radioskysims_waterfall import SystemMonitor, Waterfall, WaterfallPipeline


def test_add_and_run(wf) -> None:
    pipe = WaterfallPipeline()
    pipe.add("filter_spectral", method="median", window=3)
    pipe.add("normalize", method="minmax")
    out = pipe.run(wf)
    assert isinstance(out, Waterfall)
    assert out.shape == wf.shape


def test_steps_are_immutable_list(wf) -> None:
    pipe = WaterfallPipeline()
    pipe.add("normalize", method="minmax")
    steps = pipe.steps
    steps.append({})  # não deve alterar o pipeline
    assert len(pipe.steps) == 1


def test_describe(wf) -> None:
    pipe = WaterfallPipeline()
    pipe.add("normalize", method="minmax")
    desc = pipe.describe()
    assert len(desc) == 1
    assert "normalize" in desc[0]


def test_add_requires_name_or_func(wf) -> None:
    with pytest.raises(ValueError):
        WaterfallPipeline().add()
    with pytest.raises(ValueError):
        WaterfallPipeline().add("normalize", func=lambda w: w)


def test_run_unknown_method_raises(wf) -> None:
    pipe = WaterfallPipeline()
    pipe.add("metodo_inexistente")
    with pytest.raises(AttributeError):
        pipe.run(wf)


def test_run_func_step(wf) -> None:
    def doubler(w: Waterfall) -> Waterfall:
        return w.slice(time=slice(0, 10))

    pipe = WaterfallPipeline()
    pipe.add(func=doubler)
    out = pipe.run(wf)
    assert out.n_time == 10


def test_init_with_steps(wf) -> None:
    pipe = WaterfallPipeline([{"name": "normalize", "kwargs": {"method": "minmax"}}])
    assert len(pipe) == 1
    assert pipe.run(wf).shape == wf.shape


def test_waterfall_pipeline_method(wf) -> None:
    steps = [
        {"name": "filter_spectral", "kwargs": {"method": "median", "window": 3}},
        {"name": "normalize", "kwargs": {"method": "minmax"}},
    ]
    out = wf.pipeline(steps)
    assert isinstance(out, Waterfall)
    assert out.shape == wf.shape


def test_safe_requires_monitor(wf) -> None:
    pipe = WaterfallPipeline(safe=True)
    pipe.add("normalize", method="minmax")
    with pytest.raises(RuntimeError):
        pipe.run(wf)


def test_safe_with_monitor(wf) -> None:
    mon = SystemMonitor()
    pipe = WaterfallPipeline(safe=True, monitor=mon)
    pipe.add("normalize", method="minmax")
    out = pipe.run(wf)
    assert out.shape == wf.shape


def test_safe_with_run_monitor(wf) -> None:
    pipe = WaterfallPipeline(safe=True)
    pipe.add("normalize", method="minmax")
    out = pipe.run(wf, monitor=SystemMonitor())
    assert out.shape == wf.shape


def test_pipeline_accepts_injected_structured_logger(wf) -> None:
    """Pipeline deve emitir evento estruturado usando logger injetado."""

    class FakeLogger:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def info(self, event: str, **kwargs: object) -> None:
            self.events.append((event, kwargs))

    logger = FakeLogger()

    pipe = WaterfallPipeline(logger=logger)
    pipe.add("normalize", method="minmax")

    out = pipe.run(wf)

    assert out.shape == wf.shape
    assert logger.events == [
        (
            "waterfall_pipeline_step_completed",
            {
                "step_index": 1,
                "step_count": 1,
                "step": "normalize",
            },
        )
    ]
