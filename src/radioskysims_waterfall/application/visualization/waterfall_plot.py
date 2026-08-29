"""\
Módulo: packages/radioskysims-waterfall/src/radioskysims_waterfall/application/visualization/waterfall_plot.py

Descrição:
    Plotagem de waterfalls usando HoloViews (+ Datashader opcional) para
    preservar memória em grandes volumes (tempo × frequência).

    Funcionalidades:
    - ``plot_waterfall``: um ou dois waterfalls lado a lado, com curva de luz
      (lightcurve) e espectro (SED) em layout.
    - ``plot_sed``: curva da SED com barras de erro (std).
    - ``plot_lightcurve``: curva de luz com barras de erro (std).

    As dependências holoviews/datashader são importadas sob demanda.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from radioskysims_waterfall.domain.lightcurve import LightCurve
    from radioskysims_waterfall.domain.models import Waterfall
    from radioskysims_waterfall.domain.sed import SED


def _holoviews():
    try:
        import holoviews as hv
    except ImportError as exc:  # pragma: no cover
        msg = "holoviews é necessário para plotar. Instale: pip install holoviews"
        raise ImportError(msg) from exc
    return hv


_extension_loaded = False


def _ensure_extension() -> None:
    """Garante que uma extensão de plotagem do holoviews esteja carregada."""
    global _extension_loaded
    if _extension_loaded:
        return
    hv = _holoviews()
    hv.extension("bokeh")
    _extension_loaded = True


def _datashader():
    try:
        import datashader as ds
    except ImportError as exc:  # pragma: no cover
        msg = "datashader é necessário para rasterização. Instale: pip install datashader"
        raise ImportError(msg) from exc
    return ds


def _to_clim(
    data: Any, clim: tuple[float, float] | None, percentiles: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Deriva limites de cor a partir de percentis dos dados (robusto/lazy-safe)."""
    if clim is not None:
        return tuple(clim)
    if percentiles is None:
        return None
    vals = data
    lo = float(vals.quantile(percentiles[0] / 100.0, skipna=True).compute())
    hi = float(vals.quantile(percentiles[1] / 100.0, skipna=True).compute())
    return (lo, hi)


def _waterfall_element(
    wf: Waterfall,
    *,
    use_datashader: bool,
    cmap: str,
    clim: tuple[float, float] | None,
    title: str | None,
    width: int,
    height: int,
) -> Any:
    hv = _holoviews()
    element = hv.QuadMesh(wf.data, kdims=["time", "freq"], vdims=["value"])
    opts: dict[str, Any] = {
        "cmap": cmap,
        "width": width,
        "height": height,
    }
    if clim is not None:
        opts["clim"] = clim
    if title:
        opts["title"] = title

    if use_datashader:
        ds = _datashader()
        try:
            return ds.rasterize(element).opts(**opts)
        except Exception:
            return element.opts(**opts)
    return element.opts(**opts)


def _lightcurve_element(wf: Waterfall, *, width: int, height: int) -> Any:
    hv = _holoviews()
    lc = wf.lightcurve()
    curve = hv.Curve(lc.values, kdims="time", vdims="value", label="lightcurve")
    return curve.opts(width=width, height=height)


def _sed_element(wf: Waterfall, *, width: int, height: int) -> Any:
    hv = _holoviews()
    sed = wf.sed()
    curve = hv.Curve(sed.values, kdims="freq", vdims="value", label="SED")
    return curve.opts(width=width, height=height)


def plot_waterfall(
    wf: Waterfall,
    other: Waterfall | None = None,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    clim_percentiles: tuple[float, float] = (1.0, 99.0),
    use_datashader: bool = True,
    include_lightcurve: bool = True,
    include_sed: bool = True,
    width: int = 420,
    height: int = 320,
    **kwargs: Any,
) -> Any:
    """Plota um ou dois waterfalls lado a lado (holoviews + datashader).

    Args:
        wf: Primeiro waterfall.
        other: Segundo waterfall opcional (lado a lado para comparação).
        title: Título do gráfico.
        cmap: Colormap.
        clim: Limites de cor explícitos (vmin, vmax).
        clim_percentiles: Percentis para derivar clim dos dados (default 1-99).
        use_datashader: Rasterizar com datashader (preserva memória).
        include_lightcurve: Incluir painel de curva de luz.
        include_sed: Incluir painel de SED.
        width/height: Dimensões de cada painel.
        **kwargs: Reservado.

    Returns:
        ``hv.Layout`` com os painéis (waterfall [+ luz + SED] [+ outro]).
    """
    _ensure_extension()
    hv = _holoviews()
    lc_wf = _to_clim(wf.data, clim, clim_percentiles)

    elements: list[Any] = [
        _waterfall_element(
            wf, use_datashader=use_datashader, cmap=cmap, clim=lc_wf, title=title, width=width, height=height
        )
    ]
    if include_lightcurve:
        elements.append(_lightcurve_element(wf, width=width, height=height))
    if include_sed:
        elements.append(_sed_element(wf, width=width, height=height))

    if other is not None:
        lc_other = _to_clim(other.data, clim, clim_percentiles)
        other_title = getattr(other.meta, "name", None)
        elements.append(
            _waterfall_element(
                other,
                use_datashader=use_datashader,
                cmap=cmap,
                clim=lc_other,
                title=other_title,
                width=width,
                height=height,
            )
        )
        if include_lightcurve:
            elements.append(_lightcurve_element(other, width=width, height=height))
        if include_sed:
            elements.append(_sed_element(other, width=width, height=height))

    return hv.Layout(elements).opts(shared_axes=False).cols(2)


def plot_sed(
    sed: SED,
    *,
    title: str | None = None,
    with_error: bool = True,
    width: int = 500,
    height: int = 350,
    **kwargs: Any,
) -> Any:
    """Plota a SED (curva média por canal + barras de erro opcionais)."""
    _ensure_extension()
    hv = _holoviews()
    curve = hv.Curve(sed.values, kdims="freq", vdims="value", label=sed.name or "SED")
    if with_error and sed.std is not None:
        eb = hv.ErrorBars((sed.freq, sed.values, sed.std))
        plot: Any = curve * eb
    else:
        plot = curve
    opts: dict[str, Any] = {"width": width, "height": height}
    if title:
        opts["title"] = title
    return plot.opts(**opts)


def plot_lightcurve(
    lc: LightCurve,
    *,
    title: str | None = None,
    with_error: bool = True,
    width: int = 500,
    height: int = 350,
    **kwargs: Any,
) -> Any:
    """Plota a curva de luz (média por instante + barras de erro opcionais)."""
    _ensure_extension()
    hv = _holoviews()
    curve = hv.Curve(lc.values, kdims="time", vdims="value", label=lc.name or "LightCurve")
    if with_error and lc.std is not None:
        eb = hv.ErrorBars((lc.time, lc.values, lc.std))
        plot: Any = curve * eb
    else:
        plot = curve
    opts: dict[str, Any] = {"width": width, "height": height}
    if title:
        opts["title"] = title
    return plot.opts(**opts)


__all__ = ["plot_lightcurve", "plot_sed", "plot_waterfall"]
