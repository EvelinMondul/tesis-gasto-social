"""
components/elements.py
=======================

Componentes de interfaz reutilizables que garantizan que todas las
páginas comparten la misma identidad visual: encabezados de página,
tarjetas KPI, recuadros de metodología y el contenedor estándar de
figuras (título académico + dcc.Graph + leyenda/caption + exportación).
"""

from dash import dcc, html

from config import GRAPH_CONFIG, FIG_HEIGHT


def page_header(chapter_tag: str, title: str, subtitle: str) -> html.Div:
    """Encabezado estándar de capítulo: etiqueta + título + subtítulo."""
    return html.Div([
        html.Span(chapter_tag, className="chapter-tag"),
        html.Div(title, className="page-title"),
        html.Div(subtitle, className="page-subtitle"),
    ])


def section_title(text: str) -> html.Div:
    return html.Div(text, className="section-title")


def section_text(children) -> html.Div:
    return html.Div(children, className="section-text")


def method_card(title: str, children) -> html.Div:
    """Recuadro destacado para justificar una decisión metodológica."""
    return html.Div([
        html.Div(title, className="method-title"),
        html.Div(children),
    ], className="method-card")


def kpi_card(value: str, label: str, variant: int | None = None) -> html.Div:
    """Tarjeta KPI estándar. Si ``variant`` es 1-4, se aplica un fondo
    degradado (uso recomendado solo en la Portada, Capítulo 00)."""
    class_name = "kpi-card"
    if variant is not None:
        class_name = f"kpi-card kpi-card-gradient kpi-grad-{variant}"
    return html.Div([
        html.Div(value, className="kpi-value"),
        html.Div(label, className="kpi-label"),
    ], className=class_name)


def figure_block(fig, caption=None, fig_id=None, height=FIG_HEIGHT) -> html.Div:
    """Contenedor estándar para una figura Plotly: gráfico + leyenda
    académica, con botón de exportación PNG/SVG/PDF habilitado vía
    config.GRAPH_CONFIG."""
    fig.update_layout(height=height)
    graph = dcc.Graph(
        figure=fig,
        config=GRAPH_CONFIG,
        id=fig_id if fig_id else dash_uid(),
    )
    children = [graph]
    if caption:
        children.append(html.Div(caption, className="fig-caption"))
    return html.Div(children)


_uid_counter = {"n": 0}


def dash_uid() -> str:
    _uid_counter["n"] += 1
    return f"fig-{_uid_counter['n']}"
