"""
components/placeholder.py
==========================

Layout temporal para los capítulos aún no implementados. Mantiene la
identidad visual (encabezado de capítulo) para que la navegación y el
diseño general puedan validarse de extremo a extremo desde la primera
fase de construcción.
"""

from dash import html

from components.elements import page_header


def placeholder_layout(chapter_tag: str, title: str, subtitle: str) -> html.Div:
    return html.Div([
        page_header(chapter_tag, title, subtitle),
        html.Div(
            "Este capítulo se implementará en una siguiente fase de construcción.",
            className="section-text",
            style={"color": "var(--color-gris-medio)", "fontStyle": "italic", "marginTop": "24px"},
        ),
    ])
