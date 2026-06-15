"""
components/sidebar.py
======================

Barra de navegación lateral fija, con los 11 capítulos del trabajo de
grado en el orden lógico de lectura. El enlace correspondiente a la
página activa se resalta automáticamente mediante un callback sobre
``dcc.Location``.
"""

import dash
from dash import Input, Output, callback, dcc, html

# Estructura de navegación: (número de capítulo, etiqueta, ruta)
NAV_ITEMS = [
    ("00", "Portada", "/"),
    ("01", "Introducción", "/introduccion"),
    ("02", "Análisis exploratorio", "/eda"),
    ("03", "Análisis composicional", "/composicional"),
    ("04", "Correlaciones (CLR)", "/correlaciones"),
    ("05", "Componentes principales", "/acp"),
    ("06", "Análisis factorial", "/factorial"),
    ("07", "Clustering", "/clustering"),
    ("08", "Pobreza multidimensional", "/ipm"),
    ("09", "Mapas territoriales", "/mapas"),
    ("10", "Conclusiones", "/conclusiones"),
]


def build_sidebar() -> html.Div:
    links = []
    for num, label, href in NAV_ITEMS:
        links.append(
            dcc.Link(
                [html.Span(num, className="nav-link-num"), html.Span(label)],
                href=href,
                id={"type": "nav-link", "index": href},
                className="nav-link-custom",
            )
        )

    return html.Div(
        [
            html.Div("Análisis Composicional del", className="sidebar-brand"),
            html.Div("Gasto Social Departamental", className="sidebar-brand",
                     style={"marginBottom": "2px"}),
            html.Div("Colombia · 2024 · CoDA / CLR / ACP / Clustering",
                     className="sidebar-subtitle"),
            html.Div("Capítulos", className="sidebar-section-label"),
            html.Div(links, id="nav-links-container"),
            html.Div(
                "Tesis de Maestría en Estadística Aplicada · Universidad del Norte",
                style={
                    "fontSize": "0.72rem",
                    "color": "var(--color-gris-medio)",
                    "marginTop": "32px",
                    "paddingTop": "16px",
                    "borderTop": "1px solid var(--color-border)",
                },
            ),
        ],
        className="sidebar",
    )


@callback(
    Output("nav-links-container", "children"),
    Input("url", "pathname"),
)
def actualizar_enlace_activo(pathname):
    """Resalta el enlace de navegación correspondiente a la ruta activa."""
    links = []
    for num, label, href in NAV_ITEMS:
        is_active = (pathname == href) or (href != "/" and pathname.startswith(href))
        if href == "/" and pathname not in ("/", ""):
            is_active = False
        classes = "nav-link-custom active" if is_active else "nav-link-custom"
        links.append(
            dcc.Link(
                [html.Span(num, className="nav-link-num"), html.Span(label)],
                href=href,
                className=classes,
            )
        )
    return links
