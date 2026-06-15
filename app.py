"""
app.py
=======

Punto de entrada de la aplicación Dash multipágina.

Estructura: una barra lateral fija (components/sidebar.py) con los 11
capítulos del trabajo de grado, y un área de contenido que renderiza la
página activa mediante ``dash.page_container``. Cada capítulo se
implementa como un módulo independiente en ``pages/``, registrado con
``dash.register_page``.
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Importa y registra la plantilla visual compartida (pio.templates["tesis"])
import config  # noqa: F401

from components.sidebar import build_sidebar

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="CoDA Gasto Social Colombia 2024",
    update_title=None,
)

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div([
        build_sidebar(),
        html.Div(dash.page_container, className="content-area"),
    ], className="app-shell"),
])

server = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
