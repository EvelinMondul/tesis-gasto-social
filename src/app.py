import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

BG      = "#0D1117"
SURFACE = "#161B22"
CARD    = "#1C2333"
BORDER  = "#30363D"
TEXT1   = "#E6EDF3"
TEXT2   = "#8B949E"
ACCENT  = "#58A6FF"

NAV_PAGES = [
    {"label": "01 · Introducción",    "href": "/"},
    {"label": "02 · EDA & Tablas",    "href": "/eda"},
    {"label": "03 · Visualizaciones", "href": "/graficas"},
    {"label": "04 · Composicional",   "href": "/composicional"},
]

navbar = html.Div([
    html.Div([
        html.Div([
            html.Span("● ", style={"color": ACCENT, "fontSize": "9px"}),
            html.Span("MAESTRÍA EN ESTADÍSTICA APLICADA · UNINORTE", style={
                "color": TEXT2, "fontSize": "9px", "letterSpacing": "0.16em",
                "fontFamily": "IBM Plex Mono", "fontWeight": "600",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
        html.H1("Gasto Social · Departamentos de Colombia", style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "17px", "margin": "0",
        }),
    ], style={"padding": "18px 40px 0"}),
    html.Div([
        html.Div(id="nav-links", style={"display": "flex", "gap": "4px"}),
        html.Span("TerriData DNP · DANE · 2024", style={
            "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
        }),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "12px 40px", "marginTop": "12px",
        "borderBottom": f"1px solid {BORDER}",
    }),
    dcc.Location(id="url", refresh=False),
], style={"background": SURFACE, "borderBottom": f"1px solid {BORDER}"})

footer = html.Div([
    html.Span("Patrones de inversión en gasto social · Colombia  ·  PCA · Factorial · Clúster",
              style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px"}),
], style={
    "borderTop": f"1px solid {BORDER}", "padding": "16px 40px",
    "background": SURFACE, "marginTop": "40px", "textAlign": "center",
})

app.layout = html.Div([
    navbar,
    html.Div(dash.page_container,
             style={"background": BG, "minHeight": "calc(100vh - 140px)"}),
    footer,
], style={"background": BG, "minHeight": "100vh"})


@app.callback(Output("nav-links", "children"), Input("url", "pathname"))
def highlight_nav(pathname):
    links = []
    for p in NAV_PAGES:
        active = (pathname == p["href"]) or \
                 (p["href"] != "/" and pathname and pathname.startswith(p["href"]))
        style = {
            "color":          ACCENT if active else TEXT2,
            "textDecoration": "none",
            "fontFamily":     "IBM Plex Mono",
            "fontSize":       "11px",
            "padding":        "7px 14px",
            "borderRadius":   "6px",
            "border":         f"1px solid {ACCENT if active else 'transparent'}",
            "background":     CARD if active else "transparent",
            "fontWeight":     "600" if active else "400",
            "letterSpacing":  "0.03em",
        }
        links.append(dcc.Link(p["label"], href=p["href"], style=style))
    return links


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)