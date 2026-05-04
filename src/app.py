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
    index_string="""
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            *, *::before, *::after { box-sizing: border-box; }

            html, body {
                background-color: #0D1117 !important;
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }

            #react-entry-point,
            #_dash-app-content,
            ._dash-loading,
            ._dash-loading-callback {
                background-color: #0D1117 !important;
                min-height: 100vh;
            }

            ._dash-loading-callback {
                background: rgba(13, 17, 23, 0.85) !important;
            }

            .page-fade {
                animation: fadeIn 0.25s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to   { opacity: 1; transform: translateY(0);   }
            }

            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #0D1117; }
            ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #58A6FF; }

            /* Dash Dropdown — fondo oscuro */
            .Select-control,
            .Select-menu-outer,
            .Select-option,
            .Select--single > .Select-control .Select-value,
            .VirtualizedSelectOption {
                background-color: #161B22 !important;
                color: #E6EDF3 !important;
                border-color: #30363D !important;
            }
            .Select-option.is-focused {
                background-color: #1C2333 !important;
            }

            /* DataTable */
            .dash-table-container .dash-spreadsheet-container
            .dash-spreadsheet-inner td,
            .dash-table-container .dash-spreadsheet-container
            .dash-spreadsheet-inner th {
                font-family: 'IBM Plex Mono', monospace !important;
            }

            /* Tabs */
            .tab--selected {
                border-top: 2px solid #58A6FF !important;
                background: #1C2333 !important;
                color: #E6EDF3 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
""",
)
server = app.server

# ── PALETA ────────────────────────────────────────────────────────────────────
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
]

# ── NAVBAR ────────────────────────────────────────────────────────────────────
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

# ── FOOTER ────────────────────────────────────────────────────────────────────
footer = html.Div([
    html.Span(
        "Patrones de inversión en gasto social · Colombia  ·  "
        "PCA · Factorial · Clúster",
        style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px"},
    ),
], style={
    "borderTop": f"1px solid {BORDER}", "padding": "16px 40px",
    "background": SURFACE, "marginTop": "40px", "textAlign": "center",
})

# ── LAYOUT PRINCIPAL ──────────────────────────────────────────────────────────
app.layout = html.Div([
    navbar,
    html.Div(
        dash.page_container,
        style={"background": BG, "minHeight": "calc(100vh - 140px)"},
    ),
    footer,
], style={"background": BG, "minHeight": "100vh"})


# ── CALLBACK: resaltar nav activo ─────────────────────────────────────────────
@app.callback(Output("nav-links", "children"), Input("url", "pathname"))
def highlight_nav(pathname):
    links = []
    for p in NAV_PAGES:
        active = (pathname == p["href"]) or (
            p["href"] != "/" and pathname and pathname.startswith(p["href"])
        )
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
            "transition":     "all 0.15s ease",
        }
        links.append(dcc.Link(p["label"], href=p["href"], style=style))
    return links


# ── RUN ───────────────────────────────────────────────────────────────────────
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)