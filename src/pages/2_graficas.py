import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, REGION_COLORS, SECTORES_PC, SECTORES_ABS,
    SECTOR_LABELS, PALETTE
)

dash.register_page(__name__, path="/graficas", name="Visualizaciones", order=2)

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS Y PALETA
# ═══════════════════════════════════════════════════════════════════════════════
df = cargar_datos()
P      = PALETTE
BG     = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED    = P["red"];    PURPLE  = P["purple"]

PC_DISP  = [c for c in SECTORES_PC  if c in df.columns]
ABS_DISP = [c for c in SECTORES_ABS if c in df.columns]

# Base sin keys conflictivos
BASE = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                font=dict(color=TEXT1)),
)

DD_STYLE = {
    "background": SURFACE,
    "color": TEXT1,
    "fontFamily": "IBM Plex Mono",
    "fontSize": "11px",
    "border": f"1px solid {BORDER}",
    "width": "300px",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI  — todos antes del layout
# ═══════════════════════════════════════════════════════════════════════════════
def T(text, sub=""):
    return html.Div([
        html.H3(text, style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "12px", "margin": "0", "letterSpacing": "0.07em",
            "textTransform": "uppercase",
        }),
        html.P(sub, style={
            "color": TEXT2, "fontSize": "10px", "margin": "4px 0 0",
            "fontFamily": "IBM Plex Mono", "lineHeight": "1.5",
        }) if sub else None,
    ], style={"borderLeft": f"3px solid {ACCENT}", "paddingLeft": "12px",
               "marginBottom": "16px"})


def narrative(text):
    return html.P(text, style={
        "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
        "lineHeight": "1.8", "marginBottom": "20px",
        "borderLeft": f"2px solid {BORDER}", "paddingLeft": "14px",
    })


def card_wrap(*children, mb="24px"):
    return html.Div(list(children), style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "22px 24px", "marginBottom": mb,
    })


def kpi(title, value, sub="", color=None):
    c = color or ACCENT
    return html.Div([
        html.P(title, style={
            "color": TEXT2, "fontSize": "9px", "letterSpacing": "0.12em",
            "textTransform": "uppercase", "fontFamily": "IBM Plex Mono", "marginBottom": "5px",
        }),
        html.P(value, style={
            "color": c, "fontSize": "18px", "fontWeight": "700",
            "fontFamily": "IBM Plex Sans", "margin": "0",
        }),
        html.P(sub, style={
            "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono", "marginTop": "4px",
        }) if sub else None,
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "8px",
        "padding": "14px 16px", "borderTop": f"3px solid {c}",
    })


def seccion_header(num, titulo, descripcion, color=None):
    c = color or ACCENT
    return html.Div([
        html.Span(f"SECCIÓN {num}", style={
            "color": c, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
            "letterSpacing": "0.2em", "fontWeight": "600",
        }),
        html.H2(titulo, style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "16px", "margin": "6px 0 8px",
        }),
        html.P(descripcion, style={
            "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
            "margin": "0", "lineHeight": "1.6",
        }),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {c}", "borderRadius": "8px",
        "padding": "16px 22px", "marginBottom": "20px",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE: KPIs GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════
_pob_total    = df["poblacion"].sum()
_total_gasto  = df["total"].sum() if "total" in df.columns else 0
_gasto_pc_nac = _total_gasto / _pob_total
_max_dep      = df.loc[df["total_pc"].idxmax(), "departamento"].title()
_min_dep      = df.loc[df["total_pc"].idxmin(), "departamento"].title()
_max_pc       = df["total_pc"].max()
_min_pc       = df["total_pc"].min()
_ratio        = _max_pc / _min_pc
_cv_total     = df["total_pc"].std() / df["total_pc"].mean() * 100

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS ESTÁTICAS — VISIÓN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
def fig_donut():
    sums   = {SECTOR_LABELS.get(s, s): df[s].sum()
              for s in ABS_DISP if s in df.columns}
    total  = sum(sums.values())
    colors = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#4E9AF1", "#F1A94E"]
    fig = go.Figure(go.Pie(
        labels=list(sums.keys()),
        values=list(sums.values()),
        hole=0.58,
        marker_colors=colors[:len(sums)],
        textfont=dict(family="IBM Plex Mono", size=10, color=TEXT1),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} COP<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Composición del Gasto Social Nacional · COP totales",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        annotations=[dict(
            text=f"${total/1e12:.1f}B<br>COP",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=TEXT2, family="IBM Plex Mono"),
        )],
        height=400, margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1, size=10)),
    )
    return fig


def fig_region_bar():
    reg = df.groupby("region").apply(
        lambda g: pd.Series({
            "Gasto_pc": (g["total_pc"] * g["poblacion"]).sum() / g["poblacion"].sum(),
            "n_deptos": len(g),
        })
    ).reset_index().sort_values("Gasto_pc")
    fig = go.Figure(go.Bar(
        x=reg["Gasto_pc"],
        y=reg["region"],
        orientation="h",
        marker_color=[REGION_COLORS.get(r, ACCENT) for r in reg["region"]],
        text=[f"${v/1e6:.0f}M  (n={n})"
              for v, n in zip(reg["Gasto_pc"], reg["n_deptos"])],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color=TEXT1),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f} COP/hab<extra></extra>",
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Gasto per cápita ponderado por Región",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="COP por habitante (ponderado por población)",
                   tickfont=dict(color=TEXT1), color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), color=TEXT2),
        showlegend=False, height=320,
        margin=dict(l=100, r=120, t=60, b=60),
    )
    return fig


def fig_scatter_vision():
    fig = go.Figure()
    for reg, gdf in df.groupby("region"):
        # Tamaño proporcional al gasto total absoluto
        sizes = gdf["total"].apply(lambda x: max(8, min(35, x / 4e10)))
        fig.add_trace(go.Scatter(
            x=gdf["poblacion"],
            y=gdf["total_pc"],
            mode="markers+text",
            name=reg,
            text=gdf["departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7, color=TEXT2),
            marker=dict(
                size=sizes,
                color=REGION_COLORS.get(reg, ACCENT),
                opacity=0.85,
                line=dict(width=0.8, color=BG),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Población: %{x:,.0f}<br>"
                "Gasto pc: $%{y:,.0f} COP<extra></extra>"
            ),
        ))
    fig.update_layout(
        **BASE,
        title=dict(
            text="Población vs Gasto per cápita · tamaño = gasto absoluto · color = región",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT1),
        ),
        xaxis=dict(
            type="log", title="Población 2024 (escala logarítmica)",
            gridcolor=BORDER, linecolor=BORDER,
            tickfont=dict(color=TEXT1), color=TEXT2,
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER,
            title="Gasto total per cápita (COP/hab)",
            tickfont=dict(color=TEXT1), color=TEXT2,
        ),
        height=500, margin=dict(l=80, r=20, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS ESTÁTICAS — PERFIL REGIONAL
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar_regional():
    available  = [c for c in PC_DISP if c != "total_pc"]
    reg_means  = df.groupby("region")[available].mean()
    categorias = [SECTOR_LABELS.get(s, s) for s in available]
    global_max = max(reg_means[c].max() for c in available)
    fig = go.Figure()
    for reg, row in reg_means.iterrows():
        vals = [v / global_max * 100 for v in row.values]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categorias + [categorias[0]],
            name=reg, fill="toself", opacity=0.5,
            line=dict(color=REGION_COLORS.get(reg, ACCENT), width=1.5),
            marker=dict(color=REGION_COLORS.get(reg, ACCENT)),
        ))
    fig.update_layout(
        **BASE,
        title=dict(text="Perfil de Gasto por Región · Normalizado al máximo global",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                            tickfont=dict(size=8, color=TEXT2), ticksuffix="%"),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=TEXT2)),
        ),
        showlegend=True, height=480,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def fig_boxplot_regional():
    fig = px.box(df, x="region", y="total_pc", color="region",
                  color_discrete_map=REGION_COLORS, points="all",
                  hover_name="departamento",
                  labels={"total_pc": "Gasto total pc (COP/hab)", "region": "Región"})
    fig.update_layout(
        **BASE,
        title=dict(text="Distribución del Gasto Total per cápita por Región",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1),
                   title="COP / habitante"),
        showlegend=False, height=420,
        margin=dict(l=80, r=20, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE FIGURAS ESTÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════
_fig_donut        = fig_donut()
_fig_region_bar   = fig_region_bar()
_fig_scatter      = fig_scatter_vision()
_fig_radar        = fig_radar_regional()
_fig_box_regional = fig_boxplot_regional()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
layout = html.Div([

    # ── PORTADA ───────────────────────────────────────────────────────────────
    html.Div([
        html.Span("VISUALIZACIONES · CONTINUACIÓN DEL EDA", style={
            "color": ACCENT, "fontSize": "9px", "letterSpacing": "0.2em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.H1("De los datos a los patrones territoriales", style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "22px", "margin": "10px 0 12px",
        }),
        html.P(
            "Esta sección profundiza en la estructura visual del gasto social departamental, "
            "complementando la narrativa analítica del EDA. Cada visualización responde a "
            "una pregunta específica y conecta con las técnicas multivariadas subsiguientes.",
            style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                   "lineHeight": "1.8", "maxWidth": "860px", "margin": "0"}
        ),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {ACCENT}", "borderRadius": "10px",
        "padding": "24px 32px", "marginBottom": "28px",
    }),

    # TABS
    dcc.Tabs(id="graf-tabs", value="vision", children=[
        dcc.Tab(label="Visión General",     value="vision"),
        dcc.Tab(label="Clasificaciones",    value="rank"),
        dcc.Tab(label="Comparativo",        value="comp"),
        dcc.Tab(label="Análisis Regional",  value="region"),
    ], style={"fontFamily": "IBM Plex Mono", "fontSize": "11px"},
       colors={"border": BORDER, "primary": ACCENT, "background": SURFACE}),

    html.Div(id="graf-content", style={"paddingTop": "28px"}),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
@callback(Output("graf-content", "children"), Input("graf-tabs", "value"))
def render_tab(tab):

    # ══════════════════════════════════════════════════════════════════════════
    # VISIÓN GENERAL
    # ══════════════════════════════════════════════════════════════════════════
    if tab == "vision":
        return html.Div([
            seccion_header("A", "Panorama Nacional del Gasto Social",
                           "¿Cuánto invierte Colombia en gasto social? "
                           "¿Cómo se distribuye entre sectores y regiones? "
                           "¿Cuál es la brecha entre el departamento que más y menos invierte?"),

            # KPIs
            html.Div([
                kpi("Gasto Nacional Total",
                    f"${_total_gasto/1e12:.2f} Billones",
                    "COP · 33 departamentos · 2024"),
                kpi("Gasto pc Promedio Ponderado",
                    f"${_gasto_pc_nac:,.0f}",
                    "COP por habitante · promedio nacional", color=GREEN),
                kpi("Mayor inversión pc",
                    _max_dep,
                    f"${_max_pc:,.0f} COP/hab", color=ACCENT),
                kpi("Menor inversión pc",
                    _min_dep,
                    f"${_min_pc:,.0f} COP/hab", color=RED),
                kpi("Brecha territorial",
                    f"{_ratio:.1f}x",
                    f"Ratio {_max_dep} / {_min_dep}", color=ORANGE),
                kpi("CV gasto total pc",
                    f"{_cv_total:.1f}%",
                    "Dispersión interdepartamental", color=PURPLE),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)",
                       "gap": "14px", "marginBottom": "28px"}),

            # Donut + Región bar
            card_wrap(
                T("Estructura del Gasto Social · Nacional y Regional",
                  "Izquierda: composición sectorial del gasto total · "
                  "Derecha: gasto per cápita ponderado por región"),
                narrative(
                    "Educación concentra el 59% del gasto social total, determinado "
                    "principalmente por las transferencias del SGP. La región Amazónica "
                    "presenta el mayor gasto per cápita ponderado, efecto de su baja "
                    "densidad poblacional más que de mayor capacidad fiscal."
                ),
                html.Div([
                    html.Div(dcc.Graph(figure=_fig_donut,
                                       config={"displayModeBar": False}),
                             style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=_fig_region_bar,
                                       config={"displayModeBar": False}),
                             style={"flex": "1.3"}),
                ], style={"display": "flex", "gap": "16px"}),
            ),

            # Scatter
            card_wrap(
                T("Relación Población – Gasto per cápita",
                  "Eje X logarítmico · Tamaño del punto = gasto absoluto · "
                  "Color = región geográfica"),
                narrative(
                    "La relación inversa entre población y gasto per cápita es estructural: "
                    "departamentos con menos habitantes reciben montos absolutos similares "
                    "pero divididos entre muy pocos habitantes. Bogotá, con la mayor población, "
                    "tiene el menor gasto per cápita. Este efecto debe considerarse en la "
                    "interpretación de los clusters."
                ),
                dcc.Graph(figure=_fig_scatter, config={"displayModeBar": False}),
            ),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # CLASIFICACIONES (RANKINGS)
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "rank":
        return html.Div([
            seccion_header("B", "Clasificaciones Departamentales",
                           "Top 10 y Bottom 10 por sector · Gasto per cápita · "
                           "Brecha territorial entre extremos",
                           color=GREEN),
            card_wrap(
                T("Seleccionar sector para el ranking"),
                html.Div([
                    html.Label("Sector:", style={
                        "color": TEXT2, "fontSize": "10px",
                        "fontFamily": "IBM Plex Mono",
                        "marginBottom": "6px", "display": "block",
                    }),
                    dcc.Dropdown(
                        id="rank-dd",
                        options=[{"label": SECTOR_LABELS.get(s, s), "value": s}
                                 for s in PC_DISP],
                        value="total_pc" if "total_pc" in df.columns else PC_DISP[0],
                        clearable=False,
                        style=DD_STYLE,
                    ),
                ], style={"marginBottom": "20px"}),
                html.Div(id="rank-output"),
            ),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARATIVO
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "comp":
        return html.Div([
            seccion_header("C", "Comparativo entre Departamentos",
                           "Composición del gasto per cápita por sector · "
                           "Filtrable por región · Barras apiladas",
                           color=ORANGE),
            card_wrap(
                T("Configurar comparativo"),
                html.Div([
                    html.Div([
                        html.Label("Sectores a mostrar:", style={
                            "color": TEXT2, "fontSize": "10px",
                            "fontFamily": "IBM Plex Mono",
                            "marginBottom": "6px", "display": "block",
                        }),
                        dcc.Dropdown(
                            id="comp-sectors-dd",
                            multi=True,
                            options=[{"label": SECTOR_LABELS.get(s, s), "value": s}
                                     for s in PC_DISP if s != "total_pc"],
                            value=[c for c in PC_DISP if c != "total_pc"],
                            style={**DD_STYLE, "width": "100%"},
                        ),
                    ], style={"flex": "1.5"}),
                    html.Div([
                        html.Label("Filtrar por región:", style={
                            "color": TEXT2, "fontSize": "10px",
                            "fontFamily": "IBM Plex Mono",
                            "marginBottom": "6px", "display": "block",
                        }),
                        dcc.Dropdown(
                            id="comp-region-dd",
                            multi=True,
                            options=[{"label": r, "value": r}
                                     for r in sorted(df["region"].dropna().unique())],
                            placeholder="Todas las regiones",
                            style={**DD_STYLE, "width": "100%"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
                html.Div(id="comp-output"),
            ),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ANÁLISIS REGIONAL
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "region":
        return html.Div([
            seccion_header("D", "Análisis Regional del Gasto Social",
                           "Radar por región · Distribución intra-regional · "
                           "Perfiles de inversión por área geográfica",
                           color=PURPLE),

            card_wrap(
                T("Radar · Perfil multisectorial por Región",
                  "Valores normalizados al máximo global de cada sector · "
                  "Permite comparar la estructura relativa del gasto entre regiones"),
                narrative(
                    "El perfil radar revela que la región Amazónica sobresale en "
                    "prácticamente todos los sectores por el efecto poblacional ya "
                    "discutido. La región Andina presenta el perfil más equilibrado, "
                    "mientras que Caribe y Pacífica muestran menor gasto relativo "
                    "en sectores de libre asignación. Estas diferencias de perfil "
                    "anticipan la formación de clusters con base geográfica en el "
                    "análisis de clasificación."
                ),
                dcc.Graph(figure=_fig_radar, config={"displayModeBar": False}),
            ),

            card_wrap(
                T("Boxplot · Distribución del Gasto Total pc por Región",
                  "Cada punto = un departamento · Muestra dispersión intra e inter-regional"),
                narrative(
                    "La dispersión intra-regional es sustancial en todas las regiones, "
                    "lo que indica que la región geográfica no captura completamente "
                    "la heterogeneidad del gasto. Esto justifica el uso de técnicas "
                    "de clustering multivariado que vayan más allá de la clasificación "
                    "geográfica tradicional."
                ),
                dcc.Graph(figure=_fig_box_regional, config={"displayModeBar": False}),
            ),

            # Tabla resumen por región
            card_wrap(
                T("Resumen Estadístico por Región",
                  "Media ponderada, mín, máx y CV del gasto total per cápita"),
                _tabla_region(),
            ),
        ])

    return html.Div()


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE FIGURAS DINÁMICAS
# ═══════════════════════════════════════════════════════════════════════════════
def _tabla_region():
    from dash import dash_table
    rows = []
    for reg, gdf in df.groupby("region"):
        v = gdf["total_pc"]
        rows.append({
            "Región":      reg,
            "n":           len(gdf),
            "Media pc":    f"${v.mean():,.0f}",
            "Mediana pc":  f"${v.median():,.0f}",
            "Mín pc":      f"${v.min():,.0f}",
            "Máx pc":      f"${v.max():,.0f}",
            "CV (%)":      f"{v.std()/v.mean()*100:.1f}%",
        })
    tdf = pd.DataFrame(rows).sort_values("Región")
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tdf.columns],
        style_cell={
            "background": CARD, "color": TEXT1, "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "padding": "9px 14px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Región"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600"},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}", "fontFamily": "IBM Plex Sans",
            "fontSize": "10px",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": SURFACE},
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS DINÁMICOS
# ═══════════════════════════════════════════════════════════════════════════════
@callback(Output("rank-output", "children"), Input("rank-dd", "value"))
def update_rank(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    label   = SECTOR_LABELS.get(sector, sector)
    sorted_ = df.sort_values(sector)
    top10   = sorted_.tail(10)
    bot10   = sorted_.head(10)
    max_row = df.loc[df[sector].idxmax()]
    min_row = df.loc[df[sector].idxmin()]
    ratio   = max_row[sector] / min_row[sector] if min_row[sector] > 0 else float("inf")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Top 10 · Mayor {label}", f"Bottom 10 · Menor {label}"],
    )
    fig.add_trace(go.Bar(
        x=top10[sector], y=top10["departamento"].str.title(),
        orientation="h", marker_color=GREEN,
        text=[f"${v:,.0f}" for v in top10[sector]],
        textposition="outside",
        textfont=dict(size=8, color=TEXT1),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=bot10[sector], y=bot10["departamento"].str.title(),
        orientation="h", marker_color=RED,
        text=[f"${v:,.0f}" for v in bot10[sector]],
        textposition="outside",
        textfont=dict(size=8, color=TEXT1),
    ), row=1, col=2)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="IBM Plex Mono", color=TEXT1, size=11),
        title=dict(text=f"Rankings · {label} (COP/hab)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=420, showlegend=False,
        margin=dict(l=160, r=120, t=80, b=40),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1, size=11))

    return html.Div([
        html.Div([
            kpi("Máximo", max_row["departamento"].title(),
                f"${max_row[sector]:,.0f} COP/hab", color=GREEN),
            kpi("Mínimo", min_row["departamento"].title(),
                f"${min_row[sector]:,.0f} COP/hab", color=RED),
            kpi("Brecha Máx/Mín", f"{ratio:.1f}x",
                "Desigualdad territorial en este sector", color=ORANGE),
            kpi("Promedio nacional", f"${df[sector].mean():,.0f}",
                "COP/hab"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
                   "gap": "12px", "marginBottom": "20px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


@callback(
    Output("comp-output", "children"),
    Input("comp-sectors-dd", "value"),
    Input("comp-region-dd",  "value"),
)
def update_comp(sectors, regions):
    if not sectors:
        return html.P("Selecciona al menos un sector.",
                      style={"color": TEXT2, "fontFamily": "IBM Plex Mono"})
    fdf = df if not regions else df[df["region"].isin(regions)]
    fdf = fdf.sort_values("total_pc") if "total_pc" in fdf.columns else fdf
    colors_list = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#4E9AF1", "#F1A94E"]
    fig = go.Figure()
    for i, s in enumerate(sectors):
        if s in fdf.columns:
            fig.add_trace(go.Bar(
                name=SECTOR_LABELS.get(s, s),
                x=fdf["departamento"].str.title(),
                y=fdf[s],
                marker_color=colors_list[i % len(colors_list)],
            ))
    n_deptos = len(fdf)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="IBM Plex Mono", color=TEXT1, size=11),
        title=dict(
            text=f"Gasto per cápita · {n_deptos} departamentos · "
                 f"{'Todas las regiones' if not regions else ', '.join(regions)}",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT1),
        ),
        barmode="stack", height=520,
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickangle=-45, title="Departamento",
                   tickfont=dict(color=TEXT1), color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="COP por habitante",
                   tickfont=dict(color=TEXT1), color=TEXT2),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1)),
        margin=dict(l=80, r=20, t=70, b=120),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})