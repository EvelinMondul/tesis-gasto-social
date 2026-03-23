import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, REGION_COLORS, SECTORES_PC, SECTORES_ABS,
    SECTOR_LABELS, PALETTE, PLOTLY_TEMPLATE
)

dash.register_page(__name__, path="/graficas", name="Visualizaciones", order=2)

# ── DATOS ─────────────────────────────────────────────────────────────────────
df = cargar_datos()
P       = PALETTE
BG      = P["bg"];   SURFACE = P["surface"]; CARD   = P["card"]
BORDER  = P["border"]; TEXT1 = P["text1"];   TEXT2  = P["text2"]
ACCENT  = P["accent"]; GREEN = P["green"];   ORANGE = P["orange"]; RED = P["red"]

PC_DISP  = [c for c in SECTORES_PC if c in df.columns]
ABS_DISP = [c for c in SECTORES_ABS if c in df.columns]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def section_title(text, sub=""):
    return html.Div([
        html.H3(text, style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "600",
            "fontSize": "12px", "margin": "0", "letterSpacing": "0.06em",
            "textTransform": "uppercase",
        }),
        html.P(sub, style={
            "color": TEXT2, "fontSize": "10px", "margin": "4px 0 0",
            "fontFamily": "IBM Plex Mono",
        }) if sub else None,
    ], style={"borderLeft": f"3px solid {ACCENT}", "paddingLeft": "12px",
               "marginBottom": "18px"})


def stat_card(title, value, sub="", color=None):
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


def dd_sector(id_, value=None):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": SECTOR_LABELS.get(s, s), "value": s} for s in PC_DISP],
        value=value or (PC_DISP[0] if PC_DISP else None),
        clearable=False,
        style={"background": CARD, "color": TEXT1, "fontFamily": "IBM Plex Mono",
               "fontSize": "11px", "width": "260px"},
    )


# ── FIGURA ESTÁTICA: Donut composición nacional ───────────────────────────────
def make_donut():
    sums   = {SECTOR_LABELS.get(s, s): df[s].sum() for s in ABS_DISP if s in df.columns}
    colors = [ACCENT, GREEN, RED, ORANGE, P["purple"], "#4E9AF1", "#F1A94E"]
    fig = go.Figure(go.Pie(
        labels=list(sums.keys()), values=list(sums.values()),
        hole=0.55,
        marker_colors=colors[:len(sums)],
        textfont=dict(family="IBM Plex Mono", size=10),
    ))
    total = sum(sums.values())
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Composición del Gasto Social Nacional",
        annotations=[dict(text=f"${total/1e12:.1f}B\nCOP", x=0.5, y=0.5,
                          font=dict(size=11, color=TEXT2), showarrow=False)],
        height=380,
    )
    return fig


def make_region_bar():
    reg = df.groupby("region").agg(
        Gasto_pc=("total_pc", lambda g: (g * df.loc[g.index, "poblacion"]).sum()
                  / df.loc[g.index, "poblacion"].sum())
    ).reset_index().sort_values("Gasto_pc") if "total_pc" in df.columns else pd.DataFrame()
    if reg.empty:
        return go.Figure()
    fig = px.bar(reg, x="Gasto_pc", y="region", orientation="h",
                 color="region", color_discrete_map=REGION_COLORS,
                 labels={"Gasto_pc": "Gasto per cápita ponderado (COP)", "region": "Región"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                       showlegend=False, height=300,
                       title="Gasto per cápita ponderado por Región")
    return fig


# ── LAYOUT ────────────────────────────────────────────────────────────────────
layout = html.Div([

    dcc.Tabs(id="graf-tabs", value="vision", children=[
        dcc.Tab(label="Visión General",  value="vision"),
        dcc.Tab(label="Rankings",        value="rank"),
        dcc.Tab(label="Comparativo",     value="comp"),
        dcc.Tab(label="Evolución Regional", value="region"),
    ], style={"fontFamily": "IBM Plex Mono", "fontSize": "11px"},
       colors={"border": BORDER, "primary": ACCENT, "background": SURFACE}),

    html.Div(id="graf-content", style={"paddingTop": "28px"}),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")


# ── CALLBACK TABS ─────────────────────────────────────────────────────────────
@callback(Output("graf-content", "children"), Input("graf-tabs", "value"))
def render_graf(tab):

    if tab == "vision":
        pob_total  = df["poblacion"].sum() if "poblacion" in df.columns else 1
        total_gasto = df["total"].sum() if "total" in df.columns else 0
        gasto_pc_nac = total_gasto / pob_total
        max_dep = df.loc[df["total_pc"].idxmax(), "departamento"].title() if "total_pc" in df.columns else "—"
        min_dep = df.loc[df["total_pc"].idxmin(), "departamento"].title() if "total_pc" in df.columns else "—"

        fig_scatter = go.Figure()
        for reg, gdf in df.groupby("region"):
            fig_scatter.add_trace(go.Scatter(
                x=gdf["poblacion"],
                y=gdf["total_pc"] if "total_pc" in gdf.columns else [0]*len(gdf),
                mode="markers+text",
                name=reg,
                text=gdf["departamento"].str.title(),
                textposition="top center",
                textfont=dict(size=8, color=TEXT2),
                marker=dict(size=gdf["total"].apply(lambda x: max(6, min(30, x/5e10))),
                            color=REGION_COLORS.get(reg, ACCENT), opacity=0.85,
                            line=dict(width=0.5, color=BG)),
            ))
        fig_scatter.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Población vs Gasto per cápita · tamaño = gasto absoluto",
            xaxis=dict(type="log", title="Población 2024 (log)"),
            yaxis_title="COP por habitante", height=480,
        )

        return html.Div([
            html.Div([
                stat_card("Gasto Nacional Total",
                          f"${total_gasto/1e12:.2f}B COP", "33 departamentos"),
                stat_card("Gasto pc promedio ponderado",
                          f"${gasto_pc_nac:,.0f}", "COP/habitante", color=GREEN),
                stat_card("Mayor inversión pc", max_dep, "", color=GREEN),
                stat_card("Menor inversión pc", min_dep, "", color=RED),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
                       "gap": "14px", "marginBottom": "28px"}),
            html.Div([
                html.Div(dcc.Graph(figure=make_donut(), config={"displayModeBar": False}),
                         style={"flex": "1"}),
                html.Div(dcc.Graph(figure=make_region_bar(), config={"displayModeBar": False}),
                         style={"flex": "1.3"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
            dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}),
        ])

    elif tab == "rank":
        return html.Div([
            section_title("Rankings Departamentales",
                          "Top 10 y Bottom 10 por sector · Gasto per cápita"),
            html.Div([
                html.Label("Sector:", style={
                    "color": TEXT2, "fontSize": "10px",
                    "fontFamily": "IBM Plex Mono", "marginBottom": "6px", "display": "block",
                }),
                dd_sector("rank-sector-dd", value="total_pc" if "total_pc" in df.columns else None),
            ], style={"marginBottom": "20px"}),
            html.Div(id="rank-output"),
        ])

    elif tab == "comp":
        return html.Div([
            section_title("Comparativo entre Departamentos",
                          "Barras apiladas · filtrable por sector y región"),
            html.Div([
                html.Div([
                    html.Label("Sectores:", style={
                        "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
                        "marginBottom": "6px", "display": "block",
                    }),
                    dcc.Dropdown(
                        id="comp-sectors-dd", multi=True,
                        options=[{"label": SECTOR_LABELS.get(s, s), "value": s} for s in PC_DISP],
                        value=PC_DISP[:3],
                        style={"background": CARD, "color": TEXT1,
                               "fontFamily": "IBM Plex Mono", "fontSize": "11px"},
                    ),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Filtrar región:", style={
                        "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
                        "marginBottom": "6px", "display": "block",
                    }),
                    dcc.Dropdown(
                        id="comp-region-dd", multi=True,
                        options=[{"label": r, "value": r}
                                 for r in sorted(df["region"].dropna().unique())],
                        placeholder="Todas las regiones",
                        style={"background": CARD, "color": TEXT1,
                               "fontFamily": "IBM Plex Mono", "fontSize": "11px"},
                    ),
                ], style={"flex": "0.6"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
            html.Div(id="comp-output"),
        ])

    elif tab == "region":
        # Radar por región
        available  = [c for c in PC_DISP if c != "total_pc"]
        reg_means  = df.groupby("region")[available].mean()
        categorias = [SECTOR_LABELS.get(s, s) for s in available]
        global_max = max(reg_means[c].max() for c in available)

        fig_radar = go.Figure()
        for reg, row in reg_means.iterrows():
            vals = [v / global_max * 100 for v in row.values]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categorias + [categorias[0]],
                name=reg, fill="toself", opacity=0.5,
                line=dict(color=REGION_COLORS.get(reg, ACCENT), width=1.5),
            ))
        fig_radar.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                                tickfont=dict(size=8, color=TEXT2), ticksuffix="%"),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=TEXT2)),
            ),
            title="Perfil de Gasto por Región (% sobre máximo global)",
            showlegend=True, height=480,
        )

        # Box plots por región
        fig_box = px.box(df, x="region", y="total_pc" if "total_pc" in df.columns else PC_DISP[0],
                          color="region", color_discrete_map=REGION_COLORS,
                          points="all", hover_name="departamento",
                          labels={"total_pc": "Gasto Total pc (COP)", "region": "Región"})
        fig_box.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False,
                               title="Distribución del Gasto Total pc por Región", height=400)

        return html.Div([
            section_title("Análisis Regional", "Perfil y distribución del gasto por región geográfica"),
            dcc.Graph(figure=fig_radar, config={"displayModeBar": False},
                      style={"marginBottom": "20px"}),
            dcc.Graph(figure=fig_box, config={"displayModeBar": False}),
        ])


# ── CALLBACK RANKING ──────────────────────────────────────────────────────────
@callback(Output("rank-output", "children"), Input("rank-sector-dd", "value"))
def update_rank(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    label   = SECTOR_LABELS.get(sector, sector)
    sorted_ = df.sort_values(sector)
    top10   = sorted_.tail(10)
    bot10   = sorted_.head(10)
    max_row = df.loc[df[sector].idxmax()]
    min_row = df.loc[df[sector].idxmin()]
    ratio   = max_row[sector] / min_row[sector]

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=(f"Top 10 · Mayor {label}",
                                         f"Bottom 10 · Menor {label}"))
    fig.add_trace(go.Bar(x=top10[sector], y=top10["departamento"].str.title(),
                          orientation="h", marker_color=GREEN, name="Top 10"), row=1, col=1)
    fig.add_trace(go.Bar(x=bot10[sector], y=bot10["departamento"].str.title(),
                          orientation="h", marker_color=RED, name="Bottom 10"), row=1, col=2)
    layout = PLOTLY_TEMPLATE["layout"].copy()
    layout.update(height=420, showlegend=False,
                   title=f"Rankings · {label} (COP/hab)")
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER)

    return html.Div([
        html.Div([
            stat_card("Máximo",        max_row["departamento"].title(),
                      f"{max_row[sector]:,.0f} COP/hab", color=GREEN),
            stat_card("Mínimo",        min_row["departamento"].title(),
                      f"{min_row[sector]:,.0f} COP/hab", color=RED),
            stat_card("Ratio Máx/Mín", f"{ratio:.1f}x",
                      "Brecha territorial", color=ORANGE),
            stat_card("Promedio nacional", f"{df[sector].mean():,.0f}",
                      "COP/hab"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
                   "gap": "12px", "marginBottom": "20px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


# ── CALLBACK COMPARATIVO ──────────────────────────────────────────────────────
@callback(Output("comp-output", "children"),
          Input("comp-sectors-dd", "value"),
          Input("comp-region-dd",  "value"))
def update_comp(sectors, regions):
    if not sectors:
        return html.P("Selecciona al menos un sector.",
                      style={"color": TEXT2, "fontFamily": "IBM Plex Mono"})
    fdf = df if not regions else df[df["region"].isin(regions)]
    fdf = fdf.sort_values("total_pc") if "total_pc" in fdf.columns else fdf
    colors = [ACCENT, GREEN, RED, ORANGE, P["purple"], "#4E9AF1", "#F1A94E"]
    fig = go.Figure()
    for i, s in enumerate(sectors):
        if s in fdf.columns:
            fig.add_trace(go.Bar(
                name=SECTOR_LABELS.get(s, s),
                x=fdf["departamento"].str.title(), y=fdf[s],
                marker_color=colors[i % len(colors)],
            ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        barmode="stack", height=500,
        title="Gasto per cápita por Departamento y Sector (COP/hab)",
        xaxis_tickangle=-45,
        xaxis_title="Departamento", yaxis_title="COP por habitante",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})