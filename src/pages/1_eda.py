import dash
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, REGION_COLORS, SECTORES_PC, SECTORES_ABS,
    SECTOR_LABELS, PALETTE, PLOTLY_TEMPLATE
)

dash.register_page(__name__, path="/eda", name="EDA & Tablas", order=1)

# ── DATOS ─────────────────────────────────────────────────────────────────────
df = cargar_datos()
P       = PALETTE
BG      = P["bg"];   SURFACE = P["surface"]; CARD   = P["card"]
BORDER  = P["border"]; TEXT1 = P["text1"];   TEXT2  = P["text2"]
ACCENT  = P["accent"]; GREEN = P["green"];   ORANGE = P["orange"]; RED = P["red"]

ABS_DISPONIBLES = [c for c in SECTORES_ABS if c in df.columns]
PC_DISPONIBLES  = [c for c in SECTORES_PC  if c in df.columns]

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


def tabla_depto(cols_disponibles, fmt_fn=None):
    """Genera una DataTable con los departamentos y las columnas indicadas."""
    show = ["departamento", "region"] + cols_disponibles
    show = [c for c in show if c in df.columns]
    tdf  = df[show].copy()
    tdf["departamento"] = tdf["departamento"].str.title()
    tdf["region"]       = tdf["region"].fillna("—")
    for c in cols_disponibles:
        if c in tdf.columns:
            tdf[c] = tdf[c].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "—")
    col_labels = {"departamento": "Departamento", "region": "Región"}
    col_labels.update({c: SECTOR_LABELS.get(c, c) for c in cols_disponibles})
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": col_labels.get(c, c), "id": c} for c in show],
        style_table={"overflowX": "auto"},
        style_cell={
            "background": CARD, "color": TEXT1, "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "padding": "9px 13px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "departamento"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600", "minWidth": "150px"},
            {"if": {"column_id": "region"}, "textAlign": "left", "color": TEXT2},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}", "fontFamily": "IBM Plex Sans", "fontSize": "10px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native", filter_action="native", page_size=15,
    )


# ── FIGURAS ESTÁTICAS ─────────────────────────────────────────────────────────
def make_barras_apiladas():
    """Barras apiladas de gasto per cápita por departamento."""
    cols  = [c for c in PC_DISPONIBLES if c != "total_pc"]
    dfs   = df.sort_values("total_pc", ascending=True) if "total_pc" in df.columns else df
    deptos = dfs["departamento"].str.title()
    colors = [ACCENT, GREEN, RED, ORANGE, P["purple"], "#4E9AF1", "#F1A94E"]
    fig   = go.Figure()
    for i, c in enumerate(cols):
        if c in dfs.columns:
            fig.add_trace(go.Bar(
                name=SECTOR_LABELS.get(c, c),
                x=dfs[c], y=deptos, orientation="h",
                marker_color=colors[i % len(colors)],
            ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        barmode="stack", height=700,
        title="Composición del Gasto Social per cápita por Departamento (COP/hab)",
        xaxis_title="COP por habitante", yaxis_title="",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=160, r=20, t=80, b=50),
    )
    return fig


def make_scatter_pob_gasto():
    fig = go.Figure()
    for reg, gdf in df.groupby("region"):
        fig.add_trace(go.Scatter(
            x=gdf["poblacion"], y=gdf["total_pc"] if "total_pc" in gdf.columns else gdf[PC_DISPONIBLES[0]],
            mode="markers+text",
            name=reg,
            text=gdf["departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=8, color=TEXT2),
            marker=dict(size=9, color=REGION_COLORS.get(reg, ACCENT),
                        opacity=0.85, line=dict(width=0.5, color=BG)),
        ))
    fig.update_layout(
        **{k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k != "xaxis"},
        title="Población vs Gasto Total per cápita · coloreado por Región",
        xaxis=dict(type="log", title="Población 2024 (escala log)", gridcolor="#30363D", linecolor="#30363D"),
        yaxis_title="Gasto per cápita (COP/hab)",
        height=480,
    )
    return fig




def make_cv_barplot():
    """Barplot del coeficiente de variación por sector per cápita."""
    available = [c for c in PC_DISPONIBLES if c != "total_pc"]
    cvs   = [(SECTOR_LABELS.get(c, c), df[c].std() / df[c].mean() * 100) for c in available if c in df.columns]
    cvs   = sorted(cvs, key=lambda x: x[1], reverse=True)
    labels, values = zip(*cvs)

    colors_bar = ["#F78166" if v > 60 else "#D29922" if v > 40 else "#58A6FF" for v in values]

    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker_color=colors_bar,
        marker_line=dict(color=BG, width=0.5),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color=TEXT1),
    ))
    fig.add_hline(y=33, line_dash="dash", line_color=GREEN,
                  annotation_text="CV = 33% (referencia baja)",
                  annotation_font_color=GREEN, annotation_font_size=10)
    fig.add_hline(y=60, line_dash="dash", line_color=RED,
                  annotation_text="CV = 60% (referencia alta)",
                  annotation_font_color=RED, annotation_font_size=10)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="IBM Plex Mono", color=TEXT1, size=11),
        title=dict(text="Coeficiente de Variación por Sector · Desigualdad territorial",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-20),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="CV (%)", range=[0, max(values)*1.2]),
        margin=dict(l=50, r=20, t=60, b=80),
        height=420,
        showlegend=False,
    )
    return fig

# ── LAYOUT ────────────────────────────────────────────────────────────────────
layout = html.Div([

    # Tabs principales
    dcc.Tabs(id="eda-tabs", value="abs", children=[
        dcc.Tab(label="Datos Absolutos",       value="abs"),
        dcc.Tab(label="Datos per cápita",      value="pc"),
        dcc.Tab(label="Distribución · EDA",    value="dist"),
        dcc.Tab(label="Correlaciones",         value="corr"),
    ], style={"fontFamily": "IBM Plex Mono", "fontSize": "11px"},
       colors={"border": BORDER, "primary": ACCENT, "background": SURFACE}),

    html.Div(id="eda-tab-content",
             style={"padding": "28px 0", "background": BG}),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(Output("eda-tab-content", "children"), Input("eda-tabs", "value"))
def render_eda_tab(tab):

    # ── ABSOLUTOS ─────────────────────────────────────────────────────────────
    if tab == "abs":
        total_nac = df["total"].sum() if "total" in df.columns else 0
        return html.Div([
            section_title("Gasto Social Absoluto por Departamento",
                          "Cifras en pesos colombianos (COP) · Año 2024"),
            html.Div([
                stat_card("Gasto Nacional Total",
                          f"${total_nac/1e12:.2f} Billones COP",
                          "Suma 33 departamentos"),
                stat_card("Sector de mayor peso", "Educación",
                          "~59% del gasto total", color=GREEN),
                stat_card("Segundo sector", "Salud",
                          "~24% del gasto total", color=ORANGE),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)",
                       "gap": "14px", "marginBottom": "24px"}),
            tabla_depto(ABS_DISPONIBLES),
        ])

    # ── PER CÁPITA ────────────────────────────────────────────────────────────
    elif tab == "pc":
        return html.Div([
            section_title("Gasto Social per cápita por Departamento",
                          "COP por habitante · Año 2024"),

            # Selector de sector para los KPIs dinámicos
            html.Div([
                html.Label("Sector a analizar:", style={
                    "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
                    "marginBottom": "6px", "display": "block",
                }),
                dcc.Dropdown(
                    id="pc-sector-dd",
                    options=[{"label": SECTOR_LABELS.get(s, s), "value": s}
                             for s in PC_DISPONIBLES],
                    value=PC_DISPONIBLES[0] if PC_DISPONIBLES else None,
                    clearable=False,
                    style={"background": CARD, "color": TEXT1, "fontFamily": "IBM Plex Mono",
                           "fontSize": "11px", "width": "280px"},
                ),
            ], style={"marginBottom": "20px"}),

            html.Div(id="pc-kpis", style={"marginBottom": "24px"}),

            html.Div([
                html.Div([
                    section_title("Tabla · Gasto per cápita", "Ordenable y filtrable"),
                    tabla_depto(PC_DISPONIBLES),
                ], style={"flex": "1.2"}),
                html.Div([
                    section_title("Composición por Departamento", "Barras apiladas"),
                    dcc.Graph(figure=make_barras_apiladas(),
                              config={"displayModeBar": False}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "24px"}),

            dcc.Graph(figure=make_scatter_pob_gasto(),
                      config={"displayModeBar": False}),
        ])

    # ── DISTRIBUCIÓN ──────────────────────────────────────────────────────────
    elif tab == "dist":
        return html.Div([
            section_title("Distribución del Gasto per cápita",
                          "Histograma · KDE · Q-Q plot · Boxplot por región"),
            html.Div([
                html.Label("Seleccionar sector:", style={
                    "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
                    "marginBottom": "6px", "display": "block",
                }),
                dcc.Dropdown(
                    id="dist-sector-dd",
                    options=[{"label": SECTOR_LABELS.get(s, s), "value": s}
                             for s in PC_DISPONIBLES],
                    value=PC_DISPONIBLES[0] if PC_DISPONIBLES else None,
                    clearable=False,
                    style={"background": CARD, "color": TEXT1, "fontFamily": "IBM Plex Mono",
                           "fontSize": "11px", "width": "280px"},
                ),
            ], style={"marginBottom": "20px"}),
            html.Div(id="dist-output"),
        ])

    # ── CORRELACIONES ─────────────────────────────────────────────────────────
    elif tab == "corr":
        import numpy as np
        corr_cols = [c for c in PC_DISPONIBLES + ["total_pc", "poblacion"] if c in df.columns]
        labels    = [SECTOR_LABELS.get(c, c) for c in corr_cols]
        corr_mat  = df[corr_cols].rename(columns=dict(zip(corr_cols, labels))).corr(method="spearman")

        fig_heat = go.Figure(go.Heatmap(
            z=corr_mat.values, x=corr_mat.columns, y=corr_mat.columns,
            colorscale=[[0, "#F78166"], [0.5, "#1C2333"], [1, "#58A6FF"]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_mat.values, 2), texttemplate="%{text}",
            textfont=dict(size=9, family="IBM Plex Mono"),
            colorbar=dict(title="ρ", titlefont=dict(size=10)),
        ))
        fig_heat.update_layout(
            **{k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k != "xaxis"},
            title="Matriz de Correlación de Spearman · Gasto per cápita",
            height=520, xaxis=dict(tickangle=-30, gridcolor="#30363D", linecolor="#30363D"),
        )

        # Scatter matrix
        dims = [c for c in ["educacion_pc","salud_pc","agua_pc","total_pc"] if c in df.columns]
        dim_labels = {c: SECTOR_LABELS.get(c,c) for c in dims}
        fig_scat = px.scatter_matrix(
            df, dimensions=dims, color="region",
            color_discrete_map=REGION_COLORS,
            labels=dim_labels, hover_name="departamento",
        )
        fig_scat.update_traces(diagonal_visible=False,
                                marker=dict(size=4, opacity=0.8))
        fig_scat.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Scatter Matrix · Sectores Principales", height=480,
        )

        return html.Div([
            section_title("Correlación de Spearman",
                          "No paramétrica · detecta asociaciones monotónicas"),
            dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": BORDER, "margin": "28px 0"}),
            section_title("Scatter Matrix · Sectores Principales por Región"),
            dcc.Graph(figure=fig_scat, config={"displayModeBar": False}),
        ])


# ── CALLBACK KPIs per cápita ───────────────────────────────────────────────
@callback(Output("pc-kpis", "children"), Input("pc-sector-dd", "value"))
def update_pc_kpis(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    v   = df[sector].dropna()
    label = SECTOR_LABELS.get(sector, sector)
    max_r = df.loc[df[sector].idxmax()]
    min_r = df.loc[df[sector].idxmin()]
    return html.Div([
        stat_card(f"Máximo · {label}",
                  max_r["departamento"].title(),
                  f"{max_r[sector]:,.0f} COP/hab", color=GREEN),
        stat_card(f"Mínimo · {label}",
                  min_r["departamento"].title(),
                  f"{min_r[sector]:,.0f} COP/hab", color=RED),
        stat_card("Media nacional",
                  f"{v.mean():,.0f} COP/hab",
                  f"Mediana: {v.median():,.0f}", color=ACCENT),
        stat_card("Brecha territorial",
                  f"{max_r[sector]/min_r[sector]:.1f}x",
                  "Ratio máximo / mínimo", color=ORANGE),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)", "gap": "14px"})


# ── CALLBACK distribución ──────────────────────────────────────────────────
@callback(Output("dist-output", "children"), Input("dist-sector-dd", "value"))
def update_dist(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    label = SECTOR_LABELS.get(sector, sector)
    v     = df[sector].dropna()
    mean_, med_, std_ = v.mean(), v.median(), v.std()
    skew_ = stats.skew(v)
    kurt_ = stats.kurtosis(v)
    cv_   = std_ / mean_ * 100
    _, pval = stats.shapiro(v)
    norm_ok = pval > 0.05

    # Histograma + KDE
    import numpy as np
    kde_x = np.linspace(v.min(), v.max(), 200)
    kde   = stats.gaussian_kde(v)
    scale = len(v) * (v.max() - v.min()) / 12
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=v, nbinsx=12, name="Frecuencia",
                                  marker_color=ACCENT, opacity=0.7,
                                  marker_line=dict(color=BG, width=0.5)))
    fig_h.add_trace(go.Scatter(x=kde_x, y=kde(kde_x)*scale, mode="lines",
                                name="KDE", line=dict(color=ORANGE, width=2)))
    fig_h.add_vline(x=mean_, line_dash="dash", line_color=GREEN,
                    annotation_text=f"Media: {mean_:,.0f}",
                    annotation_font_color=GREEN)
    fig_h.add_vline(x=med_, line_dash="dot", line_color=RED,
                    annotation_text=f"Mediana: {med_:,.0f}",
                    annotation_font_color=RED)
    fig_h.update_layout(**PLOTLY_TEMPLATE["layout"],
                         title=f"Distribución · {label} (COP/hab)",
                         xaxis_title="COP/hab", yaxis_title="Frecuencia")

    # Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(v)
    line_y = np.array(osm) * slope + intercept
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                 marker=dict(color=ACCENT, size=7, opacity=0.85)))
    fig_qq.add_trace(go.Scatter(x=osm, y=line_y, mode="lines",
                                 line=dict(color=ORANGE, width=1.5, dash="dash")))
    fig_qq.update_layout(**PLOTLY_TEMPLATE["layout"],
                          title="Q-Q Plot · Normalidad",
                          xaxis_title="Cuantiles teóricos",
                          yaxis_title="Cuantiles muestrales")

    # Boxplot por región
    fig_box = px.box(df, x="region", y=sector, color="region",
                     color_discrete_map=REGION_COLORS, points="all",
                     hover_name="departamento",
                     labels={sector: f"{label} (COP/hab)", "region": "Región"})
    fig_box.update_layout(**PLOTLY_TEMPLATE["layout"],
                           showlegend=False,
                           title=f"Boxplot por Región · {label}")

    norm_color = GREEN if norm_ok else RED
    norm_text  = "✓ No se rechaza normalidad" if norm_ok else "✗ Se rechaza normalidad"

    return html.Div([
        html.Div([
            stat_card("Media",          f"{mean_:,.0f} COP",  f"CV = {cv_:.1f}%"),
            stat_card("Mediana",        f"{med_:,.0f} COP",   "Percentil 50"),
            stat_card("Asimetría",      f"{skew_:.3f}",       "Skewness", color=ORANGE),
            stat_card("Curtosis",       f"{kurt_:.3f}",       "Fisher (exceso)"),
            stat_card("Shapiro-Wilk",   norm_text,            f"p-value = {pval:.4f}",
                      color=norm_color),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)",
                   "gap": "12px", "marginBottom": "20px"}),
        html.Div([
            html.Div(dcc.Graph(figure=fig_h,  config={"displayModeBar": False}),
                     style={"flex": "1.5"}),
            html.Div(dcc.Graph(figure=fig_qq, config={"displayModeBar": False}),
                     style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
        dcc.Graph(figure=fig_box, config={"displayModeBar": False}),
    ])