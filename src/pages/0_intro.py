import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, VARIABLES_INFO, REGION_MAP, REGION_COLORS,
    SECTORES_PC, SECTOR_LABELS, PALETTE
)

dash.register_page(__name__, path="/", name="Introducción", order=0)

# ── DATOS ─────────────────────────────────────────────────────────────────────
df = cargar_datos()

P       = PALETTE
BG      = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER  = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT  = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED     = P["red"];    PURPLE  = P["purple"]

# KPIs
n_deptos     = len(df)
HAS_TOTAL_PC = "total_pc" in df.columns
max_dep  = df.loc[df["total_pc"].idxmax(), "departamento"].title() if HAS_TOTAL_PC else "—"
min_dep  = df.loc[df["total_pc"].idxmin(), "departamento"].title() if HAS_TOTAL_PC else "—"
cv_total = (df["total_pc"].std() / df["total_pc"].mean() * 100) if HAS_TOTAL_PC else 0

BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI  — todos antes del layout
# ═══════════════════════════════════════════════════════════════════════════════
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


def kpi_card(title, value, sub="", color=None):
    c = color or ACCENT
    return html.Div([
        html.P(title, style={
            "color": TEXT2, "fontSize": "9px", "letterSpacing": "0.12em",
            "textTransform": "uppercase", "fontFamily": "IBM Plex Mono",
            "marginBottom": "6px",
        }),
        html.P(value, style={
            "color": c, "fontSize": "19px", "fontWeight": "700",
            "fontFamily": "IBM Plex Sans", "margin": "0", "lineHeight": "1",
        }),
        html.P(sub, style={
            "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
            "marginTop": "5px",
        }) if sub else None,
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "8px",
        "padding": "16px 18px", "borderTop": f"3px solid {c}",
    })


def badge(text, color):
    return html.Div(text, style={
        "background": f"{color}22", "border": f"1px solid {color}",
        "color": color, "padding": "5px 14px", "borderRadius": "20px",
        "fontSize": "10px", "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        "letterSpacing": "0.08em", "textAlign": "center",
    })


def ficha_row(label, value):
    return html.Div([
        html.Span(label + ":", style={
            "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "minWidth": "200px", "display": "inline-block", "letterSpacing": "0.03em",
        }),
        html.Span(value, style={
            "color": TEXT1, "fontFamily": "IBM Plex Mono",
            "fontSize": "11px", "fontWeight": "500",
        }),
    ], style={"padding": "9px 0", "borderBottom": f"1px solid {BORDER}",
               "display": "flex", "alignItems": "flex-start"})


def variable_card(info):
    return html.Div([
        html.Div([
            html.Div([
                html.Span(info["etiqueta"], style={
                    "color": TEXT1, "fontFamily": "IBM Plex Sans",
                    "fontWeight": "600", "fontSize": "13px",
                }),
                html.Span(f"  ·  {info['tipo']}", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono",
                    "fontSize": "10px", "marginLeft": "8px",
                }),
            ]),
            html.Code(info["variable"], style={
                "color": ACCENT, "background": BG, "padding": "2px 8px",
                "borderRadius": "4px", "fontSize": "10px",
                "fontFamily": "IBM Plex Mono", "border": f"1px solid {BORDER}",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "10px"}),
        html.P(info["descripcion"], style={
            "color": TEXT2, "fontSize": "12px", "fontFamily": "IBM Plex Mono",
            "lineHeight": "1.7", "margin": "0 0 12px",
        }),
        html.Div([
            html.Div([
                html.Span("Unidad: ", style={"color": TEXT2, "fontSize": "10px",
                                              "fontFamily": "IBM Plex Mono"}),
                html.Span(info["unidad"], style={"color": GREEN, "fontSize": "10px",
                                                   "fontFamily": "IBM Plex Mono",
                                                   "fontWeight": "600"}),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Span("Fuente: ", style={"color": TEXT2, "fontSize": "10px",
                                              "fontFamily": "IBM Plex Mono"}),
                html.Span(info["fuente"], style={"color": ORANGE, "fontSize": "10px",
                                                   "fontFamily": "IBM Plex Mono",
                                                   "fontWeight": "600"}),
            ]),
        ], style={"display": "flex"}),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "8px",
        "padding": "18px 20px", "marginBottom": "10px",
        "borderLeft": f"3px solid {ACCENT}",
    })


def build_summary_table():
    available = [s for s in SECTORES_PC if s in df.columns]
    rows = []
    for s in available:
        v = df[s].dropna()
        rows.append({
            "Variable":   SECTOR_LABELS.get(s, s),
            "n":          str(len(v)),
            "Missing":    str(df[s].isnull().sum()),
            "Mín":        f"{v.min():,.0f}",
            "Q1":         f"{v.quantile(0.25):,.0f}",
            "Mediana":    f"{v.median():,.0f}",
            "Media":      f"{v.mean():,.0f}",
            "Q3":         f"{v.quantile(0.75):,.0f}",
            "Máx":        f"{v.max():,.0f}",
            "Desv. Est.": f"{v.std():,.0f}",
            "CV (%)":     f"{v.std()/v.mean()*100:.1f}",
            "Asimetría":  f"{stats.skew(v):.3f}",
        })
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tdf.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "background": CARD, "color": TEXT1, "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "padding": "9px 13px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Variable"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600", "minWidth": "140px"},
            {"if": {"column_id": "Missing"}, "color": RED},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Sans", "fontSize": "10px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════
def make_radar():
    available  = [s for s in SECTORES_PC if s in df.columns]
    reg_means  = df.groupby("region")[available].mean()
    categorias = [SECTOR_LABELS.get(s, s) for s in available]
    global_max = max(reg_means[s].max() for s in available)
    fig = go.Figure()
    for reg, row in reg_means.iterrows():
        vals_norm = [v / global_max * 100 for v in row.values]
        fig.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=categorias + [categorias[0]],
            name=reg, fill="toself", opacity=0.5,
            line=dict(color=REGION_COLORS.get(reg, ACCENT), width=1.5),
            marker=dict(color=REGION_COLORS.get(reg, ACCENT)),
        ))
    fig.update_layout(
        **BASE,
        title=dict(text="Perfil de Gasto per cápita por Región (normalizado al máximo global)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                            tickfont=dict(size=8, color=TEXT2), ticksuffix="%"),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=TEXT2)),
        ),
        showlegend=True, height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_completitud():
    check = [s for s in SECTORES_PC + ["total_pc", "poblacion"] if s in df.columns]
    miss  = df[["departamento"] + check].set_index("departamento").isnull().astype(int)
    anns  = []
    for i in range(len(miss.index)):
        for j in range(len(miss.columns)):
            val = miss.iloc[i, j]
            anns.append(dict(
                x=j, y=i, text="✗" if val == 1 else "✓", showarrow=False,
                font=dict(color=RED if val == 1 else GREEN,
                          size=10, family="IBM Plex Mono"),
            ))
    fig = go.Figure(go.Heatmap(
        z=miss.values,
        x=[SECTOR_LABELS.get(c, c) for c in miss.columns],
        y=[d.title() for d in miss.index],
        colorscale=[[0, "rgba(63,185,80,0.25)"], [1, "rgba(247,129,102,0.25)"]],
        showscale=False,
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Completitud del Dataset · ✓ dato presente  ✗ missing",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        annotations=anns,
        height=max(440, len(miss) * 17),
        xaxis=dict(tickangle=-30, gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        margin=dict(l=160, r=20, t=60, b=80),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
layout = html.Div([

    # BLOQUE 1 — Encabezado
    html.Div([
        html.Div([
            html.Div([
                html.Span("TESIS DE MAESTRÍA · ESTADÍSTICA APLICADA", style={
                    "color": ACCENT, "fontSize": "9px", "letterSpacing": "0.2em",
                    "fontFamily": "IBM Plex Mono", "fontWeight": "600",
                }),
                html.H2(
                    "Patrones de inversión en gasto social y su relación con "
                    "indicadores de bienestar en los departamentos de Colombia "
                    "mediante técnicas de análisis multivariado",
                    style={
                        "color": TEXT1, "fontFamily": "IBM Plex Sans",
                        "fontWeight": "700", "fontSize": "19px",
                        "lineHeight": "1.45", "margin": "10px 0 16px",
                        "maxWidth": "820px",
                    }
                ),
                html.P(
                    "En Colombia, el gasto público social constituye un instrumento "
                    "fundamental para mejorar las condiciones de vida de la población "
                    "mediante la inversión en sectores como salud, educación, agua potable, "
                    "cultura y deporte. El presente estudio analiza los PATRONES DE "
                    "PRIORIZACIÓN del gasto social en los departamentos a través de "
                    "proporciones sectoriales, aplicando análisis composicional de datos "
                    "(CoDa), PCA sobre transformación CLR y clustering K-Means para "
                    "identificar perfiles territoriales de inversión.",
                    style={
                        "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "12px",
                        "lineHeight": "1.8", "maxWidth": "780px", "margin": "0",
                    }
                ),
            ], style={"flex": "1"}),
            html.Div([
                badge("CoDa · CLR", ACCENT),
                badge("PCA",        PURPLE),
                badge("Clúster",    GREEN),
                badge("EDA",        ORANGE),
            ], style={"display": "flex", "flexDirection": "column", "gap": "8px",
                       "marginLeft": "40px", "flexShrink": "0"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderRadius": "10px", "padding": "28px 32px", "marginBottom": "28px",
        "borderLeft": f"4px solid {ACCENT}",
    }),

    # BLOQUE 2 — KPIs
    html.Div([
        kpi_card("Departamentos",        str(n_deptos),         "Unidades territoriales"),
        kpi_card("Variables analizadas", str(len(SECTORES_PC)), "Sectores de gasto"),
        kpi_card("Mayor gasto pc",       max_dep,               "Departamento", color=GREEN),
        kpi_card("Menor gasto pc",       min_dep,               "Departamento", color=RED),
        kpi_card("CV gasto total pc",    f"{cv_total:.1f}%",
                 "Heterogeneidad territorial", color=ORANGE),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)",
               "gap": "14px", "marginBottom": "32px"}),

    # BLOQUE 3 — Ficha técnica + Radar
    html.Div([
        html.Div([
            section_title("Ficha Técnica", "Diseño de investigación"),
            ficha_row("Unidad de análisis",     "Departamentos de Colombia (n = 33)"),
            ficha_row("Período",                "Año fiscal 2024"),
            ficha_row("Fuente principal",       "TerriData – DNP"),
            ficha_row("Fuente complementaria",  "DANE – Proyecciones de población 2024"),
            ficha_row("Variables de análisis",  "Proporciones del gasto total por sector"),
            ficha_row("Tratamiento CoDa",       "Transformación CLR (Aitchison 1986)"),
            ficha_row("Técnicas",               "PCA · Clúster K-Means · EDA composicional"),
            ficha_row("Software",               "Python · Dash · scikit-learn · scipy"),
            ficha_row("Tipo de estudio",        "Descriptivo–Exploratorio · Corte transversal"),
        ], style={"flex": "1", "background": CARD, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "22px 24px"}),
        html.Div([
            section_title("Perfil de Gasto per cápita por Región",
                          "Normalizado al máximo global · Contexto descriptivo"),
            dcc.Graph(figure=make_radar(), config={"displayModeBar": False}),
        ], style={"flex": "1.4", "background": CARD, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "22px 24px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "32px"}),

    # BLOQUE 4 — Nota metodológica composicional
    html.Div([
        html.Span("ENFOQUE METODOLÓGICO · ANÁLISIS COMPOSICIONAL DE DATOS", style={
            "color": ORANGE, "fontSize": "9px", "letterSpacing": "0.18em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.Div([
            html.P(
                "El análisis principal trabaja sobre las PROPORCIONES del gasto total "
                "destinado a cada sector (prop_s = gasto_s / gasto_total), no sobre "
                "valores absolutos ni per cápita. Este enfoque identifica patrones de "
                "PRIORIZACIÓN sectorial: cómo cada departamento distribuye sus recursos, "
                "independientemente del tamaño poblacional o del nivel total de inversión.",
                style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                       "lineHeight": "1.7", "marginBottom": "8px"}
            ),
            html.P(
                "Dado que las proporciones suman 1 (restricción del simplex), se aplica "
                "la transformación CLR (centred log-ratio) antes del PCA para eliminar "
                "la dependencia composicional y obtener resultados estadísticamente "
                "no sesgados (Aitchison 1986). Los ceros estructurales (Bogotá: "
                "libre_destinacion = 0) se tratan con reemplazo multiplicativo "
                "(Martín-Fernández et al. 2003).",
                style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                       "lineHeight": "1.7", "margin": "0"}
            ),
        ], style={"borderLeft": f"3px solid {ORANGE}", "paddingLeft": "14px",
                   "marginTop": "10px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {ORANGE}40",
        "borderRadius": "8px", "padding": "20px 24px", "marginBottom": "28px",
    }),

    # BLOQUE 5 — Descripción de variables
    html.Div([
        section_title("Descripción de Variables",
                      "Definición operacional, unidad de medida y fuente · "
                      "Variables per cápita disponibles para contexto descriptivo"),
        html.Div([variable_card(v) for v in VARIABLES_INFO]),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "24px 26px", "marginBottom": "28px",
    }),

    # BLOQUE 6 — Calidad del dato
    html.Div([
        section_title("Revisión de Calidad del Dato",
                      "Completitud por variable y departamento"),
        dcc.Graph(figure=make_completitud(), config={"displayModeBar": False},
                   style={"marginBottom": "28px"}),
        section_title("Estadísticos de Resumen · Variables per cápita",
                      "Contexto descriptivo · El análisis principal usa proporciones"),
        build_summary_table(),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "24px 26px", "marginBottom": "28px",
    }),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")