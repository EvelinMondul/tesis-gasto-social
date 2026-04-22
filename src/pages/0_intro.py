import dash
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, cargar_proporciones, calcular_clr,
    calcular_kmo_bartlett_prop,
    VARIABLES_INFO, REGION_COLORS, SECTORES_ABS,
    SECTOR_LABELS, SECTOR_COLORS, PALETTE
)

dash.register_page(__name__, path="/", name="Introducción", order=0)

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = cargar_datos()          # datos originales (solo para completitud)
df     = cargar_proporciones()   # datos con proporciones calculadas

X_clr, PROP_COLS, LABELS = calcular_clr(df)
KMO, CHI2, GL, PVAL_BART  = calcular_kmo_bartlett_prop(df)

P       = PALETTE
BG      = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER  = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT  = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED     = P["red"];    PURPLE  = P["purple"]

# ── KPIs composicionales ──────────────────────────────────────────────────────
n_deptos = len(df)
n_sector = len(PROP_COLS)

# Sector con mayor variabilidad (CV más alto)
cvs = {SECTOR_LABELS.get(s.replace("prop_",""), s): df[s].std()/df[s].mean()*100
       for s in PROP_COLS}
sector_max_cv = max(cvs, key=cvs.get)
cv_max        = cvs[sector_max_cv]

# Sector más homogéneo
sector_min_cv = min(cvs, key=cvs.get)
cv_min        = cvs[sector_min_cv]

# Proporción promedio de educación (sector dominante)
prop_educ_mean = df["prop_educacion"].mean() * 100

# BASE sin legend (evita conflicto)
BASE = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
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


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar_proporciones():
    """Radar de proporciones medias por región."""
    sectores = [c for c in PROP_COLS if c in df.columns]
    reg_means = df.groupby("region")[sectores].mean() * 100
    categorias = [SECTOR_LABELS.get(s.replace("prop_",""), s) for s in sectores]
    global_max = reg_means.values.max()

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
        title=dict(text="Perfil de Priorización del Gasto por Región (%)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                            tickfont=dict(size=8, color=TEXT2), ticksuffix="%"),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=TEXT2)),
        ),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1)),
        showlegend=True, height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def fig_barras_proporciones():
    """Barras apiladas de proporciones medias por región."""
    sectores = [c for c in PROP_COLS if c in df.columns]
    reg_means = df.groupby("region")[sectores].mean() * 100
    colors_list = list(SECTOR_COLORS.values())

    fig = go.Figure()
    for i, (s, label) in enumerate(zip(sectores, LABELS)):
        fig.add_trace(go.Bar(
            name=label,
            x=reg_means.index.tolist(),
            y=reg_means[s].tolist(),
            marker_color=colors_list[i],
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **BASE,
        barmode="stack", height=380,
        title=dict(text="Composición del Gasto por Región · Proporciones promedio (%)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="% del gasto total", tickfont=dict(color=TEXT1),
                   range=[0, 105]),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor=CARD,
                    bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT1)),
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def tabla_proporciones():
    """Tabla de estadísticos descriptivos de proporciones."""
    rows = []
    for s, label in zip(PROP_COLS, LABELS):
        v = df[s] * 100
        _, pval_sw = stats.shapiro(v)
        rows.append({
            "Sector":      label,
            "Media (%)":   f"{v.mean():.2f}",
            "Mediana (%)": f"{v.median():.2f}",
            "Mín (%)":     f"{v.min():.2f}",
            "Máx (%)":     f"{v.max():.2f}",
            "DE (%)":      f"{v.std():.2f}",
            "CV (%)":      f"{v.std()/v.mean()*100:.1f}",
            "Asimetría":   f"{stats.skew(v):.3f}",
            "SW p-valor":  f"{pval_sw:.3f}",
        })
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tdf.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "background": CARD, "color": TEXT1,
            "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "padding": "9px 13px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Sector"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600", "minWidth": "160px"},
            {"if": {"column_id": "CV (%)"}, "color": RED, "fontWeight": "600"},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Sans", "fontSize": "10px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native",
    )


def fig_completitud():
    """Heatmap de completitud del dataset original."""
    sectores_orig = [c for c in SECTORES_ABS if c in df_raw.columns]
    check = sectores_orig + (["poblacion"] if "poblacion" in df_raw.columns else [])
    miss  = df_raw[["departamento"] + check].set_index("departamento").isnull().astype(int)
    anns  = []
    for i in range(len(miss.index)):
        for j in range(len(miss.columns)):
            val = miss.iloc[i, j]
            anns.append(dict(
                x=j, y=i, text="✗" if val == 1 else "✓", showarrow=False,
                font=dict(color=RED if val == 1 else GREEN,
                          size=10, family="IBM Plex Mono"),
            ))
    labels_x = [SECTOR_LABELS.get(c, c) for c in miss.columns]
    fig = go.Figure(go.Heatmap(
        z=miss.values, x=labels_x,
        y=[d.title() for d in miss.index],
        colorscale=[[0,"rgba(63,185,80,0.25)"],[1,"rgba(247,129,102,0.25)"]],
        showscale=False,
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Completitud del Dataset · ✓ dato presente  ✗ missing",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        annotations=anns,
        height=max(440, len(miss) * 17),
        xaxis=dict(tickangle=-30, gridcolor=BORDER,
                   linecolor=BORDER, tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
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
                html.Span("TESIS DE MAESTRÍA · ESTADÍSTICA APLICADA · UNINORTE", style={
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
                    "El presente estudio analiza los PATRONES DE PRIORIZACIÓN del gasto "
                    "social departamental en Colombia mediante proporciones sectoriales "
                    "(prop_s = gasto_s / gasto_total). Este enfoque captura cómo cada "
                    "departamento distribuye sus recursos entre sectores, "
                    "independientemente del tamaño poblacional o del nivel total de "
                    "inversión. Se aplica análisis composicional de datos (CoDa) con "
                    "transformación CLR, PCA y clustering K-Means para identificar "
                    "perfiles territoriales de priorización.",
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

    # BLOQUE 2 — KPIs composicionales
    html.Div([
        kpi_card("Departamentos",          str(n_deptos),
                 "Unidades territoriales"),
        kpi_card("Sectores de gasto",      str(n_sector),
                 "Proporciones analizadas"),
        kpi_card("Sector más heterogéneo", sector_max_cv,
                 f"CV = {cv_max:.1f}% · mayor variabilidad en priorización",
                 color=RED),
        kpi_card("Sector más homogéneo",   sector_min_cv,
                 f"CV = {cv_min:.1f}% · priorización más uniforme entre dptos",
                 color=GREEN),
        kpi_card("Educación (promedio)",   f"{prop_educ_mean:.1f}%",
                 "Fracción media del gasto total · sector dominante via SGP",
                 color=ORANGE),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)",
               "gap": "14px", "marginBottom": "32px"}),

    # BLOQUE 3 — Ficha técnica
    html.Div([
        html.Div([
            section_title("Ficha Técnica", "Diseño de investigación"),
            ficha_row("Unidad de análisis",    "Departamentos de Colombia (n = 33)"),
            ficha_row("Período",               "Año fiscal 2024"),
            ficha_row("Fuente principal",      "TerriData – DNP"),
            ficha_row("Fuente complementaria", "DANE – Proyecciones de población 2024"),
            ficha_row("Variable de análisis",  "Proporciones sectoriales del gasto total"),
            ficha_row("Tratamiento CoDa",      "Transformación CLR (Aitchison 1986)"),
            ficha_row("Cero estructural",      "Bogotá: libre_destinacion = 0 → "
                                               "reemplazo multiplicativo δ=0.0001"),
            ficha_row("KMO (proporciones)",    f"{KMO:.4f} · Mediocre — factorizabilidad "
                                               "confirmada por Bartlett"),
            ficha_row("Bartlett",              f"χ²={CHI2:.2f} · gl={GL} · p={PVAL_BART:.2e}"),
            ficha_row("Técnicas",              "PCA composicional · K-Means · EDA"),
            ficha_row("Software",              "Python · Dash · scikit-learn · scipy"),
            ficha_row("Tipo de estudio",       "Descriptivo–Exploratorio · Corte transversal"),
        ], style={"flex": "1", "background": CARD, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "22px 24px"}),

        html.Div([
            section_title("Perfil de Priorización por Región",
                          "Proporciones promedio normalizadas · "
                          "Muestra diferencias en la estructura de asignación sectorial"),
            dcc.Graph(figure=fig_radar_proporciones(),
                      config={"displayModeBar": False}),
        ], style={"flex": "1.4", "background": CARD, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "22px 24px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

    # BLOQUE 4 — Composición por región
    html.Div([
        section_title("Composición del Gasto Social por Región",
                      "Proporciones medias por región · "
                      "Las diferencias en libre destinación e inversión son las más informativas"),
        dcc.Graph(figure=fig_barras_proporciones(),
                  config={"displayModeBar": False}),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "22px 24px", "marginBottom": "28px",
    }),

    # BLOQUE 5 — Nota metodológica CoDa
    html.Div([
        html.Span("FUNDAMENTO METODOLÓGICO · ANÁLISIS COMPOSICIONAL DE DATOS (CoDa)",
                  style={"color": ORANGE, "fontSize": "9px", "letterSpacing": "0.18em",
                         "fontFamily": "IBM Plex Mono", "fontWeight": "600"}),
        html.Div([
            html.P(
                "Las proporciones sectoriales satisfacen la restricción del simplex: "
                "Σ prop_s = 1 para cada departamento. Esta dependencia estructural "
                "invalida el uso directo de técnicas estadísticas estándar (PCA, "
                "correlación de Pearson) sobre proporciones brutas, ya que producen "
                "correlaciones espurias y eigenvalores sesgados.",
                style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                       "lineHeight": "1.7", "marginBottom": "8px"}
            ),
            html.P(
                "La transformación CLR — centred log-ratio (Aitchison 1986) — proyecta "
                "los datos del simplex al espacio euclidiano: CLR(xᵢ) = log(xᵢ) − "
                "(1/p)·Σlog(xⱼ). Esto elimina la dependencia composicional y permite "
                "aplicar PCA, correlación de Spearman y clustering sin sesgo. "
                "Una correlación negativa en CLR indica un TRADE-OFF real: "
                "priorizar un sector implica necesariamente reducir otro.",
                style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                       "lineHeight": "1.7", "margin": "0"}
            ),
        ], style={"borderLeft": f"3px solid {ORANGE}", "paddingLeft": "14px",
                   "marginTop": "10px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {ORANGE}40",
        "borderRadius": "8px", "padding": "20px 24px", "marginBottom": "28px",
    }),

    # BLOQUE 6 — Estadísticos de proporciones
    html.Div([
        section_title("Estadísticos Descriptivos · Proporciones del Gasto (%)",
                      "Unidad: % del gasto total · CV alto = alta heterogeneidad "
                      "en la priorización de ese sector entre departamentos"),
        tabla_proporciones(),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "24px 26px", "marginBottom": "28px",
    }),

    # BLOQUE 7 — Descripción de variables originales
    html.Div([
        section_title("Descripción de Variables Originales",
                      "Variables de gasto absoluto · Base para el cálculo de proporciones · "
                      "Fuente: TerriData DNP · DANE 2024"),
        html.Div([variable_card(v) for v in VARIABLES_INFO]),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "24px 26px", "marginBottom": "28px",
    }),

    # BLOQUE 8 — Calidad del dato
    html.Div([
        section_title("Revisión de Calidad del Dato",
                      "Completitud por variable y departamento · "
                      "Bogotá: libre_destinacion = 0 (régimen especial, no missing)"),
        dcc.Graph(figure=fig_completitud(), config={"displayModeBar": False}),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "24px 26px", "marginBottom": "28px",
    }),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")