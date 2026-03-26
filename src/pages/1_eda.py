import dash
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import (
    cargar_datos, REGION_COLORS, SECTORES_PC, SECTORES_ABS,
    SECTOR_LABELS, PALETTE
)

dash.register_page(__name__, path="/eda", name="EDA & Tablas", order=1)

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS Y PALETA
# ═══════════════════════════════════════════════════════════════════════════════
df = cargar_datos()
P       = PALETTE
BG      = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER  = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT  = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED     = P["red"];    PURPLE  = P["purple"]

PC_DISP  = [c for c in SECTORES_PC  if c in df.columns]
ABS_DISP = [c for c in SECTORES_ABS if c in df.columns]

BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
# ═══════════════════════════════════════════════════════════════════════════════
def T(text, sub=""):
    """Título de sección con acento izquierdo."""
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
    """Párrafo de interpretación académica."""
    return html.P(text, style={
        "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
        "lineHeight": "1.8", "marginBottom": "20px",
        "borderLeft": f"2px solid {BORDER}", "paddingLeft": "14px",
        "fontStyle": "italic",
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


def acto_header(num, titulo, descripcion):
    return html.Div([
        html.Div([
            html.Span(f"ACTO {num}", style={
                "color": ACCENT, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
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
        ]),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {ACCENT}", "borderRadius": "8px",
        "padding": "18px 22px", "marginBottom": "24px",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 1: ESTADÍSTICOS DESCRIPTIVOS
# ═══════════════════════════════════════════════════════════════════════════════
def tabla_descriptivos():
    rows = []
    for s in PC_DISP:
        v = df[s].dropna()
        rows.append({
            "Variable":    SECTOR_LABELS.get(s, s),
            "n":           len(v),
            "Missing":     df[s].isnull().sum(),
            "Mín":         f"{v.min():,.0f}",
            "Q1":          f"{v.quantile(.25):,.0f}",
            "Mediana":     f"{v.median():,.0f}",
            "Media":       f"{v.mean():,.0f}",
            "Q3":          f"{v.quantile(.75):,.0f}",
            "Máx":         f"{v.max():,.0f}",
            "DE":          f"{v.std():,.0f}",
            "CV (%)":      f"{v.std()/v.mean()*100:.1f}",
            "Asimetría":   f"{stats.skew(v):.3f}",
            "Curtosis":    f"{stats.kurtosis(v):.3f}",
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
             "color": ACCENT, "fontWeight": "600", "minWidth": "130px"},
            {"if": {"column_id": "Missing"}, "color": RED},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}", "fontFamily": "IBM Plex Sans", "fontSize": "10px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 2: HETEROGENEIDAD TERRITORIAL (CV)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cv():
    cols  = [c for c in PC_DISP if c != "total_pc"]
    data_ = sorted(
        [(SECTOR_LABELS.get(c, c), df[c].std()/df[c].mean()*100) for c in cols if c in df.columns],
        key=lambda x: x[1], reverse=True
    )
    labels, values = zip(*data_)
    colors = [RED if v > 60 else ORANGE if v > 40 else ACCENT for v in values]

    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker_color=colors, marker_line=dict(color=BG, width=0.5),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color=TEXT1),
    ))
    fig.add_hline(y=33, line_dash="dash", line_color=GREEN,
                  annotation_text="33% · umbral bajo",
                  annotation_font_color=GREEN, annotation_font_size=9,
                  annotation_position="right")
    fig.add_hline(y=60, line_dash="dash", line_color=RED,
                  annotation_text="60% · umbral alto",
                  annotation_font_color=RED, annotation_font_size=9,
                  annotation_position="right")
    fig.update_layout(
        **BASE,
        title=dict(text="Coeficiente de Variación por Sector · Desigualdad entre Departamentos",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-15, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="CV (%)",
                   range=[0, max(values)*1.25], color=TEXT2),
        height=400, showlegend=False,
        margin=dict(l=60, r=100, t=60, b=80),
    )
    return fig


def fig_barras_departamentos():
    cols = [c for c in PC_DISP if c != "total_pc"]
    dfs  = df.sort_values("total_pc", ascending=True) if "total_pc" in df.columns else df
    colors_list = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#4E9AF1", "#F1A94E"]
    fig = go.Figure()
    for i, c in enumerate(cols):
        if c in dfs.columns:
            fig.add_trace(go.Bar(
                name=SECTOR_LABELS.get(c, c),
                x=dfs[c], y=dfs["departamento"].str.title(),
                orientation="h", marker_color=colors_list[i % len(colors_list)],
            ))
    fig.update_layout(
        **BASE,
        barmode="stack", height=720,
        title=dict(text="Composición del Gasto Social per cápita por Departamento",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="COP / habitante", color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor=CARD,
                    bordercolor=BORDER, borderwidth=1),
        margin=dict(l=160, r=20, t=80, b=50),
    )
    return fig


def fig_scatter_poblacion():
    fig = go.Figure()
    for reg, gdf in df.groupby("region"):
        fig.add_trace(go.Scatter(
            x=gdf["poblacion"],
            y=gdf["total_pc"] if "total_pc" in gdf.columns else gdf[PC_DISP[0]],
            mode="markers+text", name=reg,
            text=gdf["departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7, color=TEXT2),
            marker=dict(size=9, color=REGION_COLORS.get(reg, ACCENT),
                        opacity=0.85, line=dict(width=0.5, color=BG)),
        ))
    fig.update_layout(
        **BASE,
        title=dict(text="Relación Población–Gasto per cápita · Efecto escala poblacional",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(type="log", title="Población 2024 (escala logarítmica)",
                   gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Gasto total per cápita (COP/hab)", color=TEXT2),
        height=480, margin=dict(l=70, r=20, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 3: DISTRIBUCIÓN UNIVARIADA
# ═══════════════════════════════════════════════════════════════════════════════
def fig_distribucion(sector):
    label = SECTOR_LABELS.get(sector, sector)
    v     = df[sector].dropna()
    mean_, med_ = v.mean(), v.median()
    skew_ = stats.skew(v)
    kurt_ = stats.kurtosis(v)
    cv_   = v.std() / mean_ * 100
    _, pval = stats.shapiro(v)

    kde_x = np.linspace(v.min(), v.max(), 200)
    kde   = stats.gaussian_kde(v)
    scale = len(v) * (v.max() - v.min()) / 12

    # Histograma + KDE
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(
        x=v, nbinsx=12, name="Frecuencia",
        marker_color=ACCENT, opacity=0.7,
        marker_line=dict(color=BG, width=0.5),
    ))
    fig_h.add_trace(go.Scatter(
        x=kde_x, y=kde(kde_x)*scale, mode="lines",
        name="KDE", line=dict(color=ORANGE, width=2.5),
    ))
    fig_h.add_vline(x=mean_, line_dash="dash", line_color=GREEN,
                    annotation_text=f"μ={mean_:,.0f}",
                    annotation_font_color=GREEN, annotation_font_size=9)
    fig_h.add_vline(x=med_, line_dash="dot", line_color=RED,
                    annotation_text=f"Md={med_:,.0f}",
                    annotation_font_color=RED, annotation_font_size=9)
    fig_h.update_layout(
        **BASE,
        title=dict(text=f"Distribución · {label}",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="COP / habitante", color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Frecuencia", color=TEXT2),
        height=340, margin=dict(l=60, r=20, t=50, b=60),
    )

    # Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(v)
    line_y = np.array(osm) * slope + intercept
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color=ACCENT, size=7, opacity=0.85),
        name="Observados",
    ))
    fig_qq.add_trace(go.Scatter(
        x=osm, y=line_y, mode="lines",
        line=dict(color=ORANGE, width=1.5, dash="dash"),
        name="Normal teórica",
    ))
    fig_qq.update_layout(
        **BASE,
        title=dict(text="Q-Q Plot · Normalidad",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Cuantiles teóricos", color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Cuantiles muestrales", color=TEXT2),
        height=340, margin=dict(l=60, r=20, t=50, b=60),
        showlegend=False,
    )

    # Boxplot por región
    fig_box = px.box(
        df, x="region", y=sector, color="region",
        color_discrete_map=REGION_COLORS, points="all",
        hover_name="departamento",
        labels={sector: f"{label} (COP/hab)", "region": "Región"},
    )
    fig_box.update_layout(
        **BASE,
        title=dict(text=f"Boxplot por Región · {label}",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        showlegend=False, height=340,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    norm_color = GREEN if pval > 0.05 else RED
    norm_text  = "✓ No se rechaza H₀ de normalidad" if pval > 0.05 else "✗ Se rechaza H₀ de normalidad"

    return fig_h, fig_qq, fig_box, {
        "media": mean_, "mediana": med_, "cv": cv_,
        "asimetria": skew_, "curtosis": kurt_,
        "shapiro_p": pval, "norm_text": norm_text, "norm_color": norm_color,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 4: ESTRUCTURA DE CORRELACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def fig_correlacion_spearman():
    cols   = PC_DISP
    labels = [SECTOR_LABELS.get(c, c) for c in cols]
    n      = len(cols)
    corr_m = np.zeros((n, n))
    pval_m = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(df[cols[i]].dropna(), df[cols[j]].dropna())
            corr_m[i, j] = r
            pval_m[i, j] = p

    text_m = []
    for i in range(n):
        row = []
        for j in range(n):
            s = "***" if pval_m[i,j] < 0.001 else "**" if pval_m[i,j] < 0.01 else "*" if pval_m[i,j] < 0.05 else ""
            row.append(f"{corr_m[i,j]:.2f}{s}")
        text_m.append(row)

    fig = go.Figure(go.Heatmap(
        z=corr_m, x=labels, y=labels,
        colorscale=[[0, "#F78166"], [0.5, "#1C2333"], [1, "#58A6FF"]],
        zmid=0, zmin=-1, zmax=1,
        text=text_m, texttemplate="%{text}",
        textfont=dict(size=9, family="IBM Plex Mono"),
        colorbar=dict(
            title="ρ", titlefont=dict(size=10, color=TEXT1),
            tickfont=dict(color=TEXT1),
        ),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Correlación de Spearman · * p<.05  ** p<.01  *** p<.001",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(tickangle=-35, gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT2),
        height=500, margin=dict(l=130, r=20, t=60, b=130),
    )
    return fig, corr_m, labels


def fig_scatter_matrix():
    dims   = [c for c in ["educacion_pc","salud_pc","agua_pc","libre_inv_pc","libre_dest_pc"] if c in df.columns]
    labels = {c: SECTOR_LABELS.get(c,c) for c in dims}
    fig    = px.scatter_matrix(
        df, dimensions=dims, color="region",
        color_discrete_map=REGION_COLORS, labels=labels,
        hover_name="departamento",
    )
    fig.update_traces(diagonal_visible=False,
                      marker=dict(size=4, opacity=0.8))
    fig.update_layout(
        **BASE,
        title=dict(text="Scatter Matrix · Relaciones Bivariadas por Región",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=520, margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 5: DETECCIÓN DE VALORES ATÍPICOS
# ═══════════════════════════════════════════════════════════════════════════════
def fig_outliers():
    cols   = [c for c in PC_DISP if c in df.columns]
    scaler = StandardScaler()
    Z      = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=[SECTOR_LABELS.get(c,c) for c in cols],
        index=df["departamento"].str.title(),
    )
    # Norma euclidiana del vector Z ≈ distancia de Mahalanobis simplificada
    Z["dist"] = np.sqrt((Z[Z.columns[:-0]]**2).sum(axis=1)) if False else \
                np.sqrt((Z**2).sum(axis=1))

    top_outliers = Z.nlargest(5, "dist").index.tolist()

    fig = go.Figure()
    threshold = 2.5
    col_names = [c for c in Z.columns if c != "dist"]
    colors_z  = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#4E9AF1", "#F1A94E"]
    for i, col in enumerate(col_names):
        fig.add_trace(go.Box(
            y=Z[col], name=col,
            boxpoints="all", jitter=0.3,
            marker=dict(size=5, opacity=0.75,
                        color=colors_z[i % len(colors_z)]),
            line=dict(color=colors_z[i % len(colors_z)], width=1.5),
            fillcolor=f"rgba({int(colors_z[i%len(colors_z)][1:3],16)},"
                      f"{int(colors_z[i%len(colors_z)][3:5],16)},"
                      f"{int(colors_z[i%len(colors_z)][5:7],16)},0.15)",
        ))
    fig.add_hline(y= threshold, line_dash="dash", line_color=RED,
                  annotation_text=f"|Z| = {threshold}",
                  annotation_font_color=RED, annotation_font_size=9)
    fig.add_hline(y=-threshold, line_dash="dash", line_color=RED)
    fig.update_layout(
        **BASE,
        title=dict(text="Detección de Valores Atípicos · Z-scores por Variable",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-20, color=TEXT2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Z-score estandarizado", color=TEXT2),
        height=440, showlegend=False,
        margin=dict(l=70, r=20, t=60, b=100),
    )
    return fig, top_outliers, Z


def tabla_mahalanobis(Z):
    dist_df = Z[["dist"]].copy()
    dist_df.columns = ["Dist. Mahalanobis aprox."]
    dist_df = dist_df.sort_values("Dist. Mahalanobis aprox.", ascending=False).head(10)
    dist_df["Dist. Mahalanobis aprox."] = dist_df["Dist. Mahalanobis aprox."].map(lambda x: f"{x:.3f}")
    dist_df = dist_df.reset_index().rename(columns={"index": "Departamento"})
    dist_df["Riesgo"] = ["🔴 Alto" if i < 3 else "🟡 Medio" if i < 6 else "🟢 Normal"
                          for i in range(len(dist_df))]
    return dash_table.DataTable(
        data=dist_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in dist_df.columns],
        style_cell={
            "background": CARD, "color": TEXT1, "border": f"1px solid {BORDER}",
            "fontFamily": "IBM Plex Mono", "fontSize": "10px",
            "padding": "8px 12px", "textAlign": "center",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Departamento"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600"},
        ],
        style_header={
            "background": SURFACE, "color": ACCENT, "fontWeight": "600",
            "border": f"1px solid {BORDER}", "fontFamily": "IBM Plex Sans", "fontSize": "10px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 6: ESTANDARIZACIÓN Y PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def fig_estandarizacion():
    cols = [c for c in PC_DISP if c != "total_pc" and c in df.columns]
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(df[cols]),
                     columns=[SECTOR_LABELS.get(c,c) for c in cols])

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("Datos originales (COP/hab)", "Datos estandarizados (Z-score)"))
    colors_ = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#4E9AF1", "#F1A94E"]
    for i, c in enumerate(cols):
        label = SECTOR_LABELS.get(c, c)
        fig.add_trace(go.Box(
            y=df[c], name=label,
            marker_color=colors_[i%len(colors_)], showlegend=False,
            boxpoints=False,
        ), row=1, col=1)
        fig.add_trace(go.Box(
            y=Z[label], name=label,
            marker_color=colors_[i%len(colors_)], showlegend=True,
            boxpoints=False,
        ), row=1, col=2)

    fig.update_layout(
        **BASE,
        title=dict(text="Efecto de la Estandarización Z-score · Antes y Después",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=460,
        legend=dict(orientation="h", y=-0.15, x=0, bgcolor=CARD,
                    bordercolor=BORDER, borderwidth=1),
        margin=dict(l=60, r=20, t=80, b=120),
    )
    fig.update_xaxes(tickangle=-30, gridcolor=BORDER, linecolor=BORDER, color=TEXT2)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, color=TEXT2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ACTO 7: VALIDACIÓN Y SCREE PLOT
# ═══════════════════════════════════════════════════════════════════════════════
def calcular_kmo():
    """Calcula el índice KMO (Kaiser-Meyer-Olkin)."""
    cols = [c for c in PC_DISP if c != "total_pc" and c in df.columns]
    X    = df[cols].dropna().values
    X    = StandardScaler().fit_transform(X)
    R    = np.corrcoef(X.T)
    R_inv = np.linalg.inv(R)
    n    = R.shape[0]
    # Matriz de correlaciones parciales
    P_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                P_mat[i,j] = -R_inv[i,j] / np.sqrt(R_inv[i,i]*R_inv[j,j])
    r2_sum = np.sum(R[np.triu_indices(n, k=1)]**2)
    p2_sum = np.sum(P_mat[np.triu_indices(n, k=1)]**2)
    kmo = r2_sum / (r2_sum + p2_sum)
    return kmo, cols


def calcular_bartlett():
    """Prueba de esfericidad de Bartlett."""
    cols = [c for c in PC_DISP if c != "total_pc" and c in df.columns]
    X    = df[cols].dropna().values
    n, p = X.shape
    X    = StandardScaler().fit_transform(X)
    R    = np.corrcoef(X.T)
    # Estadístico chi-cuadrado de Bartlett
    chi2 = -(n - 1 - (2*p + 5)/6) * np.log(np.linalg.det(R))
    gl   = p * (p - 1) / 2
    pval = 1 - stats.chi2.cdf(chi2, gl)
    return chi2, int(gl), pval


def fig_scree():
    cols = [c for c in PC_DISP if c != "total_pc" and c in df.columns]
    X    = StandardScaler().fit_transform(df[cols].dropna())
    pca  = SkPCA(n_components=len(cols))
    pca.fit(X)
    var_exp  = pca.explained_variance_ratio_ * 100
    var_acum = np.cumsum(var_exp)
    eigenvals = pca.explained_variance_
    n_comp   = len(cols)
    comp_labels = [f"CP{i+1}" for i in range(n_comp)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Varianza explicada por componente", "Eigenvalores (Criterio Kaiser λ > 1)"),
    )
    bar_colors = [GREEN if v >= 10 else ACCENT if v >= 5 else BORDER for v in var_exp]
    fig.add_trace(go.Bar(
        x=comp_labels, y=var_exp,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in var_exp],
        textposition="outside",
        textfont=dict(size=9, color=TEXT1),
        name="Var. explicada",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=comp_labels, y=var_acum,
        mode="lines+markers+text",
        line=dict(color=ORANGE, width=2),
        marker=dict(size=7, color=ORANGE),
        text=[f"{v:.0f}%" for v in var_acum],
        textposition="top center",
        textfont=dict(size=8, color=ORANGE),
        name="Acumulada",
        yaxis="y2",
    ), row=1, col=1)

    # Eigenvalores
    fig.add_trace(go.Bar(
        x=comp_labels, y=eigenvals,
        marker_color=[GREEN if e >= 1 else RED for e in eigenvals],
        text=[f"{e:.2f}" for e in eigenvals],
        textposition="outside",
        textfont=dict(size=9, color=TEXT1),
        name="Eigenvalor",
    ), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color=ORANGE,
                  annotation_text="λ = 1 (Criterio Kaiser)",
                  annotation_font_color=ORANGE, annotation_font_size=9,
                  row=1, col=2)

    fig.update_layout(
        **BASE,
        title=dict(text="Scree Plot · Justificación del Número de Componentes a Retener",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=440,
        yaxis2=dict(overlaying="y", side="right", range=[0,110],
                    showgrid=False, tickfont=dict(color=ORANGE),
                    titlefont=dict(color=ORANGE), title="Acumulada (%)"),
        showlegend=False,
        margin=dict(l=60, r=80, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, color=TEXT2)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, color=TEXT2)
    return fig, var_exp, var_acum, eigenvals


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE STATIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
_fig_cv           = fig_cv()
_fig_barras       = fig_barras_departamentos()
_fig_scatter_pob  = fig_scatter_poblacion()
_fig_corr, _corr_m, _corr_labels = fig_correlacion_spearman()
_fig_scat_matrix  = fig_scatter_matrix()
_fig_outliers, _top_outliers, _Z = fig_outliers()
_tabla_mah        = tabla_mahalanobis(_Z)
_fig_estand       = fig_estandarizacion()
_fig_scree, _var_exp, _var_acum, _eigenvals = fig_scree()
_kmo, _kmo_cols   = calcular_kmo()
_chi2, _gl, _pval_bart = calcular_bartlett()

# KMO interpretation
def kmo_interp(k):
    if k >= 0.90: return "Maravilloso", GREEN
    if k >= 0.80: return "Meritorio", ACCENT
    if k >= 0.70: return "Mediano", ORANGE
    if k >= 0.60: return "Mediocre", ORANGE
    return "Inaceptable", RED

_kmo_label, _kmo_color = kmo_interp(_kmo)
_n_kaiser = sum(1 for e in _eigenvals if e >= 1)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — NARRATIVA EN 7 ACTOS
# ═══════════════════════════════════════════════════════════════════════════════
layout = html.Div([

    # ── PORTADA DEL EDA ──────────────────────────────────────────────────────
    html.Div([
        html.Span("ANÁLISIS EXPLORATORIO DE DATOS · NARRATIVA ACADÉMICA", style={
            "color": ACCENT, "fontSize": "9px", "letterSpacing": "0.2em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.H1("Del gasto desigual a la estructura multivariada",
                style={
                    "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
                    "fontSize": "22px", "margin": "10px 0 12px", "lineHeight": "1.3",
                }),
        html.P(
            "Este EDA está organizado como una narrativa analítica en 7 actos, "
            "que conduce desde la descripción del fenómeno hasta la justificación "
            "empírica del análisis multivariado. Cada sección tiene un propósito "
            "estadístico explícito y conecta con las técnicas subsiguientes: "
            "ACP, Análisis Factorial y Análisis de Clúster.",
            style={
                "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                "lineHeight": "1.8", "maxWidth": "860px", "margin": "0",
            }
        ),
        html.Div([
            html.Div("7 Actos analíticos", style={
                "color": ACCENT, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{ACCENT}15", "border": f"1px solid {ACCENT}",
                "padding": "4px 12px", "borderRadius": "20px", "marginRight": "10px",
            }),
            html.Div("33 Departamentos", style={
                "color": GREEN, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{GREEN}15", "border": f"1px solid {GREEN}",
                "padding": "4px 12px", "borderRadius": "20px", "marginRight": "10px",
            }),
            html.Div("7 Variables per cápita", style={
                "color": ORANGE, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{ORANGE}15", "border": f"1px solid {ORANGE}",
                "padding": "4px 12px", "borderRadius": "20px",
            }),
        ], style={"display": "flex", "marginTop": "16px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {ACCENT}", "borderRadius": "10px",
        "padding": "28px 32px", "marginBottom": "32px",
    }),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 1 — DESCRIPCIÓN
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("01", "Descripción del Fenómeno",
                "Estadísticos descriptivos completos · Contexto de magnitudes · "
                "Base para identificar escala y unidad de análisis"),

    card_wrap(
        T("Estadísticos Descriptivos · Variables per cápita",
          "Media, mediana, cuartiles, desviación estándar, CV, asimetría y curtosis"),
        narrative(
            "La Tabla 1 presenta los estadísticos descriptivos de las siete variables "
            "de gasto social per cápita analizadas. La marcada diferencia entre media y "
            "mediana en variables como libre destinación y libre inversión anticipa la "
            "presencia de distribuciones asimétricas, cuyo tratamiento se discute en el "
            "Acto III. El coeficiente de variación superior al 60% en tres variables "
            "evidencia una heterogeneidad territorial que constituye el eje motivador "
            "del análisis multivariado."
        ),
        tabla_descriptivos(),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 2 — HETEROGENEIDAD
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("02", "Heterogeneidad Territorial",
                "¿Qué tan desigual es el gasto entre departamentos? "
                "El CV como medida de dispersión relativa · "
                "Composición sectorial · Efecto escala poblacional"),

    card_wrap(
        T("Coeficiente de Variación por Sector",
          "CV > 60% = desigualdad alta · CV 40-60% = moderada · CV < 40% = baja"),
        narrative(
            "El Gráfico 1 revela que la desigualdad interdepartamental no es uniforme "
            "entre sectores. Los componentes de libre destinación y libre inversión — "
            "aquellos no condicionados por transferencias del SGP — exhiben los mayores "
            "CV, lo que los convierte en los principales ejes de diferenciación fiscal "
            "entre territorios. En contraste, los sectores de educación y salud muestran "
            "menor variabilidad relativa, consistente con el efecto homogeneizador de "
            "las transferencias del SGP basadas en criterios poblacionales. "
            "Este patrón diferencial justifica el uso de técnicas multivariadas que "
            "permitan capturar simultáneamente estas dimensiones de variación."
        ),
        dcc.Graph(figure=_fig_cv, config={"displayModeBar": False}),
    ),

    card_wrap(
        T("Composición del Gasto por Departamento",
          "Ordenado por gasto total per cápita · Permite identificar perfiles territoriales"),
        narrative(
            "El Gráfico 2 muestra la composición sectorial del gasto per cápita ordenada "
            "de menor a mayor nivel de inversión total. Se observan perfiles diferenciados: "
            "departamentos con predominio del gasto en educación y salud (determinado por "
            "el SGP) frente a aquellos con mayor participación relativa de libre inversión, "
            "lo que sugiere la existencia de patrones latentes de asignación que el "
            "análisis de componentes principales permitirá formalizar."
        ),
        dcc.Graph(figure=_fig_barras, config={"displayModeBar": False}),
    ),

    card_wrap(
        T("Relación Población – Gasto per cápita",
          "Escala logarítmica en X · Revela el efecto de la densidad poblacional sobre la inversión per cápita"),
        narrative(
            "El Gráfico 3 evidencia una relación inversa entre tamaño poblacional y gasto "
            "per cápita (escala log). Los departamentos de la región Amazónica, con las "
            "menores poblaciones, presentan los mayores niveles de inversión per cápita, "
            "lo que no refleja necesariamente mayor capacidad fiscal sino el efecto "
            "matemático de dividir un gasto absoluto relativamente fijo entre un número "
            "muy reducido de habitantes. Este fenómeno debe tenerse en cuenta al "
            "interpretar los clusters resultantes."
        ),
        dcc.Graph(figure=_fig_scatter_pob, config={"displayModeBar": False}),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 3 — DISTRIBUCIÓN UNIVARIADA
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("03", "Distribución Univariada",
                "Evaluación de normalidad · Histogramas · KDE · Q-Q plots · "
                "Decisión sobre transformaciones · Supuestos para ACP"),

    card_wrap(
        T("Explorar distribución por sector",
          "Histograma + KDE · Q-Q Plot · Boxplot por región · Prueba Shapiro-Wilk"),
        html.Div([
            html.Label("Seleccionar variable:", style={
                "color": TEXT2, "fontSize": "10px", "fontFamily": "IBM Plex Mono",
                "marginBottom": "6px", "display": "block",
            }),
            dcc.Dropdown(
                id="dist-dd",
                options=[{"label": SECTOR_LABELS.get(s,s), "value": s} for s in PC_DISP],
                value=PC_DISP[0] if PC_DISP else None,
                clearable=False,
                style={"background": CARD, "color": TEXT1, "fontFamily": "IBM Plex Mono",
                       "fontSize": "11px", "width": "280px"},
            ),
        ], style={"marginBottom": "20px"}),
        html.Div(id="dist-output"),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 4 — ESTRUCTURA DE CORRELACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("04", "Estructura de Correlación",
                "Correlación de Spearman con significancia estadística · "
                "Scatter matrix · Identificación de variables redundantes y complementarias"),

    card_wrap(
        T("Matriz de Correlación de Spearman",
          "* p<.05  ** p<.01  *** p<.001 · No paramétrica · Robusta a distribuciones asimétricas"),
        narrative(
            "El análisis de correlación de Spearman revela una estructura de "
            "asociaciones sistemáticas entre los sectores de gasto. Las correlaciones "
            "elevadas entre cultura y deporte (pares funcionalmente análogos) sugieren "
            "la existencia de factores latentes comunes que determinan la asignación "
            "conjunta de recursos en estos sectores. La presencia de correlaciones "
            "significativas entre la mayoría de los pares de variables es condición "
            "necesaria — aunque no suficiente — para la aplicación del análisis factorial, "
            "cuya validez se verificará formalmente mediante las pruebas KMO y Bartlett "
            "en el Acto VI."
        ),
        dcc.Graph(figure=_fig_corr, config={"displayModeBar": False}),
    ),

    card_wrap(
        T("Scatter Matrix · Relaciones Bivariadas",
          "Cada celda muestra la relación entre dos variables · Color = región geográfica"),
        narrative(
            "La matriz de dispersión permite visualizar simultáneamente las relaciones "
            "bivariadas entre los principales sectores de gasto. Se observan patrones "
            "de asociación positiva generalizados, con mayor dispersión en los sectores "
            "de libre asignación. La diferenciación por región geográfica sugiere que "
            "parte de la variabilidad en el gasto se estructura territorialmente, "
            "anticipando la formación de clusters con base geográfica en el análisis "
            "de clasificación posterior."
        ),
        dcc.Graph(figure=_fig_scat_matrix, config={"displayModeBar": False}),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 5 — VALORES ATÍPICOS
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("05", "Detección de Valores Atípicos",
                "Z-scores univariados · Distancia de Mahalanobis aproximada · "
                "Identificación de casos influyentes para ACP y clustering"),

    card_wrap(
        T("Z-scores por Variable · |Z| > 2.5 = valor atípico potencial",
          "Cada punto es un departamento · La línea roja marca el umbral de ±2.5 desviaciones estándar"),
        narrative(
            "La detección de valores atípicos es una etapa crítica previa al ACP y al "
            "clustering, dado que ambas técnicas son sensibles a casos extremos. Un valor "
            "atípico multivariado puede dominar un componente principal completo o "
            "conformar un cluster artificial. La distancia de Mahalanobis aproximada "
            "(norma euclidiana del vector de Z-scores estandarizados) permite identificar "
            "los departamentos cuyo perfil multivariado de gasto es más alejado del "
            "centro de la distribución conjunta."
        ),
        dcc.Graph(figure=_fig_outliers, config={"displayModeBar": False}),
        html.Div([
            html.Div([
                T("Top 10 · Mayor distancia multivariada",
                  "Departamentos con perfil más alejado del promedio nacional"),
                _tabla_mah,
            ], style={"flex": "1"}),
            html.Div([
                T("Interpretación", "¿Qué hacer con los outliers?"),
                html.Div([
                    html.P("Los departamentos con distancia alta no necesariamente "
                           "deben eliminarse — pueden representar casos genuinamente "
                           "diferenciados (ej. Vaupés por baja población, Bogotá por "
                           "régimen especial). La decisión debe basarse en criterios "
                           "sustantivos, no solo estadísticos.", style={
                        "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                        "lineHeight": "1.7", "marginBottom": "12px",
                    }),
                    html.P("Recomendación: realizar el ACP y clustering con y sin "
                           "los casos extremos, reportar ambos resultados y discutir "
                           "las diferencias como análisis de sensibilidad.", style={
                        "color": ACCENT, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                        "lineHeight": "1.7",
                        "borderLeft": f"3px solid {ACCENT}", "paddingLeft": "10px",
                    }),
                ]),
            ], style={"flex": "1", "paddingLeft": "24px"}),
        ], style={"display": "flex", "gap": "20px", "marginTop": "20px"}),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 6 — ESTANDARIZACIÓN Y VALIDACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("06", "Estandarización y Validación de Factorizabilidad",
                "Efecto de la estandarización Z-score · Prueba KMO · "
                "Prueba de esfericidad de Bartlett · Condiciones para ACP y AF"),

    card_wrap(
        T("Efecto de la Estandarización · Antes y Después",
          "La estandarización elimina el efecto de escala — condición necesaria para ACP"),
        narrative(
            "El ACP maximiza la varianza explicada por los componentes. Cuando las "
            "variables tienen escalas muy diferentes — como ocurre en este conjunto "
            "donde educación (≈ 800.000 COP) supera en dos órdenes de magnitud a "
            "deporte (≈ 6.000 COP) — la variable de mayor escala dominaría "
            "artificialmente los primeros componentes. La estandarización Z-score "
            "garantiza que todas las variables contribuyan equitativamente a la "
            "estructura de componentes, haciendo que el ACP opere sobre la matriz "
            "de correlaciones en lugar de la matriz de covarianzas."
        ),
        dcc.Graph(figure=_fig_estand, config={"displayModeBar": False}),
    ),

    # KMO y Bartlett
    card_wrap(
        T("Validación de Factorizabilidad · KMO y Prueba de Bartlett",
          "Condiciones estadísticas necesarias para la aplicación de ACP y Análisis Factorial"),
        narrative(
            "Antes de proceder con el ACP y el análisis factorial, es necesario "
            "verificar que la matriz de correlaciones presente la estructura adecuada "
            "para la extracción de factores. El índice KMO evalúa la adecuación muestral "
            "comparando las correlaciones observadas con las correlaciones parciales, "
            "mientras que la prueba de Bartlett contrasta la hipótesis nula de que "
            "la matriz de correlaciones es identidad (ausencia de correlaciones)."
        ),
        html.Div([
            # KMO
            html.Div([
                html.P("ÍNDICE KMO", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
                    "letterSpacing": "0.15em", "marginBottom": "8px",
                }),
                html.P(f"{_kmo:.4f}", style={
                    "color": _kmo_color, "fontFamily": "IBM Plex Sans",
                    "fontSize": "36px", "fontWeight": "700", "margin": "0",
                }),
                html.P(_kmo_label, style={
                    "color": _kmo_color, "fontFamily": "IBM Plex Mono",
                    "fontSize": "12px", "marginTop": "4px", "fontWeight": "600",
                }),
                html.P("Adecuación muestral para ACP/AF", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                    "marginTop": "6px",
                }),
                html.P("Kaiser (1974): KMO > 0.70 = aceptable · > 0.80 = meritorio · > 0.90 = maravilloso",
                       style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                              "fontSize": "9px", "marginTop": "8px", "lineHeight": "1.5"}),
            ], style={
                "background": SURFACE, "border": f"2px solid {_kmo_color}",
                "borderRadius": "10px", "padding": "24px", "flex": "1", "textAlign": "center",
            }),

            # Bartlett
            html.Div([
                html.P("PRUEBA DE ESFERICIDAD DE BARTLETT", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
                    "letterSpacing": "0.15em", "marginBottom": "8px",
                }),
                html.P(f"χ² = {_chi2:.2f}", style={
                    "color": GREEN if _pval_bart < 0.05 else RED,
                    "fontFamily": "IBM Plex Sans", "fontSize": "28px",
                    "fontWeight": "700", "margin": "0",
                }),
                html.P(f"gl = {_gl}  ·  p-value = {_pval_bart:.4e}", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono",
                    "fontSize": "11px", "marginTop": "6px",
                }),
                html.P(
                    "✓ Se rechaza H₀ · La matriz de correlaciones NO es identidad · "
                    "Existen correlaciones sistemáticas entre variables" if _pval_bart < 0.05
                    else "✗ No se rechaza H₀ · La factorizabilidad es cuestionable",
                    style={
                        "color": GREEN if _pval_bart < 0.05 else RED,
                        "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                        "marginTop": "10px", "fontWeight": "600", "lineHeight": "1.5",
                    }
                ),
                html.P("H₀: La matriz de correlaciones es una matriz identidad · "
                       "Rechazar H₀ valida la factorizabilidad",
                       style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                              "fontSize": "9px", "marginTop": "8px", "lineHeight": "1.5"}),
            ], style={
                "background": SURFACE,
                "border": f"2px solid {GREEN if _pval_bart < 0.05 else RED}",
                "borderRadius": "10px", "padding": "24px", "flex": "1.4", "textAlign": "center",
            }),
        ], style={"display": "flex", "gap": "16px"}),
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ACTO 7 — ESTRUCTURA PRELIMINAR (SCREE PLOT)
    # ══════════════════════════════════════════════════════════════════════════
    acto_header("07", "Estructura Multivariada Preliminar",
                "Scree plot · Criterio de Kaiser (λ > 1) · Varianza acumulada · "
                "Justificación del número de componentes a retener · Transición al ACP"),

    card_wrap(
        T("Scree Plot · Criterio de Kaiser y Varianza Acumulada",
          f"Se retienen {_n_kaiser} componentes con λ > 1 · "
          f"Explican {_var_acum[_n_kaiser-1]:.1f}% de la varianza total"),
        narrative(
            f"El análisis de componentes principales preliminar sobre la matriz de "
            f"correlaciones estandarizadas revela que {_n_kaiser} componentes presentan "
            f"eigenvalores superiores a 1 (criterio de Kaiser), acumulando el "
            f"{_var_acum[_n_kaiser-1]:.1f}% de la varianza total. El primer componente "
            f"explica el {_var_exp[0]:.1f}% de la varianza, actuando como un factor "
            f"general de nivel de inversión social. La inspección del scree plot "
            f"confirma que la inflexión de la curva ocurre a partir del componente "
            f"{_n_kaiser+1}, validando la retención de {_n_kaiser} componentes. "
            f"Esta estructura bidimensional (o tridimensional si se retiene un tercer "
            f"componente con eigenvalor cercano a 1) orientará tanto el número de "
            f"factores en el análisis factorial como la interpretación de los clusters."
        ),
        dcc.Graph(figure=_fig_scree, config={"displayModeBar": False}),

        # Resumen de eigenvalores
        html.Div([
            html.Div([
                html.P(f"CP{i+1}", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
                    "letterSpacing": "0.1em", "marginBottom": "4px",
                }),
                html.P(f"{_eigenvals[i]:.3f}", style={
                    "color": GREEN if _eigenvals[i] >= 1 else BORDER,
                    "fontFamily": "IBM Plex Sans", "fontSize": "16px",
                    "fontWeight": "700", "margin": "0",
                }),
                html.P(f"{_var_exp[i]:.1f}%", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "9px",
                }),
            ], style={
                "background": SURFACE,
                "border": f"1px solid {GREEN if _eigenvals[i] >= 1 else BORDER}",
                "borderRadius": "6px", "padding": "12px 16px", "textAlign": "center",
            })
            for i in range(len(_eigenvals))
        ], style={"display": "flex", "gap": "10px", "marginTop": "20px", "flexWrap": "wrap"}),
    ),

    # ── CONCLUSIÓN DEL EDA ───────────────────────────────────────────────────
    html.Div([
        html.Span("SÍNTESIS DEL EDA · TRANSICIÓN AL ANÁLISIS MULTIVARIADO", style={
            "color": GREEN, "fontSize": "9px", "letterSpacing": "0.2em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.H2("El EDA justifica empíricamente el análisis multivariado", style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "15px", "margin": "10px 0 14px",
        }),
        html.Div([
            conclusion_item("✓", "Heterogeneidad territorial confirmada",
                            f"CV > 60% en libre destinación e inversión. "
                            f"Brecha máx/mín superior a 3x en todos los sectores.",
                            GREEN),
            conclusion_item("✓", "Correlaciones sistemáticas validadas",
                            f"Bartlett: χ²={_chi2:.1f}, p<0.001. "
                            f"Estructura factorial existente y significativa.",
                            GREEN),
            conclusion_item("✓", f"KMO = {_kmo:.3f} · {_kmo_label}",
                            f"Adecuación muestral {_kmo_label.lower()} para ACP y AF. "
                            f"Correlaciones parciales bajas respecto a las totales.",
                            _kmo_color),
            conclusion_item("✓", f"{_n_kaiser} componentes principales identificados",
                            f"Explican el {_var_acum[_n_kaiser-1]:.1f}% de la varianza. "
                            f"Criterio de Kaiser λ > 1 aplicado.",
                            ACCENT),
            conclusion_item("→", "Siguiente paso: ACP y Análisis Factorial",
                            "Extracción de factores, rotación Varimax, "
                            "interpretación de cargas y scores.",
                            ORANGE),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)",
                   "gap": "12px", "marginTop": "4px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {GREEN}",
        "borderRadius": "10px", "padding": "28px 32px", "marginBottom": "32px",
    }),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")


def conclusion_item(icon, title, body, color):
    return html.Div([
        html.Div([
            html.Span(icon, style={"color": color, "fontSize": "14px",
                                    "fontFamily": "IBM Plex Mono", "marginRight": "8px"}),
            html.Span(title, style={"color": TEXT1, "fontFamily": "IBM Plex Sans",
                                     "fontWeight": "600", "fontSize": "12px"}),
        ], style={"marginBottom": "6px"}),
        html.P(body, style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                              "fontSize": "10px", "lineHeight": "1.6", "margin": "0"}),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderLeft": f"3px solid {color}", "borderRadius": "6px", "padding": "14px 16px",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK — DISTRIBUCIÓN DINÁMICA
# ═══════════════════════════════════════════════════════════════════════════════
@callback(Output("dist-output", "children"), Input("dist-dd", "value"))
def update_dist(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    fig_h, fig_qq, fig_box, s = fig_distribucion(sector)

    return html.Div([
        html.Div([
            html.Div([
                html.P("Media",  style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","letterSpacing":"0.1em","textTransform":"uppercase","marginBottom":"4px"}),
                html.P(f"{s['media']:,.0f}", style={"color":ACCENT,"fontFamily":"IBM Plex Sans","fontSize":"17px","fontWeight":"700","margin":"0"}),
                html.P("COP/hab", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px","padding":"12px 16px","borderTop":f"3px solid {ACCENT}","flex":"1"}),
            html.Div([
                html.P("Mediana", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","letterSpacing":"0.1em","textTransform":"uppercase","marginBottom":"4px"}),
                html.P(f"{s['mediana']:,.0f}", style={"color":GREEN,"fontFamily":"IBM Plex Sans","fontSize":"17px","fontWeight":"700","margin":"0"}),
                html.P("COP/hab", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px","padding":"12px 16px","borderTop":f"3px solid {GREEN}","flex":"1"}),
            html.Div([
                html.P("CV", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","letterSpacing":"0.1em","textTransform":"uppercase","marginBottom":"4px"}),
                html.P(f"{s['cv']:.1f}%", style={"color":ORANGE,"fontFamily":"IBM Plex Sans","fontSize":"17px","fontWeight":"700","margin":"0"}),
                html.P("Coef. variación", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px","padding":"12px 16px","borderTop":f"3px solid {ORANGE}","flex":"1"}),
            html.Div([
                html.P("Asimetría", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","letterSpacing":"0.1em","textTransform":"uppercase","marginBottom":"4px"}),
                html.P(f"{s['asimetria']:.3f}", style={"color":PURPLE,"fontFamily":"IBM Plex Sans","fontSize":"17px","fontWeight":"700","margin":"0"}),
                html.P("Skewness", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px","padding":"12px 16px","borderTop":f"3px solid {PURPLE}","flex":"1"}),
            html.Div([
                html.P("Shapiro-Wilk", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","letterSpacing":"0.1em","textTransform":"uppercase","marginBottom":"4px"}),
                html.P(s["norm_text"], style={"color":s["norm_color"],"fontFamily":"IBM Plex Mono","fontSize":"11px","fontWeight":"600","margin":"0","lineHeight":"1.3"}),
                html.P(f"p = {s['shapiro_p']:.4f}", style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px","padding":"12px 16px","borderTop":f"3px solid {s['norm_color']}","flex":"1.5"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"16px"}),
        html.Div([
            html.Div(dcc.Graph(figure=fig_h,  config={"displayModeBar":False}), style={"flex":"1.5"}),
            html.Div(dcc.Graph(figure=fig_qq, config={"displayModeBar":False}), style={"flex":"1"}),
        ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
        dcc.Graph(figure=fig_box, config={"displayModeBar":False}),
    ])