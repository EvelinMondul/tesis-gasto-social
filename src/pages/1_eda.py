"""
Página 04 · Análisis Composicional del Gasto Social
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enfoque: proporciones del gasto total → patrones de PRIORIZACIÓN sectorial
Metodología: CoDa (Aitchison 1986), CLR, PCA composicional, K-Means
"""
import dash
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import cargar_datos, SECTORES_ABS, REGION_COLORS, SECTOR_LABELS, PALETTE

dash.register_page(__name__, path="/composicional",
                   name="Análisis Composicional", order=3)

# ═══════════════════════════════════════════════════════════════════════════════
# PALETA Y CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
P      = PALETTE
BG     = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED    = P["red"];    PURPLE  = P["purple"]

# Paleta suave para gráficos composicionales (tema profesional / claro)
SECTOR_COLORS = {
    "Agua Potable":      "#4A90D9",
    "Educación":         "#2ECC71",
    "Salud":             "#E74C3C",
    "Cultura":           "#9B59B6",
    "Deporte":           "#F39C12",
    "Libre Destinación": "#1ABC9C",
    "Libre Inversión":   "#E67E22",
}

CLUSTER_COLORS = {1: "#2E86AB", 2: "#E84855", 3: "#3BB273", 4: "#F4A261"}

BASE = dict(paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11))

DD = {"background": SURFACE, "color": TEXT1,
      "fontFamily": "IBM Plex Mono", "fontSize": "11px",
      "border": f"1px solid {BORDER}"}

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
df_raw  = cargar_datos()
SECTORES = [c for c in SECTORES_ABS if c in df_raw.columns]
LABELS   = [SECTOR_LABELS.get(s, s) for s in SECTORES]
PROP_COLS = [f"prop_{s}" for s in SECTORES]

# 1. Calcular proporciones
df = df_raw[["departamento", "region"]].copy()
for s in SECTORES:
    df[f"prop_{s}"] = df_raw[s] / df_raw["total"]

X_prop = df[PROP_COLS].values  # (33, 7)

# 2. Reemplazo multiplicativo de ceros (Martín-Fernández et al. 2003)
# Bogotá: libre_destinacion = 0 (régimen especial)
DELTA = 0.0001
X_rep = X_prop.copy()
for i in range(len(X_rep)):
    zeros = X_rep[i] == 0
    if zeros.sum() > 0:
        X_rep[i, zeros] = DELTA
        X_rep[i, ~zeros] = X_rep[i, ~zeros] * (1 - zeros.sum() * DELTA)

# 3. Transformación CLR (centred log-ratio)
X_clr = np.log(X_rep) - np.log(X_rep).mean(axis=1, keepdims=True)

# 4. Estandarizar CLR para PCA y clustering
X_std = StandardScaler().fit_transform(X_clr)

# 5. PCA composicional sobre CLR estandarizado
pca = PCA()
pca.fit(X_std)
VE     = pca.explained_variance_ratio_ * 100
EV     = pca.explained_variance_
VA     = np.cumsum(VE)
SCORES = pca.transform(X_std)
# Solo 5 componentes válidos (rank = p-1 = 6, pero últimos 2 ≈ 0)
N_VALID = sum(1 for e in EV if e > 0.01)
N_KAISER = sum(1 for e in EV if e >= 1)
LOADINGS  = pca.components_  # (7, 7)

# 6. KMO y Bartlett sobre proporciones (p-1 variables, robusto)
_sel = ["prop_agua_potable", "prop_salud", "prop_cultura", "prop_libre_destinacion"]
_X   = StandardScaler().fit_transform(df[_sel].values)
_R   = np.corrcoef(_X.T)
_Ri  = np.linalg.pinv(_R)
_n   = _R.shape[0]
_P   = np.zeros((_n, _n))
for _i in range(_n):
    for _j in range(_n):
        if _i != _j:
            _d = np.sqrt(abs(_Ri[_i,_i] * _Ri[_j,_j]))
            if _d > 1e-10:
                _P[_i,_j] = -_Ri[_i,_j] / _d
_r2 = np.sum(_R[np.triu_indices(_n, k=1)]**2)
_p2 = np.sum(_P[np.triu_indices(_n, k=1)]**2)
KMO  = _r2 / (_r2 + _p2)
_no, _p = _X.shape
_det = np.linalg.det(_R)
CHI2 = -(_no - 1 - (2*_p+5)/6) * np.log(_det)
GL   = int(_p * (_p-1) / 2)
PVAL_BART = 1 - stats.chi2.cdf(CHI2, GL)

# 7. Clustering K-Means sobre scores CP1-CP2
_sc2 = SCORES[:, :2]
K_RANGE = range(2, 7)
INERTIAS = []
SILHOUETTES = []
for _k in K_RANGE:
    _km = KMeans(n_clusters=_k, random_state=42, n_init=20)
    _lab = _km.fit_predict(_sc2)
    INERTIAS.append(_km.inertia_)
    SILHOUETTES.append(silhouette_score(_sc2, _lab))

# Clustering final con k=3 (balance silueta/interpretabilidad)
K_FINAL = 3
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(_sc2) + 1

# 8. Perfiles de clusters en escala de proporciones
PROFILE = df.groupby("cluster")[PROP_COLS].mean() * 100

# Nombres descriptivos de clusters
CLUSTER_NAMES = {
    1: "Equilibrio SGP",
    2: "Alta educación básica",
    3: "Perfil diferencial",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS UI
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


def narr(text):
    return html.Div([
        html.Span("▸ ", style={"color": ACCENT, "fontSize": "10px"}),
        html.Span(text, style={
            "color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
            "lineHeight": "1.8",
        }),
    ], style={
        "borderLeft": f"2px solid {BORDER}", "paddingLeft": "14px",
        "marginBottom": "18px", "display": "flex", "alignItems": "flex-start",
        "gap": "4px",
    })


def card(*children, mb="22px", border_color=None):
    bc = border_color or BORDER
    return html.Div(list(children), style={
        "background": CARD, "border": f"1px solid {bc}",
        "borderRadius": "8px", "padding": "20px 22px", "marginBottom": mb,
    })


def acto(num, titulo, desc, color=None):
    c = color or ACCENT
    return html.Div([
        html.Span(f"ETAPA {num}", style={
            "color": c, "fontFamily": "IBM Plex Mono",
            "fontSize": "9px", "letterSpacing": "0.2em", "fontWeight": "600",
        }),
        html.H2(titulo, style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans",
            "fontWeight": "700", "fontSize": "15px", "margin": "6px 0 6px",
        }),
        html.P(desc, style={
            "color": TEXT2, "fontFamily": "IBM Plex Mono",
            "fontSize": "11px", "margin": "0", "lineHeight": "1.6",
        }),
    ], style={
        "background": SURFACE, "border": f"1px solid {c}40",
        "borderLeft": f"4px solid {c}", "borderRadius": "8px",
        "padding": "16px 22px", "marginBottom": "22px",
    })


def kpi(t, v, s="", c=None):
    col = c or ACCENT
    return html.Div([
        html.P(t, style={"color": TEXT2, "fontSize": "9px", "letterSpacing": "0.1em",
                          "textTransform": "uppercase", "fontFamily": "IBM Plex Mono",
                          "marginBottom": "4px"}),
        html.P(v, style={"color": col, "fontSize": "18px", "fontWeight": "700",
                          "fontFamily": "IBM Plex Sans", "margin": "0"}),
        html.P(s, style={"color": TEXT2, "fontSize": "9px",
                          "fontFamily": "IBM Plex Mono", "marginTop": "3px"}) if s else None,
    ], style={"background": CARD, "border": f"1px solid {BORDER}",
               "borderRadius": "6px", "padding": "12px 16px",
               "borderTop": f"3px solid {col}"})


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 1: PROPORCIONES (DESCRIPCIÓN)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_proporciones_barras():
    """Barras apiladas de proporciones ordenadas por proporción de educación."""
    df_sorted = df.sort_values("prop_educacion", ascending=True)
    fig = go.Figure()
    colors = list(SECTOR_COLORS.values())
    for i, (s, label) in enumerate(zip(PROP_COLS, LABELS)):
        fig.add_trace(go.Bar(
            name=label,
            x=df_sorted[s] * 100,
            y=df_sorted["departamento"].str.title(),
            orientation="h",
            marker_color=colors[i],
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **BASE,
        barmode="stack", height=780,
        title=dict(text="Estructura de Priorización del Gasto Social · Proporciones (%)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(range=[0, 100], title="% del gasto total",
                   gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), ticksuffix="%"),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor=CARD,
                    bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT1)),
        margin=dict(l=160, r=20, t=80, b=50),
    )
    return fig


def fig_boxplot_proporciones():
    """Boxplots de proporciones para mostrar dispersión por sector."""
    data_long = []
    for s, label in zip(PROP_COLS, LABELS):
        for _, row in df.iterrows():
            data_long.append({"Sector": label, "Proporción (%)": row[s]*100,
                               "Departamento": row["departamento"].title()})
    dfl = pd.DataFrame(data_long)
    fig = px.box(dfl, x="Sector", y="Proporción (%)", color="Sector",
                  points="all", hover_name="Departamento",
                  color_discrete_map=SECTOR_COLORS)
    fig.update_layout(
        **BASE,
        title=dict(text="Distribución de Proporciones por Sector",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-20,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1),
                   title="% del gasto total"),
        showlegend=False, height=420,
        margin=dict(l=70, r=20, t=60, b=80),
    )
    return fig


def fig_cv_proporciones():
    """CV de proporciones — mide heterogeneidad en priorización."""
    cvs = [(l, df[s].std()/df[s].mean()*100) for s,l in zip(PROP_COLS, LABELS)]
    cvs.sort(key=lambda x: x[1], reverse=True)
    labels_cv, values_cv = zip(*cvs)
    colors_cv = [RED if v > 40 else ORANGE if v > 25 else ACCENT for v in values_cv]
    fig = go.Figure(go.Bar(
        x=list(labels_cv), y=list(values_cv),
        marker_color=colors_cv,
        text=[f"{v:.1f}%" for v in values_cv],
        textposition="outside",
        textfont=dict(size=10, color=TEXT1),
    ))
    fig.add_hline(y=25, line_dash="dash", line_color=GREEN,
                  annotation_text="25% · variabilidad moderada",
                  annotation_font=dict(color=GREEN, size=9))
    fig.add_hline(y=40, line_dash="dash", line_color=RED,
                  annotation_text="40% · variabilidad alta",
                  annotation_font=dict(color=RED, size=9))
    fig.update_layout(
        **BASE,
        title=dict(text="Heterogeneidad en la Priorización del Gasto (CV de proporciones)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-15,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="CV (%)",
                   range=[0, max(values_cv)*1.3], tickfont=dict(color=TEXT1)),
        height=400, showlegend=False,
        margin=dict(l=60, r=120, t=60, b=80),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 2: CORRELACIÓN (TRADE-OFFS)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_correlacion_tradeoffs():
    """Heatmap Spearman sobre CLR — revela trade-offs entre sectores."""
    n = len(PROP_COLS)
    corr_m = np.zeros((n, n))
    pval_m = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(X_clr[:, i], X_clr[:, j])
            corr_m[i, j] = r
            pval_m[i, j] = p
    text_m = []
    for i in range(n):
        row = []
        for j in range(n):
            s = ("***" if pval_m[i,j] < 0.001
                 else "**" if pval_m[i,j] < 0.01
                 else "*"  if pval_m[i,j] < 0.05
                 else "")
            row.append(f"{corr_m[i,j]:.2f}{s}")
        text_m.append(row)
    fig = go.Figure(go.Heatmap(
        z=corr_m, x=LABELS, y=LABELS,
        colorscale=[[0,"#E74C3C"],[0.5,"#1C2333"],[1,"#3498DB"]],
        zmid=0, zmin=-1, zmax=1,
        text=text_m, texttemplate="%{text}",
        textfont=dict(size=9, family="IBM Plex Mono"),
        colorbar=dict(title="ρ", titlefont=dict(size=10, color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(
            text="Correlación de Spearman sobre CLR · Trade-offs entre sectores · * p<.05  ** p<.01  *** p<.001",
            font=dict(family="IBM Plex Sans", size=12, color=TEXT1)),
        xaxis=dict(tickangle=-35, gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        height=520, margin=dict(l=140, r=20, t=70, b=140),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 3: CLR Y TRANSFORMACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def fig_clr_antes_despues():
    """Comparación proporciones vs CLR para justificar la transformación."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Proporciones brutas (%)", "Tras transformación CLR"],
    )
    colors = list(SECTOR_COLORS.values())
    for i, (s, label) in enumerate(zip(PROP_COLS, LABELS)):
        fig.add_trace(go.Box(
            y=df[s].values * 100, name=label,
            marker_color=colors[i], showlegend=False, boxpoints=False,
        ), row=1, col=1)
        fig.add_trace(go.Box(
            y=X_clr[:, i], name=label,
            marker_color=colors[i], showlegend=True, boxpoints=False,
        ), row=1, col=2)
    fig.update_layout(
        **BASE,
        title=dict(text="Efecto de la Transformación CLR sobre las Proporciones del Gasto",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=460,
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor=CARD,
                    bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT1)),
        margin=dict(l=60, r=20, t=80, b=140),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1), tickangle=-30)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1, size=11))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 4: PCA COMPOSICIONAL
# ═══════════════════════════════════════════════════════════════════════════════
def fig_scree_composicional():
    n_valid = N_VALID
    labels_cp = [f"CP{i+1}" for i in range(n_valid)]
    ve_valid = VE[:n_valid]
    ev_valid = EV[:n_valid]
    va_valid = np.cumsum(ve_valid)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Varianza explicada (%)", "Eigenvalores · Criterio Kaiser (λ > 1)"],
    )
    bar_colors = [GREEN if e >= 1 else ACCENT if e >= 0.5 else BORDER for e in ev_valid]
    fig.add_trace(go.Bar(
        x=labels_cp, y=list(ve_valid),
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in ve_valid],
        textposition="outside",
        textfont=dict(size=9, color=TEXT1),
        name="Var. %",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=labels_cp, y=list(va_valid),
        mode="lines+markers",
        line=dict(color=ORANGE, width=2),
        marker=dict(size=7, color=ORANGE),
        name="Acumulada",
        yaxis="y3",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels_cp, y=list(ev_valid),
        marker_color=[GREEN if e >= 1 else RED for e in ev_valid],
        text=[f"{e:.3f}" for e in ev_valid],
        textposition="outside",
        textfont=dict(size=9, color=TEXT1),
        name="λ",
    ), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color=ORANGE,
                  annotation_text="λ=1 (Kaiser)",
                  annotation_font=dict(color=ORANGE, size=9), row=1, col=2)
    fig.update_layout(
        **BASE,
        title=dict(text="PCA Composicional · Scree Plot sobre Datos CLR",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=420, showlegend=False,
        yaxis3=dict(overlaying="y", side="right", range=[0, 115],
                    showgrid=False, tickfont=dict(color=ORANGE),
                    titlefont=dict(color=ORANGE)),
        margin=dict(l=60, r=80, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1, size=11))
    return fig


def fig_biplot():
    """Biplot PCA composicional — scores + loadings."""
    sc = SCORES
    ld = LOADINGS
    scale = np.sqrt(EV[0]) * 1.5

    fig = go.Figure()

    # Scores coloreados por región
    for reg, gdf in df.groupby("region"):
        idx = gdf.index
        fig.add_trace(go.Scatter(
            x=sc[idx, 0], y=sc[idx, 1],
            mode="markers+text",
            name=reg,
            text=gdf["departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7, color=TEXT2),
            marker=dict(size=9, color=REGION_COLORS.get(reg, ACCENT),
                        opacity=0.85, line=dict(width=0.5, color=BG)),
        ))

    # Vectores de cargas (loadings)
    for i, label in enumerate(LABELS):
        lx, ly = ld[0, i] * scale, ld[1, i] * scale
        fig.add_annotation(
            x=lx, y=ly, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor=SECTOR_COLORS.get(label, ACCENT),
        )
        fig.add_annotation(
            x=lx * 1.12, y=ly * 1.12,
            text=label, showarrow=False,
            font=dict(size=9, color=SECTOR_COLORS.get(label, ACCENT),
                      family="IBM Plex Mono"),
        )

    # Ejes
    fig.add_hline(y=0, line_color=BORDER, line_width=0.5)
    fig.add_vline(x=0, line_color=BORDER, line_width=0.5)

    fig.update_layout(
        **BASE,
        title=dict(
            text=f"Biplot PCA Composicional · CP1 ({VE[0]:.1f}%) vs CP2 ({VE[1]:.1f}%) · "
                 f"Acumulado: {VA[1]:.1f}%",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT1),
        ),
        xaxis=dict(title=f"CP1 — {VE[0]:.1f}% varianza",
                   gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), zeroline=False),
        yaxis=dict(title=f"CP2 — {VE[1]:.1f}% varianza",
                   gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), zeroline=False),
        height=580,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1)),
        margin=dict(l=80, r=20, t=70, b=60),
    )
    return fig


def fig_loadings_heatmap():
    """Mapa de calor de cargas factoriales."""
    n_cp = N_VALID
    z = LOADINGS[:n_cp].T  # (n_vars, n_cp)
    cp_labels = [f"CP{i+1} ({VE[i]:.1f}%)" for i in range(n_cp)]
    text_z = [[f"{v:.3f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=cp_labels, y=LABELS,
        colorscale=[[0,"#E74C3C"],[0.5,"#1C2333"],[1,"#3498DB"]],
        zmid=0, zmin=-0.7, zmax=0.7,
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=9, family="IBM Plex Mono"),
        colorbar=dict(title="Carga", titlefont=dict(size=10, color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Cargas Factoriales · Contribución de cada sector a los componentes",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(tickangle=-20, gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        height=380, margin=dict(l=140, r=20, t=60, b=80),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 5: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
def fig_codo_silueta():
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["Método del Codo (Inercia)",
                                          "Coeficiente de Silueta"])
    ks = list(K_RANGE)
    fig.add_trace(go.Scatter(
        x=ks, y=INERTIAS, mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8, color=ACCENT),
        name="Inercia",
    ), row=1, col=1)
    best_k_idx = SILHOUETTES.index(max(SILHOUETTES))
    sil_colors = [GREEN if i == best_k_idx else ACCENT for i in range(len(ks))]
    fig.add_trace(go.Bar(
        x=ks, y=SILHOUETTES,
        marker_color=sil_colors,
        text=[f"{s:.3f}" for s in SILHOUETTES],
        textposition="outside",
        textfont=dict(size=9, color=TEXT1),
        name="Silueta",
    ), row=1, col=2)
    fig.update_layout(
        **BASE,
        title=dict(text="Selección del Número de Clusters · Codo y Silueta",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        height=380, showlegend=False,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER,
                     title="Número de clusters (k)", tickfont=dict(color=TEXT1))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1, size=11))
    return fig


def fig_scatter_clusters():
    """Clusters en espacio PCA con elipses de confianza."""
    fig = go.Figure()
    for k in sorted(df["cluster"].unique()):
        mask = df["cluster"] == k
        idx  = df.index[mask]
        fig.add_trace(go.Scatter(
            x=SCORES[idx, 0], y=SCORES[idx, 1],
            mode="markers+text",
            name=f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            text=df.loc[mask,"departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7, color=TEXT2),
            marker=dict(size=10, color=CLUSTER_COLORS.get(k, ACCENT),
                        opacity=0.85, line=dict(width=1, color=BG)),
        ))
    fig.add_hline(y=0, line_color=BORDER, line_width=0.5)
    fig.add_vline(x=0, line_color=BORDER, line_width=0.5)
    fig.update_layout(
        **BASE,
        title=dict(
            text=f"Clusters K-Means (k={K_FINAL}) en Espacio PCA Composicional",
            font=dict(family="IBM Plex Sans", size=13, color=TEXT1),
        ),
        xaxis=dict(title=f"CP1 — {VE[0]:.1f}%",
                   gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), zeroline=False),
        yaxis=dict(title=f"CP2 — {VE[1]:.1f}%",
                   gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1), zeroline=False),
        height=520,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1)),
        margin=dict(l=80, r=20, t=70, b=60),
    )
    return fig


def fig_perfil_clusters():
    """Radar de perfiles de clusters en escala de proporciones."""
    fig = go.Figure()
    for k in sorted(df["cluster"].unique()):
        vals = [PROFILE.loc[k, c] for c in PROP_COLS]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=LABELS + [LABELS[0]],
            name=f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            fill="toself", opacity=0.5,
            line=dict(color=CLUSTER_COLORS.get(k, ACCENT), width=2),
            marker=dict(color=CLUSTER_COLORS.get(k, ACCENT)),
        ))
    fig.update_layout(
        **BASE,
        title=dict(text="Perfil de Priorización del Gasto por Cluster (%)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, max(PROFILE.values.max()*1.1, 60)],
                            gridcolor=BORDER,
                            tickfont=dict(size=8, color=TEXT2), ticksuffix="%"),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9, color=TEXT2)),
        ),
        showlegend=True, height=480,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT1)),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def fig_heatmap_clusters():
    """Heatmap de proporciones medias por cluster."""
    z = PROFILE[PROP_COLS].values  # (k, 7)
    text_z = [[f"{v:.1f}%" for v in row] for row in z]
    row_labels = [f"C{k} · {CLUSTER_NAMES.get(k,'')}" for k in sorted(df.cluster.unique())]
    fig = go.Figure(go.Heatmap(
        z=z, x=LABELS, y=row_labels,
        colorscale=[[0, "#EAF4FB"], [0.5, "#3498DB"], [1, "#1A5276"]],
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=10, family="IBM Plex Mono", color=TEXT1),
        colorbar=dict(title="%", titlefont=dict(size=10, color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Proporciones Medias del Gasto por Cluster (%)",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(tickangle=-25, gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        height=280, margin=dict(l=200, r=20, t=60, b=80),
    )
    return fig


def tabla_clusters():
    rows = []
    for k in sorted(df["cluster"].unique()):
        deps = df.loc[df.cluster==k, "departamento"].str.title().tolist()
        n    = len(deps)
        prof = {SECTOR_LABELS.get(c.replace("prop_",""),c): f"{PROFILE.loc[k,c]:.1f}%"
                for c in PROP_COLS}
        dominant = max(prof, key=lambda x: float(prof[x].replace("%","")))
        rows.append({
            "Cluster": f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            "n": str(n),
            "Sector dominante": dominant,
            "Proporción dominante": prof[dominant],
            "Departamentos": ", ".join(deps),
        })
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tdf.columns],
        style_table={"overflowX": "auto"},
        style_cell={"background": CARD, "color": TEXT1,
                     "border": f"1px solid {BORDER}",
                     "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                     "padding": "9px 13px", "textAlign": "left",
                     "whiteSpace": "normal", "height": "auto"},
        style_cell_conditional=[
            {"if": {"column_id": "Cluster"}, "color": ACCENT, "fontWeight": "600",
             "minWidth": "180px"},
            {"if": {"column_id": "n"}, "textAlign": "center", "maxWidth": "40px"},
        ],
        style_header={"background": SURFACE, "color": ACCENT, "fontWeight": "600",
                       "border": f"1px solid {BORDER}",
                       "fontFamily": "IBM Plex Sans", "fontSize": "10px"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
_fig_prop_barras  = fig_proporciones_barras()
_fig_box_prop     = fig_boxplot_proporciones()
_fig_cv_prop      = fig_cv_proporciones()
_fig_corr         = fig_correlacion_tradeoffs()
_fig_clr          = fig_clr_antes_despues()
_fig_scree        = fig_scree_composicional()
_fig_biplot       = fig_biplot()
_fig_loadings     = fig_loadings_heatmap()
_fig_codo         = fig_codo_silueta()
_fig_scatter_cl   = fig_scatter_clusters()
_fig_perfil_cl    = fig_perfil_clusters()
_fig_heatmap_cl   = fig_heatmap_clusters()
_tabla_cl         = tabla_clusters()

def _kmo_label(k):
    if k >= 0.90: return "Maravilloso", GREEN
    if k >= 0.80: return "Meritorio",   ACCENT
    if k >= 0.70: return "Mediano",     ORANGE
    if k >= 0.60: return "Mediocre",    ORANGE
    return "Inaceptable", RED

_kl, _kc = _kmo_label(KMO)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
layout = html.Div([

    # PORTADA
    html.Div([
        html.Span("ANÁLISIS COMPOSICIONAL DEL GASTO SOCIAL · CoDa + PCA + K-MEANS", style={
            "color": ACCENT, "fontSize": "9px", "letterSpacing": "0.2em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.H1("Patrones de Priorización del Gasto Social Departamental", style={
            "color": TEXT1, "fontFamily": "IBM Plex Sans", "fontWeight": "700",
            "fontSize": "21px", "margin": "10px 0 12px",
        }),
        html.P(
            "El análisis se realiza sobre las PROPORCIONES del gasto total destinado a cada sector, "
            "no sobre valores absolutos ni per cápita. Este enfoque permite identificar patrones de "
            "PRIORIZACIÓN: cómo cada departamento distribuye sus recursos entre sectores. "
            "Dado que las proporciones suman 1 (simplex constraint), se aplica la transformación "
            "CLR (centred log-ratio, Aitchison 1986) antes del PCA para eliminar la dependencia "
            "composicional y obtener resultados no sesgados.",
            style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "11px",
                   "lineHeight": "1.8", "maxWidth": "900px", "margin": "0"}
        ),
        html.Div([
            html.Div(f"KMO = {KMO:.3f} · {_kl}", style={
                "color": _kc, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{_kc}18", "border": f"1px solid {_kc}",
                "padding": "4px 14px", "borderRadius": "20px", "marginRight": "10px",
            }),
            html.Div(f"Bartlett p = {PVAL_BART:.2e}", style={
                "color": GREEN, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{GREEN}18", "border": f"1px solid {GREEN}",
                "padding": "4px 14px", "borderRadius": "20px", "marginRight": "10px",
            }),
            html.Div(f"Kaiser: {N_KAISER} CP · {VA[N_KAISER-1]:.1f}% varianza", style={
                "color": ACCENT, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{ACCENT}18", "border": f"1px solid {ACCENT}",
                "padding": "4px 14px", "borderRadius": "20px", "marginRight": "10px",
            }),
            html.Div(f"k = {K_FINAL} clusters · silueta = {max(SILHOUETTES):.3f}", style={
                "color": ORANGE, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                "background": f"{ORANGE}18", "border": f"1px solid {ORANGE}",
                "padding": "4px 14px", "borderRadius": "20px",
            }),
        ], style={"display": "flex", "marginTop": "16px", "flexWrap": "wrap", "gap": "8px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {ACCENT}", "borderRadius": "10px",
        "padding": "26px 32px", "marginBottom": "30px",
    }),

    # ══ ETAPA 1 ══════════════════════════════════════════════════════════════
    acto("01", "Descripción de la Estructura Composicional",
         "¿Cómo prioriza cada departamento su gasto social? · "
         "Proporciones sectoriales · Heterogeneidad en la priorización"),

    card(
        T("Tabla 1 · Estadísticos Descriptivos · Proporciones del Gasto (%)",
          "Unidad: % del gasto total · Cada fila suma 100%"),
        narr(
            "Las proporciones representan PRIORIDADES de gasto, no niveles de inversión. "
            "Educación concentra en promedio el 57.5% del gasto (rango 42.4%–70.0%), "
            "determinado principalmente por el SGP. Libre destinación (CV=47.2%) y "
            "libre inversión (CV=34.1%) son los sectores con mayor variabilidad en la "
            "priorización, lo que indica diferencias sustanciales en la autonomía fiscal "
            "de los territorios."
        ),
        _tabla_descriptivos_prop(),
    ),

    card(
        T("Gráfico 1 · Estructura de Priorización por Departamento",
          "Barras apiladas de proporciones · Ordenado por proporción de educación"),
        narr(
            "La estructura composicional revela que todos los departamentos priorizan "
            "educación como el componente de mayor peso, pero existen diferencias "
            "sustanciales en los sectores de libre asignación. Bogotá (prop. libre "
            "destinación = 0) se distingue estructuralmente del resto, reflejando su "
            "régimen especial de distrito capital."
        ),
        dcc.Graph(figure=_fig_prop_barras, config={"displayModeBar": False}),
    ),

    html.Div([
        html.Div(card(
            T("Gráfico 2 · Distribución por Sector",
              "Dispersión de proporciones · Outliers composicionales"),
            narr(
                "La distribución de las proporciones confirma la dominancia de educación "
                "y salud. La alta dispersión en libre destinación evidencia asimetrías en "
                "la capacidad fiscal autónoma entre departamentos."
            ),
            dcc.Graph(figure=_fig_box_prop, config={"displayModeBar": False}),
        ), style={"flex": "1"}),
        html.Div(card(
            T("Gráfico 3 · CV de Proporciones",
              "Heterogeneidad en la priorización · Rojo > 40%"),
            narr(
                "La libre destinación presenta el mayor CV (47.2%), indicando que "
                "las decisiones de asignación autónoma varían ampliamente entre "
                "departamentos, a diferencia de los sectores SGP que están más "
                "estandarizados."
            ),
            dcc.Graph(figure=_fig_cv_prop, config={"displayModeBar": False}),
        ), style={"flex": "1"}),
    ], style={"display": "flex", "gap": "16px"}),

    # ══ ETAPA 2 ══════════════════════════════════════════════════════════════
    acto("02", "Análisis de Trade-offs entre Sectores",
         "¿Qué sectores compiten por recursos? · "
         "Correlación de Spearman sobre CLR · Trade-offs composicionales",
         color=GREEN),

    card(
        T("Gráfico 4 · Correlaciones Spearman sobre CLR · Trade-offs de Priorización",
          "Correlación negativa = trade-off: priorizar uno implica reducir el otro"),
        narr(
            "En datos composicionales, una correlación negativa fuerte indica un "
            "TRADE-OFF estructural: los departamentos que priorizan un sector "
            "necesariamente destinan menos proporción a otro. La correlación "
            "negativa entre educación y libre destinación (ρ ≈ -0.81***) revela "
            "el principal trade-off del sistema: los departamentos que destinan "
            "mayor fracción al sector educativo (vía SGP) tienen menos espacio "
            "fiscal para gasto discrecional. La correlación positiva entre cultura "
            "y deporte (ρ ≈ +1.00***) confirma que se trata de un único patrón "
            "de priorización conjunta de sectores complementarios."
        ),
        dcc.Graph(figure=_fig_corr, config={"displayModeBar": False}),
    ),

    # ══ ETAPA 3 ══════════════════════════════════════════════════════════════
    acto("03", "Transformación CLR · Justificación Metodológica",
         "Las proporciones suman 1 (simplex constraint) → dependencia estructural. "
         "La transformación CLR elimina el sesgo composicional antes del PCA.",
         color=ORANGE),

    card(
        T("Gráfico 5 · Efecto de la Transformación CLR",
          "Izquierda: proporciones brutas · Derecha: tras CLR · "
          "La CLR elimina la escala y la dependencia entre proporciones"),
        narr(
            "La transformación CLR (Aitchison 1986) convierte proporciones en "
            "log-ratios centrados: CLR(xᵢ) = log(xᵢ) - (1/p)·Σlog(xⱼ). "
            "Esto tiene tres efectos: (1) elimina la dependencia estructural "
            "entre proporciones, (2) hace las variables más simétricas y "
            "comparables, y (3) permite interpretar el PCA en términos de "
            "ratios relativos entre sectores. El cero de Bogotá en libre "
            "destinación requiere reemplazo multiplicativo previo (δ=0.0001, "
            "Martín-Fernández et al. 2003)."
        ),
        dcc.Graph(figure=_fig_clr, config={"displayModeBar": False}),
    ),

    # Validación KMO y Bartlett
    card(
        T("Validación de Factorizabilidad · KMO y Bartlett",
          "Aplicados sobre proporciones estandarizadas (p-1 variables) · "
          "Nota: la matriz CLR es singular por construcción (rank = p-1)"),
        html.Div([
            html.Div([
                html.P("KMO (proporciones)", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono",
                    "fontSize": "9px", "letterSpacing": "0.12em", "marginBottom": "8px",
                }),
                html.P(f"{KMO:.4f}", style={
                    "color": _kc, "fontFamily": "IBM Plex Sans",
                    "fontSize": "40px", "fontWeight": "700", "margin": "0",
                }),
                html.P(_kl, style={
                    "color": _kc, "fontFamily": "IBM Plex Mono",
                    "fontSize": "12px", "fontWeight": "600", "marginTop": "4px",
                }),
                html.P("Adecuación muestral para análisis factorial · Kaiser 1974",
                       style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                              "fontSize": "9px", "marginTop": "8px", "lineHeight": "1.5"}),
                html.P("⚠ KMO calculado sobre 4 proporciones no collineales. "
                       "La matriz CLR completa es singular (rango p-1) por la "
                       "restricción composicional — esto es esperado y no es un error.",
                       style={"color": ORANGE, "fontFamily": "IBM Plex Mono",
                              "fontSize": "9px", "marginTop": "8px", "lineHeight": "1.5",
                              "borderLeft": f"2px solid {ORANGE}", "paddingLeft": "8px"}),
            ], style={"flex": "1", "background": SURFACE,
                       "border": f"2px solid {_kc}", "borderRadius": "10px",
                       "padding": "22px", "textAlign": "center"}),

            html.Div([
                html.P("PRUEBA DE BARTLETT", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono",
                    "fontSize": "9px", "letterSpacing": "0.12em", "marginBottom": "8px",
                }),
                html.P(f"χ² = {CHI2:.2f}", style={
                    "color": GREEN, "fontFamily": "IBM Plex Sans",
                    "fontSize": "30px", "fontWeight": "700", "margin": "0",
                }),
                html.P(f"gl = {GL}   ·   p = {PVAL_BART:.2e}", style={
                    "color": TEXT2, "fontFamily": "IBM Plex Mono",
                    "fontSize": "12px", "marginTop": "8px",
                }),
                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.P(
                    "✓ Se rechaza H₀ · Existen correlaciones sistemáticas entre "
                    "proporciones · Factorizabilidad confirmada",
                    style={"color": GREEN, "fontFamily": "IBM Plex Mono",
                           "fontSize": "11px", "fontWeight": "600", "lineHeight": "1.6"}
                ),
                html.P("H₀: R = I · Las proporciones son estadísticamente independientes.\n"
                       "Rechazar H₀ confirma la existencia de estructura factorial.",
                       style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                              "fontSize": "9px", "marginTop": "8px",
                              "lineHeight": "1.6", "whiteSpace": "pre-line"}),
            ], style={"flex": "1.4", "background": SURFACE,
                       "border": f"2px solid {GREEN}", "borderRadius": "10px",
                       "padding": "22px"}),
        ], style={"display": "flex", "gap": "16px"}),
    ),

    # ══ ETAPA 4 ══════════════════════════════════════════════════════════════
    acto("04", "PCA Composicional sobre CLR",
         f"CP1 explica {VE[0]:.1f}% · CP2 explica {VE[1]:.1f}% · "
         f"Total acumulado 2 CP: {VA[1]:.1f}% · Kaiser: {N_KAISER} componentes",
         color=PURPLE),

    card(
        T("Gráfico 6 · Scree Plot · Varianza y Eigenvalores",
          f"Los últimos 2 eigenvalores son ≈0 por la restricción composicional (rank = p-1 = {len(SECTORES)-1})"),
        narr(
            f"El PCA sobre los datos CLR estandarizados identifica {N_KAISER} componentes "
            f"con eigenvalor > 1 (criterio Kaiser). CP1 ({VE[0]:.1f}%) captura el eje "
            f"principal de diferenciación en la priorización del gasto, mientras que "
            f"CP2 ({VE[1]:.1f}%) captura una dimensión secundaria ortogonal. "
            f"Los últimos 2 eigenvalores son exactamente 0 — esto es esperado en datos "
            f"composicionales con p variables: la transformación CLR impone rank = p-1."
        ),
        dcc.Graph(figure=_fig_scree, config={"displayModeBar": False}),
    ),

    card(
        T("Gráfico 7 · Mapa de Cargas Factoriales",
          "Contribución de cada sector a cada componente principal"),
        narr(
            f"CP1 opone los sectores de LIBRE ASIGNACIÓN (cultura, deporte, libre inversión, "
            f"con cargas positivas) frente a LIBRE DESTINACIÓN (carga negativa). "
            f"Esto indica que CP1 captura el trade-off entre inversión discrecional vs. "
            f"capacidad fiscal autónoma. CP2 contrasta los sectores SGP "
            f"(educación, salud, agua) con los sectores de libre asignación, "
            f"capturando la dimensión de priorización de servicios básicos."
        ),
        dcc.Graph(figure=_fig_loadings, config={"displayModeBar": False}),
    ),

    card(
        T("Gráfico 8 · Biplot PCA Composicional",
          "Puntos = departamentos · Vectores = dirección de máxima variación de cada sector · "
          "Color = región geográfica"),
        narr(
            "El biplot permite leer simultáneamente los scores (posición de cada departamento "
            "en el espacio de componentes) y las cargas (dirección de influencia de cada sector). "
            "Departamentos en la misma dirección que un vector priorizan más ese sector. "
            "Departamentos en la dirección opuesta priorizan menos. Bogotá aparece aislado "
            "en el extremo del eje de libre destinación por su valor cero en ese sector."
        ),
        dcc.Graph(figure=_fig_biplot, config={"displayModeBar": False}),
    ),

    # ══ ETAPA 5 ══════════════════════════════════════════════════════════════
    acto("05", "Análisis de Clúster · Clasificación de Patrones de Priorización",
         f"K-Means sobre CP1-CP2 · Selección por silueta · k={K_FINAL} clusters · "
         f"Silueta={max(SILHOUETTES):.3f}",
         color=RED),

    card(
        T("Gráfico 9 · Selección del Número de Clusters",
          "Método del codo (inercia) y coeficiente de silueta"),
        narr(
            f"El coeficiente de silueta alcanza su máximo en k=2 ({SILHOUETTES[0]:.3f}), "
            f"pero k={K_FINAL} ofrece mayor riqueza interpretativa con silueta "
            f"aceptable ({SILHOUETTES[K_FINAL-2]:.3f}). El método del codo confirma "
            f"que k=3 es un punto de inflexión razonable en la curva de inercia. "
            f"La elección de k=3 equilibra parsimonia estadística e "
            f"interpretabilidad sustantiva para la política pública."
        ),
        dcc.Graph(figure=_fig_codo, config={"displayModeBar": False}),
    ),

    card(
        T(f"Gráfico 10 · Clusters K-Means (k={K_FINAL}) en Espacio PCA",
          "Cada punto = un departamento · Color = cluster asignado"),
        narr(
            "La separación visual de los clusters en el espacio PCA confirma la "
            "coherencia del agrupamiento. Los clusters reflejan patrones distintos "
            "de priorización del gasto y no se solapan significativamente, "
            "lo que indica que las diferencias entre grupos son sustantivas "
            "y no artefactos del método."
        ),
        dcc.Graph(figure=_fig_scatter_cl, config={"displayModeBar": False}),
    ),

    # ══ ETAPA 6 ══════════════════════════════════════════════════════════════
    acto("06", "Caracterización de los Clusters",
         "¿Qué distingue a cada grupo? · Perfiles de priorización · "
         "Interpretación sustantiva para política pública",
         color=GREEN),

    card(
        T("Gráfico 11 · Radar · Perfil de Priorización por Cluster",
          "Valores en % del gasto total · Permite comparar patrones relativos"),
        narr(
            "Los perfiles composicionales revelan patrones diferenciados de "
            "priorización. Las diferencias en los sectores de libre asignación "
            "son las más informativas, pues reflejan decisiones deliberadas de "
            "los gobiernos departamentales, a diferencia de los sectores SGP "
            "que están en gran medida determinados por fórmulas de transferencia."
        ),
        dcc.Graph(figure=_fig_perfil_cl, config={"displayModeBar": False}),
    ),

    card(
        T("Gráfico 12 · Heatmap de Proporciones Medias por Cluster",
          "Permite identificar sectores diferenciadores entre grupos"),
        dcc.Graph(figure=_fig_heatmap_cl, config={"displayModeBar": False}),
        html.Div(style={"marginTop": "20px"}),
        T("Tabla · Composición y Perfil de cada Cluster"),
        _tabla_cl,
    ),

    # ══ NOTA METODOLÓGICA ══════════════════════════════════════════════════
    html.Div([
        html.Span("NOTA METODOLÓGICA · ANÁLISIS DE DATOS COMPOSICIONALES", style={
            "color": ORANGE, "fontSize": "9px", "letterSpacing": "0.18em",
            "fontFamily": "IBM Plex Mono", "fontWeight": "600",
        }),
        html.Div([
            html.P("Fundamento: Aitchison (1986). The Statistical Analysis of Compositional Data.",
                   style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                          "lineHeight": "1.6", "marginBottom": "6px"}),
            html.P("Reemplazo de ceros: Martín-Fernández et al. (2003). Dealing with zeros and missing values in compositional data sets using nonparametric imputation.",
                   style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                          "lineHeight": "1.6", "marginBottom": "6px"}),
            html.P("Transformación CLR: log(xᵢ) − (1/p)·Σlog(xⱼ) · La restricción Σ CLR = 0 implica que la matriz de covarianza CLR tiene rango p-1. Los últimos eigenvalores son exactamente cero — esto es una propiedad matemática de la transformación, no un problema de los datos.",
                   style={"color": TEXT2, "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                          "lineHeight": "1.6", "marginBottom": "0"}),
        ], style={"borderLeft": f"3px solid {ORANGE}", "paddingLeft": "14px",
                   "marginTop": "10px"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {ORANGE}40",
        "borderRadius": "8px", "padding": "20px 24px", "marginBottom": "32px",
    }),

], style={"padding": "30px 40px", "background": BG, "minHeight": "100vh"},
   className="page-fade")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN TABLA DESCRIPTIVOS PROPORCIONES
# ═══════════════════════════════════════════════════════════════════════════════
def _tabla_descriptivos_prop():
    rows = []
    for s, label in zip(PROP_COLS, LABELS):
        v = df[s] * 100
        rows.append({
            "Variable":   label,
            "Media (%)":  f"{v.mean():.2f}",
            "Mediana (%)":f"{v.median():.2f}",
            "Mín (%)":    f"{v.min():.2f}",
            "Máx (%)":    f"{v.max():.2f}",
            "DE (%)":     f"{v.std():.2f}",
            "CV (%)":     f"{v.std()/v.mean()*100:.1f}",
            "Asimetría":  f"{stats.skew(v):.3f}",
        })
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tdf.columns],
        style_table={"overflowX": "auto"},
        style_cell={"background": CARD, "color": TEXT1,
                     "border": f"1px solid {BORDER}",
                     "fontFamily": "IBM Plex Mono", "fontSize": "10px",
                     "padding": "9px 13px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "Variable"}, "textAlign": "left",
             "color": ACCENT, "fontWeight": "600", "minWidth": "140px"},
            {"if": {"column_id": "CV (%)"},
             "color": RED, "fontWeight": "600"},
        ],
        style_header={"background": SURFACE, "color": ACCENT, "fontWeight": "600",
                       "border": f"1px solid {BORDER}",
                       "fontFamily": "IBM Plex Sans", "fontSize": "10px"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native",
    )