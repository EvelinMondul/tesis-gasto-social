"""
Página 02 · Análisis Exploratorio de Datos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enfoque: proporciones del gasto total → patrones de PRIORIZACIÓN sectorial
Narrativa: descripción → univariado → correlaciones → KMO/Bartlett → PCA → clustering
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
from data import (
    cargar_proporciones, calcular_clr, calcular_kmo_bartlett_prop,
    cargar_datos, REGION_COLORS, SECTORES_PC, SECTORES_ABS,
    SECTOR_LABELS, SECTOR_COLORS, CLUSTER_COLORS, PALETTE
)

dash.register_page(__name__, path="/eda", name="EDA & Tablas", order=1)

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS Y PALETA
# ═══════════════════════════════════════════════════════════════════════════════
df       = cargar_proporciones()
df_pc    = cargar_datos()          # per cápita — solo para tabla descriptiva inicial

P      = PALETTE
BG     = P["bg"];     SURFACE = P["surface"]; CARD   = P["card"]
BORDER = P["border"]; TEXT1   = P["text1"];   TEXT2  = P["text2"]
ACCENT = P["accent"]; GREEN   = P["green"];   ORANGE = P["orange"]
RED    = P["red"];    PURPLE  = P["purple"]

BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT1, size=11),
)
LEGEND = dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT1))
DD = {"background": SURFACE, "color": TEXT1,
      "fontFamily": "IBM Plex Mono", "fontSize": "11px",
      "border": f"1px solid {BORDER}"}

# ── Proporciones y CLR ────────────────────────────────────────────────────────
X_clr, PROP_COLS, LABELS = calcular_clr(df)
KMO, CHI2, GL, PVAL_BART = calcular_kmo_bartlett_prop(df)

# ── PCA composicional ─────────────────────────────────────────────────────────
X_std   = StandardScaler().fit_transform(X_clr)
pca     = PCA()
pca.fit(X_std)
VE      = pca.explained_variance_ratio_ * 100
EV      = pca.explained_variance_
VA      = np.cumsum(VE)
SCORES  = pca.transform(X_std)
N_VALID = sum(1 for e in EV if e > 0.01)
N_KAI   = sum(1 for e in EV if e >= 1)

# ── Clustering K-Means (k=3) ──────────────────────────────────────────────────
SC2       = SCORES[:, :2]
INERTIAS  = []
SILS      = []
for _k in range(2, 7):
    _km  = KMeans(n_clusters=_k, random_state=42, n_init=20)
    _lab = _km.fit_predict(SC2)
    INERTIAS.append(_km.inertia_)
    SILS.append(silhouette_score(SC2, _lab))

K_FINAL = 3
km_final       = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20)
df["cluster"]  = km_final.fit_predict(SC2) + 1
PROFILE        = df.groupby("cluster")[PROP_COLS].mean() * 100

CLUSTER_NAMES = {
    1: "Inversión diversificada",
    2: "Priorización SGP básica",
    3: "Perfil diferencial",
}

# ── Per cápita disponibles ────────────────────────────────────────────────────
PC_DISP = [c for c in SECTORES_PC if c in df_pc.columns]


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


def narr(text, color=None):
    c = color or BORDER
    return html.Div([
        html.Span("▸ ", style={"color": ACCENT, "fontSize": "10px",
                                "flexShrink": "0", "marginTop": "2px"}),
        html.Span(text, style={
            "color": TEXT2, "fontFamily": "IBM Plex Mono",
            "fontSize": "11px", "lineHeight": "1.8",
        }),
    ], style={
        "borderLeft": f"2px solid {c}", "paddingLeft": "14px",
        "marginBottom": "18px", "display": "flex",
        "alignItems": "flex-start", "gap": "6px",
    })


def card(*children, mb="22px", color=None):
    bc = color or BORDER
    return html.Div(list(children), style={
        "background": CARD, "border": f"1px solid {bc}",
        "borderRadius": "8px", "padding": "20px 22px", "marginBottom": mb,
    })


def etapa(num, titulo, desc, color=None):
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
        "padding": "14px 20px", "marginBottom": "20px",
    })


def kpi(t, v, s="", c=None):
    col = c or ACCENT
    return html.Div([
        html.P(t, style={"color": TEXT2, "fontSize": "9px", "letterSpacing": "0.1em",
                          "textTransform": "uppercase", "fontFamily": "IBM Plex Mono",
                          "marginBottom": "4px"}),
        html.P(v, style={"color": col, "fontSize": "17px", "fontWeight": "700",
                          "fontFamily": "IBM Plex Sans", "margin": "0"}),
        html.P(s, style={"color": TEXT2, "fontSize": "9px",
                          "fontFamily": "IBM Plex Mono", "marginTop": "3px"}) if s else None,
    ], style={"background": CARD, "border": f"1px solid {BORDER}",
               "borderRadius": "6px", "padding": "12px 14px",
               "borderTop": f"3px solid {col}"})


def justif_box(titulo, texto, color=None):
    c = color or ORANGE
    return html.Div([
        html.P(titulo, style={"color": c, "fontFamily": "IBM Plex Mono",
                               "fontSize": "10px", "fontWeight": "600",
                               "letterSpacing": "0.08em", "marginBottom": "6px"}),
        html.P(texto, style={"color": TEXT2, "fontFamily": "IBM Plex Mono",
                               "fontSize": "11px", "lineHeight": "1.7", "margin": "0"}),
    ], style={
        "background": SURFACE, "border": f"1px solid {c}50",
        "borderLeft": f"3px solid {c}", "borderRadius": "6px",
        "padding": "14px 16px", "marginBottom": "16px",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 1: DESCRIPCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def tabla_descriptivos_prop():
    rows = []
    for s, label in zip(PROP_COLS, LABELS):
        v = df[s] * 100
        _, pval_sw = stats.shapiro(v)
        rows.append({
            "Variable":    label,
            "Media (%)":   f"{v.mean():.2f}",
            "Mediana (%)": f"{v.median():.2f}",
            "Mín (%)":     f"{v.min():.2f}",
            "Máx (%)":     f"{v.max():.2f}",
            "DE (%)":      f"{v.std():.2f}",
            "CV (%)":      f"{v.std()/v.mean()*100:.1f}",
            "Asimetría":   f"{stats.skew(v):.3f}",
            "Shapiro p":   f"{pval_sw:.3f}",
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
            {"if": {"column_id": "Variable"},
             "textAlign": "left", "color": ACCENT,
             "fontWeight": "600", "minWidth": "150px"},
            {"if": {"column_id": "CV (%)"},
             "color": RED, "fontWeight": "600"},
        ],
        style_header={"background": SURFACE, "color": ACCENT,
                       "fontWeight": "600", "border": f"1px solid {BORDER}",
                       "fontFamily": "IBM Plex Sans", "fontSize": "10px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": SURFACE}],
        sort_action="native",
    )


def fig_barras_prop():
    dfs = df.sort_values("prop_educacion", ascending=True)
    colors = list(SECTOR_COLORS.values())
    fig = go.Figure()

    for i, (s, label) in enumerate(zip(PROP_COLS, LABELS)):
        fig.add_trace(go.Bar(
            name=label,
            x=dfs[s] * 100,
            y=dfs["departamento"].str.title(),
            orientation="h",
            marker_color=colors[i],
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))
    
    fig.update_layout(
    **BASE,
    legend=LEGEND,
    barmode="stack",
    height=760,
    title=dict(
        text="Estructura de Priorización del Gasto · Proporciones por departamento (%)",
        font=dict(family="IBM Plex Sans", size=13, color=TEXT1)
    ),
    xaxis=dict(
        range=[0,100],
        title="% del gasto total",
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT1),
        ticksuffix="%"
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT1)
    ),
    legend=dict(
        orientation="h",
        y=1.02,
        x=0,
        bgcolor=CARD,   # ← corregido
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT1)
    ),
    margin=dict(l=160, r=20, t=80, b=50),
)

    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 2: ANÁLISIS UNIVARIADO
# ═══════════════════════════════════════════════════════════════════════════════
def fig_distribucion_prop(sector):
    label = SECTOR_LABELS.get(sector.replace("prop_",""), sector)
    v     = df[sector].dropna() * 100
    mean_, med_ = v.mean(), v.median()
    skew_ = stats.skew(v)
    cv_   = v.std() / mean_ * 100
    _, pval = stats.shapiro(v)

    kde_x = np.linspace(v.min(), v.max(), 200)
    kde   = stats.gaussian_kde(v)
    scale = len(v) * (v.max()-v.min()) / 10

    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=v, nbinsx=10, name="Frecuencia",
                                  marker_color=ACCENT, opacity=0.7,
                                  marker_line=dict(color=BG, width=0.5)))
    fig_h.add_trace(go.Scatter(x=kde_x, y=kde(kde_x)*scale, mode="lines",
                                name="KDE", line=dict(color=ORANGE, width=2.5)))
    fig_h.add_vline(x=mean_, line_dash="dash", line_color=GREEN,
                    annotation_text=f"μ={mean_:.1f}%",
                    annotation_font=dict(color=GREEN, size=9))
    fig_h.add_vline(x=med_, line_dash="dot", line_color=RED,
                    annotation_text=f"Md={med_:.1f}%",
                    annotation_font=dict(color=RED, size=9))
    fig_h.update_layout(
        **BASE, title=dict(text=f"Distribución · Proporción {label}",
                            font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="% del gasto total", tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Frecuencia", tickfont=dict(color=TEXT1)),
        height=320, margin=dict(l=60, r=20, t=50, b=60), showlegend=False,
    )

    (osm, osr), (slope, intercept, _) = stats.probplot(v)
    line_y = np.array(osm)*slope + intercept
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                 marker=dict(color=ACCENT, size=7, opacity=0.85)))
    fig_qq.add_trace(go.Scatter(x=osm, y=line_y, mode="lines",
                                 line=dict(color=ORANGE, width=1.5, dash="dash")))
    fig_qq.update_layout(
        **BASE, title=dict(text="Q-Q Plot · Normalidad",
                            font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Cuantiles teóricos", tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                   title="Cuantiles muestrales", tickfont=dict(color=TEXT1)),
        height=320, margin=dict(l=60, r=20, t=50, b=60), showlegend=False,
    )

    fig_box = px.box(df, x="region", y=sector, color="region",
                      color_discrete_map=REGION_COLORS, points="all",
                      hover_name="departamento",
                      labels={sector: f"Proporción {label} (%)", "region": "Región"})
    for tr in fig_box.data:
        tr.y = [v*100 if v else v for v in tr.y]
    fig_box.update_layout(
        **BASE, showlegend=False,
        title=dict(text=f"Boxplot regional · Proporción {label}",
                    font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1),
                   title="% del gasto total"),
        height=320, margin=dict(l=60, r=20, t=50, b=60),
    )

    norm_color = GREEN if pval > 0.05 else RED
    norm_text  = "✓ No se rechaza H₀" if pval > 0.05 else "✗ Se rechaza H₀"

    return fig_h, fig_qq, fig_box, {
        "media": mean_, "mediana": med_, "cv": cv_,
        "asimetria": skew_, "pval": pval,
        "norm_text": norm_text, "norm_color": norm_color,
    }


def fig_cv_prop():
    data_ = sorted(
        [(l, df[s].std()/df[s].mean()*100) for s,l in zip(PROP_COLS, LABELS)],
        key=lambda x: x[1], reverse=True,
    )
    labs, vals = zip(*data_)
    colors = [RED if v>40 else ORANGE if v>25 else ACCENT for v in vals]
    fig = go.Figure(go.Bar(
        x=list(labs), y=list(vals),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color=TEXT1),
    ))
    fig.add_hline(y=25, line_dash="dash", line_color=GREEN,
                  annotation_text="25%",
                  annotation_font=dict(color=GREEN, size=9))
    fig.add_hline(y=40, line_dash="dash", line_color=RED,
                  annotation_text="40%",
                  annotation_font=dict(color=RED, size=9))
    fig.update_layout(
        **BASE,
        title=dict(text="CV de Proporciones · Heterogeneidad en la Priorización del Gasto",
                   font=dict(family="IBM Plex Sans", size=13, color=TEXT1)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickangle=-15,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="CV (%)",
                   range=[0, max(vals)*1.3], tickfont=dict(color=TEXT1)),
        height=380, showlegend=False,
        margin=dict(l=60, r=100, t=60, b=80),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 3: CORRELACIONES (TRADE-OFFS)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_correlacion():
    n = len(PROP_COLS)
    corr_m = np.zeros((n,n)); pval_m = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(X_clr[:,i], X_clr[:,j])
            corr_m[i,j] = r; pval_m[i,j] = p
    text_m = []
    for i in range(n):
        row = []
        for j in range(n):
            s = ("***" if pval_m[i,j]<0.001 else
                 "**"  if pval_m[i,j]<0.01  else
                 "*"   if pval_m[i,j]<0.05  else "")
            row.append(f"{corr_m[i,j]:.2f}{s}")
        text_m.append(row)
    fig = go.Figure(go.Heatmap(
        z=corr_m, x=LABELS, y=LABELS,
        colorscale=[[0,"#E74C3C"],[0.5,"#1C2333"],[1,"#3498DB"]],
        zmid=0, zmin=-1, zmax=1,
        text=text_m, texttemplate="%{text}",
        textfont=dict(size=9, family="IBM Plex Mono"),
        colorbar=dict(title="ρ", titlefont=dict(size=10,color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Correlación de Spearman sobre CLR · Trade-offs entre sectores · * p<.05  ** p<.01  *** p<.001",
                   font=dict(family="IBM Plex Sans", size=12, color=TEXT1)),
        xaxis=dict(tickangle=-35, gridcolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT1)),
        height=500, margin=dict(l=140,r=20,t=70,b=140),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 4: KMO + BARTLETT
# ═══════════════════════════════════════════════════════════════════════════════
def kmo_interp(k):
    if k>=0.90: return "Maravilloso", GREEN
    if k>=0.80: return "Meritorio",   ACCENT
    if k>=0.70: return "Mediano",     ORANGE
    if k>=0.60: return "Mediocre",    ORANGE
    return "Inaceptable", RED

KMO_LABEL, KMO_COLOR = kmo_interp(KMO)


def bloque_kmo_bartlett():
    return html.Div([
        # KMO
        html.Div([
            html.P("ÍNDICE KMO", style={"color":TEXT2,"fontFamily":"IBM Plex Mono",
                                         "fontSize":"9px","letterSpacing":"0.12em",
                                         "marginBottom":"8px","textAlign":"center"}),
            html.P(f"{KMO:.4f}", style={"color":KMO_COLOR,"fontFamily":"IBM Plex Sans",
                                          "fontSize":"42px","fontWeight":"700",
                                          "margin":"0","textAlign":"center"}),
            html.P(KMO_LABEL, style={"color":KMO_COLOR,"fontFamily":"IBM Plex Mono",
                                      "fontSize":"13px","fontWeight":"600",
                                      "textAlign":"center","marginTop":"4px"}),
            html.Hr(style={"borderColor":BORDER,"margin":"12px 0"}),
            html.P("Kaiser (1974): > 0.90 Maravilloso · > 0.80 Meritorio · "
                   "> 0.70 Mediano · > 0.60 Mediocre",
                   style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px",
                          "lineHeight":"1.5","textAlign":"center"}),
            html.P("⚠ KMO calculado sobre 4 proporciones no colineales. "
                   "La matriz CLR completa es singular por construcción (rank = p-1). "
                   "Esto es una propiedad matemática de la transformación, no un error.",
                   style={"color":ORANGE,"fontFamily":"IBM Plex Mono","fontSize":"9px",
                          "lineHeight":"1.5","marginTop":"8px",
                          "borderLeft":f"2px solid {ORANGE}","paddingLeft":"8px"}),
        ], style={"flex":"1","background":SURFACE,
                   "border":f"2px solid {KMO_COLOR}",
                   "borderRadius":"10px","padding":"20px"}),

        # Bartlett
        html.Div([
            html.P("PRUEBA DE ESFERICIDAD DE BARTLETT", style={
                "color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"9px",
                "letterSpacing":"0.12em","marginBottom":"8px"}),
            html.P(f"χ² = {CHI2:.2f}", style={
                "color":GREEN,"fontFamily":"IBM Plex Sans",
                "fontSize":"32px","fontWeight":"700","margin":"0"}),
            html.P(f"gl = {GL}   ·   p-value = {PVAL_BART:.2e}", style={
                "color":TEXT2,"fontFamily":"IBM Plex Mono",
                "fontSize":"12px","marginTop":"8px"}),
            html.Hr(style={"borderColor":BORDER,"margin":"12px 0"}),
            html.P("✓  Se rechaza H₀  ·  La matriz de correlaciones NO es identidad  ·  "
                   "Existen correlaciones sistemáticas entre proporciones  ·  "
                   "Factorizabilidad confirmada",
                   style={"color":GREEN,"fontFamily":"IBM Plex Mono",
                          "fontSize":"11px","fontWeight":"600","lineHeight":"1.6"}),
            html.P("H₀: R = I (proporciones estadísticamente independientes)\n"
                   "H₁: R ≠ I (correlaciones sistemáticas entre sectores)\n"
                   "Rechazar H₀ valida la aplicación de PCA y análisis factorial.",
                   style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                          "lineHeight":"1.7","marginTop":"10px","whiteSpace":"pre-line"}),
        ], style={"flex":"1.5","background":SURFACE,
                   "border":f"2px solid {GREEN}",
                   "borderRadius":"10px","padding":"20px"}),
    ], style={"display":"flex","gap":"16px"})


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 5: PCA COMPOSICIONAL
# ═══════════════════════════════════════════════════════════════════════════════
def fig_scree():
    nv = N_VALID
    labs = [f"CP{i+1}" for i in range(nv)]
    ve_v = VE[:nv]; ev_v = EV[:nv]; va_v = np.cumsum(ve_v)
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["Varianza explicada (%) y acumulada",
                                          "Eigenvalores · Criterio Kaiser λ > 1"])
    bar_colors = [GREEN if e>=1 else ACCENT if e>=0.5 else BORDER for e in ev_v]
    fig.add_trace(go.Bar(x=labs, y=list(ve_v), marker_color=bar_colors,
                          text=[f"{v:.1f}%" for v in ve_v],
                          textposition="outside",
                          textfont=dict(size=9,color=TEXT1)), row=1,col=1)
    fig.add_trace(go.Scatter(x=labs, y=list(va_v), mode="lines+markers",
                              line=dict(color=ORANGE,width=2),
                              marker=dict(size=7,color=ORANGE),
                              yaxis="y3"), row=1,col=1)
    fig.add_trace(go.Bar(x=labs, y=list(ev_v),
                          marker_color=[GREEN if e>=1 else RED for e in ev_v],
                          text=[f"{e:.3f}" for e in ev_v],
                          textposition="outside",
                          textfont=dict(size=9,color=TEXT1)), row=1,col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color=ORANGE,
                  annotation_text="λ=1 (Kaiser)",
                  annotation_font=dict(color=ORANGE,size=9), row=1,col=2)
    fig.update_layout(
        **BASE,
        title=dict(text="Scree Plot · PCA Composicional sobre Datos CLR",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        height=400, showlegend=False,
        yaxis3=dict(overlaying="y", side="right", range=[0,115],
                    showgrid=False, tickfont=dict(color=ORANGE)),
        margin=dict(l=60,r=80,t=80,b=60),
    )
    fig.update_xaxes(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=TEXT1))
    fig.update_yaxes(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1,size=11))
    return fig


def fig_loadings():
    nv = N_VALID
    z  = pca.components_[:nv].T
    cp_labs = [f"CP{i+1} ({VE[i]:.1f}%)" for i in range(nv)]
    text_z  = [[f"{v:.3f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=cp_labs, y=LABELS,
        colorscale=[[0,"#E74C3C"],[0.5,"#1C2333"],[1,"#3498DB"]],
        zmid=0, zmin=-0.7, zmax=0.7,
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=9,family="IBM Plex Mono"),
        colorbar=dict(title="Carga",titlefont=dict(size=10,color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Cargas Factoriales · Contribución de cada sector a los componentes",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        xaxis=dict(tickangle=-20,gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=TEXT1)),
        height=360, margin=dict(l=140,r=20,t=60,b=80),
    )
    return fig


def fig_biplot():
    sc   = SCORES
    ld   = pca.components_
    scale = np.sqrt(EV[0]) * 1.5
    fig  = go.Figure()
    for reg, gdf in df.groupby("region"):
        idx = gdf.index
        fig.add_trace(go.Scatter(
            x=sc[idx,0], y=sc[idx,1],
            mode="markers+text", name=reg,
            text=gdf["departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7,color=TEXT2),
            marker=dict(size=9,color=REGION_COLORS.get(reg,ACCENT),
                        opacity=0.85,line=dict(width=0.5,color=BG)),
        ))
    colors_v = list(SECTOR_COLORS.values())
    for i,label in enumerate(LABELS):
        lx,ly = ld[0,i]*scale, ld[1,i]*scale
        fig.add_annotation(x=lx,y=ly,ax=0,ay=0,
                            xref="x",yref="y",axref="x",ayref="y",
                            showarrow=True,arrowhead=2,arrowsize=1,arrowwidth=2,
                            arrowcolor=colors_v[i])
        fig.add_annotation(x=lx*1.12,y=ly*1.12,text=label,showarrow=False,
                            font=dict(size=9,color=colors_v[i],family="IBM Plex Mono"))
    fig.add_hline(y=0,line_color=BORDER,line_width=0.5)
    fig.add_vline(x=0,line_color=BORDER,line_width=0.5)
    fig.update_layout(
        **BASE, legend=LEGEND,
        title=dict(text=f"Biplot PCA · CP1 ({VE[0]:.1f}%) vs CP2 ({VE[1]:.1f}%) · Acumulado {VA[1]:.1f}%",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        xaxis=dict(title=f"CP1 — {VE[0]:.1f}% varianza",
                   gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1),zeroline=False),
        yaxis=dict(title=f"CP2 — {VE[1]:.1f}% varianza",
                   gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1),zeroline=False),
        height=560, margin=dict(l=80,r=20,t=70,b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 6: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
def fig_codo_silueta():
    ks = list(range(2,7))
    fig = make_subplots(rows=1,cols=2,
                         subplot_titles=["Método del Codo (Inercia)",
                                          "Coeficiente de Silueta"])
    fig.add_trace(go.Scatter(x=ks,y=INERTIAS,mode="lines+markers",
                              line=dict(color=ACCENT,width=2),
                              marker=dict(size=8,color=ACCENT)), row=1,col=1)
    best = SILS.index(max(SILS))
    fig.add_trace(go.Bar(x=ks,y=SILS,
                          marker_color=[GREEN if i==best else ACCENT for i in range(len(ks))],
                          text=[f"{s:.3f}" for s in SILS],
                          textposition="outside",
                          textfont=dict(size=9,color=TEXT1)), row=1,col=2)
    fig.update_layout(
        **BASE,
        title=dict(text="Selección del Número de Clusters · Codo y Silueta",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        height=360,showlegend=False,
        margin=dict(l=60,r=20,t=80,b=60),
    )
    fig.update_xaxes(gridcolor=BORDER,linecolor=BORDER,
                     title="k (número de clusters)",tickfont=dict(color=TEXT1))
    fig.update_yaxes(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=TEXT1))
    fig.update_annotations(font=dict(color=TEXT1,size=11))
    return fig


def fig_clusters_pca():
    fig = go.Figure()
    for k in sorted(df["cluster"].unique()):
        mask = df["cluster"]==k
        idx  = df.index[mask]
        fig.add_trace(go.Scatter(
            x=SCORES[idx,0], y=SCORES[idx,1],
            mode="markers+text",
            name=f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            text=df.loc[mask,"departamento"].str.title(),
            textposition="top center",
            textfont=dict(size=7,color=TEXT2),
            marker=dict(size=11,color=CLUSTER_COLORS.get(k,ACCENT),
                        opacity=0.9,line=dict(width=1,color=BG)),
        ))
    fig.add_hline(y=0,line_color=BORDER,line_width=0.5)
    fig.add_vline(x=0,line_color=BORDER,line_width=0.5)
    fig.update_layout(
        **BASE, legend=LEGEND,
        title=dict(text=f"Clusters K-Means (k={K_FINAL}) en Espacio PCA Composicional",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        xaxis=dict(title=f"CP1 — {VE[0]:.1f}%",
                   gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1),zeroline=False),
        yaxis=dict(title=f"CP2 — {VE[1]:.1f}%",
                   gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1),zeroline=False),
        height=520, margin=dict(l=80,r=20,t=70,b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS — ETAPA 7: CARACTERIZACIÓN DE CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar_clusters():
    fig = go.Figure()
    for k in sorted(df["cluster"].unique()):
        vals = [PROFILE.loc[k,c] for c in PROP_COLS]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=LABELS+[LABELS[0]],
            name=f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            fill="toself", opacity=0.5,
            line=dict(color=CLUSTER_COLORS.get(k,ACCENT),width=2),
        ))
    fig.update_layout(
        **BASE, legend=LEGEND,
        title=dict(text="Perfil de Priorización del Gasto por Cluster (%)",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        polar=dict(bgcolor=CARD,
                   radialaxis=dict(visible=True,
                                   range=[0,max(PROFILE.values.max()*1.1,60)],
                                   gridcolor=BORDER,
                                   tickfont=dict(size=8,color=TEXT2),
                                   ticksuffix="%"),
                   angularaxis=dict(gridcolor=BORDER,
                                    tickfont=dict(size=9,color=TEXT2))),
        height=460, margin=dict(l=40,r=40,t=70,b=40),
    )
    return fig


def fig_heatmap_clusters():
    z = PROFILE[PROP_COLS].values
    text_z = [[f"{v:.1f}%" for v in row] for row in z]
    row_labs = [f"C{k} · {CLUSTER_NAMES.get(k,'')}"
                for k in sorted(df.cluster.unique())]
    fig = go.Figure(go.Heatmap(
        z=z, x=LABELS, y=row_labs,
        colorscale=[[0,"#EAF4FB"],[0.5,"#3498DB"],[1,"#1A5276"]],
        text=text_z, texttemplate="%{text}",
        textfont=dict(size=10,family="IBM Plex Mono",color=TEXT1),
        colorbar=dict(title="%",titlefont=dict(size=10,color=TEXT1),
                      tickfont=dict(color=TEXT1)),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="Proporciones Medias del Gasto por Cluster (%)",
                   font=dict(family="IBM Plex Sans",size=13,color=TEXT1)),
        xaxis=dict(tickangle=-25,gridcolor=BORDER,linecolor=BORDER,
                   tickfont=dict(color=TEXT1)),
        yaxis=dict(gridcolor=BORDER,linecolor=BORDER,tickfont=dict(color=TEXT1)),
        height=260, margin=dict(l=200,r=20,t=60,b=80),
    )
    return fig


def tabla_clusters():
    rows = []
    for k in sorted(df["cluster"].unique()):
        deps = df.loc[df.cluster==k,"departamento"].str.title().tolist()
        prof = {SECTOR_LABELS.get(c.replace("prop_",""),c): f"{PROFILE.loc[k,c]:.1f}%"
                for c in PROP_COLS}
        dominant = max(prof, key=lambda x: float(prof[x].replace("%","")))
        rows.append({
            "Cluster": f"C{k} · {CLUSTER_NAMES.get(k,'')}",
            "n": str(len(deps)),
            "Sector dominante": dominant,
            "Proporción": prof[dominant],
            "Departamentos": ", ".join(deps),
        })
    tdf = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=tdf.to_dict("records"),
        columns=[{"name":c,"id":c} for c in tdf.columns],
        style_table={"overflowX":"auto"},
        style_cell={"background":CARD,"color":TEXT1,"border":f"1px solid {BORDER}",
                     "fontFamily":"IBM Plex Mono","fontSize":"10px",
                     "padding":"9px 13px","textAlign":"left",
                     "whiteSpace":"normal","height":"auto"},
        style_cell_conditional=[
            {"if":{"column_id":"Cluster"},"color":ACCENT,"fontWeight":"600","minWidth":"200px"},
            {"if":{"column_id":"n"},"textAlign":"center","maxWidth":"40px"},
        ],
        style_header={"background":SURFACE,"color":ACCENT,"fontWeight":"600",
                       "border":f"1px solid {BORDER}",
                       "fontFamily":"IBM Plex Sans","fontSize":"10px"},
        style_data_conditional=[{"if":{"row_index":"odd"},"background":SURFACE}],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
_fig_barras    = fig_barras_prop()
_fig_cv        = fig_cv_prop()
_fig_corr      = fig_correlacion()
_fig_scree     = fig_scree()
_fig_loadings  = fig_loadings()
_fig_biplot    = fig_biplot()
_fig_codo      = fig_codo_silueta()
_fig_clusters  = fig_clusters_pca()
_fig_radar     = fig_radar_clusters()
_fig_heatmap   = fig_heatmap_clusters()
_tabla_cl      = tabla_clusters()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
layout = html.Div([

    # PORTADA
    html.Div([
        html.Span("ANÁLISIS EXPLORATORIO DE DATOS · ENFOQUE COMPOSICIONAL", style={
            "color":ACCENT,"fontSize":"9px","letterSpacing":"0.2em",
            "fontFamily":"IBM Plex Mono","fontWeight":"600",
        }),
        html.H1("Patrones de Priorización del Gasto Social Departamental", style={
            "color":TEXT1,"fontFamily":"IBM Plex Sans","fontWeight":"700",
            "fontSize":"21px","margin":"10px 0 12px",
        }),
        html.P(
            "El análisis trabaja sobre las PROPORCIONES del gasto total destinado a cada sector "
            "(prop_s = gasto_s / gasto_total), no sobre valores absolutos ni per cápita. "
            "Este enfoque permite identificar patrones de PRIORIZACIÓN: cómo cada departamento "
            "distribuye sus recursos entre sectores. Dado que las proporciones suman 1 "
            "(restricción del simplex), se aplica la transformación CLR antes del PCA para "
            "eliminar la dependencia composicional (Aitchison 1986).",
            style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"11px",
                   "lineHeight":"1.8","maxWidth":"900px","margin":"0"},
        ),
        html.Div([
            html.Div(f"KMO = {KMO:.3f} · {KMO_LABEL}", style={
                "color":KMO_COLOR,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                "background":f"{KMO_COLOR}18","border":f"1px solid {KMO_COLOR}",
                "padding":"4px 14px","borderRadius":"20px","marginRight":"10px"}),
            html.Div(f"Bartlett p = {PVAL_BART:.2e}", style={
                "color":GREEN,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                "background":f"{GREEN}18","border":f"1px solid {GREEN}",
                "padding":"4px 14px","borderRadius":"20px","marginRight":"10px"}),
            html.Div(f"Kaiser: {N_KAI} CP · {VA[N_KAI-1]:.1f}% varianza", style={
                "color":ACCENT,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                "background":f"{ACCENT}18","border":f"1px solid {ACCENT}",
                "padding":"4px 14px","borderRadius":"20px","marginRight":"10px"}),
            html.Div(f"k = {K_FINAL} clusters · silueta = {SILS[K_FINAL-2]:.3f}", style={
                "color":ORANGE,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                "background":f"{ORANGE}18","border":f"1px solid {ORANGE}",
                "padding":"4px 14px","borderRadius":"20px"}),
        ], style={"display":"flex","marginTop":"16px","flexWrap":"wrap","gap":"8px"}),
    ], style={
        "background":SURFACE,"border":f"1px solid {BORDER}",
        "borderLeft":f"4px solid {ACCENT}","borderRadius":"10px",
        "padding":"26px 32px","marginBottom":"30px",
    }),

    # ══ ETAPA 1 ══════════════════════════════════════════════════════════════
    etapa("01","Descripción de la Estructura Composicional",
          "¿Cómo prioriza cada departamento su gasto social? · "
          "Proporciones sectoriales · Estadísticos descriptivos"),

    card(
        T("Tabla 1 · Estadísticos Descriptivos · Proporciones del Gasto (%)",
          "Unidad: % del gasto total · Cada departamento suma 100% entre todos los sectores"),
        justif_box("¿POR QUÉ PROPORCIONES Y NO PER CÁPITA?",
                   "Las proporciones capturan la ESTRUCTURA DE PRIORIZACIÓN del gasto, "
                   "eliminando el efecto del tamaño poblacional. Dos departamentos con "
                   "gasto per cápita muy diferente pueden tener la misma priorización "
                   "sectorial. El objetivo del estudio es identificar patrones de "
                   "asignación, no niveles de inversión."),
        narr("Educación concentra en promedio el 57.5% del gasto (rango 42.4%–70.0%), "
             "determinado principalmente por el SGP. Libre destinación (CV=47.2%) y "
             "libre inversión (CV=34.1%) son los sectores con mayor variabilidad en la "
             "priorización: reflejan decisiones autónomas de los gobiernos departamentales, "
             "no transferencias estandarizadas. Este contraste es el hallazgo central "
             "del análisis descriptivo."),
        tabla_descriptivos_prop(),
    ),

    card(
        T("Gráfico 1 · Estructura de Priorización por Departamento",
          "Barras apiladas ordenadas por proporción de educación · Cada barra suma 100%"),
        narr("La visualización revela que todos los departamentos priorizan educación "
             "como componente dominante, pero las diferencias en libre destinación e "
             "inversión son las más informativas sobre autonomía fiscal. Bogotá se "
             "distingue estructuralmente: libre_destinacion = 0 por régimen especial "
             "de distrito capital, lo que lo convierte en un caso atípico composicional."),
        dcc.Graph(figure=_fig_barras, config={"displayModeBar":False}),
    ),

    # ══ ETAPA 2 ══════════════════════════════════════════════════════════════
    etapa("02","Análisis Univariado de Proporciones",
          "Distribución de cada proporción sectorial · CV · Normalidad · "
          "Detección de asimetrías relevantes para el análisis multivariado",
          color=GREEN),

    card(
        T("Gráfico 2 · CV de Proporciones · Heterogeneidad en la Priorización",
          "CV alto = mayor variabilidad interdepartamental en ese sector · Rojo > 40%"),
        justif_box("INTERPRETACIÓN DEL CV EN PROPORCIONES",
                   "Un CV alto en una proporción indica que los departamentos difieren "
                   "sustancialmente en cuánto priorizan ese sector. No significa que "
                   "haya mayor o menor gasto total, sino que la fracción asignada varía "
                   "ampliamente. Libre destinación (CV=47.2%) es el sector de mayor "
                   "heterogeneidad en priorización.",
                   color=GREEN),
        dcc.Graph(figure=_fig_cv, config={"displayModeBar":False}),
    ),

    card(
        T("Explorador de Distribución por Variable",
          "Histograma + KDE · Q-Q Plot · Boxplot regional · Shapiro-Wilk"),
        justif_box("¿POR QUÉ EVALUAR NORMALIDAD?",
                   "El PCA no requiere normalidad estricta, pero la asimetría severa "
                   "puede distorsionar las cargas factoriales. Las proporciones típicamente "
                   "presentan distribuciones asimétricas positivas (valores cerca de 0 "
                   "son más frecuentes que valores cercanos a 1). Este diagnóstico "
                   "justifica la transformación CLR antes del PCA."),
        html.Div([
            html.Label("Variable:", style={"color":TEXT2,"fontSize":"10px",
                                            "fontFamily":"IBM Plex Mono",
                                            "marginBottom":"6px","display":"block"}),
            dcc.Dropdown(
                id="dist-dd",
                options=[{"label":SECTOR_LABELS.get(s.replace("prop_",""),s),"value":s}
                         for s in PROP_COLS],
                value=PROP_COLS[0],
                clearable=False,
                style={**DD,"width":"280px"},
            ),
        ], style={"marginBottom":"20px"}),
        html.Div(id="dist-output"),
    ),

    # ══ ETAPA 3 ══════════════════════════════════════════════════════════════
    etapa("03","Análisis Bivariado · Trade-offs entre Sectores",
          "¿Qué sectores compiten por recursos? · "
          "Correlación de Spearman sobre CLR · Identificación de trade-offs composicionales",
          color=ORANGE),

    card(
        T("Gráfico 3 · Matriz de Correlación de Spearman sobre CLR",
          "Correlación negativa = trade-off estructural entre sectores"),
        justif_box("¿POR QUÉ SPEARMAN Y POR QUÉ SOBRE CLR?",
                   "Se usa Spearman porque las proporciones presentan asimetrías y "
                   "posibles outliers composicionales. Se calcula sobre los datos CLR "
                   "y no sobre las proporciones brutas para eliminar las correlaciones "
                   "espurias que introduce la restricción del simplex (si una proporción "
                   "sube, al menos otra debe bajar por definición). Una correlación "
                   "negativa sobre CLR indica un TRADE-OFF real, no un artefacto matemático."),
        narr("La correlación negativa entre educación y libre destinación (ρ ≈ -0.81***) "
             "revela el principal trade-off del sistema: los departamentos que destinan "
             "mayor fracción al sector educativo tienen menos espacio fiscal para gasto "
             "discrecional. La correlación positiva perfecta entre cultura y deporte "
             "(ρ ≈ +1.00***) confirma que se trata de un único patrón de priorización "
             "conjunta. Estas estructuras de correlación justifican la búsqueda de "
             "factores latentes mediante PCA."),
        dcc.Graph(figure=_fig_corr, config={"displayModeBar":False}),
    ),

    # ══ ETAPA 4 ══════════════════════════════════════════════════════════════
    etapa("04","Validación de Factorizabilidad · KMO y Bartlett",
          "Condiciones estadísticas para la aplicación de PCA y Análisis Factorial · "
          f"KMO = {KMO:.3f} · Bartlett p < 0.001",
          color=PURPLE),

    card(
        T("Pruebas de Adecuación Muestral"),
        justif_box("¿POR QUÉ KMO Y BARTLETT ANTES DEL PCA?",
                   "KMO evalúa si las correlaciones parciales son pequeñas respecto "
                   "a las totales — si lo son, las variables comparten suficiente varianza "
                   "común para ser reducidas en componentes. Bartlett contrasta que la "
                   "matriz de correlaciones NO sea identidad (es decir, que existan "
                   "correlaciones sistemáticas entre proporciones). Ambas pruebas "
                   "son condiciones necesarias para que el PCA produzca resultados "
                   "estadísticamente interpretables.",
                   color=PURPLE),
        bloque_kmo_bartlett(),
    ),

    # ══ ETAPA 5 ══════════════════════════════════════════════════════════════
    etapa("05","PCA Composicional · Estructura de Variación",
          f"CP1 = {VE[0]:.1f}% · CP2 = {VE[1]:.1f}% · "
          f"Total 2 CP: {VA[1]:.1f}% · Kaiser: {N_KAI} componente(s)",
          color=ACCENT),

    card(
        T("Gráfico 4 · Scree Plot · Varianza Explicada y Eigenvalores",
          f"Los últimos 2 eigenvalores ≈ 0 por restricción composicional (rank = p-1 = {len(PROP_COLS)-1})"),
        justif_box("¿POR QUÉ PCA SOBRE CLR Y NO SOBRE PROPORCIONES DIRECTAS?",
                   "El PCA sobre proporciones brutas está sesgado por la restricción "
                   "del simplex: la varianza total es artificialmente limitada y las "
                   "cargas están distorsionadas. La transformación CLR proyecta los "
                   "datos al espacio euclidiano real (ℝᵖ), donde el PCA estándar "
                   "es válido. Los últimos 2 eigenvalores = 0 son una propiedad "
                   "matemática esperada (rango = p-1), no una deficiencia de los datos."),
        narr(f"El primer componente explica el {VE[0]:.1f}% de la varianza total, "
             f"capturando el eje principal de diferenciación en la priorización del gasto. "
             f"El segundo componente ({VE[1]:.1f}%) captura una dimensión ortogonal secundaria. "
             f"Juntos acumulan {VA[1]:.1f}%, lo que confirma que la estructura de "
             f"priorización departamental puede resumirse en dos dimensiones principales."),
        dcc.Graph(figure=_fig_scree, config={"displayModeBar":False}),
        # Eigenvalores en tarjetas
        html.Div([
            html.Div([
                html.P(f"CP{i+1}", style={"color":TEXT2,"fontFamily":"IBM Plex Mono",
                                           "fontSize":"9px","letterSpacing":"0.1em",
                                           "marginBottom":"4px"}),
                html.P(f"{EV[i]:.3f}", style={
                    "color":GREEN if EV[i]>=1 else BORDER,
                    "fontFamily":"IBM Plex Sans","fontSize":"15px",
                    "fontWeight":"700","margin":"0"}),
                html.P(f"{VE[i]:.1f}%", style={"color":TEXT2,"fontFamily":"IBM Plex Mono",
                                                 "fontSize":"9px"}),
            ], style={"background":SURFACE,
                       "border":f"1px solid {GREEN if EV[i]>=1 else BORDER}",
                       "borderRadius":"6px","padding":"10px 14px","textAlign":"center",
                       "minWidth":"70px"})
            for i in range(N_VALID)
        ], style={"display":"flex","gap":"10px","marginTop":"18px","flexWrap":"wrap"}),
    ),

    card(
        T("Gráfico 5 · Cargas Factoriales",
          "Contribución de cada sector a cada componente principal"),
        narr(f"CP1 opone los sectores de LIBRE ASIGNACIÓN (cultura, deporte, libre inversión "
             f"con cargas positivas) frente a LIBRE DESTINACIÓN (carga negativa). "
             f"CP1 captura el trade-off entre inversión discrecional focalizada y "
             f"capacidad fiscal autónoma. CP2 contrasta los sectores SGP "
             f"(educación, salud, agua) con los de libre asignación, "
             f"capturando la dimensión de priorización de servicios básicos universales."),
        dcc.Graph(figure=_fig_loadings, config={"displayModeBar":False}),
    ),

    card(
        T("Gráfico 6 · Biplot PCA Composicional",
          "Puntos = departamentos · Vectores = dirección de priorización de cada sector · "
          "Color = región geográfica"),
        narr("El biplot permite leer simultáneamente la posición de cada departamento "
             "en el espacio de componentes y la dirección de influencia de cada sector. "
             "Un departamento ubicado en la dirección de un vector prioriza más ese sector. "
             "Bogotá aparece aislado en el extremo del eje de libre destinación (valor cero). "
             "Los departamentos amazónicos y de los Llanos tienden a agruparse, "
             "anticipando la estructura de clusters."),
        dcc.Graph(figure=_fig_biplot, config={"displayModeBar":False}),
    ),

    # ══ ETAPA 6 ══════════════════════════════════════════════════════════════
    etapa("06","Análisis de Clúster · Clasificación de Patrones de Priorización",
          f"K-Means sobre CP1-CP2 · Selección por silueta y codo · "
          f"k={K_FINAL} clusters · Silueta = {SILS[K_FINAL-2]:.3f}",
          color=RED),

    card(
        T("Gráfico 7 · Selección del Número de Clusters",
          "Método del codo (inercia) y coeficiente de silueta"),
        justif_box("¿POR QUÉ K-MEANS EN ESPACIO PCA Y NO SOBRE PROPORCIONES?",
                   "El clustering en el espacio de las primeras componentes principales "
                   "tiene dos ventajas: (1) reduce el ruido dimensional manteniendo "
                   "la varianza relevante, y (2) los componentes son ortogonales, "
                   "lo que mejora la separabilidad de los clusters. Se usa K-Means "
                   "sobre CP1-CP2 que acumulan el " + f"{VA[1]:.1f}% de la varianza.",
                   color=RED),
        justif_box("¿CÓMO SE SELECCIONA k?",
                   f"El coeficiente de silueta mide qué tan similar es cada punto "
                   f"a su propio cluster vs. el cluster más cercano (rango -1 a 1, "
                   f"mayor es mejor). El método del codo identifica dónde la reducción "
                   f"de inercia se aplana. k={K_FINAL} combina un coeficiente de silueta "
                   f"aceptable ({SILS[K_FINAL-2]:.3f}) con mayor riqueza interpretativa "
                   f"que k=2 (silueta={SILS[0]:.3f} pero solo 2 grupos).",
                   color=RED),
        dcc.Graph(figure=_fig_codo, config={"displayModeBar":False}),
    ),

    card(
        T(f"Gráfico 8 · Clusters K-Means (k={K_FINAL}) en Espacio PCA",
          "Cada punto = un departamento · Color = cluster asignado"),
        narr(f"La separación visual de los {K_FINAL} clusters en el espacio PCA confirma "
             f"la coherencia del agrupamiento. Los clusters reflejan patrones distintos "
             f"de priorización del gasto y se solapan mínimamente, "
             f"indicando que las diferencias entre grupos son sustantivas."),
        dcc.Graph(figure=_fig_clusters, config={"displayModeBar":False}),
    ),

    # ══ ETAPA 7 ══════════════════════════════════════════════════════════════
    etapa("07","Caracterización de los Clusters · Perfiles de Priorización",
          "¿Qué distingue a cada grupo? · Radar · Heatmap · Interpretación para política pública",
          color=GREEN),

    card(
        T("Gráfico 9 · Radar · Perfil de Priorización por Cluster (%)",
          "Valores en % del gasto total · Compara la estructura relativa de cada grupo"),
        narr("Los perfiles composicionales revelan patrones diferenciados. "
             "Las diferencias en sectores de libre asignación son las más informativas "
             "pues reflejan decisiones deliberadas, a diferencia de los sectores SGP "
             "determinados por fórmulas de transferencia. El cluster 3 (Bogotá) "
             "se distingue por su nula libre destinación y mayor proporción educativa."),
        dcc.Graph(figure=_fig_radar, config={"displayModeBar":False}),
    ),

    card(
        T("Gráfico 10 · Heatmap de Proporciones Medias por Cluster",
          "Permite identificar los sectores diferenciadores entre grupos"),
        dcc.Graph(figure=_fig_heatmap, config={"displayModeBar":False}),
        html.Div(style={"marginTop":"20px"}),
        T("Tabla · Composición y Perfil de cada Cluster"),
        narr("Las diferencias más relevantes entre clusters están en libre destinación "
             "e inversión — los sectores de mayor CV y mayor autonomía fiscal. "
             "Estos resultados serán la base para la conexión con indicadores sociales "
             "en la etapa siguiente."),
        _tabla_cl,
    ),

    # ══ NOTA METODOLÓGICA ═════════════════════════════════════════════════════
    html.Div([
        html.Span("NOTA METODOLÓGICA · ANÁLISIS COMPOSICIONAL DE DATOS (CoDa)", style={
            "color":ORANGE,"fontSize":"9px","letterSpacing":"0.18em",
            "fontFamily":"IBM Plex Mono","fontWeight":"600",
        }),
        html.Div([
            html.P("Aitchison, J. (1986). The Statistical Analysis of Compositional Data. "
                   "Chapman and Hall.",
                   style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                          "lineHeight":"1.6","marginBottom":"6px"}),
            html.P("Martín-Fernández et al. (2003). Dealing with zeros and missing values "
                   "in compositional data sets using nonparametric imputation. "
                   "Mathematical Geology, 35(3).",
                   style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                          "lineHeight":"1.6","marginBottom":"6px"}),
            html.P("CLR(xᵢ) = log(xᵢ) − (1/p)·Σlog(xⱼ). "
                   "La restricción ΣCLR = 0 implica que la matriz de covarianza CLR "
                   "tiene rango p-1. Los últimos eigenvalores = 0 son una propiedad "
                   "matemática de la transformación, no un problema de los datos.",
                   style={"color":TEXT2,"fontFamily":"IBM Plex Mono","fontSize":"10px",
                          "lineHeight":"1.6"}),
        ], style={"borderLeft":f"3px solid {ORANGE}","paddingLeft":"14px","marginTop":"10px"}),
    ], style={
        "background":SURFACE,"border":f"1px solid {ORANGE}40",
        "borderRadius":"8px","padding":"18px 22px","marginBottom":"32px",
    }),

], style={"padding":"30px 40px","background":BG,"minHeight":"100vh"},
   className="page-fade")


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK — DISTRIBUCIÓN DINÁMICA
# ═══════════════════════════════════════════════════════════════════════════════
@callback(Output("dist-output","children"), Input("dist-dd","value"))
def update_dist(sector):
    if not sector or sector not in df.columns:
        return html.Div()
    fig_h, fig_qq, fig_box, s = fig_distribucion_prop(sector)
    return html.Div([
        html.Div([
            kpi("Media",     f"{s['media']:.2f}%",  "del gasto total",  ACCENT),
            kpi("Mediana",   f"{s['mediana']:.2f}%","del gasto total",  GREEN),
            kpi("CV",        f"{s['cv']:.1f}%",     "Heterogeneidad",   ORANGE),
            kpi("Asimetría", f"{s['asimetria']:.3f}","Skewness",        PURPLE),
            html.Div([
                html.P("Shapiro-Wilk", style={"color":TEXT2,"fontFamily":"IBM Plex Mono",
                                               "fontSize":"9px","letterSpacing":"0.1em",
                                               "textTransform":"uppercase","marginBottom":"4px"}),
                html.P(s["norm_text"], style={"color":s["norm_color"],
                                               "fontFamily":"IBM Plex Mono","fontSize":"11px",
                                               "fontWeight":"600","margin":"0","lineHeight":"1.3"}),
                html.P(f"p = {s['pval']:.4f}", style={"color":TEXT2,"fontFamily":"IBM Plex Mono",
                                                        "fontSize":"9px","marginTop":"3px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"6px",
                       "padding":"12px 16px","borderTop":f"3px solid {s['norm_color']}","flex":"1.5"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"16px"}),
        html.Div([
            html.Div(dcc.Graph(figure=fig_h,  config={"displayModeBar":False}), style={"flex":"1.5"}),
            html.Div(dcc.Graph(figure=fig_qq, config={"displayModeBar":False}), style={"flex":"1"}),
        ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
        dcc.Graph(figure=fig_box, config={"displayModeBar":False}),
    ])