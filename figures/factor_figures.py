"""
figures/factor_figures.py
===========================

Figuras del Capítulo 06 (Análisis Factorial Exploratorio): cargas
factoriales rotadas (varimax) y comparación con las cargas del ACP del
Capítulo 05.
"""

import plotly.graph_objects as go

from analysis.factor_analysis import comparar_acp_afe, run_afe
from config import COLORS


def fig_factor_loadings(df=None, factores=("F1", "F2")):
    """Cargas factoriales rotadas (varimax) de cada coordenada CLR sobre
    los factores retenidos, como barras agrupadas."""
    res = run_afe(df)
    loadings = res["loadings"]

    colores = [COLORS["petroleo"], COLORS["azul_oscuro"], COLORS["gris_medio"]]

    fig = go.Figure()
    for f, color in zip(factores, colores):
        fig.add_trace(go.Bar(
            x=loadings.index,
            y=loadings[f],
            name=f,
            marker_color=color,
            hovertemplate="%{x}<br>" + f + ": %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=COLORS["gris_medio"], line_width=1)
    fig.update_layout(
        barmode="group",
        title=dict(text="Cargas factoriales rotadas (varimax)"),
        yaxis_title="Carga factorial",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def fig_acp_vs_afe(df=None):
    """Comparación de las cargas del ACP (CP1, CP2) frente a las cargas
    del AFE (F1, F2 rotadas) para cada coordenada CLR, mediante barras
    agrupadas con anchura reducida por par."""
    comp = comparar_acp_afe(df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp["Componente (CLR)"], y=comp["ACP - CP1"], name="ACP — CP1",
        marker_color=COLORS["petroleo"], offsetgroup="dim1",
        hovertemplate="%{x}<br>ACP CP1: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=comp["Componente (CLR)"], y=comp["AFE - F1 (varimax)"], name="AFE — F1 (varimax)",
        marker_color=COLORS["petroleo_claro"], offsetgroup="dim1",
        hovertemplate="%{x}<br>AFE F1: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=comp["Componente (CLR)"], y=comp["ACP - CP2"], name="ACP — CP2",
        marker_color=COLORS["azul_oscuro"], offsetgroup="dim2",
        hovertemplate="%{x}<br>ACP CP2: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=comp["Componente (CLR)"], y=comp["AFE - F2 (varimax)"], name="AFE — F2 (varimax)",
        marker_color=COLORS["gris_medio"], offsetgroup="dim2",
        hovertemplate="%{x}<br>AFE F2: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["gris_medio"], line_width=1)
    fig.update_layout(
        barmode="group",
        title=dict(text="Comparación de cargas: ACP (sin rotar) frente a AFE (varimax)"),
        yaxis_title="Carga",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig
