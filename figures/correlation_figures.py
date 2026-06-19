"""
figures/correlation_figures.py
================================

Figuras del Capítulo 04 (Correlaciones): prueba de normalidad de
Shapiro-Wilk sobre las coordenadas CLR y matriz de correlación de
Spearman.
"""

import numpy as np
import plotly.graph_objects as go

from analysis.correlations import spearman_clr
from analysis.descriptive import shapiro_clr
from config import COLORS, ESCALA_DIVERGENTE


def fig_shapiro(df=None, alpha: float = 0.05):
    """Estadístico W de Shapiro-Wilk por componente CLR, con línea de
    referencia del nivel de significancia."""
    res = shapiro_clr(df, alpha=alpha)

    colors = [
        COLORS["petroleo"] if normal == "Sí" else COLORS["gris_medio"]
        for normal in res["Normal (alpha=0.05)"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res["Componente (CLR)"], y=res["Estadístico W"],
        marker_color=colors,
        text=[f"p={p:.4f}" for p in res["Valor p"]],
        textposition="outside",
        hovertemplate="%{x}<br>W=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Prueba de Shapiro-Wilk sobre las coordenadas CLR"),
        yaxis_title="Estadístico W",
        yaxis_range=[0, 1.15],
        height=420,
    )
    return fig


def fig_heatmap_spearman(df=None):
    """Mapa de calor de la matriz de correlación de Spearman sobre CLR,
    con los coeficientes anotados."""
    rho, pval = spearman_clr(df)

    fig = go.Figure(data=go.Heatmap(
        z=rho.values,
        x=rho.columns,
        y=rho.index,
        colorscale=ESCALA_DIVERGENTE,
        zmin=-1, zmax=1,
        text=rho.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="rho"),
        hovertemplate="%{y} vs %{x}<br>rho=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
    title=dict(text="Matriz de correlación de Spearman sobre coordenadas CLR"),
    height=520,
    yaxis=dict(autorange="reversed"),
    margin=dict(l=130, r=20, t=70, b=60),
    )
    return fig
