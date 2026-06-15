"""
figures/ipm_figures.py
========================

Figuras del Capítulo 08: contraste entre el gasto social, los clusters
del Capítulo 07 y el Índice de Pobreza Multidimensional (IPM 2025).
"""

import plotly.graph_objects as go

from analysis.ipm_analysis import correlacion_gasto_ipm, gasto_ipm_clusters, ranking_ipm
from config import COLOR_CLUSTER, COLORS


def fig_ipm_por_cluster(df=None):
    """Diagramas de caja del IPM (%) por cluster (Capítulo 07)."""
    datos = gasto_ipm_clusters(df)

    fig = go.Figure()
    for cluster in ["C1", "C3", "Atípico"]:
        sub = datos[datos["cluster"] == cluster]
        fig.add_trace(go.Box(
            y=sub["IPM_pct"],
            name=cluster,
            marker_color=COLOR_CLUSTER[cluster],
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            text=sub["Departamento"],
            hovertemplate="%{text}<br>IPM: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Índice de Pobreza Multidimensional (%) por cluster"),
        yaxis_title="IPM (%)",
        xaxis_title=None,
        showlegend=False,
        height=480,
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def fig_correlacion_gasto_ipm(df=None):
    """Correlación de Spearman entre cada componente del gasto (%) y el
    IPM departamental, como barras horizontales."""
    corr = correlacion_gasto_ipm(df)
    corr = corr.sort_values("rho (Spearman) vs. IPM")

    colores = [
        COLORS["petroleo"] if sig == "Sí" else COLORS["acento_suave"]
        for sig in corr["Significativo (alpha=0.05)"]
    ]

    fig = go.Figure(go.Bar(
        x=corr["rho (Spearman) vs. IPM"],
        y=corr["Componente del gasto"],
        orientation="h",
        marker_color=colores,
        text=[f"{v:.2f}" for v in corr["rho (Spearman) vs. IPM"]],
        textposition="outside",
        hovertemplate="%{y}<br>rho = %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=COLORS["gris_medio"], line_width=1)
    fig.update_layout(
        title=dict(text="Correlación entre la composición del gasto y el IPM"),
        xaxis_title="rho de Spearman (componente del gasto vs. IPM)",
        yaxis_title=None,
        height=420,
    )
    fig.update_xaxes(range=[-1, 1])
    return fig


def fig_ranking_ipm(df=None):
    """Ranking de los 33 departamentos por IPM (%), coloreados según su
    cluster (Capítulo 07)."""
    ranking = ranking_ipm(df).sort_values("IPM_pct", ascending=True)

    fig = go.Figure()
    for cluster in ["C1", "C3", "Atípico"]:
        sub = ranking[ranking["cluster"] == cluster]
        fig.add_trace(go.Bar(
            x=sub["IPM_pct"],
            y=sub["Departamento"],
            orientation="h",
            name=cluster,
            marker_color=COLOR_CLUSTER[cluster],
            hovertemplate="%{y}<br>IPM: %{x:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Índice de Pobreza Multidimensional por departamento (2025)"),
        xaxis_title="IPM (%)",
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Cluster"),
        height=900,
        margin=dict(l=140),
    )
    fig.update_xaxes(ticksuffix="%")
    return fig
