"""
figures/clustering_figures.py
================================

Figuras del Capítulo 07 (Clustering): dispersión CP1-CP2 coloreada por
cluster y perfil de composición promedio por cluster.
"""

import plotly.express as px
import plotly.graph_objects as go

from analysis.clustering import run_kmeans, perfil_clusters
from analysis.coda import COMPONENTES
from config import COLOR_CLUSTER, COLORS, SECUENCIA_CATEGORICA


def fig_clusters_cp1_cp2(df=None):
    """Dispersión de los 33 departamentos sobre el plano CP1-CP2. Los 32
    departamentos distintos de Bogotá se colorean según el cluster
    K-Means (k=2, C1/C3) al que pertenecen; Bogotá D.C. se resalta como
    caso atípico, excluido del K-Means."""
    res = run_kmeans(df)
    scores = res["scores"]

    fig = px.scatter(
        scores,
        x="CP1",
        y="CP2",
        color="cluster",
        hover_name="Departamento",
        category_orders={"cluster": ["C1", "C3", "Atípico"]},
        color_discrete_map=COLOR_CLUSTER,
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color=COLORS["bg"])))

    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=1)

    fig.update_layout(
        title=dict(text="Segmentación de departamentos por K-Means (k=2) sobre CP1-CP2, con Bogotá D.C. como caso atípico"),
        xaxis_title="CP1",
        yaxis_title="CP2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        height=560,
    )
    return fig


def fig_perfil_clusters(df=None):
    """Composición promedio (%) de cada componente del gasto, por
    cluster, como barras agrupadas."""
    perfil = perfil_clusters(df)

    fig = go.Figure()
    for _, row in perfil.iterrows():
        cluster = row["cluster"]
        fig.add_trace(go.Bar(
            x=COMPONENTES,
            y=[row[c] for c in COMPONENTES],
            name=f"{cluster} (n={row['n']})",
            marker_color=COLOR_CLUSTER[cluster],
            hovertemplate="%{x}<br>" + cluster + ": %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Composición promedio del gasto social por cluster (Bogotá D.C. = caso atípico, n=1)"),
        yaxis_title="Porcentaje del gasto social total (%)",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    fig.update_yaxes(ticksuffix="%")
    return fig
