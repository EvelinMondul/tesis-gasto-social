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
    """Dispersión de los 32 departamentos (excluye Bogotá D.C.) sobre el
    plano CP1-CP2, coloreados según el cluster K-Means (k=2, C1/C2)."""
    res = run_kmeans(df)
    scores = res["scores"]

    # Excluir Bogotá del gráfico
    scores_sin_bogota = scores[scores["cluster"] != "Atípico"].copy()

    # Etiquetas descriptivas para la leyenda
    labels_map = {
        "C1": "C1 — Mayor margen discrecional (n=21)",
        "C2": "C2 — Menor margen discrecional (n=11)",
    }
    scores_sin_bogota["Grupo"] = scores_sin_bogota["cluster"].map(labels_map)

    fig = px.scatter(
        scores_sin_bogota,
        x="CP1",
        y="CP2",
        color="Grupo",
        text="Departamento",
        hover_name="Departamento",
        category_orders={"Grupo": list(labels_map.values())},
        color_discrete_map={
            labels_map["C1"]: COLOR_CLUSTER["C1"],
            labels_map["C2"]: COLOR_CLUSTER["C2"],
        },
    )
    fig.update_traces(
        marker=dict(size=11, line=dict(width=1, color="white")),
        textposition="top center",
        textfont=dict(size=8, color=COLORS["texto_secundario"]),
    )
    fig.add_hline(y=0, line_color="gray", line_width=1.2, line_dash="dash")
    fig.add_vline(x=0, line_color="gray", line_width=1.2, line_dash="dash")
    fig.update_layout(
        title=dict(text="Segmentación K-Means (k=2) sobre CP1-CP2 · 32 departamentos"),
        xaxis_title="CP1 — Autonomía fiscal (52.8% varianza)",
        yaxis_title="CP2 — Sectores SGP vs discrecionales (30.0% varianza)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, title=None
        ),
        height=580,
    )
    return fig


def fig_perfil_clusters(df=None):
    """Composición promedio (%) de cada componente del gasto, por
    cluster (C1 y C2), como barras agrupadas. Excluye Bogotá D.C."""
    perfil = perfil_clusters(df)

    # Excluir Bogotá del perfil
    perfil_sin_bogota = perfil[perfil["cluster"] != "Atípico"].copy()

    fig = go.Figure()
    for _, row in perfil_sin_bogota.iterrows():
        cluster = row["cluster"]
        fig.add_trace(go.Bar(
            x=COMPONENTES,
            y=[row[c] for c in COMPONENTES],
            name=f"{cluster} (n={int(row['n'])})",
            marker_color=COLOR_CLUSTER[cluster],
            hovertemplate="%{x}<br>" + cluster + ": %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Composición promedio del gasto social por cluster"),
        yaxis_title="Porcentaje del gasto social total (%)",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    fig.update_yaxes(ticksuffix="%")
    return fig