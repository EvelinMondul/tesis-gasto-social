"""
pages/clustering.py
=====================

Capítulo 07: Clustering. Segmentación de los 32 departamentos (excluyendo
Bogotá D.C., tratada como caso atípico) mediante K-Means (k=2) sobre las
puntuaciones CP1-CP2 del ACP, y caracterización de los grupos resultantes
mediante el test de Mann-Whitney U.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from analysis.clustering import CLUSTER_LABELS, mannwhitney_clusters, perfil_clusters, run_kmeans
from analysis.coda import load_gasto
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.clustering_figures import fig_clusters_cp1_cp2, fig_perfil_clusters

dash.register_page(__name__, path="/clustering", name="Clustering", order=7)


def _tabla(data, columns, table_id, style_data_conditional=None):
    kwargs = {}
    if style_data_conditional:
        kwargs["style_data_conditional"] = style_data_conditional
    return dash_table.DataTable(
        id=table_id,
        data=data,
        columns=columns,
        style_as_list_view=True,
        style_header={
            "backgroundColor": "var(--color-bg-subtle)",
            "fontWeight": "600",
            "border": "none",
            "borderBottom": "1px solid var(--color-border)",
        },
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": "13px",
            "padding": "8px 12px",
            "border": "none",
            "borderBottom": "1px solid var(--color-border)",
        },
        **kwargs,
    )


def layout():
    df = load_gasto()
    res = run_kmeans(df)
    perfil = perfil_clusters(df)
    mw = mannwhitney_clusters(df)

    tab_metodo = html.Div([
        section_title("Método: K-Means sobre CP1-CP2"),
        section_text([
            "El plano CP1-CP2 (Capítulo 05) resume el 82.8% de la varianza "
            "de las coordenadas CLR del gasto social. Bogotá D.C. "
            "constituye un valor atípico marcado en CP1 (su gasto se "
            "concentra casi totalmente en Educación, con Libre Destinación "
            "igual a cero) y, al ser un Distrito Capital y no un "
            "departamento sujeto a las mismas reglas del Sistema General "
            "de Participaciones (SGP), se excluye del K-Means y se presenta "
            "aparte como caso atípico. Sobre las puntuaciones CP1-CP2 de "
            "los 32 departamentos restantes se aplica K-Means con k=2 "
            f"grupos, valor seleccionado por tener el mayor coeficiente de "
            f"silueta (silueta(k=2) = {res['silhouette']:.4f}, frente a "
            "0.4572 para k=3). Los dos grupos resultantes se distinguen "
            "según su margen de gasto discrecional (Libre Destinación y "
            "Libre Inversión).",
        ]),

        dbc.Row([
            dbc.Col(kpi_card(f"{int(row['n'])} dpto." + ("s" if row["n"] != 1 else ""),
                             f"{row['cluster']} — {CLUSTER_LABELS[row['cluster']]}"), md=4, className="mb-3")
            for _, row in perfil.iterrows()
        ], className="mb-3"),

        section_title("Segmentación en el plano CP1-CP2"),
        figure_block(
            fig_clusters_cp1_cp2(df),
            caption=html.Span([
                html.Strong("Figura 7.1. "),
                "Los 33 departamentos sobre el plano CP1-CP2. Los 32 "
                "departamentos distintos de Bogotá se colorean según el "
                "cluster K-Means (k=2) al que pertenecen; Bogotá D.C. se "
                "muestra aparte como caso atípico, excluido del K-Means.",
            ]),
            fig_id="fig-clust-scatter",
            height=560,
        ),
    ], className="pt-3")

    tab_perfil = html.Div([
        section_title("Perfil de composición por cluster"),
        section_text(
            "La siguiente tabla y gráfico muestran la composición promedio "
            "(%) del gasto social en cada cluster, lo que permite "
            "caracterizar el patrón de gasto típico de cada grupo."
        ),
        _tabla(
            perfil.drop(columns="Descripción").to_dict("records"),
            [{"name": c, "id": c} for c in perfil.columns if c != "Descripción"],
            "tabla-clust-perfil",
        ),
        figure_block(
            fig_perfil_clusters(df),
            caption=html.Span([
                html.Strong("Figura 7.2. "),
                "Composición promedio del gasto social (%) por cluster, "
                "para cada uno de los seis componentes.",
            ]),
            fig_id="fig-clust-perfil",
        ),
    ], className="pt-3")

    tab_mw = html.Div([
        section_title("Diferencias entre clusters: prueba de Mann-Whitney U"),
        section_text(
            "Para cada componente del gasto se aplica el test no "
            "paramétrico de Mann-Whitney U, específico para contrastar "
            "dos grupos independientes, que evalúa si su distribución "
            "porcentual difiere entre los clusters C1 y C2 (H₀: las "
            "distribuciones son iguales en ambos grupos). Bogotá D.C. "
            "(caso atípico, n=1) se excluye de este test."
        ),
        _tabla(
            mw.to_dict("records"),
            [{"name": c, "id": c} for c in mw.columns],
            "tabla-clust-mw",
            style_data_conditional=[{
                "if": {"filter_query": '{Significativo (alpha=0.05)} = "Sí"'},
                "backgroundColor": "var(--color-bg-subtle)",
            }],
        ),

        method_card(
            "Lectura de resultados y siguientes pasos",
            [
                "Las diferencias entre clusters son estadísticamente "
                "significativas (p < 0.05) en ", html.B("Libre Inversión"),
                ", ", html.B("Libre Destinación"), " y ",
                html.B("Cultura y Deporte"), ", lo que confirma que el "
                "margen de gasto discrecional es el principal criterio "
                "que distingue a los dos grupos. El cluster C1 (21 "
                "departamentos) destina, en promedio, una proporción "
                "mayor a Libre Destinación y Libre Inversión; el cluster "
                "C2 (11 departamentos) destina una proporción mayor a "
                "Salud y menor a las categorías discrecionales. Bogotá "
                "D.C., excluida del K-Means por ser un caso atípico, "
                "presenta Libre Destinación nula y la mayor proporción de "
                "gasto en Educación de los 33 departamentos. En el "
                "Capítulo 08 se contrastan estos grupos con el Índice de "
                "Pobreza Multidimensional (IPM) de cada departamento.",
            ],
        ),
    ], className="pt-3")

    return html.Div([
        page_header(
            "Capítulo 07",
            "Clustering de departamentos",
            "Segmentación de los 32 departamentos (Bogotá D.C. se trata "
            "como caso atípico) en grupos homogéneos según el patrón de su "
            "gasto social, mediante K-Means (k=2) aplicado a las "
            "puntuaciones CP1-CP2 del ACP.",
        ),

        dbc.Tabs([
            dbc.Tab(tab_metodo, label="Método y segmentación CP1-CP2", tab_id="tab-clust-metodo"),
            dbc.Tab(tab_perfil, label="Perfil por cluster", tab_id="tab-clust-perfil"),
            dbc.Tab(tab_mw, label="Mann-Whitney U", tab_id="tab-clust-mw"),
        ], id="tabs-clustering", active_tab="tab-clust-metodo"),
    ])
