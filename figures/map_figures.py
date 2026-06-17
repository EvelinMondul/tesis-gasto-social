"""
pages/mapas.py
==================

Capítulo 09: Mapas Territoriales.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from analysis.coda import load_gasto
from analysis.geo import datos_mapas
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.map_figures import (
    fig_mapa_clusters,
    fig_mapa_componente_predominante,
    fig_mapa_educacion,
    fig_mapa_ipm,
)

dash.register_page(__name__, path="/mapas", name="Mapas Territoriales", order=9)


def _tabla(data, columns=None):
    cols = columns if columns is not None else data.columns
    return dash_table.DataTable(
        data=data.to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols],
        style_as_list_view=True,
        style_table={"overflowY": "auto", "maxHeight": "440px"},
        style_header={
            "backgroundColor": "var(--color-bg-subtle)",
            "fontWeight": "600",
            "border": "none",
            "borderBottom": "1px solid var(--color-border)",
            "position": "sticky",
            "top": 0,
        },
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": "13px",
            "padding": "8px 12px",
            "border": "none",
            "borderBottom": "1px solid var(--color-border)",
        },
        style_data_conditional=[{
            "if": {"filter_query": '{Cluster} = "Atípico"'},
            "backgroundColor": "var(--color-bg-subtle)",
        }],
    )


def layout():
    df = load_gasto()
    datos = datos_mapas(df)

    n_total = len(datos)
    n_educacion = (datos["Componente predominante"] == "Educación").sum()
    cluster_counts = datos["cluster"].value_counts()
    top_ipm = datos.sort_values("IPM_pct", ascending=False).iloc[0]
    bottom_ipm = datos.sort_values("IPM_pct").iloc[0]

    tabla_datos = (
        datos[["Departamento", "region", "cluster", "Componente predominante", "Educación (%)", "IPM_pct"]]
        .rename(columns={"region": "Región", "cluster": "Cluster", "IPM_pct": "IPM (%)"})
        .sort_values("IPM (%)", ascending=False)
        .reset_index(drop=True)
    )

    return html.Div([
        page_header(
            "Capítulo 09",
            "Mapas Territoriales",
            "Distribución geográfica de la composición del gasto social, "
            "los clusters del Capítulo 07 y el Índice de Pobreza "
            "Multidimensional (IPM 2025) en los 33 departamentos de "
            "Colombia.",
        ),

        dbc.Row([
            dbc.Col(kpi_card(f"{n_total}", "Departamentos mapeados"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{n_educacion}/{n_total}", "Departamentos con Educación como componente predominante"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{cluster_counts.get('C1', 0)} / {cluster_counts.get('C2', 0)} / {cluster_counts.get('Atípico', 0)}",
                             "Departamentos en cluster C1 / C2 / Atípico (Bogotá)"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{top_ipm['IPM_pct']:.1f}%", f"IPM más alto: {top_ipm['Departamento']}"), md=3, className="mb-3"),
        ], className="mb-3"),

        section_title("Composición del gasto en el territorio"),
        section_text(
            "El siguiente mapa identifica, para cada departamento, el "
            "componente del gasto social con la mayor proporción dentro "
            "de su composición (Capítulo 03). El predominio casi total de "
            "Educación ilustra que las diferencias entre departamentos "
            "residen más en la magnitud relativa de cada componente que "
            "en cuál de ellos concentra la mayor parte del gasto."
        ),
        figure_block(
            fig_mapa_componente_predominante(df),
            caption=html.Span([
                html.Strong("Figura 9.1. "),
                "Componente de gasto social con mayor proporción dentro "
                "de la composición de cada departamento (Capítulo 03).",
            ]),
            fig_id="fig-mapa-componente",
            height=620,
        ),

        section_title("Clusters de gasto social"),
        section_text(
            "Este mapa ubica geográficamente los clusters de gasto social "
            "identificados mediante K-Means sobre las coordenadas "
            "factoriales (Capítulo 07): C1 (mayor margen discrecional) y "
            "C2 (mayor peso del gasto en Salud, menor margen "
            "discrecional). Bogotá D.C. se excluye del K-Means y se "
            "muestra aparte como caso atípico."
        ),
        figure_block(
            fig_mapa_clusters(df),
            caption=html.Span([
                html.Strong("Figura 9.2. "),
                "Distribución territorial de los clusters C1 y C2 de "
                "gasto social y de Bogotá D.C. como caso atípico "
                "(Capítulo 07).",
            ]),
            fig_id="fig-mapa-clusters",
            height=620,
        ),

        section_title("Índice de Pobreza Multidimensional"),
        section_text(
            "El IPM (%) de 2025 presenta un marcado patrón centro-periferia: "
            "los departamentos de la región Amazónica y Orinoquía "
            "(Vichada, Guainía, Vaupés) y del Pacífico (Chocó) registran "
            "los niveles más altos, mientras que Bogotá y los "
            "departamentos del centro y eje cafetero presentan los "
            "niveles más bajos."
        ),
        figure_block(
            fig_mapa_ipm(df),
            caption=html.Span([
                html.Strong("Figura 9.3. "),
                "Índice de Pobreza Multidimensional (%) por departamento "
                "(2025).",
            ]),
            fig_id="fig-mapa-ipm",
            height=620,
        ),

        section_title("Gasto en Educación"),
        section_text(
            "Dado que la proporción de gasto en Educación es el componente "
            "con la correlación más fuerte (e inversa) con el IPM "
            "(Capítulo 08), este mapa permite contrastar visualmente su "
            "distribución territorial con la del IPM de la figura "
            "anterior."
        ),
        figure_block(
            fig_mapa_educacion(df),
            caption=html.Span([
                html.Strong("Figura 9.4. "),
                "Proporción del gasto social destinada a Educación (%) "
                "por departamento.",
            ]),
            fig_id="fig-mapa-educacion",
            height=620,
        ),

        section_title("Tabla departamental"),
        section_text(
            "Resumen por departamento: región, cluster de gasto (Capítulo "
            "07), componente de gasto predominante (Capítulo 03), "
            "proporción de gasto en Educación e IPM (%), ordenado de "
            "mayor a menor IPM."
        ),
        _tabla(tabla_datos),

        method_card(
            "Lectura territorial y cierre del análisis",
            [
                "La superposición de los mapas confirma, a escala "
                "territorial, los hallazgos de los Capítulos 07 y 08: los "
                "departamentos con mayor IPM (", html.B("Vichada"), ", ",
                html.B("Guainía"), ", ", html.B("Vaupés"), " y ",
                html.B("Chocó"), ") pertenecen al cluster C1 pero se "
                "ubican en el extremo inferior de la distribución de "
                "gasto en Educación (entre 42% y 44%), mientras que ",
                html.B("Bogotá"), " (caso atípico, excluida del "
                "clustering, IPM = 2.2%) y ",
                html.B("San Andrés y Providencia"), " (IPM = 4.3%) "
                "destinan la mayor proporción de su gasto a este "
                "componente (70.6% y 70.0%, respectivamente). Esto "
                "sugiere que, más allá del cluster al que pertenece un "
                "departamento, la proporción específica de gasto en "
                "Educación está asociada con su nivel de pobreza "
                "multidimensional. En el Capítulo 10 se sintetizan estas "
                "conclusiones junto con las de los capítulos anteriores.",
            ],
        ),
    ])