"""
pages/ipm.py
=============

Capítulo 08: contraste entre el gasto social, los clusters del
Capítulo 07 y el Índice de Pobreza Multidimensional (IPM 2025).
"""

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from analysis.coda import load_gasto
from analysis.ipm_analysis import correlacion_gasto_ipm, ipm_por_cluster, privaciones_por_cluster, ranking_ipm
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.ipm_figures import fig_correlacion_gasto_ipm, fig_ipm_por_cluster, fig_ranking_ipm

dash.register_page(__name__, path="/ipm", name="Contraste con el IPM", order=8)


def _tabla(data, columns=None):
    cols = columns if columns is not None else data.columns
    return dash_table.DataTable(
        data=data.to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols],
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
        style_data_conditional=[{
            "if": {"filter_query": '{Significativo (alpha=0.05)} = "Sí"'},
            "backgroundColor": "var(--color-bg-subtle)",
        }],
    )


def layout():
    df = load_gasto()
    resumen_cluster, kw = ipm_por_cluster(df)
    privaciones = privaciones_por_cluster(df)
    corr = correlacion_gasto_ipm(df)
    ranking = ranking_ipm(df)

    nacional = resumen_cluster["Media"].mul(resumen_cluster["n"]).sum() / resumen_cluster["n"].sum()
    top = ranking.iloc[0]
    bottom = ranking.iloc[-1]

    return html.Div([
        page_header(
            "Capítulo 08",
            "Contraste con el Índice de Pobreza Multidimensional",
            "Relación entre la composición del gasto social, los clusters "
            "departamentales identificados en el Capítulo 07 y el Índice "
            "de Pobreza Multidimensional (IPM 2025) de cada departamento.",
        ),

        dbc.Row([
            dbc.Col(kpi_card(f"{nacional:.1f}%", "IPM promedio (33 departamentos)"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{top['IPM_pct']:.1f}%", f"IPM más alto: {top['Departamento']}"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{bottom['IPM_pct']:.1f}%", f"IPM más bajo: {bottom['Departamento']}"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{corr.iloc[0]['rho (Spearman) vs. IPM']:.2f}",
                             f"Correlación más fuerte con el gasto: {corr.iloc[0]['Componente del gasto']}"),
                    md=3, className="mb-3"),
        ], className="mb-3"),

        section_title("IPM por cluster de gasto"),
        section_text(
            "Se contrasta el IPM (%) de cada departamento con el cluster "
            "de gasto social al que pertenece (Capítulo 07), para evaluar "
            "si los patrones de asignación presupuestal se asocian con "
            "los niveles de pobreza multidimensional."
        ),
        figure_block(
            fig_ipm_por_cluster(df),
            caption=html.Span([
                html.Strong("Figura 8.1. "),
                "Distribución del IPM (%) en los clusters C1 y C3 de "
                "gasto social identificados en el Capítulo 07, y en "
                "Bogotá D.C. (caso atípico, n=1).",
            ]),
            fig_id="fig-ipm-cluster",
        ),
        _tabla(resumen_cluster.drop(columns="Descripción")),
        method_card(
            "Prueba de Kruskal-Wallis (IPM por cluster)",
            (
                f"H = {kw['H (Kruskal-Wallis)']}, valor p = {kw['Valor p']} "
                f"({kw['Significativo (alpha=0.05)']} significativo al "
                f"nivel α = 0.05). {kw['nota']} El cluster C1 (mayor margen "
                "discrecional de gasto) y el cluster C3 (mayor peso del "
                "gasto en salud) presentan niveles de IPM medio similares "
                "(18.3% y 14.2%, respectivamente), por lo que el patrón de "
                "composición del gasto detectado en los Capítulos 05-07 no "
                "se traduce, por sí solo, en diferencias estadísticamente "
                "significativas en pobreza multidimensional entre estos "
                "grupos."
            ),
        ),

        section_title("Privaciones del IPM por cluster"),
        section_text(
            "La siguiente tabla muestra, para cada uno de los 15 "
            "indicadores de privación que componen el IPM, el porcentaje "
            "medio y mediano de personas que presentan dicha privación en "
            "cada cluster de gasto (C1, C3 y el caso atípico, Bogotá D.C.)."
        ),
        html.Div(
            _tabla(privaciones),
            style={"overflowX": "auto"},
        ),

        section_title("Correlación entre composición del gasto e IPM"),
        section_text(
            "Para cada componente del gasto se calcula la correlación de "
            "Spearman entre su proporción departamental y el IPM (%), "
            "evaluando si una mayor proporción relativa de gasto en un "
            "componente se asocia con mayor o menor pobreza "
            "multidimensional."
        ),
        figure_block(
            fig_correlacion_gasto_ipm(df),
            caption=html.Span([
                html.Strong("Figura 8.2. "),
                "Correlación de Spearman entre la proporción de cada "
                "componente del gasto social y el IPM (%) de los 33 "
                "departamentos. Las barras en azul petróleo son "
                "significativas al nivel α = 0.05.",
            ]),
            fig_id="fig-ipm-corr",
            height=420,
        ),
        _tabla(corr),

        section_title("Ranking departamental del IPM"),
        section_text(
            "El siguiente gráfico ordena a los 33 departamentos según su "
            "IPM (%), coloreados según el cluster de gasto social al que "
            "pertenecen (Capítulo 07)."
        ),
        figure_block(
            fig_ranking_ipm(df),
            caption=html.Span([
                html.Strong("Figura 8.3. "),
                "Índice de Pobreza Multidimensional (%) por departamento "
                "(2025), coloreado según el cluster de gasto social.",
            ]),
            fig_id="fig-ipm-ranking",
            height=900,
        ),

        method_card(
            "Lectura de resultados y siguientes pasos",
            [
                "La proporción de gasto en ", html.B("Educación"),
                " muestra la correlación más fuerte con el IPM (rho = "
                f"{corr.iloc[0]['rho (Spearman) vs. IPM']:.2f}, p = "
                f"{corr.iloc[0]['Valor p']:.4f}): los departamentos que "
                "destinan una mayor proporción de su gasto social a "
                "Educación tienden a presentar un IPM más bajo. En sentido "
                "inverso, mayores proporciones de gasto en ",
                html.B("Salud"), ", ", html.B("Libre Destinación"), " y ",
                html.B("Agua potable"), " se asocian con un IPM más alto, "
                "lo que es consistente con que estos departamentos "
                "destinan más recursos a atender carencias estructurales "
                "ya existentes. Estas relaciones son de naturaleza "
                "correlacional y no permiten establecer causalidad. En el "
                "Capítulo 09 se presentan estos resultados en mapas "
                "territoriales, y en el Capítulo 10 se sintetizan las "
                "conclusiones generales del análisis.",
            ],
        ),
    ])
