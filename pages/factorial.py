"""
pages/factorial.py
====================

Capítulo 06: Análisis Factorial Exploratorio (AFE), utilizado como
contraste metodológico del Análisis de Componentes Principales del
Capítulo 05.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from analysis.coda import load_gasto
from analysis.factor_analysis import run_afe
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.factor_figures import fig_acp_vs_afe, fig_factor_loadings

dash.register_page(__name__, path="/factorial", name="Análisis Factorial (AFE)", order=6)


def layout():
    df = load_gasto()
    res = run_afe(df)

    comunalidades = res["communalities"]
    varianza = res["varianza"]

    return html.Div([
        page_header(
            "Capítulo 06",
            "Análisis Factorial Exploratorio",
            "Contraste del ACP del Capítulo 05 mediante un Análisis "
            "Factorial Exploratorio (AFE) con extracción por componentes "
            "principales y rotación varimax, que busca factores "
            "interpretables más simples.",
        ),

        section_title("Extracción y rotación"),
        section_text([
            "Se extraen dos factores --el mismo número retenido por el "
            "criterio de Kaiser en el ACP-- y se aplica una rotación "
            "ortogonal varimax, que maximiza la varianza de las cargas al "
            "cuadrado dentro de cada factor para obtener una estructura "
            "más simple: cada coordenada CLR tiende a cargar fuertemente "
            "en un solo factor.",
        ]),

        dbc.Row([
            dbc.Col(kpi_card(f"{varianza.loc[0, 'Varianza explicada (%)']:.2f}%",
                             "Varianza explicada por F1"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{varianza.loc[1, 'Varianza explicada (%)']:.2f}%",
                             "Varianza explicada por F2"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{varianza.loc[1, 'Varianza acumulada (%)']:.2f}%",
                             "Varianza acumulada (F1+F2)"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{comunalidades['Comunalidad'].mean():.3f}",
                             "Comunalidad promedio"), md=3, className="mb-3"),
        ], className="mb-3"),

        section_title("Cargas factoriales rotadas"),
        section_text(
            "La rotación varimax redistribuye la varianza entre F1 y F2 "
            "buscando que cada coordenada CLR cargue principalmente sobre "
            "un único factor, lo que facilita su interpretación frente a "
            "las cargas sin rotar del ACP."
        ),
        figure_block(
            fig_factor_loadings(df),
            caption=html.Span([
                html.Strong("Figura 6.1. "),
                "Cargas factoriales rotadas (varimax) de cada coordenada "
                "CLR sobre los factores F1 y F2.",
            ]),
            fig_id="fig-afe-loadings",
        ),

        section_title("Comunalidades"),
        section_text(
            "La comunalidad de cada coordenada CLR indica la proporción de "
            "su varianza explicada conjuntamente por F1 y F2. Valores "
            "cercanos a 1 indican que el componente queda casi "
            "completamente representado por los dos factores retenidos."
        ),
        dash_table.DataTable(
            data=comunalidades.to_dict("records"),
            columns=[{"name": c, "id": c} for c in comunalidades.columns],
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
        ),

        section_title("Comparación con el ACP"),
        section_text(
            "El siguiente gráfico compara, para cada coordenada CLR, las "
            "cargas obtenidas en el ACP (sin rotar) con las cargas "
            "rotadas del AFE. La similitud entre ambas soluciones es una "
            "evidencia de robustez: la estructura de dos dimensiones no es "
            "un artefacto del método de extracción."
        ),
        figure_block(
            fig_acp_vs_afe(df),
            caption=html.Span([
                html.Strong("Figura 6.2. "),
                "Cargas del ACP (CP1, CP2, sin rotar) frente a las cargas "
                "del AFE (F1, F2, rotación varimax) para cada coordenada "
                "CLR.",
            ]),
            fig_id="fig-afe-vs-acp",
        ),

        method_card(
            "Lectura de resultados y siguientes pasos",
            [
                "La rotación varimax separa con mayor claridad dos "
                "patrones: F1 está dominado por ", html.B("Educación"),
                ", ", html.B("Salud"), " y, con signo opuesto, ",
                html.B("Libre Destinación"), "; F2 está dominado por ",
                html.B("Cultura y Deporte"), " y ", html.B("Libre Inversión"),
                ". Esta estructura es coherente con el ACP del Capítulo 05 "
                "(que mezcla ambos patrones en CP1), lo que respalda la "
                "validez de una solución de dos dimensiones para resumir el "
                "gasto social. En el Capítulo 07 se utilizan las "
                "puntuaciones CP1-CP2 del ACP como insumo para segmentar "
                "los departamentos mediante K-Means.",
            ],
        ),
    ])
