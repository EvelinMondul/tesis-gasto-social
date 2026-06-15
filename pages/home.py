"""
pages/home.py
==============

Capítulo 00: Portada. Presenta el título del trabajo de grado, un resumen
ejecutivo del enfoque metodológico (CoDA / CLR / ACP / AFE / Clustering /
IPM) y una guía de navegación hacia los diez capítulos del dashboard.
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

from components.elements import kpi_card, section_title, section_text
from components.sidebar import NAV_ITEMS

dash.register_page(__name__, path="/", name="Portada", order=0)


CAPITULOS_DESC = {
    "/introduccion": "Planteamiento del problema, objetivos y estructura del trabajo.",
    "/eda": "Estadística descriptiva de la composición del gasto social por departamento.",
    "/composicional": "Cierre, perturbación y transformación CLR de las seis componentes del gasto.",
    "/correlaciones": "Matriz de correlación de Spearman sobre coordenadas CLR.",
    "/acp": "KMO, Bartlett, scree plot y biplot composicional del ACP.",
    "/factorial": "Modelo de factores latentes del análisis factorial exploratorio.",
    "/clustering": "Agrupamiento K-Means de los 33 departamentos según su perfil de gasto.",
    "/ipm": "Contraste entre los perfiles de gasto y el Índice de Pobreza Multidimensional.",
    "/mapas": "Visualización geográfica de composición, clusters e IPM.",
    "/conclusiones": "Síntesis de hallazgos, limitaciones y recomendaciones.",
}


def _chapter_card(num, label, href):
    return dbc.Col(
        dash.html.A(
            html.Div([
                html.Div(num, style={
                    "fontFamily": "var(--font-heading)",
                    "fontSize": "0.85rem",
                    "color": "var(--color-petroleo)",
                    "fontWeight": "600",
                    "marginBottom": "4px",
                }),
                html.Div(label, style={
                    "fontWeight": "600",
                    "color": "var(--color-azul-oscuro)",
                    "marginBottom": "6px",
                }),
                html.Div(CAPITULOS_DESC.get(href, ""), style={
                    "fontSize": "0.82rem",
                    "color": "var(--color-texto-secundario)",
                }),
            ], className="kpi-card", style={"height": "100%"}),
            href=href,
            className="chapter-card-link",
        ),
        md=4, className="mb-3",
    )


def layout():
    chapter_cards = [
        _chapter_card(num, label, href)
        for num, label, href in NAV_ITEMS if href != "/"
    ]

    return html.Div([
        html.Span("Trabajo de grado · Maestría en Estadística Aplicada", className="chapter-tag"),
        html.Div(
            "Análisis Composicional del Gasto Social Departamental en Colombia (2024) "
            "y su relación con la Pobreza Multidimensional",
            className="page-title",
            style={"fontSize": "1.85rem", "maxWidth": "880px"},
        ),
        html.Div(
            "Un enfoque de Análisis de Datos Composicionales (CoDA) mediante "
            "transformación log-razón centrada (CLR), Análisis de Componentes "
            "Principales, Análisis Factorial Exploratorio y agrupamiento "
            "K-Means sobre la estructura de asignación del gasto social en "
            "los 33 departamentos de Colombia, contrastada con el Índice de "
            "Pobreza Multidimensional (IPM) 2025.",
            className="page-subtitle",
            style={"maxWidth": "880px"},
        ),

        dbc.Row([
            dbc.Col(kpi_card("33", "Departamentos / distritos analizados", variant=1), md=3, className="mb-3"),
            dbc.Col(kpi_card("6", "Componentes del gasto social (D)", variant=2), md=3, className="mb-3"),
            dbc.Col(kpi_card("2024", "Año fiscal de referencia", variant=3), md=3, className="mb-3"),
            dbc.Col(kpi_card("C1 / C3 + Atípico", "Perfiles de gasto (Bogotá D.C. como caso atípico)", variant=4), md=3, className="mb-3"),
        ], className="mb-3"),

        section_title("Resumen del enfoque metodológico"),
        section_text([
            "El gasto social departamental se modela como un dato ",
            html.B("composicional"),
            ": cada departamento distribuye su gasto total entre seis "
            "componentes (Agua potable, Cultura y Deporte, Educación, Libre "
            "Destinación, Libre Inversión y Salud) que suman necesariamente "
            "el 100 % del presupuesto. Esta restricción de suma constante "
            "(la ",
            html.Em("simplex"),
            ") impide aplicar directamente técnicas estadísticas clásicas "
            "como la correlación de Pearson o el Análisis de Componentes "
            "Principales sobre las proporciones brutas, por lo que el "
            "análisis se realiza sobre las coordenadas log-razón centradas "
            "(CLR), siguiendo el marco de Aitchison (1986).",
        ]),
        section_text([
            "A partir de estas coordenadas se construye la matriz de "
            "correlación de Spearman (Capítulo 4), se evalúa la adecuación "
            "muestral mediante el índice KMO y la prueba de esfericidad de "
            "Bartlett, y se ejecuta un Análisis de Componentes Principales "
            "(Capítulo 5) y un Análisis Factorial Exploratorio (Capítulo 6) "
            "para reducir la dimensionalidad del problema. Sobre las "
            "puntuaciones factoriales de 32 departamentos se aplica un "
            "agrupamiento K-Means (Capítulo 7) que revela dos perfiles de "
            "gasto social (C1 y C3); Bogotá D.C. se excluye del K-Means y "
            "se presenta como caso atípico. Los tres grupos resultantes "
            "(C1, C3 y Atípico) se contrastan con el Índice de Pobreza "
            "Multidimensional (Capítulo 8) y se visualizan territorialmente "
            "(Capítulo 9).",
        ]),

        section_title("Navegación por capítulos"),
        dbc.Row(chapter_cards),
    ])
