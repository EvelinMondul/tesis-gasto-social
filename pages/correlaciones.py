"""
pages/correlaciones.py
========================

Capítulo 04: Correlaciones. Justifica el uso del coeficiente de Spearman
mediante la prueba de Shapiro-Wilk sobre las coordenadas CLR y presenta la
matriz de correlación resultante.
"""

import dash
from dash import dash_table, html

from analysis.coda import load_gasto
from analysis.correlations import pares_significativos
from components.elements import figure_block, method_card, page_header, section_title, section_text
from figures.correlation_figures import fig_heatmap_spearman, fig_shapiro

dash.register_page(__name__, path="/correlaciones", name="Correlaciones (CLR)", order=4)


def layout():
    df = load_gasto()
    pares = pares_significativos(df)

    return html.Div([
        page_header(
            "Capítulo 04",
            "Correlaciones",
            "Asociación entre los componentes del gasto social, medida "
            "sobre las coordenadas CLR mediante el coeficiente de "
            "correlación de Spearman.",
        ),

        section_title("Justificación: Spearman frente a Pearson"),
        section_text([
            "El coeficiente de correlación de Pearson asume que ambas "
            "variables siguen una distribución aproximadamente normal y "
            "que su relación es lineal. La prueba de Shapiro-Wilk, aplicada "
            "a cada una de las seis coordenadas CLR, evalúa esta condición: "
            "la hipótesis nula H", html.Sub("0"),
            " establece que la variable proviene de una distribución "
            "normal, y se rechaza cuando el valor p es menor que el nivel "
            "de significancia (α = 0.05).",
        ]),
        figure_block(
            fig_shapiro(df),
            caption=html.Span([
                html.Strong("Figura 4.1. "),
                "Estadístico W de Shapiro-Wilk por componente CLR. Las "
                "barras en gris grafito corresponden a componentes cuya "
                "hipótesis de normalidad se rechaza (p < 0.05).",
            ]),
            fig_id="fig-corr-shapiro",
        ),
        method_card(
            "Decisión metodológica",
            "Cuatro de los seis componentes (Cultura y Deporte, Educación, "
            "Libre Destinación y Libre Inversión) rechazan la hipótesis de "
            "normalidad sobre sus coordenadas CLR (p < 0.05); solo Agua "
            "potable y Salud son compatibles con la normalidad. Dado que la "
            "mayoría de las variables no son normales, se utiliza el "
            "coeficiente de correlación de Spearman —basado en rangos, no "
            "paramétrico y robusto a valores atípicos— para construir la "
            "matriz de asociación entre componentes.",
        ),

        section_title("Matriz de correlación de Spearman (CLR)"),
        section_text(
            "El mapa de calor muestra el coeficiente de correlación de "
            "Spearman (rho) entre cada par de coordenadas CLR. Valores "
            "cercanos a 1 (gris grafito) indican asociación positiva fuerte; "
            "valores cercanos a -1 (azul oscuro), asociación negativa "
            "fuerte; valores cercanos a 0 (blanco), ausencia de asociación "
            "monótona."
        ),
        figure_block(
            fig_heatmap_spearman(df),
            caption=html.Span([
                html.Strong("Figura 4.2. "),
                "Matriz de correlación de Spearman entre las coordenadas "
                "CLR de los seis componentes del gasto social (n=33).",
            ]),
            fig_id="fig-corr-heatmap",
            height=520,
        ),

        section_title("Pares de componentes con correlación significativa"),
        section_text(
            "La siguiente tabla ordena los 15 pares de componentes según el "
            "valor absoluto de su coeficiente de correlación de Spearman, "
            "indicando cuáles resultan estadísticamente significativos al "
            "nivel α = 0.05."
        ),
        dash_table.DataTable(
            data=pares.to_dict("records"),
            columns=[{"name": c, "id": c} for c in pares.columns],
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
        ),

        method_card(
            "Lectura de resultados",
            [
                "Las asociaciones más fuertes se observan entre ",
                html.B("Educación y Libre Destinación"), " (negativa), ",
                html.B("Libre Destinación y Salud"), " (negativa), y ",
                html.B("Libre Inversión y Salud"), " (negativa). En términos "
                "composicionales, esto indica que los departamentos que "
                "destinan una mayor proporción relativa de su gasto a "
                "Educación tienden a destinar una proporción relativamente "
                "menor a Libre Destinación, y que Libre Inversión y Salud "
                "compiten por una porción similar del presupuesto. Estos "
                "siete pares significativos motivan la reducción de "
                "dimensionalidad mediante ACP (Capítulo 05) y AFE "
                "(Capítulo 06).",
            ],
        ),
    ])
