"""
pages/composicional.py
=======================

Capítulo 03: Análisis Composicional. Presenta las operaciones básicas del
marco CoDA (cierre, reemplazo de ceros) y la transformación log-razón
centrada (CLR) aplicada a los seis componentes del gasto social.
"""

import dash
from dash import dash_table, html
import dash_bootstrap_components as dbc

from analysis.coda import load_gasto
from analysis.descriptive import resumen_clr
from components.elements import figure_block, method_card, page_header, section_title, section_text
from figures.coda_figures import fig_clr_boxplot, fig_reemplazo_ceros_bogota, fig_simplex_vs_clr

dash.register_page(__name__, path="/composicional", name="Análisis Composicional", order=3)


def layout():
    df = load_gasto()
    resumen = resumen_clr(df)

    return html.Div([
        page_header(
            "Capítulo 03",
            "Análisis Composicional",
            "El gasto social departamental como dato composicional: cierre, "
            "reemplazo de ceros y transformación log-razón centrada (CLR).",
        ),

        section_title("El gasto social como dato composicional"),
        section_text([
            "Cada departamento distribuye su gasto social total entre seis "
            "componentes que, por definición, suman el 100 % del "
            "presupuesto. Esta restricción de suma constante define la "
            "región muestral del problema como un símplex D-dimensional "
            "S",
            html.Sup("D"),
            " = {x = (x", html.Sub("1"), ", …, x", html.Sub("D"),
            ") : x", html.Sub("i"), " > 0, Σx", html.Sub("i"), " = 1}.",
            " Cualquier operación válida sobre estos datos debe respetar "
            "esta geometría; la operación de ", html.B("cierre"),
            " (closure) reescala un vector positivo cualquiera para que "
            "sus componentes sumen 1, y es la operación que se aplicó al "
            "construir las proporciones P_* del Capítulo 02.",
        ]),

        section_title("Reemplazo de ceros: el caso de Bogotá D.C."),
        section_text([
            "La transformación CLR requiere tomar el logaritmo de cada "
            "componente, lo cual no está definido cuando una componente es "
            "exactamente cero. Bogotá D.C. no recibe recursos por el rubro "
            "de ", html.I("Libre Destinación"), ", lo que constituye un ",
            html.B("cero estructural"), ". Siguiendo a Martín-Fernández et "
            "al. (2003), este cero se sustituye por un valor pequeño "
            "(δ = 0.0001) y el resto de componentes de la fila se reescala "
            "multiplicativamente para preservar el cierre (suma = 1).",
        ]),
        figure_block(
            fig_reemplazo_ceros_bogota(df),
            caption=html.Span([
                html.Strong("Figura 3.1. "),
                "Composición de Bogotá D.C. antes y después del reemplazo "
                "multiplicativo de ceros (δ = 0.0001). El resto de "
                "componentes se reduce marginalmente para mantener la suma "
                "en 100 %.",
            ]),
            fig_id="fig-coda-zero-replacement",
        ),

        section_title("Transformación log-razón centrada (CLR)"),
        section_text([
            "La transformación CLR proyecta cada composición x = (x",
            html.Sub("1"), ", …, x", html.Sub("D"), ") de la simplex S",
            html.Sup("D"), " a un vector en R", html.Sup("D"),
            " mediante:",
        ]),
        html.Div(
            "clr(x)ᵢ = ln(xᵢ) − (1/D) · Σⱼ ln(xⱼ)",
            style={
                "fontFamily": "var(--font-heading)",
                "fontSize": "1.05rem",
                "textAlign": "center",
                "padding": "14px",
                "backgroundColor": "var(--color-bg-subtle)",
                "borderRadius": "6px",
                "marginBottom": "16px",
                "maxWidth": "520px",
            },
        ),
        section_text([
            "Las coordenadas CLR resultantes suman 0 por construcción "
            "(rango D−1) y ya no están sujetas a la restricción de suma "
            "constante, lo que habilita el cálculo de correlaciones, ACP, "
            "AFE y distancias euclídeas estándar —estas últimas equivalentes "
            "a la distancia de Aitchison en la simplex original.",
        ]),
        figure_block(
            fig_simplex_vs_clr(df, "Bogotá"),
            caption=html.Span([
                html.Strong("Figura 3.2. "),
                "Comparación, para Bogotá D.C., entre las proporciones "
                "originales (barras, eje izquierdo, suma=100%) y las "
                "coordenadas CLR correspondientes (línea, eje derecho, "
                "suma=0).",
            ]),
            fig_id="fig-coda-simplex-vs-clr",
        ),

        section_title("Distribución de las coordenadas CLR"),
        section_text(
            "A diferencia de las proporciones brutas del Capítulo 02 "
            "(acotadas entre 0 % y 100 % y sesgadas a la derecha), las "
            "coordenadas CLR se distribuyen alrededor de 0 y permiten "
            "comparar la magnitud relativa de cada componente frente a la "
            "media geométrica de la composición de cada departamento."
        ),
        figure_block(
            fig_clr_boxplot(df),
            caption=html.Span([
                html.Strong("Figura 3.3. "),
                "Distribución de las coordenadas CLR de los seis "
                "componentes del gasto social (n=33). Educación y Salud "
                "presentan coordenadas CLR positivas (por encima de la "
                "media geométrica de cada departamento); Cultura y Deporte "
                "y Libre Destinación, las más negativas.",
            ]),
            fig_id="fig-coda-clr-boxplot",
        ),

        section_title("Estadísticos descriptivos de las coordenadas CLR"),
        dash_table.DataTable(
            data=resumen.to_dict("records"),
            columns=[{"name": c, "id": c} for c in resumen.columns],
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

        method_card(
            "Nota metodológica",
            [
                "Los coeficientes de asimetría y curtosis de varios "
                "componentes (en particular Libre Destinación y Cultura y "
                "Deporte) sugieren desviaciones importantes de la "
                "normalidad. El Capítulo 04 formaliza esta evaluación "
                "mediante la prueba de Shapiro-Wilk y, con base en sus "
                "resultados, justifica el uso del coeficiente de "
                "correlación de Spearman —en lugar de Pearson— para el "
                "análisis de asociación entre componentes.",
            ],
        ),
    ])
