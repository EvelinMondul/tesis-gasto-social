"""
pages/eda.py
============

Capítulo 02: Análisis Exploratorio de Datos. Caracteriza la composición
del gasto social departamental (D=6) en proporciones brutas, antes de
aplicar la transformación CLR (Capítulo 03).
"""

import dash
from dash import dash_table, html
import dash_bootstrap_components as dbc

from analysis.coda import load_gasto
from analysis.descriptive import resumen_composicion
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.eda_figures import (
    fig_boxplot_componentes,
    fig_composicion_departamentos,
    fig_composicion_regional,
    fig_poblacion_vs_gasto,
)

dash.register_page(__name__, path="/eda", name="Análisis exploratorio", order=2)


def layout():
    df = load_gasto()
    resumen = resumen_composicion(df)

    tab_composicion = html.Div([
        section_title("Composición del gasto por departamento"),
        section_text(
            "Cada barra representa el 100 % del gasto social de un "
            "departamento, distribuido entre los seis componentes "
            "considerados. El orden de los departamentos corresponde a la "
            "proporción destinada a Educación, el componente con mayor peso "
            "promedio en la composición."
        ),
        figure_block(
            fig_composicion_departamentos(df),
            caption=html.Span([
                html.Strong("Figura 2.1. "),
                "Composición porcentual del gasto social por departamento, 2024. "
                "Fuente: elaboración propia a partir de datos de ejecución "
                "presupuestal departamental.",
            ]),
            fig_id="fig-eda-composicion-departamentos",
            height=900,
        ),
    ], className="pt-3")

    tab_distribucion = html.Div([
        section_title("Distribución de cada componente"),
        section_text(
            "Los diagramas de caja muestran la dispersión entre "
            "departamentos de la proporción destinada a cada componente. "
            "Educación y Salud concentran la mayor parte del gasto, mientras "
            "que Cultura y Deporte y Libre Destinación presentan los valores "
            "más bajos y, en el caso de Libre Destinación, el coeficiente de "
            "variación más alto —incluyendo el cero estructural de Bogotá "
            "D.C., abordado en el Capítulo 03."
        ),
        figure_block(
            fig_boxplot_componentes(df),
            caption=html.Span([
                html.Strong("Figura 2.2. "),
                "Distribución de la proporción de cada componente del gasto "
                "social entre los 33 departamentos (n=33). Cada punto "
                "representa un departamento.",
            ]),
            fig_id="fig-eda-boxplot",
        ),
    ], className="pt-3")

    tab_regional = html.Div([
        dbc.Row([
            dbc.Col([
                section_title("Composición promedio por región"),
                section_text(
                    "Agregación de los perfiles departamentales según las "
                    "seis regiones geográficas de Colombia (Andina, "
                    "Caribe, Pacífica, Orinoquía, Amazonía e Insular)."
                ),
                figure_block(
                    fig_composicion_regional(df),
                    caption=html.Span([
                        html.Strong("Figura 2.3. "),
                        "Composición promedio del gasto social por región "
                        "geográfica.",
                    ]),
                    fig_id="fig-eda-regional",
                ),
            ], md=6),
            dbc.Col([
                section_title("Población y gasto en Educación"),
                section_text(
                    "Relación entre el tamaño poblacional del departamento "
                    "(escala logarítmica) y la proporción de su gasto social "
                    "destinada a Educación."
                ),
                figure_block(
                    fig_poblacion_vs_gasto(df),
                    caption=html.Span([
                        html.Strong("Figura 2.4. "),
                        "Población 2024 vs. proporción del gasto en "
                        "Educación, por departamento.",
                    ]),
                    fig_id="fig-eda-poblacion",
                ),
            ], md=6),
        ]),
    ], className="pt-3")

    tab_estadisticos = html.Div([
        section_title("Estadísticos descriptivos por componente"),
        section_text(
            "Media, mediana, desviación estándar y coeficiente de variación "
            "(CV) de la proporción de cada componente, calculados sobre las "
            "proporciones brutas (sin transformar)."
        ),
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
            "Las proporciones brutas presentadas en este capítulo permiten "
            "una primera caracterización del gasto social, pero no son "
            "adecuadas para calcular correlaciones o aplicar ACP "
            "directamente: al estar restringidas a sumar 1 (cierre de la "
            "simplex), inducen correlaciones espurias entre componentes. El "
            "Capítulo 03 introduce la transformación log-razón centrada "
            "(CLR), que remueve esta restricción y habilita el análisis "
            "multivariado posterior.",
        ),
    ], className="pt-3")

    return html.Div([
        page_header(
            "Capítulo 02",
            "Análisis Exploratorio de Datos",
            "Composición del gasto social en los 33 departamentos de "
            "Colombia (2024), en proporciones brutas, previo a la "
            "transformación log-razón centrada (CLR).",
        ),

        dbc.Row([
            dbc.Col(kpi_card(f"{df['Población 2024'].sum():,.0f}".replace(",", "."),
                             "Población total cubierta (2024)"), md=3, className="mb-3"),
            dbc.Col(kpi_card("6", "Componentes del gasto social"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{resumen.loc[resumen['Componente']=='Educación', 'Media (%)'].values[0]:.1f}%",
                             "Gasto promedio en Educación"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{resumen.loc[resumen['Componente']=='Libre Destinación', 'CV'].values[0]:.2f}",
                             "Coef. de variación más alto: Libre Destinación"), md=3, className="mb-3"),
        ], className="mb-3"),

        dbc.Tabs([
            dbc.Tab(tab_composicion, label="Composición por departamento", tab_id="tab-eda-composicion"),
            dbc.Tab(tab_distribucion, label="Distribución por componente", tab_id="tab-eda-distribucion"),
            dbc.Tab(tab_regional, label="Análisis regional y poblacional", tab_id="tab-eda-regional"),
            dbc.Tab(tab_estadisticos, label="Estadísticos descriptivos", tab_id="tab-eda-estadisticos"),
        ], id="tabs-eda", active_tab="tab-eda-composicion"),
    ])
