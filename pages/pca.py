"""
pages/pca.py
=============

Capítulo 05: Análisis de Componentes Principales (ACP) sobre las
coordenadas CLR estandarizadas, precedido por las pruebas de adecuación
muestral KMO (vía pseudoinversa) y Bartlett.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from analysis.coda import load_gasto
from analysis.pca import kmo_bartlett, run_pca
from components.elements import figure_block, kpi_card, method_card, page_header, section_title, section_text
from figures.pca_figures import fig_biplot, fig_esquema_pca, fig_loadings, fig_scree

dash.register_page(__name__, path="/acp", name="Componentes Principales (ACP)", order=5)


def _tabla(df, table_id):
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
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
        page_size=12,
    )


def layout():
    df = load_gasto()
    kb = kmo_bartlett(df)
    res = run_pca(df)

    kmo_tabla = kb["kmo_por_variable"].rename(columns={"MSA": "KMO (MSA)"})
    scores_tabla = res["scores"][["Departamento", "region", "CP1", "CP2"]].rename(
        columns={"region": "Región"}
    )

    tab_adecuacion = html.Div([
        section_title("Adecuación muestral: KMO y prueba de Bartlett"),
        section_text([
            "Antes de extraer los componentes principales se evalúa si la "
            "matriz de correlación de las coordenadas CLR contiene "
            "suficiente información compartida para que la reducción de "
            "dimensionalidad sea pertinente. Dado que esta matriz es "
            "singular (rango D-1 = 5), el índice KMO se calcula a partir "
            "de la pseudoinversa de Moore-Penrose de la matriz de "
            "correlaciones.",
        ]),

        dbc.Row([
            dbc.Col(kpi_card(f"{kb['kmo_total']:.4f}", "Índice KMO global"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{kb['bartlett_chi2']:,.2f}".replace(",", "."),
                             "Chi² de Bartlett"), md=3, className="mb-3"),
            dbc.Col(kpi_card(f"{kb['bartlett_gl']}", "Grados de libertad"), md=3, className="mb-3"),
            dbc.Col(kpi_card("p < 0.001" if kb["bartlett_p"] < 0.001 else f"{kb['bartlett_p']:.4f}",
                             "Significancia de Bartlett"), md=3, className="mb-3"),
        ], className="mb-3"),

        _tabla(kmo_tabla, "tabla-pca-kmo"),

        method_card(
            "Interpretación",
            f"Un índice KMO de {kb['kmo_total']:.4f} se ubica en el rango "
            "\"aceptable\" (0.6-0.7) según los criterios usuales de Kaiser, "
            "lo que indica que la proporción de varianza común entre las "
            "coordenadas CLR es suficiente para justificar un ACP. El test "
            "de Bartlett rechaza con holgura la hipótesis nula de que la "
            "matriz de correlación es la identidad (p < 0.001), confirmando "
            "que las variables están suficientemente correlacionadas entre "
            "sí.",
        ),
    ], className="pt-3")

    tab_varianza = html.Div([
        section_title("Varianza explicada"),
        section_text(
            "El gráfico de sedimentación muestra el autovalor asociado a "
            "cada componente principal y la varianza acumulada explicada. "
            "Siguiendo el criterio de Kaiser (autovalor ≥ 1), se retienen "
            f"los primeros {res['n_kaiser']} componentes, que explican "
            f"conjuntamente el {res['var_acum'][res['n_kaiser']-1]:.1f}% de "
            "la varianza total."
        ),
        figure_block(
            fig_scree(df),
            caption=html.Span([
                html.Strong("Figura 5.1. "),
                "Autovalores (barras) y varianza acumulada (línea) de los "
                "seis componentes principales. La línea punteada marca el "
                "criterio de Kaiser (λ = 1).",
            ]),
            fig_id="fig-pca-scree",
        ),

        section_title("Cargas de los componentes retenidos"),
        section_text(
            "Las cargas indican la contribución de cada coordenada CLR a "
            "los componentes principales retenidos (CP1 y CP2). Cargas con "
            "el mismo signo indican que los componentes del gasto se "
            "mueven en la misma dirección dentro de la composición; cargas "
            "de signo opuesto indican una relación de compensación."
        ),
        figure_block(
            fig_loadings(df),
            caption=html.Span([
                html.Strong("Figura 5.2. "),
                "Cargas de cada coordenada CLR sobre CP1 y CP2.",
            ]),
            fig_id="fig-pca-loadings",
        ),
    ], className="pt-3")

    tab_biplot = html.Div([
        section_title("Biplot: puntuaciones departamentales y cargas"),
        section_text(
            "El biplot proyecta a los 33 departamentos sobre el plano "
            "CP1-CP2 (coloreados por región) y superpone los vectores de "
            "carga de cada coordenada CLR, permitiendo interpretar "
            "simultáneamente la posición de cada departamento y la "
            "dirección de las variables que la explican."
        ),
        figure_block(
            fig_biplot(df),
            caption=html.Span([
                html.Strong("Figura 5.3. "),
                "Biplot de las puntuaciones CP1-CP2 de los 33 departamentos, "
                "coloreados por región, con los vectores de carga de las "
                "seis coordenadas CLR.",
            ]),
            fig_id="fig-pca-biplot",
            height=600,
        ),

        method_card(
            "Lectura de resultados y siguientes pasos",
            [
                "CP1 está dominado por la oposición entre ",
                html.B("Libre Destinación"), " (carga positiva) y el resto "
                "de componentes —especialmente ", html.B("Educación"),
                " y ", html.B("Agua potable"), "— (cargas negativas), y "
                "separa claramente a Bogotá D.C. del resto de "
                "departamentos. CP2 contrasta ", html.B("Libre Inversión"),
                " y ", html.B("Cultura y Deporte"), " (cargas positivas) "
                "frente a ", html.B("Salud"), " y ", html.B("Educación"),
                " (cargas negativas). En el Capítulo 06 se contrasta esta "
                "solución con un Análisis Factorial Exploratorio (AFE), y "
                "en el Capítulo 07 las puntuaciones CP1-CP2 se emplean "
                "como insumo para la segmentación de departamentos "
                "mediante K-Means.",
            ],
        ),
    ], className="pt-3")

    tab_esquema = html.Div([
        section_title("Esquema del proceso metodológico"),
        section_text(
            "El siguiente diagrama resume el flujo completo del análisis "
            "composicional aplicado en este capítulo: desde las "
            "proporciones brutas del gasto social hasta las puntuaciones "
            "CP1-CP2 que alimentan la segmentación K-Means del Capítulo 07."
        ),
        figure_block(
            fig_esquema_pca(df),
            caption=html.Span([
                html.Strong("Figura 5.4. "),
                "Esquema metodológico del Análisis de Componentes "
                "Principales sobre coordenadas CLR.",
            ]),
            fig_id="fig-pca-esquema",
            height=300,
        ),

        section_title("Tabla de resultados del ACP"),
        section_text(
            "Puntuaciones de los 33 departamentos sobre los dos "
            "componentes retenidos (CP1 y CP2), agrupados por región."
        ),
        _tabla(scores_tabla, "tabla-pca-scores"),
    ], className="pt-3")

    return html.Div([
        page_header(
            "Capítulo 05",
            "Análisis de Componentes Principales",
            "Reducción de la dimensionalidad de las seis coordenadas CLR a "
            "un número menor de componentes ortogonales que resuman la "
            "estructura de varianza-covarianza del gasto social.",
        ),

        dbc.Tabs([
            dbc.Tab(tab_adecuacion, label="Adecuación muestral", tab_id="tab-pca-adecuacion"),
            dbc.Tab(tab_varianza, label="Varianza explicada y cargas", tab_id="tab-pca-varianza"),
            dbc.Tab(tab_biplot, label="Biplot CP1-CP2", tab_id="tab-pca-biplot"),
            dbc.Tab(tab_esquema, label="Esquema y resultados", tab_id="tab-pca-esquema"),
        ], id="tabs-pca", active_tab="tab-pca-adecuacion"),
    ])
