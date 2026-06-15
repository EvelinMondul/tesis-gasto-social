"""
figures/map_figures.py
========================

Mapas coropléticos del Capítulo 09: distribución territorial del
componente de gasto predominante (Cap. 03), los clusters de gasto
social (Cap. 07), el Índice de Pobreza Multidimensional (Cap. 08) y la
proporción de gasto en Educación (Cap. 08) en los 33 departamentos de
Colombia.
"""

import plotly.express as px

from analysis.coda import COMPONENTES
from analysis.geo import datos_mapas, load_geojson
from config import COLOR_CLUSTER, COLORS, ESCALA_CONTINUA, SECUENCIA_CATEGORICA

# Colores por componente del gasto, coherentes con la secuencia
# categórica usada en los Capítulos 03-06.
COLOR_COMPONENTE = dict(zip(COMPONENTES, SECUENCIA_CATEGORICA))


def _mapa_base(fig, titulo, colorbar_title=None):
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=COLORS["bg"],
    )
    fig.update_traces(marker_line_color=COLORS["bg"], marker_line_width=0.6)
    fig.update_layout(
        title=dict(text=titulo),
        height=620,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.01),
    )
    if colorbar_title is not None:
        fig.update_layout(coloraxis_colorbar=dict(title=colorbar_title, len=0.6))
    return fig


def fig_mapa_componente_predominante(df=None):
    """Mapa categórico del componente del gasto social con mayor
    proporción en cada departamento (Capítulo 03)."""
    datos = datos_mapas(df)
    fig = px.choropleth(
        datos,
        geojson=load_geojson(),
        locations="Departamento",
        featureidkey="properties.Departamento",
        color="Componente predominante",
        color_discrete_map=COLOR_COMPONENTE,
        category_orders={"Componente predominante": COMPONENTES},
        hover_name="Departamento",
        hover_data={"Componente predominante": True},
    )
    fig.update_layout(legend_title_text="Componente predominante")
    return _mapa_base(fig, "Componente de gasto predominante por departamento")


def fig_mapa_clusters(df=None):
    """Mapa categórico de los clusters de gasto social (Capítulo 07)."""
    datos = datos_mapas(df)
    fig = px.choropleth(
        datos,
        geojson=load_geojson(),
        locations="Departamento",
        featureidkey="properties.Departamento",
        color="cluster",
        color_discrete_map=COLOR_CLUSTER,
        category_orders={"cluster": ["C1", "C3", "Atípico"]},
        hover_name="Departamento",
        hover_data={"cluster": True, "IPM_pct": ":.1f"},
    )
    fig.update_layout(legend_title_text="Cluster")
    return _mapa_base(fig, "Clusters de gasto social por departamento (Bogotá D.C. = caso atípico)")


def fig_mapa_ipm(df=None):
    """Mapa continuo del Índice de Pobreza Multidimensional (Capítulo 08)."""
    datos = datos_mapas(df)
    fig = px.choropleth(
        datos,
        geojson=load_geojson(),
        locations="Departamento",
        featureidkey="properties.Departamento",
        color="IPM_pct",
        color_continuous_scale=ESCALA_CONTINUA,
        hover_name="Departamento",
        hover_data={"IPM_pct": ":.1f", "cluster": True},
    )
    return _mapa_base(fig, "Índice de Pobreza Multidimensional (%) por departamento", colorbar_title="IPM (%)")


def fig_mapa_educacion(df=None):
    """Mapa continuo de la proporción del gasto destinada a Educación
    (componente con la correlación más fuerte con el IPM, Capítulo 08)."""
    datos = datos_mapas(df)
    fig = px.choropleth(
        datos,
        geojson=load_geojson(),
        locations="Departamento",
        featureidkey="properties.Departamento",
        color="Educación (%)",
        color_continuous_scale=ESCALA_CONTINUA,
        hover_name="Departamento",
        hover_data={"Educación (%)": ":.1f"},
    )
    return _mapa_base(fig, "Proporción del gasto social en Educación (%) por departamento", colorbar_title="Educación (%)")
