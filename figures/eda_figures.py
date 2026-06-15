"""
figures/eda_figures.py
=======================

Figuras del Capítulo 02 (Análisis Exploratorio de Datos): composición del
gasto social por departamento y por región, y distribución de cada
componente entre los 33 departamentos.
"""

import plotly.express as px
import plotly.graph_objects as go

from analysis.coda import COMPONENTES, load_gasto
from config import COLORS, SECUENCIA_CATEGORICA


def fig_composicion_departamentos(df=None, ordenar_por="Educación"):
    """Barras horizontales 100% apiladas: composición del gasto por
    departamento, ordenadas por la proporción del componente indicado."""
    if df is None:
        df = load_gasto()

    df_sorted = df.sort_values(f"P_{ordenar_por}", ascending=True)

    fig = go.Figure()
    for comp, color in zip(COMPONENTES, SECUENCIA_CATEGORICA):
        fig.add_trace(go.Bar(
            y=df_sorted["Departamento"],
            x=df_sorted[f"P_{comp}"] * 100,
            name=comp,
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>" + comp + ": %{x:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Composición del gasto social por departamento (2024)"),
        xaxis_title="Porcentaje del gasto social total (%)",
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900,
        margin=dict(l=140),
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    return fig


def fig_boxplot_componentes(df=None):
    """Diagramas de caja de la proporción de cada componente (%) a través
    de los 33 departamentos."""
    if df is None:
        df = load_gasto()

    fig = go.Figure()
    for comp, color in zip(COMPONENTES, SECUENCIA_CATEGORICA):
        fig.add_trace(go.Box(
            y=df[f"P_{comp}"] * 100,
            name=comp,
            marker_color=color,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            text=df["Departamento"],
            hovertemplate="%{text}<br>" + comp + ": %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Distribución de cada componente del gasto social (n=33)"),
        yaxis_title="Porcentaje del gasto social total (%)",
        showlegend=False,
        height=480,
    )
    return fig


def fig_composicion_regional(df=None):
    """Composición promedio del gasto social por región geográfica."""
    if df is None:
        df = load_gasto()

    reg = df.groupby("region")[[f"P_{c}" for c in COMPONENTES]].mean() * 100
    reg = reg.rename(columns=lambda c: c[2:])
    reg = reg.reset_index()

    fig = go.Figure()
    for comp, color in zip(COMPONENTES, SECUENCIA_CATEGORICA):
        fig.add_trace(go.Bar(
            x=reg["region"],
            y=reg[comp],
            name=comp,
            marker_color=color,
            hovertemplate="%{x}<br>" + comp + ": %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Composición promedio del gasto social por región"),
        yaxis_title="Porcentaje del gasto social total (%)",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def fig_poblacion_vs_gasto(df=None):
    """Dispersión: población 2024 vs. proporción de gasto en Educación,
    coloreado por región (relación tamaño/composición)."""
    if df is None:
        df = load_gasto()

    fig = px.scatter(
        df,
        x="Población 2024",
        y=df["P_Educación"] * 100,
        color="region",
        hover_name="Departamento",
        color_discrete_sequence=SECUENCIA_CATEGORICA,
        labels={"y": "Gasto en Educación (%)", "Población 2024": "Población (2024)"},
        log_x=True,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color=COLORS["bg"])))
    fig.update_layout(
        title=dict(text="Población departamental y proporción del gasto en Educación"),
        height=480,
        yaxis_title="Gasto en Educación (%)",
        xaxis_title="Población 2024 (escala logarítmica)",
    )
    return fig
