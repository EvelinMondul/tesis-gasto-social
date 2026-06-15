"""
figures/coda_figures.py
========================

Figuras del Capítulo 03 (Análisis Composicional): efecto del reemplazo
multiplicativo de ceros (caso Bogotá D.C.) y distribución de las
coordenadas log-razón centradas (CLR) para los seis componentes del
gasto social.
"""

import numpy as np
import plotly.graph_objects as go

from analysis.coda import (
    COMPONENTES,
    DELTA_ZERO_REPLACEMENT,
    closure,
    get_clr_coords,
    get_composicion,
    load_gasto,
    multiplicative_replacement,
)
from config import COLORS, SECUENCIA_CATEGORICA


def fig_reemplazo_ceros_bogota(df=None):
    """Compara la composición de Bogotá D.C. antes y después del
    reemplazo multiplicativo de ceros aplicado a 'Libre Destinación'."""
    if df is None:
        df = load_gasto()

    bogota = df[df["Departamento"] == "Bogotá"]
    comp_original = closure(bogota[[f"P_{c}" for c in COMPONENTES]].values)[0] * 100
    comp_reemplazada = multiplicative_replacement(
        bogota[[f"P_{c}" for c in COMPONENTES]].values
    )[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=COMPONENTES, y=comp_original, name="Proporción original (con cero)",
        marker_color=COLORS["gris_medio"],
    ))
    fig.add_trace(go.Bar(
        x=COMPONENTES, y=comp_reemplazada, name="Tras reemplazo multiplicativo",
        marker_color=COLORS["petroleo"],
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Efecto del reemplazo multiplicativo de ceros — Bogotá D.C."),
        yaxis_title="Porcentaje del gasto social (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def fig_clr_boxplot(df=None):
    """Diagramas de caja de las coordenadas CLR de cada componente."""
    Z = get_clr_coords(df)

    fig = go.Figure()
    for comp, color in zip(Z.columns, SECUENCIA_CATEGORICA):
        fig.add_trace(go.Box(
            y=Z[comp], name=comp, marker_color=color,
            boxpoints="all", jitter=0.4, pointpos=0,
            hovertemplate=comp + ": %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gris_medio"])
    fig.update_layout(
        title=dict(text="Distribución de las coordenadas CLR por componente"),
        yaxis_title="Coordenada CLR",
        showlegend=False,
        height=480,
    )
    return fig


def fig_simplex_vs_clr(df=None, departamento="Bogotá"):
    """Compara, para un departamento, las proporciones originales
    (restringidas a sumar 1) con las coordenadas CLR correspondientes
    (sin restricción de suma, centradas en torno a 0)."""
    if df is None:
        df = load_gasto()

    comp = get_composicion(df) * 100
    Z = get_clr_coords(df)
    idx = df.index[df["Departamento"] == departamento][0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=COMPONENTES, y=comp.loc[idx].values, name="Proporción (%) — simplex (suma=100)",
        marker_color=COLORS["acento_suave"], yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=COMPONENTES, y=Z.loc[idx].values, name="Coordenada CLR (suma=0)",
        mode="markers+lines", marker=dict(size=10, color=COLORS["azul_oscuro"]),
        line=dict(color=COLORS["azul_oscuro"], dash="dot"), yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text=f"Proporciones vs. coordenadas CLR — {departamento}"),
        yaxis=dict(title="Proporción (%)", side="left"),
        yaxis2=dict(title="Coordenada CLR", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig
