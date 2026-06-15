"""
figures/pca_figures.py
========================

Figuras del Capítulo 05 (Análisis de Componentes Principales): gráfico de
sedimentación (scree plot), cargas de los componentes retenidos y biplot
de puntuaciones CP1-CP2.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis.pca import run_pca
from config import COLORS, SECUENCIA_CATEGORICA


def fig_scree(df=None):
    """Gráfico de sedimentación: autovalores (barras) y varianza
    acumulada (línea), con la línea de referencia del criterio de
    Kaiser (lambda = 1)."""
    res = run_pca(df)
    scree = res["scree"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scree["Componente"],
        y=scree["Autovalor"],
        name="Autovalor",
        marker_color=COLORS["petroleo"],
        hovertemplate="%{x}<br>Autovalor: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scree["Componente"],
        y=scree["Varianza acumulada (%)"],
        name="Varianza acumulada (%)",
        mode="lines+markers",
        marker=dict(color=COLORS["azul_oscuro"], size=8),
        line=dict(color=COLORS["azul_oscuro"], width=2),
        yaxis="y2",
        hovertemplate="%{x}<br>Varianza acumulada: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(
        y=1, line_dash="dash", line_color=COLORS["gris_medio"],
        annotation_text="Criterio de Kaiser (λ = 1)",
        annotation_position="top right",
        annotation_font=dict(size=11, color=COLORS["texto_secundario"]),
    )

    fig.update_layout(
        title=dict(text="Gráfico de sedimentación (scree plot)"),
        xaxis_title="Componente principal",
        yaxis=dict(title="Autovalor"),
        yaxis2=dict(
            title="Varianza acumulada (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
            ticksuffix="%",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def fig_loadings(df=None, componentes=("CP1", "CP2")):
    """Cargas (loadings) de cada coordenada CLR sobre los componentes
    principales retenidos, como barras agrupadas."""
    res = run_pca(df)
    loadings = res["loadings"]

    colores = [COLORS["petroleo"], COLORS["azul_oscuro"], COLORS["gris_medio"]]

    fig = go.Figure()
    for cp, color in zip(componentes, colores):
        fig.add_trace(go.Bar(
            x=loadings.index,
            y=loadings[cp],
            name=cp,
            marker_color=color,
            hovertemplate="%{x}<br>" + cp + ": %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=COLORS["gris_medio"], line_width=1)
    fig.update_layout(
        barmode="group",
        title=dict(text="Cargas de los componentes retenidos sobre las coordenadas CLR"),
        yaxis_title="Carga (loading)",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def fig_biplot(df=None):
    """Biplot: puntuaciones de los 33 departamentos sobre CP1-CP2,
    coloreadas por región, con los vectores de carga de cada
    coordenada CLR superpuestos."""
    res = run_pca(df)
    scores = res["scores"]
    loadings = res["loadings"]

    fig = px.scatter(
        scores,
        x="CP1",
        y="CP2",
        color="region",
        hover_name="Departamento",
        color_discrete_sequence=SECUENCIA_CATEGORICA,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color=COLORS["bg"])))

    # Vectores de carga, escalados al rango de las puntuaciones para que
    # sean visibles junto a los departamentos.
    escala = 0.85 * max(scores["CP1"].abs().max(), scores["CP2"].abs().max()) \
        / max(loadings["CP1"].abs().max(), loadings["CP2"].abs().max())

    for comp in loadings.index:
        x_end = loadings.loc[comp, "CP1"] * escala
        y_end = loadings.loc[comp, "CP2"] * escala
        fig.add_annotation(
            x=x_end, y=y_end, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=1.5,
            arrowcolor=COLORS["grafito"],
        )
        fig.add_annotation(
            x=x_end * 1.12, y=y_end * 1.12,
            xref="x", yref="y",
            text=f"<b>{comp}</b>",
            showarrow=False,
            font=dict(size=11, color=COLORS["grafito"]),
        )

    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=1)

    var_exp = res["var_exp"]
    fig.update_layout(
        title=dict(text="Biplot: puntuaciones departamentales y cargas (CP1-CP2)"),
        xaxis_title=f"CP1 ({var_exp[0]:.1f}% de varianza explicada)",
        yaxis_title=f"CP2 ({var_exp[1]:.1f}% de varianza explicada)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
    )
    return fig


def fig_esquema_pca(df=None):
    """Diagrama de flujo del proceso metodológico del ACP composicional:
    desde las proporciones brutas del gasto hasta las puntuaciones CP1-CP2
    empleadas en el Capítulo 07."""
    res = run_pca(df)
    var_acum_2 = res["var_acum"][1]

    pasos = [
        ("1. Proporciones\nbrutas del gasto\n(D=6, Cap. 02)", COLORS["gris_medio"]),
        ("2. Cierre + reemplazo\nde ceros + CLR\n(Cap. 03)", COLORS["petroleo_claro"]),
        ("3. Matriz de\ncorrelación\nSpearman (Cap. 04)", COLORS["petroleo"]),
        ("4. KMO y test de\nBartlett\n(adecuación muestral)", COLORS["petroleo"]),
        ("5. Extracción de\ncomponentes\n(criterio de Kaiser)", COLORS["azul_oscuro"]),
        (f"6. Puntuaciones\nCP1-CP2\n({var_acum_2:.1f}% var.)", COLORS["azul_oscuro"]),
    ]

    n = len(pasos)
    box_w, box_h, gap = 1.0, 0.62, 0.35
    total_w = n * box_w + (n - 1) * gap

    fig = go.Figure()
    for i, (texto, color) in enumerate(pasos):
        x0 = i * (box_w + gap)
        x1 = x0 + box_w
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=box_h,
            line=dict(color=color, width=1.5),
            fillcolor=color, opacity=0.92,
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=box_h / 2, text=texto.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=11.5, color=COLORS["bg"], family="Inter, sans-serif"),
            align="center",
        )
        if i < n - 1:
            fig.add_annotation(
                x=x1 + gap / 2, y=box_h / 2,
                ax=x1, ay=box_h / 2, axref="x", ayref="y", xref="x", yref="y",
                text="", showarrow=True, arrowhead=3, arrowwidth=1.5,
                arrowcolor=COLORS["grafito"],
            )

    fig.update_xaxes(visible=False, range=[-0.15, total_w + 0.15])
    fig.update_yaxes(visible=False, range=[-0.25, box_h + 0.25])
    fig.update_layout(
        title=dict(text="Esquema del proceso metodológico: de la composición del gasto a las puntuaciones CP1-CP2"),
        height=260,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )
    return fig
