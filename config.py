"""
config.py
==========

Configuración global del dashboard: paleta de colores, tipografía y
plantilla de Plotly compartida (``pio.templates["tesis"]``) usada por
todas las figuras del proyecto para garantizar una identidad visual
única, coherente con un reporte ejecutivo / artículo científico
(fondo blanco, minimalista, gama azul petróleo / azul oscuro / gris
grafito / gris claro).
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paleta de colores (debe coincidir con assets/style.css :root)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "bg_subtle": "#F4F6F7",
    "border": "#D9DEE2",
    "grafito": "#3D4A52",
    "azul_oscuro": "#1B3A4B",
    "petroleo": "#2E6E73",
    "petroleo_claro": "#7FAEB3",
    "gris_medio": "#8C97A0",
    "acento_suave": "#C9D6D9",
    "texto": "#26323A",
    "texto_secundario": "#5C6B73",
}

# Secuencia categórica principal (hasta 6 componentes / clusters / regiones)
SECUENCIA_CATEGORICA = [
    COLORS["petroleo"],       # 1. Azul petróleo
    COLORS["azul_oscuro"],    # 2. Azul oscuro
    COLORS["gris_medio"],     # 3. Gris medio
    COLORS["petroleo_claro"], # 4. Azul petróleo claro
    COLORS["grafito"],        # 5. Gris grafito
    COLORS["acento_suave"],   # 6. Acento suave
]

# Escala continua (para mapas / heatmaps) — monocromática azul petróleo
ESCALA_CONTINUA = [
    [0.0, "#FFFFFF"],
    [0.25, COLORS["acento_suave"]],
    [0.5, COLORS["petroleo_claro"]],
    [0.75, COLORS["petroleo"]],
    [1.0, COLORS["azul_oscuro"]],
]

# Escala divergente (correlaciones: -1 a 1) — petróleo / blanco / grafito
ESCALA_DIVERGENTE = [
    [0.0, COLORS["azul_oscuro"]],
    [0.5, COLORS["bg"]],
    [1.0, COLORS["grafito"]],
]

# Colores fijos por cluster (Capítulo 7), reutilizados en mapas e IPM
COLOR_CLUSTER = {
    "C1": COLORS["petroleo"],       # Mayor margen discrecional
    "C2": COLORS["gris_medio"],     # Menor margen discrecional
    "Atípico": COLORS["azul_oscuro"],  # Bogotá D.C. (caso atípico)
}

# ---------------------------------------------------------------------------
# Tipografía
# ---------------------------------------------------------------------------
FONT_FAMILY_BODY = "Inter, -apple-system, Helvetica, Arial, sans-serif"
FONT_FAMILY_HEADINGS = "'Source Serif 4', Georgia, serif"

# ---------------------------------------------------------------------------
# Plantilla Plotly compartida: pio.templates["tesis"]
# ---------------------------------------------------------------------------
_tesis_layout = go.Layout(
    font=dict(family=FONT_FAMILY_BODY, size=13, color=COLORS["texto"]),
    title=dict(
        font=dict(family=FONT_FAMILY_HEADINGS, size=18, color=COLORS["azul_oscuro"]),
        x=0.0,
        xanchor="left",
        pad=dict(t=10, b=10),
    ),
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    colorway=SECUENCIA_CATEGORICA,
    margin=dict(l=60, r=30, t=70, b=60),
    xaxis=dict(
        showgrid=True,
        gridcolor=COLORS["border"],
        gridwidth=1,
        zeroline=False,
        linecolor=COLORS["gris_medio"],
        ticks="outside",
        tickcolor=COLORS["gris_medio"],
        title=dict(font=dict(size=13, color=COLORS["texto_secundario"])),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=COLORS["border"],
        gridwidth=1,
        zeroline=False,
        linecolor=COLORS["gris_medio"],
        ticks="outside",
        tickcolor=COLORS["gris_medio"],
        title=dict(font=dict(size=13, color=COLORS["texto_secundario"])),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        bordercolor=COLORS["border"],
        borderwidth=0,
        font=dict(size=12),
    ),
    hoverlabel=dict(
        bgcolor=COLORS["bg"],
        bordercolor=COLORS["border"],
        font=dict(family=FONT_FAMILY_BODY, size=12, color=COLORS["texto"]),
    ),
    coloraxis=dict(colorbar=dict(outlinewidth=0, ticks="outside")),
)

pio.templates["tesis"] = go.layout.Template(layout=_tesis_layout)
pio.templates.default = "tesis"

# ---------------------------------------------------------------------------
# Configuración estándar de exportación para dcc.Graph (PNG/SVG/PDF)
# ---------------------------------------------------------------------------
GRAPH_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "figura_tesis",
        "scale": 2,
    },
    "modeBarButtonsToAdd": ["toggleSpikelines"],
}

# Tamaño estándar de figuras (para exportación PDF/PNG consistente)
FIG_HEIGHT = 480
FIG_HEIGHT_TALL = 600
