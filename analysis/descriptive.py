"""
analysis/descriptive.py
========================

Estadística descriptiva para el Capítulo 2 (Análisis Exploratorio de Datos)
y soporte metodológico para el Capítulo 3 (justificación de Spearman sobre
Pearson mediante la prueba de normalidad de Shapiro-Wilk aplicada a las
coordenadas CLR, no a las proporciones brutas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.coda import COMPONENTES, get_clr_coords, get_composicion, load_gasto


def resumen_composicion(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estadísticos descriptivos de la composición en proporciones (%).

    Para cada componente: media, mediana, desviación estándar, mínimo,
    máximo y coeficiente de variación (CV), expresados en porcentaje del
    gasto social total.
    """
    comp = get_composicion(df) * 100  # a porcentaje
    out = pd.DataFrame({
        "Componente": comp.columns,
        "Media (%)": comp.mean().values,
        "Mediana (%)": comp.median().values,
        "Desv. estándar (%)": comp.std().values,
        "Mínimo (%)": comp.min().values,
        "Máximo (%)": comp.max().values,
        "CV": (comp.std() / comp.mean()).values,
    })
    return out.round(3)


def resumen_clr(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estadísticos descriptivos de las coordenadas CLR por componente."""
    Z = get_clr_coords(df)
    out = pd.DataFrame({
        "Componente": Z.columns,
        "Media CLR": Z.mean().values,
        "Mediana CLR": Z.median().values,
        "Desv. estándar CLR": Z.std().values,
        "Mínimo CLR": Z.min().values,
        "Máximo CLR": Z.max().values,
        "Asimetría": Z.skew().values,
        "Curtosis": Z.kurtosis().values,
    })
    return out.round(4)


def shapiro_clr(df: pd.DataFrame | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Prueba de normalidad de Shapiro-Wilk sobre cada coordenada CLR.

    Esta prueba es la que justifica, en el Capítulo 3, el uso del
    coeficiente de correlación de Spearman (no paramétrico) en lugar de
    Pearson: si una o más coordenadas CLR se desvían significativamente de
    la normalidad, Spearman es la opción metodológicamente más robusta.
    """
    Z = get_clr_coords(df)
    rows = []
    for col in Z.columns:
        stat, p = stats.shapiro(Z[col])
        rows.append({
            "Componente (CLR)": col,
            "Estadístico W": stat,
            "Valor p": p,
            "Normal (alpha=0.05)": "Sí" if p > alpha else "No",
        })
    return pd.DataFrame(rows).round({"Estadístico W": 4, "Valor p": 4})


def top_bottom_departamentos(componente: str, n: int = 5, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top-n y bottom-n departamentos según la proporción de un componente."""
    if df is None:
        df = load_gasto()
    col = f"P_{componente}"
    sub = df[["Departamento", "region", col]].copy()
    sub[col] = sub[col] * 100
    sub = sub.rename(columns={col: f"{componente} (%)"})
    top = sub.nlargest(n, f"{componente} (%)")
    bottom = sub.nsmallest(n, f"{componente} (%)")
    return top, bottom


if __name__ == "__main__":
    print("=== Resumen composición (proporciones, %) ===")
    print(resumen_composicion())
    print("\n=== Resumen CLR ===")
    print(resumen_clr())
    print("\n=== Shapiro-Wilk sobre CLR ===")
    print(shapiro_clr())
