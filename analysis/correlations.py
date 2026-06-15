"""
analysis/correlations.py
=========================

Matriz de correlación de Spearman sobre las coordenadas CLR (Capítulo 4),
con su correspondiente matriz de valores p, usada para construir el mapa
de calor de correlaciones y la tabla de pares significativos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.coda import get_clr_coords


def spearman_clr(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matrices de correlación de Spearman (rho) y valores p sobre CLR.

    Devuelve ``(rho, pval)``, ambas matrices D x D indexadas por el nombre
    de los componentes.
    """
    Z = get_clr_coords(df)
    cols = Z.columns
    D = len(cols)
    rho = pd.DataFrame(np.eye(D), index=cols, columns=cols)
    pval = pd.DataFrame(np.zeros((D, D)), index=cols, columns=cols)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                pval.iloc[i, j] = 0.0
                continue
            r, p = stats.spearmanr(Z[ci], Z[cj])
            rho.iloc[i, j] = r
            pval.iloc[i, j] = p
    return rho, pval


def pares_significativos(df: pd.DataFrame | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Tabla de pares (componente_i, componente_j, rho, p) sin duplicados,
    ordenada por |rho| descendente, marcando significancia al nivel alpha."""
    rho, pval = spearman_clr(df)
    cols = rho.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({
                "Componente A": cols[i],
                "Componente B": cols[j],
                "rho (Spearman, CLR)": rho.iloc[i, j],
                "Valor p": pval.iloc[i, j],
                "Significativo (alpha=0.05)": "Sí" if pval.iloc[i, j] < alpha else "No",
            })
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho (Spearman, CLR)"].abs()
    out = out.sort_values("abs_rho", ascending=False).drop(columns="abs_rho")
    return out.round({"rho (Spearman, CLR)": 4, "Valor p": 4}).reset_index(drop=True)


if __name__ == "__main__":
    rho, pval = spearman_clr()
    print("=== rho (Spearman, CLR) ===")
    print(rho.round(3))
    print("\n=== Pares ordenados por |rho| ===")
    print(pares_significativos())
