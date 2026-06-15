"""
analysis/factor_analysis.py
=============================

Análisis Factorial Exploratorio (AFE) sobre las coordenadas CLR
estandarizadas (Capítulo 6), utilizado como contraste metodológico del
Análisis de Componentes Principales (Capítulo 5).

Se extraen 2 factores --el mismo número de dimensiones retenidas por el
criterio de Kaiser en el ACP-- mediante extracción de componentes
principales (``method="principal"``) y rotación varimax, lo que permite
comparar directamente las cargas de cada coordenada CLR sobre F1/F2 frente
a CP1/CP2.
"""

from __future__ import annotations

import warnings

import pandas as pd
from factor_analyzer import FactorAnalyzer

from analysis.coda import load_gasto
from analysis.pca import _standardized_clr, run_pca


def run_afe(df: pd.DataFrame | None = None, n_factors: int = 2, rotation: str = "varimax") -> dict:
    """Ejecuta el AFE sobre las coordenadas CLR estandarizadas.

    Devuelve un diccionario con las cargas factoriales rotadas, las
    comunalidades, la varianza explicada por factor y las puntuaciones
    factoriales de cada departamento.
    """
    Xs, cols = _standardized_clr(df)
    gasto = df if df is not None else load_gasto()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="principal")
        fa.fit(Xs)
        scores = fa.transform(Xs)

    factor_names = [f"F{i+1}" for i in range(n_factors)]

    loadings = pd.DataFrame(fa.loadings_, index=cols, columns=factor_names)

    communalities = pd.DataFrame({
        "Componente (CLR)": cols,
        "Comunalidad": fa.get_communalities(),
    })

    _, prop_var, cum_var = fa.get_factor_variance()
    varianza = pd.DataFrame({
        "Factor": factor_names,
        "Varianza explicada (%)": prop_var * 100,
        "Varianza acumulada (%)": cum_var * 100,
    })

    scores_df = pd.DataFrame(scores, columns=factor_names)
    scores_df.insert(0, "Departamento", gasto["Departamento"].values)
    scores_df.insert(1, "region", gasto["region"].values)

    return {
        "loadings": loadings.round(4),
        "communalities": communalities.round(4),
        "varianza": varianza.round(4),
        "scores": scores_df.round(4),
        "cols": cols,
        "n_factors": n_factors,
        "rotation": rotation,
    }


def comparar_acp_afe(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla comparativa de las cargas del ACP (CP1, CP2) frente a las
    cargas rotadas del AFE (F1, F2) para cada coordenada CLR."""
    pca_res = run_pca(df)
    afe_res = run_afe(df)

    return pd.DataFrame({
        "Componente (CLR)": pca_res["cols"],
        "ACP - CP1": pca_res["loadings"]["CP1"].values,
        "AFE - F1 (varimax)": afe_res["loadings"]["F1"].values,
        "ACP - CP2": pca_res["loadings"]["CP2"].values,
        "AFE - F2 (varimax)": afe_res["loadings"]["F2"].values,
    }).round(4)


if __name__ == "__main__":
    res = run_afe()
    print("Varianza explicada:")
    print(res["varianza"])
    print("\nCargas (varimax):")
    print(res["loadings"])
    print("\nComunalidades:")
    print(res["communalities"])
    print("\nComparación ACP vs AFE:")
    print(comparar_acp_afe())
