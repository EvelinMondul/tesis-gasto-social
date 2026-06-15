"""
analysis/pca.py
================

Análisis de Componentes Principales (ACP) sobre las coordenadas CLR
estandarizadas (Capítulo 5), precedido por las pruebas de adecuación
muestral KMO (Kaiser-Meyer-Olkin, calculado vía pseudoinversa porque la
matriz de covarianzas CLR es singular de rango D-1) y el test de
esfericidad de Bartlett.

Resultados de referencia reportados en la tesis (n=33, D=6):
    KMO total      = 0.6583
    Bartlett chi2   = 20147.62, gl = 15, p < 0.001
    CP1            = 52.8 % de varianza explicada
    CP2            = 30.0 % de varianza explicada
    CP1 + CP2      = 82.8 % (criterio de Kaiser: lambda >= 1 retiene 2 CP)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.coda import COMPONENTES, get_clr_coords, load_gasto


def _standardized_clr(df: pd.DataFrame | None = None) -> tuple[np.ndarray, list[str]]:
    Z = get_clr_coords(df)
    Xs = StandardScaler().fit_transform(Z.values)
    return Xs, list(Z.columns)


def kmo_bartlett(df: pd.DataFrame | None = None) -> dict:
    """Calcula el índice KMO global (vía pseudoinversa) y el test de
    esfericidad de Bartlett sobre las coordenadas CLR estandarizadas.

    La matriz de correlación de las coordenadas CLR es singular (rango
    D-1), por lo que su inversa no existe; se utiliza la pseudoinversa de
    Moore-Penrose (``np.linalg.pinv``) para obtener la matriz de
    correlaciones anti-imagen requerida por el índice KMO.
    """
    Xs, cols = _standardized_clr(df)
    n, D = Xs.shape

    R = np.corrcoef(Xs, rowvar=False)
    R_inv = np.linalg.pinv(R)

    # Matriz de correlaciones parciales (anti-imagen)
    diag = np.sqrt(np.diag(R_inv))
    denom = np.outer(diag, diag)
    partial_corr = -R_inv / denom
    np.fill_diagonal(partial_corr, 1.0)

    r2_sum = np.sum(R**2) - np.sum(np.diag(R) ** 2)
    p2_sum = np.sum(partial_corr**2) - np.sum(np.diag(partial_corr) ** 2)
    kmo_total = r2_sum / (r2_sum + p2_sum)

    # KMO por variable (MSA individual)
    r2_i = (np.sum(R**2, axis=1) - 1)
    p2_i = (np.sum(partial_corr**2, axis=1) - 1)
    kmo_per_var = r2_i / (r2_i + p2_i)

    # Test de esfericidad de Bartlett
    corr_det = np.linalg.det(R)
    corr_det = max(corr_det, 1e-300)  # evitar log(0)
    chi2_stat = -(n - 1 - (2 * D + 5) / 6) * np.log(corr_det)
    dof = D * (D - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)

    return {
        "kmo_total": float(kmo_total),
        "kmo_por_variable": pd.DataFrame({"Componente": cols, "MSA": kmo_per_var}).round(4),
        "bartlett_chi2": float(chi2_stat),
        "bartlett_gl": int(dof),
        "bartlett_p": float(p_value),
        "n": n,
        "D": D,
    }


def run_pca(df: pd.DataFrame | None = None) -> dict:
    """Ejecuta el ACP sobre las coordenadas CLR estandarizadas.

    Devuelve un diccionario con: varianza explicada, varianza acumulada,
    autovalores, número de componentes retenidos por el criterio de Kaiser
    (lambda >= 1), matriz de cargas (loadings) y puntuaciones (scores) de
    cada departamento sobre cada componente.
    """
    Xs, cols = _standardized_clr(df)
    gasto = df if df is not None else load_gasto()
    n, D = Xs.shape

    pca = PCA(n_components=D)
    scores = pca.fit_transform(Xs)

    eigenvalues = pca.explained_variance_
    var_exp = pca.explained_variance_ratio_ * 100
    var_acum = np.cumsum(var_exp)

    n_kaiser = int(np.sum(eigenvalues >= 1))

    comp_names = [f"CP{i+1}" for i in range(D)]

    loadings = pd.DataFrame(
        (pca.components_.T * np.sqrt(eigenvalues)),  # cargas = autovector * sqrt(autovalor)
        index=cols,
        columns=comp_names,
    )

    scores_df = pd.DataFrame(scores, columns=comp_names)
    scores_df.insert(0, "Departamento", gasto["Departamento"].values)
    scores_df.insert(1, "region", gasto["region"].values)

    scree = pd.DataFrame({
        "Componente": comp_names,
        "Autovalor": eigenvalues,
        "Varianza explicada (%)": var_exp,
        "Varianza acumulada (%)": var_acum,
    })

    return {
        "scree": scree.round(4),
        "loadings": loadings.round(4),
        "scores": scores_df.round(4),
        "n_kaiser": n_kaiser,
        "eigenvalues": eigenvalues,
        "var_exp": var_exp,
        "var_acum": var_acum,
        "components": comp_names,
        "cols": cols,
    }


if __name__ == "__main__":
    res = kmo_bartlett()
    print("KMO total:", round(res["kmo_total"], 4), "(referencia tesis: 0.6583)")
    print("Bartlett chi2:", round(res["bartlett_chi2"], 2),
          "gl:", res["bartlett_gl"], "p:", res["bartlett_p"], "(referencia tesis: 20147.62, gl=15)")
    print("\nKMO por variable:")
    print(res["kmo_por_variable"])

    pca_res = run_pca()
    print("\nScree:")
    print(pca_res["scree"])
    print("\nComponentes retenidos (Kaiser, lambda>=1):", pca_res["n_kaiser"])
    print("\nCargas (loadings):")
    print(pca_res["loadings"])
    print("\nScores (primeras filas):")
    print(pca_res["scores"].head())
