"""
analysis/ipm_analysis.py
==========================

Contraste entre el gasto social (composición CLR, ACP y clusters de los
Capítulos 03-07) y el Índice de Pobreza Multidimensional (IPM 2025) de
cada departamento (Capítulo 08).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kruskal, spearmanr

from analysis.clustering import CLUSTER_LABELS, run_kmeans
from analysis.coda import COMPONENTES, load_gasto, load_ipm


def gasto_ipm_clusters(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla con la composición del gasto, el cluster asignado (Cap. 07)
    y el IPM de cada departamento."""
    gasto = df if df is not None else load_gasto()
    ipm = load_ipm()
    res = run_kmeans(gasto)
    clusters = res["scores"][["Departamento", "cluster"]]

    out = gasto.merge(clusters, on="Departamento").merge(
        ipm[["Departamento", "IPM", "IPM_pct"]], on="Departamento"
    )
    return out


def ipm_por_cluster(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Estadísticos del IPM por cluster y test de Kruskal-Wallis."""
    datos = gasto_ipm_clusters(df)

    resumen = (
        datos.groupby("cluster")["IPM_pct"]
        .agg(n="count", Media="mean", Mediana="median", Mín="min", Máx="max")
        .round(2)
        .reset_index()
    )
    resumen["Descripción"] = resumen["cluster"].map(CLUSTER_LABELS)
    orden = {"C1": 0, "C3": 1, "Atípico": 2}
    resumen = resumen.sort_values(by="cluster", key=lambda s: s.map(orden)).reset_index(drop=True)

    grupos = [datos.loc[datos["cluster"] == c, "IPM_pct"] for c in ["C1", "C3"]]
    # Kruskal-Wallis requiere al menos 2 observaciones por grupo (Atípico = Bogotá, n=1)
    grupos_validos = [g for g in grupos if len(g) >= 2]
    if len(grupos_validos) >= 2:
        h_stat, p_val = kruskal(*grupos_validos)
    else:
        h_stat, p_val = float("nan"), float("nan")

    test = {"H (Kruskal-Wallis)": round(h_stat, 3) if h_stat == h_stat else None,
            "Valor p": round(p_val, 4) if p_val == p_val else None,
            "Significativo (alpha=0.05)": "Sí" if (p_val == p_val and p_val < 0.05) else "No",
            "nota": "Atípico (Bogotá, n=1) se excluye del test por tener un solo dato."}

    return resumen, test


INDICADORES_IPM = [
    "Excretas inadecuadas", "Analfabetismo", "Bajo logro educativo",
    "Barreras acceso salud", "Primera infancia", "Desempleo largo",
    "Hacinamiento crítico", "Inasistencia escolar", "Paredes inadecuadas",
    "Pisos inadecuados", "Rezago escolar", "Sin agua mejorada",
    "Sin aseguramiento salud", "Trabajo infantil", "Trabajo informal",
]


def privaciones_por_cluster(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Media y mediana de los 15 indicadores de privación del IPM (% de
    personas que presentan cada privación), agrupados por cluster de gasto
    (C1, C3, Atípico)."""
    gasto = df if df is not None else load_gasto()
    ipm = load_ipm()
    res = run_kmeans(gasto)
    clusters = res["scores"][["Departamento", "cluster"]]

    datos = clusters.merge(ipm[["Departamento"] + INDICADORES_IPM], on="Departamento")

    orden = {"C1": 0, "C3": 1, "Atípico": 2}
    filas = []
    for indicador in INDICADORES_IPM:
        fila = {"Privación (IPM)": indicador}
        for cluster in sorted(datos["cluster"].unique(), key=lambda c: orden[c]):
            valores = datos.loc[datos["cluster"] == cluster, indicador]
            fila[f"{cluster} \u2014 Media"] = round(valores.mean(), 2)
            fila[f"{cluster} \u2014 Mediana"] = round(valores.median(), 2)
        filas.append(fila)

    return pd.DataFrame(filas)


def correlacion_gasto_ipm(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Correlación de Spearman entre la proporción de cada componente del
    gasto y el IPM departamental."""
    datos = gasto_ipm_clusters(df)

    filas = []
    for comp in COMPONENTES:
        rho, p = spearmanr(datos[f"P_{comp}"], datos["IPM_pct"])
        filas.append({
            "Componente del gasto": comp,
            "rho (Spearman) vs. IPM": round(rho, 4),
            "Valor p": round(p, 4),
            "Significativo (alpha=0.05)": "Sí" if p < 0.05 else "No",
        })

    out = pd.DataFrame(filas)
    out["abs_rho"] = out["rho (Spearman) vs. IPM"].abs()
    out = out.sort_values("abs_rho", ascending=False).drop(columns="abs_rho")
    return out.reset_index(drop=True)


def ranking_ipm(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Departamentos ordenados por IPM descendente, con su cluster."""
    datos = gasto_ipm_clusters(df)
    return datos[["Departamento", "region", "cluster", "IPM_pct"]].sort_values(
        "IPM_pct", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    resumen, test = ipm_por_cluster()
    print("IPM por cluster:")
    print(resumen)
    print("\nKruskal-Wallis:", test)
    print("\nCorrelación gasto-IPM:")
    print(correlacion_gasto_ipm())
    print("\nRanking IPM (top 5):")
    print(ranking_ipm().head())
