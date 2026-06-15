"""
analysis/clustering.py
========================

Segmentación de los departamentos mediante K-Means sobre las
puntuaciones CP1-CP2 del ACP (Capítulo 05), y caracterización de los
grupos resultantes mediante el test no paramétrico de Kruskal-Wallis
sobre la composición porcentual de cada componente del gasto.

Bogotá D.C. se trata como un caso atípico (institucionalmente, un
Distrito Capital y no un departamento bajo las mismas reglas del SGP) y
se excluye del K-Means: su posición extrema en CP1 (gasto concentrado en
Educación, con Libre Destinación igual a cero) distorsionaría la
segmentación de los 32 departamentos restantes. El K-Means se ejecuta
con k=2 sobre esas 32 observaciones, seleccionado mediante el
coeficiente de silueta (silueta(k=2) > silueta(k=3)). Los dos clusters
resultantes se etiquetan de forma determinista: C1 es el de mayor
proporción promedio de gasto de libre destinación e inversión (mayor
margen discrecional) y C3 el de menor margen discrecional / mayor peso
relativo del gasto en salud. Bogotá se reporta aparte con la etiqueta
"Atípico".
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analysis.coda import COMPONENTES, load_gasto
from analysis.pca import run_pca

DEPARTAMENTO_ATIPICO = "Bogotá"

CLUSTER_LABELS = {
    "C1": "Mayor margen discrecional (Libre Destinación / Inversión)",
    "C3": "Menor margen discrecional, mayor peso del gasto en Salud",
    "Atípico": "Bogotá D.C. (excluida del K-Means, caso atípico)",
}


def silueta_por_k(df: pd.DataFrame | None = None, ks: tuple[int, ...] = (2, 3, 4), random_state: int = 42) -> pd.DataFrame:
    """Coeficiente de silueta de K-Means para distintos valores de k,
    calculado sobre los 32 departamentos (excluye Bogotá)."""
    gasto = df if df is not None else load_gasto()
    pca_res = run_pca(gasto)
    scores = pca_res["scores"]
    X = scores.loc[scores["Departamento"] != DEPARTAMENTO_ATIPICO, ["CP1", "CP2"]].values

    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        filas.append({"k": k, "Coeficiente de silueta": round(float(sil), 4)})
    return pd.DataFrame(filas)


def run_kmeans(df: pd.DataFrame | None = None, k: int = 2, random_state: int = 42) -> dict:
    """Ejecuta K-Means (k=2) sobre las puntuaciones CP1-CP2 de los 32
    departamentos distintos de Bogotá, y renombra los clusters de forma
    determinista (ver módulo). Bogotá se añade aparte con cluster =
    "Atípico"."""
    gasto = df if df is not None else load_gasto()
    pca_res = run_pca(gasto)
    scores = pca_res["scores"].copy()

    es_atipico = scores["Departamento"] == DEPARTAMENTO_ATIPICO
    X = scores.loc[~es_atipico, ["CP1", "CP2"]].values

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X)

    gasto_aux = gasto.copy()
    cols_p = [f"P_{c}" for c in COMPONENTES]

    raw_series = pd.Series(raw_labels, index=scores.loc[~es_atipico].index)
    gasto_aux.loc[~es_atipico, "raw_cluster"] = raw_series.values
    medias = gasto_aux.loc[~es_atipico].groupby("raw_cluster")[cols_p].mean()

    # Discrecionalidad = Libre Destinación + Libre Inversión (promedio del grupo)
    discrecional = medias["P_Libre Destinación"] + medias["P_Libre Inversión"]
    mas_discrecional = discrecional.idxmax()
    menos_discrecional = [c for c in medias.index if c != mas_discrecional][0]
    mapeo = {mas_discrecional: "C1", menos_discrecional: "C3"}

    scores["cluster"] = "Atípico"
    scores.loc[~es_atipico, "cluster"] = raw_series.map(mapeo).values

    gasto_aux["cluster"] = "Atípico"
    gasto_aux.loc[~es_atipico, "cluster"] = raw_series.map(mapeo).values
    gasto_aux = gasto_aux.drop(columns="raw_cluster")

    centers_raw = pd.DataFrame(km.cluster_centers_, columns=["CP1", "CP2"])
    centers_raw.index = centers_raw.index.map(mapeo)
    centers = centers_raw.loc[["C1", "C3"]].reset_index().rename(columns={"index": "cluster"})

    sil = silhouette_score(X, raw_labels)

    return {
        "scores": scores.round(4),
        "gasto": gasto_aux,
        "centers": centers.round(4),
        "k": k,
        "silhouette": round(float(sil), 4),
    }


def perfil_clusters(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Composición promedio (%) de cada componente del gasto por cluster
    (C1, C3), junto con la fila correspondiente a Bogotá D.C. (Atípico,
    n=1, con su composición real)."""
    res = run_kmeans(df)
    gasto_aux = res["gasto"]
    cols_p = [f"P_{c}" for c in COMPONENTES]

    perfil = (gasto_aux.groupby("cluster")[cols_p].mean() * 100).round(2)
    perfil = perfil.rename(columns=lambda c: c[2:])
    perfil.insert(0, "n", gasto_aux.groupby("cluster").size())
    perfil = perfil.reset_index()
    perfil["Descripción"] = perfil["cluster"].map(CLUSTER_LABELS)

    orden = {"C1": 0, "C3": 1, "Atípico": 2}
    perfil["_orden"] = perfil["cluster"].map(orden)
    return perfil.sort_values("_orden").drop(columns="_orden").reset_index(drop=True)


def kruskal_clusters(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Test de Kruskal-Wallis: ¿difiere la composición porcentual de cada
    componente del gasto entre los clusters C1 y C3? (Bogotá, caso
    atípico con n=1, se excluye del test)."""
    res = run_kmeans(df)
    gasto_aux = res["gasto"]

    filas = []
    for comp in COMPONENTES:
        g1 = gasto_aux.loc[gasto_aux["cluster"] == "C1", f"P_{comp}"]
        g3 = gasto_aux.loc[gasto_aux["cluster"] == "C3", f"P_{comp}"]
        h_stat, p_val = kruskal(g1, g3)
        filas.append({
            "Componente": comp,
            "H (Kruskal-Wallis)": round(h_stat, 3),
            "Valor p": round(p_val, 4),
            "Significativo (alpha=0.05)": "Sí" if p_val < 0.05 else "No",
        })

    return pd.DataFrame(filas).sort_values("Valor p").reset_index(drop=True)


if __name__ == "__main__":
    print(silueta_por_k())
    print()
    print(perfil_clusters())
    print()
    print(kruskal_clusters())
