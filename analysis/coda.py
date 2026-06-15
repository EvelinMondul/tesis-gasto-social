"""
analysis/coda.py
=================

Funciones núcleo de Análisis de Datos Composicionales (CoDA), coherentes con
el marco teórico del Capítulo 3 de la tesis (Aitchison, 1986; Pawlowsky-Glahn
& Buccianti, 2011; Martín-Fernández et al., 2003).

Operaciones implementadas:
    - closure(X)                    : cierre a la simplex (suma = 1 por fila)
    - multiplicative_replacement(X) : reemplazo de ceros (Martín-Fernández et al., 2003)
    - clr(X)                        : transformación log-razón centrada
    - clr_inv(Z)                    : transformación inversa (clr -> simplex)
    - aitchison_distance(x, y)      : distancia de Aitchison entre dos composiciones

Estas funciones son usadas por todos los módulos de analysis/ (correlations,
pca, factorial, clustering) para garantizar que el preprocesamiento sea
idéntico al descrito en la tesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Componentes de la composición del gasto social (D = 6), tras la fusión
# de "Cultura" y "Deporte" justificada en el Capítulo 3 (rho_CLR = 1.00).
COMPONENTES = [
    "Agua potable",
    "Cultura y Deporte",
    "Educación",
    "Libre Destinación",
    "Libre Inversión",
    "Salud",
]

# Etiquetas cortas para gráficos (ejes, leyendas)
COMPONENTES_CORTO = {
    "Agua potable": "Agua potable",
    "Cultura y Deporte": "Cultura y Deporte",
    "Educación": "Educación",
    "Libre Destinación": "Libre Destinación",
    "Libre Inversión": "Libre Inversión",
    "Salud": "Salud",
}

DELTA_ZERO_REPLACEMENT = 0.0001


# ---------------------------------------------------------------------------
# Operaciones CoDA básicas
# ---------------------------------------------------------------------------

def closure(X: np.ndarray) -> np.ndarray:
    """Cierra cada fila de X a la simplex S^D (suma de componentes = 1)."""
    X = np.asarray(X, dtype=float)
    return X / X.sum(axis=1, keepdims=True)


def multiplicative_replacement(X: np.ndarray, delta: float = DELTA_ZERO_REPLACEMENT) -> np.ndarray:
    """Reemplazo multiplicativo de ceros (Martín-Fernández et al., 2003).

    Cada cero estructural/de redondeo se sustituye por ``delta`` y el resto
    de componentes de la fila se reescala multiplicativamente para preservar
    el cierre (suma = 1). Es el procedimiento estándar para permitir el
    cálculo de log-razones cuando existen ceros (p. ej. Bogotá D.C. en
    "Libre Destinación").
    """
    X = closure(np.asarray(X, dtype=float))
    D = X.shape[1]
    Z = X.copy()
    zero_mask = Z == 0
    n_zeros_por_fila = zero_mask.sum(axis=1)

    # Sustitución de ceros por delta
    Z[zero_mask] = delta

    # Reescalado multiplicativo de las componentes no-cero
    factor = 1 - n_zeros_por_fila * delta
    nonzero_mask = ~zero_mask
    # factor por fila aplicado solo a columnas no-cero
    Z[nonzero_mask] = (Z * factor[:, None])[nonzero_mask]

    return Z


def clr(X: np.ndarray) -> np.ndarray:
    """Transformación log-razón centrada (CLR).

    clr(x)_i = ln(x_i) - (1/D) * sum_j ln(x_j)

    Devuelve coordenadas en R^D con suma = 0 por fila (rango D-1).
    """
    X = np.asarray(X, dtype=float)
    logX = np.log(X)
    g = logX.mean(axis=1, keepdims=True)  # media geométrica en escala log
    return logX - g


def clr_inv(Z: np.ndarray) -> np.ndarray:
    """Inversa de la transformación CLR: de R^D (suma=0) a la simplex S^D."""
    Z = np.asarray(Z, dtype=float)
    expZ = np.exp(Z)
    return closure(expZ)


def aitchison_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Distancia de Aitchison entre dos composiciones (vectores en S^D)."""
    cx, cy = clr(np.atleast_2d(x)), clr(np.atleast_2d(y))
    return float(np.linalg.norm(cx - cy))


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_gasto() -> pd.DataFrame:
    """Carga data/gasto.csv (n=33, D=6 componentes del gasto social)."""
    return pd.read_csv(DATA_DIR / "gasto.csv")


def load_ipm() -> pd.DataFrame:
    """Carga data/ipm.csv (n=33, 15 indicadores de privación + IPM total)."""
    return pd.read_csv(DATA_DIR / "ipm.csv")


def get_composicion(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Devuelve la matriz de composición (n x D) en proporciones (P_*)."""
    if df is None:
        df = load_gasto()
    cols = [f"P_{c}" for c in COMPONENTES]
    return df[cols].rename(columns=lambda c: c[2:])


def get_clr_coords(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Devuelve las coordenadas CLR (n x D) tras el reemplazo de ceros."""
    comp = get_composicion(df)
    X = multiplicative_replacement(comp.values)
    Z = clr(X)
    return pd.DataFrame(Z, columns=comp.columns, index=comp.index)


if __name__ == "__main__":
    gasto = load_gasto()
    comp = get_composicion(gasto)
    print("Composición (primeras filas):")
    print(comp.head())
    print("\nSuma por fila (closure):", comp.sum(axis=1).round(6).unique())

    Z = get_clr_coords(gasto)
    print("\nCLR (primeras filas):")
    print(Z.head())
    print("\nSuma por fila (CLR, debe ser ~0):", Z.sum(axis=1).round(10).unique())
    print("\nBogotá CLR (componente Libre Destinación tras reemplazo de ceros):")
    print(Z.loc[gasto["Departamento"] == "Bogotá"])
