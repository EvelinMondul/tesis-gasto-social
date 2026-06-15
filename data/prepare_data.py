"""
Script de preparación de datos para el dashboard de tesis.

Lee el archivo fuente 'Privaciones_IPM_2025.xlsx' (hojas 'Gasto' y
'Privaciones_2025') y produce dos archivos CSV limpios y listos para el
pipeline de análisis (CoDA / CLR / PCA / AFE / Clustering / IPM):

  - data/gasto.csv : composición del gasto social por departamento (D=6
    componentes), valores absolutos, proporciones (cierre) y población.
  - data/ipm.csv   : indicadores de privación del IPM 2025 por departamento.

Transformaciones aplicadas (coherentes con la metodología del Capítulo 3
de la tesis):

1. Se excluye la fila de agregado nacional ("Total general") -> n = 33.
2. Se fusionan los componentes "Cultura" y "Deporte" en un único componente
   "Cultura y Deporte" (D pasa de 7 a 6), dado que en la tesis ambos
   presentan correlación CLR perfecta (rho = 1.00, p < 0.001) y se
   modelan como una sola partida presupuestal.
3. El valor faltante (NaN) de "Libre Destinación" para Bogotá D.C.
   corresponde a un cero estructural (la ciudad no recibe recursos del
   SGP por libre destinación) y se codifica como 0 en valores absolutos.
4. Las proporciones (P_*) se recalculan por cierre (closure) a partir de
   los valores absolutos de los D=6 componentes, garantizando suma = 1
   por fila. El reemplazo multiplicativo de ceros (Martín-Fernández et
   al., 2003, delta = 0.0001) se aplica más adelante, en analysis/coda.py,
   justo antes de la transformación CLR.
5. Se añade una columna 'region' con la clasificación geográfica estándar
   de los departamentos de Colombia, usada en los gráficos exploratorios
   del Capítulo 2 (EDA).
"""

from pathlib import Path
import pandas as pd

SRC = Path("/sessions/elegant-awesome-babbage/mnt/uploads/Privaciones_IPM_2025.xlsx")
OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Clasificación regional de los 33 departamentos / distritos de Colombia
# ---------------------------------------------------------------------------
REGION_MAP = {
    "Amazonas": "Amazonía",
    "Antioquia": "Andina",
    "Arauca": "Orinoquía",
    "Atlántico": "Caribe",
    "Bogotá": "Andina",
    "Bolívar": "Caribe",
    "Boyacá": "Andina",
    "Caldas": "Andina",
    "Caquetá": "Amazonía",
    "Casanare": "Orinoquía",
    "Cauca": "Pacífica",
    "Cesar": "Caribe",
    "Chocó": "Pacífica",
    "Córdoba": "Caribe",
    "Cundinamarca": "Andina",
    "Guainía": "Amazonía",
    "Guaviare": "Amazonía",
    "Huila": "Andina",
    "La Guajira": "Caribe",
    "Magdalena": "Caribe",
    "Meta": "Orinoquía",
    "Nariño": "Pacífica",
    "Norte de Santander": "Andina",
    "Putumayo": "Amazonía",
    "Quindío": "Andina",
    "Risaralda": "Andina",
    "San Andrés y Providencia": "Insular",
    "Santander": "Andina",
    "Sucre": "Caribe",
    "Tolima": "Andina",
    "Valle del Cauca": "Pacífica",
    "Vaupés": "Amazonía",
    "Vichada": "Orinoquía",
}

# Componentes absolutos originales (D=7) -> fusión a D=6
ABS_COLS_ORIG = [
    "Agua potable",
    "Cultura",
    "Deporte",
    "Educación",
    "Libre Destinación",
    "Libre Inversión",
    "Salud",
]

COMPONENTES_D6 = [
    "Agua potable",
    "Cultura y Deporte",
    "Educación",
    "Libre Destinación",
    "Libre Inversión",
    "Salud",
]


def prepare_gasto() -> pd.DataFrame:
    df = pd.read_excel(SRC, sheet_name="Gasto")

    # Excluir fila de agregado nacional
    df = df[df["Departamento"] != "Total general"].copy()
    assert len(df) == 33, f"Se esperaban 33 departamentos, se obtuvieron {len(df)}"

    # Cero estructural: Bogotá no tiene 'Libre Destinación'
    df["Libre Destinación"] = df["Libre Destinación"].fillna(0.0)

    # Fusión Cultura + Deporte -> Cultura y Deporte (D=7 -> D=6)
    df["Cultura y Deporte"] = df["Cultura"] + df["Deporte"]

    abs_cols = COMPONENTES_D6
    df["Total general"] = df[abs_cols].sum(axis=1)

    # Cierre: proporciones recalculadas para que sumen exactamente 1
    for c in abs_cols:
        df[f"P_{c}"] = df[c] / df["Total general"]

    df["region"] = df["Departamento"].map(REGION_MAP)
    missing_region = df[df["region"].isna()]["Departamento"].tolist()
    assert not missing_region, f"Departamentos sin región asignada: {missing_region}"

    out_cols = (
        ["Departamento", "region", "Población 2024"]
        + abs_cols
        + ["Total general"]
        + [f"P_{c}" for c in abs_cols]
    )
    df = df[out_cols].reset_index(drop=True)
    return df


def prepare_ipm() -> pd.DataFrame:
    df = pd.read_excel(SRC, sheet_name="Privaciones_2025")
    assert len(df) == 33, f"Se esperaban 33 departamentos, se obtuvieron {len(df)}"

    df = df.rename(columns={
        "Índice de pobreza multidimensional - IPM": "IPM",
        "%  IPM": "IPM_pct",
    })
    df["region"] = df["Departamento"].map(REGION_MAP)
    missing_region = df[df["region"].isna()]["Departamento"].tolist()
    assert not missing_region, f"Departamentos sin región asignada: {missing_region}"

    cols = ["Departamento", "region"] + [c for c in df.columns if c not in ("Departamento", "region")]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    gasto = prepare_gasto()
    ipm = prepare_ipm()

    gasto.to_csv(OUT_DIR / "gasto.csv", index=False)
    ipm.to_csv(OUT_DIR / "ipm.csv", index=False)

    print("gasto.csv ->", gasto.shape)
    print(gasto[["Departamento", "region"] + [f"P_{c}" for c in COMPONENTES_D6]].head())
    print()
    print("ipm.csv ->", ipm.shape)
    print(ipm[["Departamento", "region", "IPM"]].head())

    # Verificación: las proporciones suman 1
    sums = gasto[[f"P_{c}" for c in COMPONENTES_D6]].sum(axis=1)
    assert (sums.round(8) == 1).all(), "Las proporciones no cierran a 1"
    print("\nOK: todas las filas cierran a 1 (closure verificada).")
