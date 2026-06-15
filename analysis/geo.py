"""
analysis/geo.py
=================

Vínculo entre los datos departamentales (composición del gasto,
clusters del Capítulo 07 e IPM del Capítulo 08) y el GeoJSON de los 33
departamentos de Colombia, usado por los mapas coropléticos del
Capítulo 09.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.coda import COMPONENTES
from analysis.ipm_analysis import gasto_ipm_clusters

GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "geo" / "colombia_departamentos.geojson"

# Mapeo NOMBRE_DPT (GeoJSON, mayúsculas y sin tilde) -> Departamento
# (nombre usado en data/gasto.csv y data/ipm.csv).
NOMBRE_DPT_TO_DEPARTAMENTO = {
    "AMAZONAS": "Amazonas",
    "ANTIOQUIA": "Antioquia",
    "ARAUCA": "Arauca",
    "ATLANTICO": "Atlántico",
    "SANTAFE DE BOGOTA D.C": "Bogotá",
    "BOLIVAR": "Bolívar",
    "BOYACA": "Boyacá",
    "CALDAS": "Caldas",
    "CAQUETA": "Caquetá",
    "CASANARE": "Casanare",
    "CAUCA": "Cauca",
    "CESAR": "Cesar",
    "CHOCO": "Chocó",
    "CUNDINAMARCA": "Cundinamarca",
    "CORDOBA": "Córdoba",
    "GUAINIA": "Guainía",
    "GUAVIARE": "Guaviare",
    "HUILA": "Huila",
    "LA GUAJIRA": "La Guajira",
    "MAGDALENA": "Magdalena",
    "META": "Meta",
    "NARIÑO": "Nariño",
    "NORTE DE SANTANDER": "Norte de Santander",
    "PUTUMAYO": "Putumayo",
    "QUINDIO": "Quindío",
    "RISARALDA": "Risaralda",
    "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "San Andrés y Providencia",
    "SANTANDER": "Santander",
    "SUCRE": "Sucre",
    "TOLIMA": "Tolima",
    "VALLE DEL CAUCA": "Valle del Cauca",
    "VAUPES": "Vaupés",
    "VICHADA": "Vichada",
}

_GEOJSON_CACHE: dict | None = None


def load_geojson() -> dict:
    """Carga el GeoJSON de los 33 departamentos y añade la propiedad
    ``Departamento`` (nombre homologado con los datasets de gasto e IPM)
    a cada feature, para usarla como ``featureidkey`` en los mapas."""
    global _GEOJSON_CACHE
    if _GEOJSON_CACHE is None:
        with open(GEOJSON_PATH, encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj["features"]:
            nombre = feat["properties"].get("NOMBRE_DPT", "")
            feat["properties"]["Departamento"] = NOMBRE_DPT_TO_DEPARTAMENTO.get(nombre, nombre)
        _GEOJSON_CACHE = gj
    return _GEOJSON_CACHE


def datos_mapas(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla departamental con composición del gasto, componente de
    gasto predominante (Cap. 03), cluster (Cap. 07) e IPM (Cap. 08),
    lista para los mapas coropléticos del Capítulo 09."""
    datos = gasto_ipm_clusters(df)
    cols_p = [f"P_{c}" for c in COMPONENTES]
    datos["Componente predominante"] = (
        datos[cols_p].idxmax(axis=1).str.replace("P_", "", regex=False)
    )
    datos["Educación (%)"] = (datos["P_Educación"] * 100).round(1)
    return datos


if __name__ == "__main__":
    gj = load_geojson()
    nombres_geo = {f["properties"]["Departamento"] for f in gj["features"]}
    datos = datos_mapas()
    nombres_datos = set(datos["Departamento"])
    print("En GeoJSON pero no en datos:", nombres_geo - nombres_datos)
    print("En datos pero no en GeoJSON:", nombres_datos - nombres_geo)
    print("Total features:", len(gj["features"]))
