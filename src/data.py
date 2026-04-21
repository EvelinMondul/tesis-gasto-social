import pandas as pd
import os
import unicodedata
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def limpiar_texto(texto):
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = texto.replace(" ", "_")
    return texto

def normalizar_depto(texto):
    return (str(texto).lower()
            .replace(" ", "_").replace("á","a").replace("é","e")
            .replace("í","i").replace("ó","o").replace("ú","u"))

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS EMBEBIDOS
# ═══════════════════════════════════════════════════════════════════════════════
_RAW = {
    "departamento": [
        "Amazonas","Antioquia","Arauca","Atlántico","Bogotá","Bolívar","Boyacá",
        "Caldas","Caquetá","Casanare","Cauca","Cesar","Chocó","Córdoba",
        "Cundinamarca","Guainía","Guaviare","Huila","La Guajira","Magdalena",
        "Meta","Nariño","Norte de Santander","Putumayo","Quindío","Risaralda",
        "San Andrés","Santander","Sucre","Tolima","Valle del Cauca","Vaupés","Vichada"
    ],
    "agua_potable": [8722807950,387857669881,31958048323,165897400976,163597009103,
        204401852284,155908310387,62700397047,50034252703,47892802776,158007241735,
        116311501666,110038480512,204137591528,231681167329,8608212072,13901735029,
        95715614430,113965402924,142002946882,87966031353,188108372129,139193939493,
        42762070741,27959349763,43589781818,1847957054,167428569543,102349249556,
        99812681249,178165113143,9583884401,18913415140],
    "cultura": [338383636,28392190229,1415644138,10211484118,23590684588,12229394094,
        11137727798,4062680380,2713675131,2738739392,7519641244,6062704067,7110504979,
        9308970483,15457961291,562047834,750512326,5590868534,4935320980,7398234424,
        5843883657,10386630223,8216418213,1729869802,2046216347,3806448158,42462637,
        12588182952,5327313344,6688885558,15447906496,744849557,1048664349],
    "deporte": [451178181,37856253633,1887525516,13615312157,31454246117,16305858794,
        14850303715,5416907174,3618233501,3651652521,10026188325,8083605425,9480673310,
        12411960645,20610615056,749397113,1000683101,7454491375,6580427972,9864312563,
        7791844869,13848840295,10955224282,2306493069,2728288460,5075264210,56616850,
        16784243943,7103084454,8918514070,20597208655,993132741,1398219134],
    "educacion": [103146096966,4437850477590,343804333494,1849385573845,3772997778998,
        1961182524455,1307921501699,783661401374,490072931427,451909225864,1635832063874,
        1072458441765,808665916847,1960392493895,2111388948702,76733386161,124096362997,
        1152006081108,1163483945012,1500744190959,824865727988,1663058654004,1332395902660,
        463789569138,444926671934,775722249542,43917813821,1891074487091,1028940263851,
        1276652929562,2621049865436,69757575635,137228230755],
    "libre_destinacion": [6043198780,241592033964,21689325046,51966892900,0,170639182732,
        207929626284,51703388071,42689556228,42938934169,100260806506,67921267661,
        130615629037,113352706333,165495388670,9900451019,14668533695,75594567925,
        53948216095,101176693026,65567812255,183706842859,112544614859,27583120290,
        16848868343,26354622896,1089308220,151771000144,83229011991,95361601937,
        58439794100,13556276429,18792907455],
    "libre_inversion": [7180770694,493989823758,25293961624,153471661124,298815338107,
        254717914704,281026235099,81182514135,60890184844,61334015669,129546477379,
        105187749483,174218390173,151036929287,307420535319,11839024262,18045447848,
        113296095477,74017764003,142431737062,119471912714,237388169377,165676729640,
        31683094692,31398439847,63441100853,1334432231,247572285490,114351819196,
        141412880001,226098547176,16982589912,23505226815],
    "salud": [111964695775,1731045660725,155053191019,931108432396,1102832472249,
        984253697805,458723028029,268377722367,218689629487,166894230792,637058405456,
        562254357176,258673612578,814961952388,657687792060,64365819336,55314192295,
        506826597121,540432426178,607978935798,374770130374,730447350348,736415242792,
        178751473878,168232758782,285330685541,13882026985,699101313582,482548943860,
        486949769128,1269890290805,53084708609,61357086889],
    "total": [237847131982,7358584109780,581102029160,3175656757516,5393287529162,
        3603730424868,2437496733011,1257105010548,868708463321,777359601183,2678250824519,
        1938279627243,1498803207436,3265602604559,3509742408427,172758337797,227777467291,
        1956484315970,1957363503164,2511597050714,1486277343210,3026944859235,2505398071939,
        748605691610,694140593476,1203320153018,62170617798,3186320082745,1823849686252,
        2115797261505,4389688725811,164703017284,262243750537],
    "poblacion": [86318,6903721,317398,2827124,7929539,2264523,1311983,1046110,428162,
        475144,1574506,1395486,605478,3553293,1914778,57934,100497,1192273,1057252,
        1513782,1145766,1709890,1709570,388716,566048,973879,62249,2376736,1006044,
        1380948,4647367,47961,125477],
    "agua_pc": [101054.33,56180.96,100687.62,58680.62,20631.34,90262.65,118834.09,
        59936.72,116858.23,100796.40,100353.53,83348.38,181738.20,57450.26,120996.36,
        148586.53,138329.85,80279.95,107793.98,93806.73,76774.87,110011.97,81420.44,
        110008.52,49393.96,44758.93,29686.53,70444.75,101734.37,72278.38,38336.79,
        199826.62,150732.13],
    "cultura_pc": [3920.20,4112.59,4460.15,3611.97,2975.04,5400.43,8489.23,3883.61,
        6337.96,5764.02,4775.87,4344.51,11743.62,2619.82,8072.98,9701.52,7468.01,
        4689.25,4668.06,4887.25,5100.42,6074.44,4806.13,4450.22,3614.92,3908.54,
        682.14,5296.42,5295.31,4843.69,3324.01,15530.32,8357.42],
    "deporte_pc": [5226.93,5483.46,5946.87,4815.96,3966.72,7200.57,11318.98,5178.14,
        8450.62,7685.36,6367.83,5792.68,15658.16,3493.09,10763.97,12935.36,9957.34,
        6252.34,6224.09,6516.34,6800.56,8099.26,6408.18,5933.62,4819.89,5211.39,
        909.52,7061.89,7060.41,6458.25,4432.02,20707.09,11143.23],
    "educacion_pc": [1194954.67,642820.08,1083196.28,654157.93,475815.53,866046.64,
        996904.31,749119.50,1144596.98,951099.51,1038949.40,768519.67,1335582.66,
        551711.47,1102680.81,1324496.60,1234826.54,966226.76,1100479.30,991387.26,
        719925.12,972611.49,779374.87,1193132.18,786022.87,796528.37,705518.38,
        795660.30,1022758.71,924475.74,563985.99,1454464.58,1093652.47],
    "libre_dest_pc": [70010.88,34994.47,68334.79,18381.54,0.00,75353.26,158485.00,
        49424.43,99704.22,90370.36,63677.63,48672.12,215723.16,31900.75,86430.59,
        170891.89,145959.92,63403.74,51026.83,66837.03,57226.18,107437.81,65832.12,
        70959.57,29765.79,27061.50,17499.21,63856.90,82729.00,69055.17,12574.82,
        282652.08,149771.73],
    "libre_inv_pc": [83189.73,71554.14,79691.62,54285.44,37683.82,112481.93,214199.60,
        77604.19,142212.96,129085.11,82277.54,75377.14,287736.95,42506.18,160551.53,
        204353.65,179562.06,95025.30,70009.58,94089.99,104272.52,138832.42,96911.35,
        81507.05,55469.57,65142.69,21437.01,104164.82,113664.83,102402.76,48650.89,
        354091.66,187326.97],
    "salud_pc": [1297118.74,250740.96,488513.45,329348.28,139079.01,434640.63,349640.98,
        256548.28,510763.75,351249.79,404608.43,402909.35,427222.15,229354.00,343479.92,
        1111019.77,550406.40,425092.74,511167.09,401629.12,327091.33,427189.67,430760.51,
        459851.08,297205.82,292983.71,223008.03,294143.44,479649.94,352619.92,273249.41,
        1106830.73,488990.71],
    "total_pc": [2755475.47,1065886.66,1830830.78,1123281.74,680151.46,1591386.10,
        1857872.19,1201694.86,2028924.71,1636050.55,1701010.24,1388963.86,2475404.90,
        919035.56,1832976.15,2981985.32,2266510.12,1640970.08,1851368.93,1659153.73,
        1297191.00,1770257.07,1465513.59,1925842.24,1226292.81,1235595.13,998740.83,
        1340628.53,1812892.56,1532133.91,944553.92,3434103.07,2089974.66],
}


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
EXCEL_NAME = "Gasto_sectores_dptos_2.xlsx"
SHEET_NAME = "Hoja2"

_LOCAL  = r"C:\Users\eveli\OneDrive - Universidad del Norte\tesis\data"
_BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_PATH  = os.path.join(_LOCAL, EXCEL_NAME)
DOCKER_PATH = os.path.join(_BASE, "data", EXCEL_NAME)


def cargar_datos() -> pd.DataFrame:
    """
    Intenta cargar desde Excel (local o Docker).
    Si no encuentra el archivo, usa los datos embebidos.
    Siempre agrega la columna 'region'.
    """
    df = None

    for path in [LOCAL_PATH, DOCKER_PATH]:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path, sheet_name=SHEET_NAME)
                df.columns = [limpiar_texto(c) for c in df.columns]
                mask = df["departamento"].str.lower().str.strip() != "total general"
                df = df[mask].dropna(subset=["departamento"]).reset_index(drop=True)
                break
            except Exception:
                df = None

    if df is None:
        # Fallback: datos embebidos (para Render y entornos sin Excel)
        df = pd.DataFrame(_RAW)

    df["region"] = df["departamento"].map(
        lambda d: REGION_MAP.get(normalizar_depto(d), "Otra")
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMNAS Y ETIQUETAS
# ═══════════════════════════════════════════════════════════════════════════════
SECTORES_PC = [
    "agua_pc", "educacion_pc", "salud_pc",
    "cultura_pc", "deporte_pc",
    "libre_dest_pc", "libre_inv_pc",
]

SECTORES_ABS = [
    "agua_potable", "educacion", "salud",
    "cultura", "deporte",
    "libre_destinacion", "libre_inversion",
]

SECTOR_LABELS = {
    "agua_pc":           "Agua Potable",
    "educacion_pc":      "Educación",
    "salud_pc":          "Salud",
    "cultura_pc":        "Cultura",
    "deporte_pc":        "Deporte",
    "libre_dest_pc":     "Libre Destinación",
    "libre_inv_pc":      "Libre Inversión",
    "agua_potable":      "Agua Potable",
    "educacion":         "Educación",
    "salud":             "Salud",
    "cultura":           "Cultura",
    "deporte":           "Deporte",
    "libre_destinacion": "Libre Destinación",
    "libre_inversion":   "Libre Inversión",
    "total_pc":          "Total",
    "total":             "Total General",
    "poblacion":         "Población",
}


# ═══════════════════════════════════════════════════════════════════════════════
# REGIONES
# ═══════════════════════════════════════════════════════════════════════════════
REGION_MAP = {
    "amazonas":                 "Amazónica",
    "antioquia":                "Andina",
    "arauca":                   "Llanos",
    "atlantico":                "Caribe",
    "bogota":                   "Andina",
    "bogota,_d.c.":             "Andina",
    "bolivar":                  "Caribe",
    "boyaca":                   "Andina",
    "caldas":                   "Andina",
    "caqueta":                  "Amazónica",
    "casanare":                 "Llanos",
    "cauca":                    "Andina",
    "cesar":                    "Caribe",
    "choco":                    "Pacífica",
    "cordoba":                  "Caribe",
    "cundinamarca":             "Andina",
    "guainia":                  "Amazónica",
    "guaviare":                 "Amazónica",
    "huila":                    "Andina",
    "la_guajira":               "Caribe",
    "magdalena":                "Caribe",
    "meta":                     "Llanos",
    "narino":                   "Andina",
    "norte_de_santander":       "Andina",
    "putumayo":                 "Amazónica",
    "quindio":                  "Andina",
    "risaralda":                "Andina",
    "san_andres_y_providencia": "Caribe",
    "san_andres":               "Caribe",
    "santander":                "Andina",
    "sucre":                    "Caribe",
    "tolima":                   "Andina",
    "valle_del_cauca":          "Pacífica",
    "vaupes":                   "Amazónica",
    "vichada":                  "Llanos",
}

REGION_COLORS = {
    "Andina":    "#4E9AF1",
    "Caribe":    "#F1A94E",
    "Pacífica":  "#50C878",
    "Amazónica": "#C97FF1",
    "Llanos":    "#F16B6B",
    "Otra":      "#8B949E",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DESCRIPCIÓN DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
VARIABLES_INFO = [
    {
        "variable":    "agua_pc",
        "etiqueta":    "Gasto en Agua Potable per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Inversión departamental destinada a acueducto, alcantarillado y saneamiento "
            "básico dividida entre la población 2024. Refleja el esfuerzo territorial en "
            "garantizar acceso al agua potable y condiciones sanitarias básicas."
        ),
    },
    {
        "variable":    "educacion_pc",
        "etiqueta":    "Gasto en Educación per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Recursos destinados al sector educativo por habitante. Es el componente de "
            "mayor peso dentro del gasto social departamental, determinado en gran medida "
            "por las transferencias del SGP."
        ),
    },
    {
        "variable":    "salud_pc",
        "etiqueta":    "Gasto en Salud per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Inversión en aseguramiento, salud pública y prestación de servicios de salud "
            "por habitante. Incluye recursos del SGP y recursos propios orientados al sector."
        ),
    },
    {
        "variable":    "cultura_pc",
        "etiqueta":    "Gasto en Cultura per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Gasto destinado a la promoción de la cultura, patrimonio, bibliotecas y "
            "actividades artísticas por habitante."
        ),
    },
    {
        "variable":    "deporte_pc",
        "etiqueta":    "Gasto en Deporte per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Inversión en recreación, deporte comunitario e infraestructura deportiva "
            "por habitante. Altamente correlacionado con el gasto en cultura."
        ),
    },
    {
        "variable":    "libre_dest_pc",
        "etiqueta":    "Libre Destinación per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Recursos de libre destinación por habitante. Proxy de capacidad fiscal "
            "autónoma territorial."
        ),
    },
    {
        "variable":    "libre_inv_pc",
        "etiqueta":    "Libre Inversión per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Recursos de libre inversión por habitante. Refleja la capacidad del "
            "departamento para financiar proyectos fuera de los sectores obligatorios del SGP."
        ),
    },
    {
        "variable":    "total_pc",
        "etiqueta":    "Gasto Total per cápita",
        "unidad":      "COP / habitante",
        "tipo":        "Continua · Razón",
        "fuente":      "TerriData – DNP",
        "descripcion": (
            "Suma de todos los componentes del gasto social departamental dividida entre "
            "la población 2024. Variable síntesis del nivel global de inversión social."
        ),
    },
    {
        "variable":    "poblacion",
        "etiqueta":    "Población 2024",
        "unidad":      "Habitantes",
        "tipo":        "Continua · Razón",
        "fuente":      "DANE – Proyecciones",
        "descripcion": (
            "Proyección poblacional DANE para 2024. Denominador para el cálculo de "
            "indicadores per cápita y variable de control en los análisis multivariados."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# PALETA Y TEMPLATE PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "bg":      "#0D1117",
    "surface": "#161B22",
    "card":    "#1C2333",
    "border":  "#30363D",
    "text1":   "#E6EDF3",
    "text2":   "#8B949E",
    "accent":  "#58A6FF",
    "green":   "#3FB950",
    "orange":  "#D29922",
    "red":     "#F78166",
    "purple":  "#C97FF1",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor = PALETTE["card"],
        plot_bgcolor  = PALETTE["card"],
        font          = dict(family="IBM Plex Mono, monospace",
                             color=PALETTE["text1"], size=11),
        title_font    = dict(family="IBM Plex Sans, sans-serif",
                             size=13, color=PALETTE["text1"]),
        xaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"],
                   zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"],
                   zerolinecolor=PALETTE["border"]),
        legend=dict(bgcolor=PALETTE["card"], bordercolor=PALETTE["border"], borderwidth=1),
        margin=dict(l=50, r=20, t=50, b=50),
        colorway=[
            PALETTE["accent"], PALETTE["green"], PALETTE["red"],
            PALETTE["orange"], PALETTE["purple"], "#4E9AF1", "#F1A94E",
        ],
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPOSICIONAL · Proporciones + CLR
# Aitchison (1986) · Martín-Fernández et al. (2003)
# ═══════════════════════════════════════════════════════════════════════════════

# Constante para reemplazo multiplicativo de ceros
DELTA_CLR = 0.0001

# Colores por sector (para gráficos composicionales)
SECTOR_COLORS = {
    "Agua Potable":      "#4A90D9",
    "Educación":         "#2ECC71",
    "Salud":             "#E74C3C",
    "Cultura":           "#9B59B6",
    "Deporte":           "#F39C12",
    "Libre Destinación": "#1ABC9C",
    "Libre Inversión":   "#E67E22",
}

# Colores por cluster
CLUSTER_COLORS = {1: "#2E86AB", 2: "#E84855", 3: "#3BB273", 4: "#F4A261"}


def cargar_proporciones():
    """
    Carga los datos y calcula las proporciones sectoriales del gasto total.
    Devuelve df con columnas prop_<sector> para cada sector absoluto.
    """
    df = cargar_datos()
    for s in SECTORES_ABS:
        if s in df.columns and "total" in df.columns:
            df[f"prop_{s}"] = df[s] / df["total"]
    return df


def calcular_clr(df):
    """
    Transformación CLR (centred log-ratio) sobre las proporciones.
    Aplica reemplazo multiplicativo de ceros antes de log (Martín-Fernández 2003).

    Parámetros
    ----------
    df : DataFrame con columnas prop_<sector>

    Retorna
    -------
    X_clr  : np.ndarray (n, p) — datos transformados
    prop_cols : list de nombres de columnas de proporciones
    labels    : list de etiquetas legibles
    """
    prop_cols = [f"prop_{s}" for s in SECTORES_ABS if f"prop_{s}" in df.columns]
    labels    = [SECTOR_LABELS.get(s, s) for s in SECTORES_ABS if f"prop_{s}" in df.columns]

    X = df[prop_cols].values.copy().astype(float)

    # Reemplazo multiplicativo de ceros (Bogotá: libre_destinacion = 0)
    for i in range(len(X)):
        zeros = X[i] == 0
        n_zeros = zeros.sum()
        if n_zeros > 0:
            X[i, zeros]  = DELTA_CLR
            X[i, ~zeros] = X[i, ~zeros] * (1 - n_zeros * DELTA_CLR)

    # CLR: log(x_i) - mean(log(x_j))
    X_clr = np.log(X) - np.log(X).mean(axis=1, keepdims=True)

    return X_clr, prop_cols, labels


def calcular_kmo_bartlett_prop(df):
    """
    KMO y prueba de Bartlett sobre un subconjunto de proporciones
    no colineales (evita singularidad de la matriz CLR completa).

    Retorna: kmo (float), chi2 (float), gl (int), pval (float)
    """
    from scipy import stats as _stats

    # 4 proporciones con menor colinealidad
    sel = ["prop_agua_potable", "prop_salud",
           "prop_cultura", "prop_libre_destinacion"]
    sel = [c for c in sel if c in df.columns]

    from sklearn.preprocessing import StandardScaler as _SS
    X   = _SS().fit_transform(df[sel].values)
    R   = np.corrcoef(X.T)
    Ri  = np.linalg.pinv(R)
    n   = R.shape[0]
    P   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(abs(Ri[i, i] * Ri[j, j]))
                if d > 1e-10:
                    P[i, j] = -Ri[i, j] / d
    r2  = np.sum(R[np.triu_indices(n, k=1)] ** 2)
    p2  = np.sum(P[np.triu_indices(n, k=1)] ** 2)
    kmo = r2 / (r2 + p2)

    no, p = X.shape
    det   = np.linalg.det(R)
    chi2  = -(no - 1 - (2*p + 5)/6) * np.log(det)
    gl    = int(p * (p - 1) / 2)
    pval  = 1 - _stats.chi2.cdf(chi2, gl)

    return kmo, chi2, gl, pval