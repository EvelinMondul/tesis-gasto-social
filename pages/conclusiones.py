"""
pages/conclusiones.py
=========================

Capítulo 10: Conclusiones.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html

from components.elements import kpi_card, method_card, page_header, section_title, section_text

dash.register_page(__name__, path="/conclusiones", name="Conclusiones", order=10)


def layout():
    return html.Div([
        page_header(
            "Capítulo 10",
            "Conclusiones",
            "Síntesis de los hallazgos de los Capítulos 02-09, limitaciones "
            "del análisis y recomendaciones derivadas del enfoque de "
            "Análisis de Datos Composicionales (CoDA) aplicado al gasto "
            "social departamental.",
        ),

        dbc.Row([
            dbc.Col(kpi_card("33", "Departamentos analizados"), md=3, className="mb-3"),
            dbc.Col(kpi_card("6 → 2", "Componentes del gasto reducidos a 2 factores (AFE)"), md=3, className="mb-3"),
            dbc.Col(kpi_card("2 + 1", "Clusters C1/C2 (K-Means) y Bogotá D.C. como caso atípico"), md=3, className="mb-3"),
            dbc.Col(kpi_card("-0.64", "rho Educación vs. IPM (p < 0.001)"), md=3, className="mb-3"),
        ], className="mb-3"),

        section_title("Síntesis de hallazgos por capítulo"),
        section_text([
            html.B("Capítulo 02 (EDA). "),
            "La composición bruta del gasto social 2024 muestra a "
            "Educación y Salud como los componentes de mayor peso promedio "
            "en los 33 departamentos, con una alta heterogeneidad en "
            "Libre Destinación y Libre Inversión, anticipando que estos "
            "dos rubros serán los principales diferenciadores entre "
            "perfiles departamentales.",
        ]),
        section_text([
            html.B("Capítulo 03 (Análisis Composicional). "),
            "El gasto social es, por construcción, un dato composicional "
            "que vive en un símplex de suma constante. El cierre, el "
            "reemplazo multiplicativo de ceros y la transformación CLR "
            "permiten proyectar las seis proporciones a un espacio "
            "euclídeo donde son válidas las técnicas multivariadas "
            "estándar (correlación, ACP, AFE, K-Means) empleadas en los "
            "capítulos siguientes.",
        ]),
        section_text([
            html.B("Capítulo 04 (Correlaciones sobre CLR). "),
            "La matriz de correlación de Spearman sobre las coordenadas "
            "CLR identifica siete pares de componentes significativamente "
            "asociados, en particular Educación-Libre Destinación, "
            "Libre Destinación-Salud y Libre Inversión-Salud (todas "
            "negativas), lo que evidencia que los departamentos enfrentan "
            "una asignación de suma constante: privilegiar un componente "
            "implica reducir la proporción relativa de otros.",
        ]),
        section_text([
            html.B("Capítulo 05 (ACP). "),
            "Las pruebas KMO y de esfericidad de Bartlett respaldan la "
            "pertinencia del ACP sobre las coordenadas CLR. CP1 opone "
            "Libre Destinación al resto de componentes (especialmente "
            "Educación y Agua potable) y separa a Bogotá D.C. del resto "
            "de departamentos; CP2 opone Libre Inversión y Cultura y "
            "Deporte a Salud y Educación. Ambos componentes principales "
            "satisfacen el criterio de Kaiser y resumen la mayor parte de "
            "la varianza de la estructura del gasto.",
        ]),
        section_text([
            html.B("Capítulo 06 (AFE). "),
            "La rotación varimax sobre la misma estructura de "
            "correlaciones produce dos factores más interpretables: F1, "
            "dominado por Educación y Salud frente a Libre Destinación, y "
            "F2, dominado por Cultura y Deporte y Libre Inversión. Esta "
            "solución de dos dimensiones es coherente con el ACP y se "
            "adopta como insumo para la segmentación de departamentos.",
        ]),
        section_text([
            html.B("Capítulo 07 (Clustering). "),
            "Bogotá D.C. se excluye del K-Means por constituir un caso "
            "atípico (Distrito Capital, con Libre Destinación nula y la "
            "mayor proporción de gasto en Educación de los 33 "
            "departamentos). Sobre los 32 departamentos restantes, "
            "K-Means (k=2, silueta = 0.4765) identifica dos perfiles de "
            "gasto: C1 (21 departamentos, mayor margen discrecional "
            "—Libre Destinación y Libre Inversión—) y C2 (11 "
            "departamentos, mayor peso del gasto en Salud y menor margen "
            "discrecional). Las diferencias entre C1 y C2 son "
            "estadísticamente significativas (p < 0.05) en Libre "
            "Inversión, Libre Destinación y Cultura y Deporte.",
        ]),
        section_text([
            html.B("Capítulo 08 (Contraste con el IPM). "),
            "El IPM (%) no difiere significativamente entre los clusters "
            "de gasto (Mann-Whitney U, p = 0.606): pertenecer a "
            "un perfil de gasto discrecional u orientado a Salud no se "
            "asocia, por sí solo, con un IPM distinto. Sin embargo, la "
            "proporción específica de gasto en Educación sí está "
            "correlacionada negativamente con el IPM (rho = -0.64, "
            "p < 0.001), al igual que Salud, Libre Destinación y Agua "
            "potable lo están positivamente (rho entre 0.43 y 0.47, "
            "p < 0.05).",
        ]),
        section_text([
            html.B("Capítulo 09 (Mapas territoriales). "),
            "La proyección geográfica confirma un patrón centro-periferia: "
            "los departamentos con mayor IPM (Vichada, Guainía, Vaupés, "
            "La Guajira, Chocó) se concentran en las regiones Amazónica, "
            "Orinoquía, Caribe y Pacífica, pertenecen mayoritariamente al "
            "cluster C1 y destinan entre 42 % y 44 % de su gasto a "
            "Educación, frente al 70 % de Bogotá (caso atípico) y San "
            "Andrés y Providencia, los departamentos con menor IPM.",
        ]),

        section_title("Conclusión general"),
        method_card(
            "Respuesta a la pregunta de investigación",
            [
                "El enfoque CoDA permite caracterizar de forma rigurosa la "
                "estructura del gasto social departamental sin incurrir en "
                "las correlaciones espurias propias de la restricción de "
                "suma constante. Los resultados muestran que el ",
                html.B("perfil agregado de gasto"),
                " (los clusters C1/C2, basados principalmente en el "
                "margen de gasto discrecional, con Bogotá D.C. como caso "
                "atípico excluido del clustering) ", html.B("no presenta una "
                "asociación estadísticamente significativa con el IPM"),
                ", pero que ", html.B("componentes específicos de ese "
                "perfil sí la presentan"), ": en particular, una mayor "
                "proporción de gasto en Educación se asocia de forma "
                "robusta con un menor IPM, mientras que mayores "
                "proporciones en Salud, Agua potable y Libre Destinación "
                "se asocian con un IPM más alto —consistente con que estos "
                "departamentos destinan más recursos a atender carencias "
                "estructurales ya existentes—. La relación entre "
                "composición del gasto social e IPM es, por tanto, "
                "específica por componente y no se reduce a una "
                "tipología agregada de departamentos.",
            ],
        ),

        section_title("Limitaciones"),
        section_text([
            "El análisis se basa en un único corte transversal (gasto "
            "2024 / IPM 2025), por lo que no permite establecer relaciones "
            "causales ni dinámicas temporales entre la composición del "
            "gasto y la pobreza multidimensional. El tamaño muestral (n = "
            "33) limita la potencia estadística de la prueba de "
            "Mann-Whitney U, en particular para Bogotá D.C. (caso atípico, "
            "n = 1), que debió excluirse tanto del K-Means como de las "
            "comparaciones entre grupos. "
            "Adicionalmente, la agregación departamental oculta la "
            "heterogeneidad municipal dentro de cada departamento, y el "
            "reemplazo multiplicativo de ceros, aunque estándar en CoDA, "
            "introduce un supuesto sobre la magnitud de las proporciones "
            "no observadas.",
        ]),

        section_title("Recomendaciones de política pública"),
        section_text([
            "Los resultados sugieren que, más que modificar la "
            "distribución agregada del gasto entre rubros discrecionales y "
            "no discrecionales, los departamentos con mayor IPM "
            "(Vichada, Guainía, Vaupés, La Guajira y Chocó) podrían "
            "beneficiarse de un incremento focalizado en la proporción de "
            "gasto destinada a Educación, manteniendo los niveles "
            "actuales de inversión en Agua potable y Salud, que ya "
            "atienden carencias estructurales identificadas por el IPM. "
            "El monitoreo de estos perfiles composicionales —en lugar de "
            "montos absolutos— ofrece una herramienta complementaria para "
            "el seguimiento de la asignación del Sistema General de "
            "Participaciones.",
        ]),

        section_title("Líneas futuras de investigación"),
        section_text([
            "Futuros trabajos podrían: (i) construir un panel "
            "multianual del gasto social y el IPM para evaluar relaciones "
            "dinámicas y de causalidad mediante modelos de datos de "
            "panel; (ii) desagregar el análisis a nivel municipal, donde "
            "la restricción composicional y la heterogeneidad territorial "
            "son aún más pronunciadas; (iii) incorporar los 15 "
            "indicadores de privación del IPM de forma individual —y no "
            "solo el índice agregado— en el contraste con la composición "
            "del gasto mediante regresión composicional (CoDA "
            "regression); y (iv) explorar métodos de clustering "
            "composicional (p. ej. sobre coordenadas CLR directamente, sin "
            "pasar por ACP/AFE) para validar la robustez de los "
            "perfiles C1/C2 y del tratamiento de Bogotá D.C. como caso "
            "atípico identificados en el Capítulo 07.",
        ]),
    ])