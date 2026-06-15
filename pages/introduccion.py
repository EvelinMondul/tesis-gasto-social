"""
pages/introduccion.py
======================

Capítulo 01: Introducción. Plantea el problema de investigación, la
justificación, los objetivos (general y específicos) y la hoja de ruta
del documento.
"""

import dash
from dash import html

from components.elements import page_header, section_title, section_text

dash.register_page(__name__, path="/introduccion", name="Introducción", order=1)


def layout():
    return html.Div([
        page_header(
            "Capítulo 01",
            "Introducción",
            "Planteamiento del problema, justificación, objetivos y "
            "estructura del trabajo de grado.",
        ),

        section_title("Planteamiento del problema"),
        section_text([
            "El proceso de descentralización fiscal en Colombia, materializado "
            "principalmente a través del Sistema General de Participaciones "
            "(SGP), transfiere a los 32 departamentos y al Distrito Capital "
            "recursos que deben distribuirse entre sectores sociales "
            "prioritarios: agua potable, educación, salud, cultura y deporte, "
            "y los rubros de libre destinación y libre inversión. Cada "
            "departamento decide, dentro de los márgenes normativos, qué "
            "proporción de su presupuesto social destina a cada sector, lo "
            "que da lugar a perfiles de asignación heterogéneos entre "
            "territorios con niveles de pobreza, población y capacidad "
            "institucional muy distintos.",
        ]),
        section_text([
            "Estos perfiles de gasto son, por construcción, datos ",
            html.B("composicionales"),
            ": cada vector de proporciones departamentales suma 100 % del "
            "presupuesto social, por lo que sus componentes no son "
            "independientes entre sí. Analizar esta información con técnicas "
            "estadísticas estándar (correlación de Pearson, ACP sobre "
            "proporciones brutas, distancias euclídeas) puede producir "
            "resultados espurios —correlaciones negativas inducidas "
            "artificialmente por la restricción de suma constante— y "
            "conclusiones poco robustas sobre la relación entre la "
            "estructura del gasto y los resultados sociales, en particular "
            "la pobreza multidimensional medida por el IPM.",
        ]),

        section_title("Justificación"),
        section_text([
            "El Análisis de Datos Composicionales (CoDA), formalizado por "
            "Aitchison (1986) y desarrollado posteriormente por "
            "Pawlowsky-Glahn y Buccianti (2011), provee un marco geométrico "
            "y estadístico coherente para tratar este tipo de información: "
            "mediante la transformación log-razón centrada (CLR), las "
            "proporciones del simplex se proyectan a un espacio euclídeo "
            "donde sí son válidas las correlaciones, el Análisis de "
            "Componentes Principales (ACP), el Análisis Factorial "
            "Exploratorio (AFE) y los algoritmos de agrupamiento como "
            "K-Means. Aplicar este marco al gasto social departamental "
            "colombiano permite caracterizar de forma rigurosa los patrones "
            "de asignación presupuestal y evaluar si existe una asociación "
            "estadísticamente sustentable entre dichos patrones y los "
            "indicadores de privación del IPM.",
        ]),

        section_title("Objetivo general"),
        section_text(
            "Analizar los patrones de asignación del gasto social en los 33 "
            "departamentos de Colombia durante el año fiscal 2024 mediante "
            "un enfoque de Análisis de Datos Composicionales, y contrastar "
            "los perfiles resultantes con el Índice de Pobreza "
            "Multidimensional (IPM) 2025."
        ),

        section_title("Objetivos específicos"),
        html.Ul([
            html.Li("Caracterizar descriptivamente la composición del gasto social "
                    "departamental en sus seis componentes (Agua potable, Cultura y "
                    "Deporte, Educación, Libre Destinación, Libre Inversión y Salud)."),
            html.Li("Transformar la composición del gasto mediante la transformación "
                    "log-razón centrada (CLR), aplicando el reemplazo multiplicativo "
                    "de ceros donde corresponda."),
            html.Li("Estimar la matriz de correlación de Spearman sobre las "
                    "coordenadas CLR y evaluar la pertinencia del Análisis de "
                    "Componentes Principales mediante las pruebas KMO y de "
                    "esfericidad de Bartlett."),
            html.Li("Reducir la dimensionalidad de la estructura del gasto mediante "
                    "Análisis de Componentes Principales y Análisis Factorial "
                    "Exploratorio, aplicando el criterio de Kaiser."),
            html.Li("Agrupar los departamentos en perfiles de gasto homogéneos "
                    "mediante K-Means sobre las puntuaciones factoriales, "
                    "seleccionando el número de grupos mediante el método del "
                    "codo y el coeficiente de silueta."),
            html.Li("Contrastar los perfiles de gasto resultantes con los 15 "
                    "indicadores de privación del IPM 2025 mediante la prueba de "
                    "Kruskal-Wallis."),
            html.Li("Visualizar territorialmente los resultados mediante mapas "
                    "coropléticos de los 33 departamentos."),
        ], className="section-text"),

        section_title("Estructura del documento"),
        section_text(
            "El presente dashboard reproduce la estructura del trabajo de "
            "grado en diez capítulos navegables desde la barra lateral: "
            "Análisis Exploratorio de Datos (02), Análisis Composicional "
            "(03), Correlaciones sobre CLR (04), Componentes Principales "
            "(05), Análisis Factorial Exploratorio (06), Clustering (07), "
            "Pobreza Multidimensional (08), Mapas Territoriales (09) y "
            "Conclusiones (10). Cada capítulo incluye, junto a las "
            "visualizaciones, un recuadro de justificación metodológica que "
            "explica la decisión estadística tomada en ese punto del "
            "análisis."
        ),
    ])
