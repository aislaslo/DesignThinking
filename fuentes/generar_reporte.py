"""
generar_reporte.py
Genera el reporte tecnico PDF de la Actividad 5.
"""

from fpdf import FPDF, XPos, YPos
from pathlib import Path

OUTPUT = Path("/Users/islas/Documents/Development/master_ai/docs/reporte_actividad5.pdf")
ASSETS = Path(__file__).resolve().parent.parent / "datos" / "datos_limp"


class ReportePDF(FPDF):
    AZUL  = (41,  128, 185)
    GRIS  = (100, 100, 100)
    NEGRO = (30,  30,  30)
    BLANC = (255, 255, 255)
    FONDO = (245, 247, 250)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.GRIS)
        self.cell(0, 8, "Actividad 5 - Entrenamiento y Ajuste de Modelos | Alejandro Islas Lopez",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*self.AZUL)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.GRIS)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")

    def portada(self):
        self.add_page()
        self.set_fill_color(*self.AZUL)
        self.rect(0, 0, 210, 60, "F")
        self.set_y(18)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*self.BLANC)
        self.multi_cell(0, 10, "Actividad 5\nEntrenamiento y Ajuste de Modelos", align="C")
        self.set_y(62)
        datos = [
            ("Alumno",    "Alejandro Islas Lopez"),
            ("Materia",   "Inteligencia Artificial Aplicada"),
            ("Fecha",     "Junio 2025"),
            ("Dataset",   "Resenas de Restaurante - Zapopan, Jalisco"),
            ("Tarea",     "Clasificacion de Sentimientos (POS / NEU / NEG)"),
            ("Modelos",   "Regresion Logistica vs Random Forest"),
            ("Tracking",  "MLflow - Experimento: Actividad5_Sentimientos"),
        ]
        for label, valor in datos:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*self.AZUL)
            self.cell(52, 8, f"{label}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*self.NEGRO)
            self.cell(0, 8, valor, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        self.set_draw_color(*self.AZUL)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())

    def seccion(self, n, titulo):
        self.ln(5)
        self.set_fill_color(*self.AZUL)
        self.set_text_color(*self.BLANC)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {n}. {titulo}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*self.NEGRO)
        self.ln(3)

    def subseccion(self, titulo):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.AZUL)
        self.cell(0, 7, titulo, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*self.NEGRO)

    def parrafo(self, texto):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.NEGRO)
        self.multi_cell(0, 5.5, texto)
        self.ln(2)

    def tabla(self, encabezados, filas, anchos=None):
        if anchos is None:
            w = 190 // len(encabezados)
            anchos = [w] * len(encabezados)
        self.set_fill_color(*self.AZUL)
        self.set_text_color(*self.BLANC)
        self.set_font("Helvetica", "B", 9)
        for i, h in enumerate(encabezados):
            self.cell(anchos[i], 7, h, border=1, fill=True, align="C",
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln()
        for j, fila in enumerate(filas):
            self.set_fill_color(*self.FONDO)
            self.set_text_color(*self.NEGRO)
            self.set_font("Helvetica", "", 9)
            for i, celda in enumerate(fila):
                self.cell(anchos[i], 6.5, str(celda), border=1,
                          fill=(j % 2 == 0), align="C",
                          new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.ln()
        self.ln(3)

    def img(self, ruta, ancho=170, caption=""):
        p = Path(ruta)
        if p.exists():
            self.image(str(p), x=(210 - ancho) / 2, w=ancho)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*self.GRIS)
                self.multi_cell(0, 5, caption, align="C")
                self.set_text_color(*self.NEGRO)
            self.ln(3)
        else:
            self.parrafo(f"[imagen no disponible: {p.name}]")


def generar():
    pdf = ReportePDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)

    # Portada
    pdf.portada()
    pdf.add_page()

    # 1. Dataset
    pdf.seccion(1, "Descripcion del Dataset")
    pdf.parrafo(
        "El dataset utilizado es un conjunto sintetico de 400 resenas de un restaurante "
        "de cocina tradicional mexicana en Zapopan, Jalisco. Fue generado con semilla fija "
        "(seed=42) para garantizar reproducibilidad. Cubre opiniones de tres canales: "
        "Google Reviews, Facebook y Encuestas Internas."
    )
    pdf.subseccion("Diccionario de datos")
    pdf.tabla(
        ["Campo", "Tipo", "Valores", "Descripcion"],
        [
            ["id",          "Entero",     "1 - 400",                     "Identificador unico"],
            ["comentario",  "Texto",      "Variable",                    "Opinion del cliente"],
            ["sentimiento", "Categorico", "POS / NEU / NEG",             "Etiqueta de sentimiento"],
            ["fecha",       "Fecha",      "2024-01-01 a 2024-12-31",     "Fecha de publicacion"],
            ["fuente",      "Categorico", "Google / Facebook / Encuesta","Canal de origen"],
        ],
        anchos=[28, 26, 68, 68]
    )
    pdf.subseccion("Distribucion de clases")
    pdf.tabla(
        ["Sentimiento", "Cantidad", "Porcentaje"],
        [
            ["POS (Positivo)", "158", "39.5%"],
            ["NEU (Neutral)",  "132", "33.0%"],
            ["NEG (Negativo)", "110", "27.5%"],
        ],
        anchos=[63, 63, 64]
    )
    pdf.subseccion("Supuestos")
    pdf.parrafo(
        "1. El dataset es sintetico; los patrones son mas regulares que datos reales.\n"
        "2. Los tres canales tienen distribuciones de sentimiento similares.\n"
        "3. El vocabulario representa el espanol mexicano coloquial de Zapopan, Jalisco.\n"
        "4. Las etiquetas son correctas sin necesidad de revision adicional."
    )
    pdf.img(ASSETS / "eda_distribucion.png",
            caption="Figura 1. Distribucion de sentimientos y longitud de comentarios.")
    pdf.img(ASSETS / "eda_wordclouds.png",
            caption="Figura 2. Nubes de palabras por categoria de sentimiento.")

    # 2. Limpieza
    pdf.seccion(2, "Proceso de Limpieza y Estandarizacion")
    pdf.parrafo(
        "Pipeline implementado en fuentes/datos_prep.py. "
        "No se encontraron valores nulos en el dataset."
    )
    pdf.tabla(
        ["Paso", "Transformacion", "Justificacion"],
        [
            ["1", "Verificacion de nulos",          "Eliminar registros incompletos"],
            ["2", "Conversion a minusculas",         "Normalizar vocabulario"],
            ["3", "Eliminacion de caracteres esp.", "Conservar letras, acentos, espacios"],
            ["4", "Normalizacion de espacios",       "Eliminar espacios multiples"],
            ["5", "Encoding de etiquetas",           "NEG=0, NEU=1, POS=2"],
            ["6", "Vectorizacion TF-IDF",            "Unigramas+bigramas, 5000 features"],
            ["7", "Split estratificado 80/20",       "Preservar distribucion de clases"],
        ],
        anchos=[10, 80, 100]
    )
    pdf.parrafo(
        "El vectorizador TF-IDF con sublinear_tf=True reduce el peso de palabras muy "
        "frecuentes. El rango de n-gramas (1,2) captura patrones como 'muy bueno' o "
        "'tiempo de espera' que son mas informativos que palabras individuales."
    )

    # 3. Versionamiento
    pdf.seccion(3, "Estrategia de Versionamiento")
    pdf.parrafo(
        "Repositorio en GitHub: https://github.com/aislaslo/DesignThinking\n"
        "Versionamiento semantico documentado en CHANGELOG.md."
    )
    pdf.tabla(
        ["Commit", "Tipo", "Descripcion"],
        [
            ["7fb0106", "feat", "Estructura inicial: datos, fuentes, README, CHANGELOG"],
            ["f59cfa0", "fix",  "Eliminar parametro deprecated multi_class"],
        ],
        anchos=[28, 18, 144]
    )
    pdf.parrafo(
        "Convencion: feat (nueva funcionalidad), fix (correccion), docs (documentacion). "
        "Cada cambio al dataset o hiperparametros se documenta en CHANGELOG.md."
    )

    # 4. Modelos
    pdf.seccion(4, "Descripcion de los Modelos Entrenados")
    pdf.subseccion("Modelo 1 - Regresion Logistica")
    pdf.parrafo(
        "Clasificador lineal que modela la probabilidad de pertenencia a cada clase "
        "mediante funcion softmax en el caso multiclase. Es interpretable, eficiente y "
        "sirve como linea base robusta para clasificacion de texto con TF-IDF, donde la "
        "relacion entre features y etiquetas tiende a ser aproximadamente lineal."
    )
    pdf.subseccion("Modelo 2 - Random Forest")
    pdf.parrafo(
        "Ensemble de arboles de decision entrenados con bootstrap sampling y seleccion "
        "aleatoria de features. Robusto al ruido, captura interacciones no lineales entre "
        "terminos y maneja bien la alta dimensionalidad del espacio TF-IDF."
    )

    # 5. Hiperparametros
    pdf.seccion(5, "Configuracion de Hiperparametros")
    pdf.parrafo("GridSearchCV con validacion cruzada estratificada 5-fold, scoring=f1_macro.")
    pdf.subseccion("Regresion Logistica")
    pdf.tabla(
        ["Hiperparametro", "Valores evaluados", "Optimo"],
        [
            ["C (regularizacion)", "0.01, 0.1, 1, 10", "10"],
            ["solver",             "lbfgs, saga",       "lbfgs"],
            ["max_iter",           "1000 (fijo)",       "1000"],
        ],
        anchos=[60, 70, 60]
    )
    pdf.subseccion("Random Forest")
    pdf.tabla(
        ["Hiperparametro", "Valores evaluados", "Optimo"],
        [
            ["n_estimators",      "100, 200",      "100"],
            ["max_depth",         "None, 10, 20",  "None"],
            ["min_samples_split", "2, 5",          "5"],
        ],
        anchos=[60, 70, 60]
    )

    # 6. Metricas
    pdf.seccion(6, "Metricas Obtenidas")
    pdf.tabla(
        ["Metrica", "Regresion Logistica", "Random Forest"],
        [
            ["Accuracy",          "1.0000", "0.9500"],
            ["Precision (macro)", "1.0000", "0.9630"],
            ["Recall (macro)",    "1.0000", "0.9464"],
            ["F1-Score (macro)",  "1.0000", "0.9522"],
            ["ROC-AUC (OVR)",     "1.0000", "0.9995"],
            ["CV F1 mean",        "1.0000", "0.9783"],
            ["CV F1 std",         "0.0000", "0.0232"],
        ],
        anchos=[63, 63, 64]
    )
    pdf.parrafo(
        "La Regresion Logistica alcanzo metricas perfectas (1.0). Esto es caracteristico "
        "de datasets sinteticos donde los patrones son altamente regulares y separables "
        "linealmente. En produccion con datos reales se esperan valores mas representativos."
    )
    pdf.img(ASSETS / "comparativa_modelos.png",
            caption="Figura 3. Comparativa de metricas en conjunto de prueba.")

    # 7. Comparacion estadistica
    pdf.seccion(7, "Comparacion Estadistica de Resultados")
    pdf.tabla(
        ["Criterio", "Regresion Logistica", "Random Forest", "Ventaja"],
        [
            ["Accuracy",          "1.000",  "0.950",  "Logistica (+5.0%)"],
            ["F1-macro",          "1.000",  "0.952",  "Logistica (+4.8%)"],
            ["ROC-AUC",           "1.000",  "0.999",  "Logistica (~igual)"],
            ["Tiempo entrenamiento", "< 1s", "~15s",  "Logistica"],
            ["Interpretabilidad", "Alta",   "Media",  "Logistica"],
            ["No linealidad",     "No",     "Si",     "Random Forest"],
        ],
        anchos=[52, 42, 42, 54]
    )
    pdf.parrafo(
        "Para datos reales, Random Forest podria reducir la brecha gracias a su capacidad "
        "de capturar interacciones no lineales entre terminos."
    )
    pdf.img(ASSETS / "confusion_regresion_logistica.png", ancho=130,
            caption="Figura 4. Matriz de confusion - Regresion Logistica.")
    pdf.img(ASSETS / "confusion_random_forest.png", ancho=130,
            caption="Figura 5. Matriz de confusion - Random Forest.")
    pdf.img(ASSETS / "roc_regresion_logistica.png", ancho=150,
            caption="Figura 6. Curvas ROC (OVR) - Regresion Logistica.")
    pdf.img(ASSETS / "roc_random_forest.png", ancho=150,
            caption="Figura 7. Curvas ROC (OVR) - Random Forest.")

    # 8. MLflow
    pdf.seccion(8, "Evidencia del Uso de MLflow")
    pdf.parrafo(
        "MLflow configurado en modo local (backend-store-uri: mlruns/). "
        "Experimento 'Actividad5_Sentimientos' con dos runs registrados."
    )
    pdf.tabla(
        ["Tipo", "Que se registro"],
        [
            ["Parametros", "C, solver, n_estimators, max_depth, min_samples_split, cv_folds, ngram_range, max_features"],
            ["Metricas",   "accuracy, precision_macro, recall_macro, f1_macro, roc_auc_ovr, cv_f1_macro_mean, cv_f1_macro_std"],
            ["Artefactos", "Matriz de confusion (PNG), Curvas ROC (PNG), modelo serializado (sklearn format)"],
        ],
        anchos=[32, 158]
    )
    pdf.parrafo(
        "Panel disponible en: http://127.0.0.1:5000\n"
        "Comando: mlflow ui --backend-store-uri file:///ruta/mlruns --port 5000"
    )

    # 9. Deuda tecnica
    pdf.seccion(9, "Reflexion sobre Deuda Tecnica y MLOps")
    pdf.subseccion("Deuda tecnica identificada")
    pdf.tabla(
        ["Deuda", "Descripcion", "Prioridad"],
        [
            ["Dataset sintetico",   "Patrones regulares; puede no generalizar a datos reales", "Alta"],
            ["Fuga de datos",       "Vectorizador ajustado antes del split; debe ir en Pipeline", "Alta"],
            ["Sin Pipeline sklearn","Preprocesamiento y modelo separados dificultan despliegue", "Alta"],
            ["Sin stopwords",       "No se filtran palabras vacias en espanol", "Media"],
            ["Sin drift monitoring","No se detecta cambio en vocabulario de clientes", "Media"],
        ],
        anchos=[42, 118, 30]
    )
    pdf.subseccion("Mejoras MLOps propuestas")
    pdf.tabla(
        ["Mejora", "Impacto"],
        [
            ["sklearn.Pipeline para preprocesamiento + modelo",          "Elimina fuga de datos y simplifica despliegue"],
            ["MLflow Model Registry con staging dev->prod",             "Versionamiento formal y promocion controlada"],
            ["Dataset real de resenas etiquetadas manualmente",         "Mejora la generalizacion en produccion"],
            ["CI/CD con GitHub Actions para reentrenamiento",           "Automatiza el ciclo de mejora del modelo"],
            ["Endpoint serverless (Cloud Run / AWS Lambda)",            "Integracion con sistema POS del restaurante"],
        ],
        anchos=[105, 85]
    )

    pdf.output(str(OUTPUT))
    print(f"PDF generado: {OUTPUT}")


if __name__ == "__main__":
    generar()
