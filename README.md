# Actividad 5 — Entrenamiento, Ajuste y Registro de Modelos de Clasificación

**Alumno:** Alejandro Islas López  
**Materia:** Inteligencia Artificial Aplicada  
**Fecha:** Junio 2025

---

## Descripción

Comparación de **Regresión Logística** vs **Random Forest** para clasificación de sentimientos (POS / NEU / NEG) en reseñas de un restaurante de Zapopan, Jalisco. Se aplica GridSearchCV con validación cruzada estratificada 5-fold y se registran todos los experimentos en MLflow.

---

## Estructura del repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/    ← CSV original (400 reseñas)
│   └── datos_limp/   ← CSV limpio, vectorizador, gráficas y artefactos
├── fuentes/
│   ├── datos_prep.py ← pipeline de limpieza y vectorización
│   ├── train.py      ← entrenamiento + GridSearch + MLflow
│   └── entrena.ipynb ← notebook Colab completo (EDA → modelos → comparativa)
├── mlruns/           ← tracking local de MLflow (generado al ejecutar)
├── README.md
└── CHANGELOG.md
```

---

## Dataset

| Campo | Descripción |
|---|---|
| `id` | Identificador único (1–400) |
| `comentario` | Texto de la reseña del cliente |
| `sentimiento` | Etiqueta: POS / NEU / NEG |
| `fecha` | Fecha de publicación |
| `fuente` | Canal: Google Reviews / Facebook / Encuesta Interna |

**Distribución:** POS 39.5% — NEU 33.0% — NEG 27.5%  
**Origen:** Sintético, generado con `seed=42` en español mexicano.

---

## Cómo reproducir

### Opción A — Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/Actividad5.git
cd Actividad5

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Preparar datos
python fuentes/datos_prep.py

# 4. Entrenar modelos y registrar en MLflow
python fuentes/train.py

# 5. Ver panel de MLflow
mlflow ui --backend-store-uri mlruns/
# Abrir http://localhost:5000
```

### Opción B — Google Colab

1. Abrir `fuentes/entrena.ipynb` en Google Colab
2. Ejecutar la celda de instalación (Celda 0)
3. Ejecutar todas las celdas en orden

---

## Dependencias

```
pandas>=2.0
scikit-learn>=1.4
mlflow>=2.12
matplotlib>=3.8
seaborn>=0.13
wordcloud>=1.9
```

---

## Decisiones de diseño

| Decisión | Justificación |
|---|---|
| TF-IDF (1,2)-gramas, 5000 features | Captura bigramas informativos sin explotar la dimensionalidad |
| Split 80/20 estratificado | Preserva la distribución de clases en train y test |
| GridSearchCV 5-fold | Balance entre robustez estadística y tiempo de cómputo |
| Scoring = f1_macro | Métrica adecuada para clases desbalanceadas |
| MLflow local (`mlruns/`) | Sin dependencias de servidor externo, reproducible offline |
| seed=42 en todo | Garantiza reproducibilidad completa |

---

## Resultados (se completan después de ejecutar)

| Modelo | Accuracy | F1-macro | ROC-AUC |
|---|---|---|---|
| Regresión Logística | — | — | — |
| Random Forest | — | — | — |
