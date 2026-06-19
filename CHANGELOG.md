# CHANGELOG — Actividad 5

## [1.0.0] — 2025-06-18

### Estructura inicial del proyecto
- Creación del repositorio con estructura `datos/`, `fuentes/`
- Copia del dataset original a `datos/datos_ini/resenas_restaurante_zapopan.csv`

### Dataset
- **Fuente:** Datos sintéticos del proyecto POSAnalisisSentimientos (Actividad 4)
- **Idioma:** Español mexicano
- **Registros:** 400 reseñas de restaurante en Zapopan, Jalisco
- **Distribución:** POS 158 (39.5%) · NEU 132 (33%) · NEG 110 (27.5%)
- **Supuestos:** Dataset balanceado sintéticamente con `seed=42`; representa opiniones de Google Reviews, Facebook y Encuestas internas

### Limpieza y estandarización (`datos_prep.py`)
- Conversión de texto a minúsculas
- Eliminación de caracteres no alfabéticos (conservando acentos y ñ)
- Normalización de espacios en blanco
- Verificación y eliminación de valores nulos en columnas críticas
- Codificación de etiquetas: NEG=0, NEU=1, POS=2
- Vectorización TF-IDF: unigramas + bigramas, 5000 features, `sublinear_tf=True`
- Split estratificado 80/20 con `seed=42`

### Modelos entrenados (`train.py`)
- **Regresión Logística** — GridSearchCV sobre `C` y `solver`
- **Random Forest** — GridSearchCV sobre `n_estimators`, `max_depth`, `min_samples_split`
- Validación cruzada estratificada 5-fold
- Scoring: F1-macro

### Métricas registradas en MLflow
- accuracy, precision_macro, recall_macro, f1_macro, roc_auc_ovr
- cv_f1_macro_mean, cv_f1_macro_std

### Artefactos generados
- Matrices de confusión por modelo (PNG)
- Curvas ROC por modelo (PNG)
- Modelos serializados en MLflow
- Vectorizador TF-IDF serializado (pickle)
- Gráficas EDA: distribución de clases, longitud de comentarios, nubes de palabras
- Gráfica comparativa de métricas

---

## Deuda técnica y reflexión MLOps

### Deuda técnica identificada
1. **Dataset sintético:** Los modelos están entrenados con datos generados, no con reseñas reales. El rendimiento en producción puede diferir significativamente.
2. **Sin preprocesamiento de stopwords:** No se eliminaron palabras vacías en español, lo que puede introducir ruido en el TF-IDF.
3. **Vectorizador acoplado al split:** El vectorizador se ajusta antes del split, lo que introduce una leve fuga de datos. En producción debe ajustarse solo con train.
4. **Sin pipeline sklearn formal:** La limpieza y vectorización están separadas del modelo, dificultando el despliegue como un único artefacto.

### Mejoras MLOps propuestas
| Mejora | Prioridad | Impacto |
|---|---|---|
| Usar `sklearn.Pipeline` para empaquetar preprocesamiento + modelo | Alta | Elimina fuga de datos y simplifica despliegue |
| Registrar el modelo en MLflow Model Registry | Alta | Permite versionamiento y staging (dev→prod) |
| Agregar dataset real de reseñas | Alta | Mejora generalización del modelo |
| Implementar monitoreo de drift de datos | Media | Detecta cuando el vocabulario de clientes cambia |
| CI/CD con GitHub Actions para reentrenar automáticamente | Media | Automatiza el ciclo de mejora del modelo |
| Serverless endpoint (AWS Lambda / Cloud Run) | Baja | Permite integración con el sistema POS del restaurante |
