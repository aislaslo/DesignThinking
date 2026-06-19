"""
train.py
Entrena Regresión Logística y Random Forest con GridSearchCV + validación cruzada.
Registra parámetros, métricas y artefactos en MLflow.
"""

import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).parent))
from datos_prep import preparar

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MLFLOW_URI  = f"file://{ROOT / 'mlruns'}"
ARTIFACT_DIR = ROOT / "datos" / "datos_limp"

SEED = 42
CV_FOLDS = 5
CLASSES = [0, 1, 2]
CLASS_NAMES = ["NEG", "NEU", "POS"]


# ── Utilidades ─────────────────────────────────────────────────────────────────
def calcular_metricas(y_true, y_pred, y_prob):
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr":     roc_auc_score(
            label_binarize(y_true, classes=CLASSES),
            y_prob, multi_class="ovr", average="macro"
        ),
    }


def guardar_confusion_matrix(y_true, y_pred, nombre: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title(f"Matriz de Confusión — {nombre}")
    path = ARTIFACT_DIR / f"confusion_{nombre.lower().replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def guardar_roc_curves(y_true, y_prob, nombre: str) -> Path:
    y_bin = label_binarize(y_true, classes=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            y_bin[:, i], y_prob[:, i], name=cls, ax=ax
        )
    ax.set_title(f"Curvas ROC (OVR) — {nombre}")
    path = ARTIFACT_DIR / f"roc_{nombre.lower().replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Entrenamiento con MLflow ───────────────────────────────────────────────────
def entrenar_modelo(nombre, estimator, param_grid, X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Actividad5_Sentimientos")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    with mlflow.start_run(run_name=nombre):

        # GridSearchCV
        grid = GridSearchCV(
            estimator, param_grid, cv=cv,
            scoring="f1_macro", n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        # CV score con mejor estimador
        cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="f1_macro")

        # Evaluación en test
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)
        metricas = calcular_metricas(y_test, y_pred, y_prob)

        # ── Registrar en MLflow ────────────────────────────────────────────────
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("vectorizer_ngram_range", "(1,2)")
        mlflow.log_param("vectorizer_max_features", 5000)

        mlflow.log_metrics(metricas)
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std",  cv_scores.std())

        # Artefactos: gráficas
        cm_path  = guardar_confusion_matrix(y_test, y_pred, nombre)
        roc_path = guardar_roc_curves(y_test, y_prob, nombre)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        # Artefacto: modelo
        mlflow.sklearn.log_model(best, artifact_path="modelo")

        print(f"\n{'='*50}")
        print(f"  {nombre}")
        print(f"  Mejores params: {grid.best_params_}")
        print(f"  CV F1-macro:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        for k, v in metricas.items():
            print(f"  {k:20s}: {v:.4f}")
        print(f"{'='*50}\n")

    return metricas, grid.best_params_


# ── Configuración de modelos ───────────────────────────────────────────────────
MODELOS = [
    (
        "Regresion Logistica",
        LogisticRegression(max_iter=1000, random_state=SEED),
        {
            "C":       [0.01, 0.1, 1, 10],
            "solver":  ["lbfgs", "saga"],
        },
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": [100, 200],
            "max_depth":    [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    ),
]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preparar()

    resultados = {}
    for nombre, estimator, param_grid in MODELOS:
        metricas, best_params = entrenar_modelo(
            nombre, estimator, param_grid, X_train, X_test, y_train, y_test
        )
        resultados[nombre] = {"metricas": metricas, "params": best_params}

    # Comparativa final
    print("\n📊 COMPARATIVA FINAL")
    print(f"{'Modelo':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 75)
    for nombre, res in resultados.items():
        m = res["metricas"]
        print(
            f"{nombre:<25} {m['accuracy']:>9.4f} {m['precision_macro']:>10.4f} "
            f"{m['recall_macro']:>8.4f} {m['f1_macro']:>8.4f} {m['roc_auc_ovr']:>9.4f}"
        )
