"""
datos_prep.py
Limpieza, codificación y vectorización del dataset de reseñas.
Genera datos/datos_limp/resenas_limpias.csv y el vectorizador serializado.
"""

import re
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_CSV    = ROOT / "datos" / "datos_ini"  / "resenas_restaurante_zapopan.csv"
CLEAN_CSV  = ROOT / "datos" / "datos_limp" / "resenas_limpias.csv"
TFIDF_PKL  = ROOT / "datos" / "datos_limp" / "tfidf_vectorizer.pkl"

# ── Mapa de etiquetas ──────────────────────────────────────────────────────────
LABEL_MAP = {"POS": 2, "NEU": 1, "NEG": 0}


def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def cargar_y_limpiar(ruta_csv: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv)

    # Verificar nulos
    nulos = df.isnull().sum()
    if nulos.any():
        print(f"[datos_prep] Valores nulos encontrados:\n{nulos[nulos > 0]}")
        df = df.dropna(subset=["comentario", "sentimiento"])

    # Limpiar texto
    df["comentario_limpio"] = df["comentario"].astype(str).apply(limpiar_texto)

    # Codificar etiqueta
    df["label"] = df["sentimiento"].map(LABEL_MAP)

    print(f"[datos_prep] Dataset cargado: {len(df)} registros")
    print(f"[datos_prep] Distribución:\n{df['sentimiento'].value_counts()}")

    return df


def vectorizar(df: pd.DataFrame, max_features: int = 5000, ngram_range: tuple = (1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["comentario_limpio"])
    y = df["label"].values
    return X, y, vectorizer


def preparar(ruta_csv: Path = RAW_CSV, test_size: float = 0.2, seed: int = 42):
    """Pipeline completo: carga → limpia → vectoriza → split → guarda."""
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = cargar_y_limpiar(ruta_csv)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[datos_prep] Datos limpios guardados en: {CLEAN_CSV}")

    X, y, vectorizer = vectorizar(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    with open(TFIDF_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[datos_prep] Vectorizador guardado en: {TFIDF_PKL}")

    print(f"[datos_prep] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    preparar()
