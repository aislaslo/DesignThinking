"""
generar_evidencias.py
Genera evidencias visuales del Punto 5:
  - Panel de experimentos MLflow (estilo UI)
  - Detalle de cada run con parametros y metricas
  - Tabla comparativa de desempeno
  - Resumen consolidado
"""

import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
MLRUNS   = ROOT / "mlruns"
ASSETS   = ROOT / "datos" / "datos_limp"
EVIDENCIAS = ROOT / "docs" / "evidencias"
EVIDENCIAS.mkdir(parents=True, exist_ok=True)

AZUL  = "#2980b9"
VERDE = "#27ae60"
NARANJA = "#e67e22"
GRIS  = "#95a5a6"
FONDO = "#f5f7fa"
NEGRO = "#1e1e1e"

mlflow.set_tracking_uri(f"file://{MLRUNS}")
client = mlflow.tracking.MlflowClient()


def obtener_runs():
    experimento = client.get_experiment_by_name("Actividad5_Sentimientos")
    if not experimento:
        print("ERROR: No se encontro el experimento 'Actividad5_Sentimientos'")
        return []
    runs = client.search_runs(
        experiment_ids=[experimento.experiment_id],
        order_by=["metrics.f1_macro DESC"]
    )
    return runs


# ── Evidencia 1: Panel de experimentos estilo MLflow UI ───────────────────────
def panel_experimentos(runs):
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.axis("off")

    # Titulo estilo MLflow
    fig.text(0.02, 0.92, "MLflow", fontsize=18, fontweight="bold", color=AZUL)
    fig.text(0.095, 0.92, "Experiments", fontsize=18, color=NEGRO)
    fig.text(0.02, 0.82, "Experiment: Actividad5_Sentimientos", fontsize=11,
             color=GRIS, style="italic")

    cols = ["Run Name", "Status", "f1_macro", "accuracy", "roc_auc_ovr",
            "precision_macro", "recall_macro", "cv_f1_macro_mean"]
    anchos = [0.16, 0.07, 0.09, 0.09, 0.10, 0.12, 0.10, 0.14]
    x_pos = [0.02]
    for a in anchos[:-1]:
        x_pos.append(x_pos[-1] + a)

    y_header = 0.70
    for i, (col, x) in enumerate(zip(cols, x_pos)):
        ax.text(x, y_header, col, transform=fig.transFigure,
                fontsize=8.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=AZUL, edgecolor="none"))

    colores_fila = ["#ffffff", "#eaf4fb"]
    for r_idx, run in enumerate(runs):
        y = 0.54 - r_idx * 0.18
        fcolor = colores_fila[r_idx % 2]
        rect = FancyBboxPatch((0.02, y - 0.06), 0.96, 0.14,
                              transform=fig.transFigure, clip_on=False,
                              boxstyle="round,pad=0.01",
                              facecolor=fcolor, edgecolor="#d5d8dc", linewidth=0.5)
        fig.add_artist(rect)

        nombre = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        metricas = run.data.metrics
        valores = [
            nombre,
            "FINISHED",
            f"{metricas.get('f1_macro', 0):.4f}",
            f"{metricas.get('accuracy', 0):.4f}",
            f"{metricas.get('roc_auc_ovr', 0):.4f}",
            f"{metricas.get('precision_macro', 0):.4f}",
            f"{metricas.get('recall_macro', 0):.4f}",
            f"{metricas.get('cv_f1_macro_mean', 0):.4f}",
        ]
        for i, (val, x) in enumerate(zip(valores, x_pos)):
            color = NEGRO
            fw = "normal"
            if i == 0:
                color = AZUL; fw = "bold"
            elif i == 1:
                color = VERDE; fw = "bold"
            ax.text(x + 0.005, y + 0.01, val, transform=fig.transFigure,
                    fontsize=9, color=color, fontweight=fw)

    plt.tight_layout()
    out = EVIDENCIAS / "mlflow_panel_experimentos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  Generada: {out.name}")
    return out


# ── Evidencia 2: Detalle de cada run ─────────────────────────────────────────
def detalle_run(run, idx):
    nombre = run.data.tags.get("mlflow.runName", f"Run {idx}")
    params  = run.data.params
    metricas = run.data.metrics

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(FONDO)

    # Header
    fig.text(0.03, 0.93, f"MLflow  /  Actividad5_Sentimientos  /  {nombre}",
             fontsize=12, color=NEGRO)
    fig.text(0.03, 0.87, f"Run ID: {run.info.run_id[:16]}...",
             fontsize=8, color=GRIS)
    fig.text(0.03, 0.83, "Status: FINISHED", fontsize=9,
             color=VERDE, fontweight="bold")

    ax_sep = fig.add_axes([0.03, 0.81, 0.94, 0.005])
    ax_sep.set_facecolor(AZUL)
    ax_sep.axis("off")

    # Panel parametros
    ax_p = fig.add_axes([0.03, 0.44, 0.42, 0.34])
    ax_p.set_facecolor("white")
    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.axis("off")
    ax_p.text(0.05, 0.93, "Parameters", fontsize=11,
              fontweight="bold", color=AZUL, transform=ax_p.transAxes)
    ax_p.plot([0, 1], [0.88, 0.88], color=AZUL, linewidth=1.5, transform=ax_p.transAxes, clip_on=False)

    items_p = [(k, v) for k, v in sorted(params.items())]
    for i, (k, v) in enumerate(items_p):
        y = 0.80 - i * 0.12
        c = "#f0f3f4" if i % 2 == 0 else "white"
        ax_p.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.12,
                       transform=ax_p.transAxes, facecolor=c, zorder=0))
        ax_p.text(0.04, y + 0.02, k, transform=ax_p.transAxes,
                  fontsize=8.5, color=GRIS, fontweight="bold")
        ax_p.text(0.55, y + 0.02, str(v), transform=ax_p.transAxes,
                  fontsize=8.5, color=NEGRO)

    # Panel metricas
    ax_m = fig.add_axes([0.52, 0.44, 0.45, 0.34])
    ax_m.set_facecolor("white")
    ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1); ax_m.axis("off")
    ax_m.text(0.05, 0.93, "Metrics", fontsize=11,
              fontweight="bold", color=AZUL, transform=ax_m.transAxes)
    ax_m.plot([0, 1], [0.88, 0.88], color=AZUL, linewidth=1.5, transform=ax_m.transAxes, clip_on=False)

    orden_metricas = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "roc_auc_ovr", "cv_f1_macro_mean", "cv_f1_macro_std"
    ]
    for i, key in enumerate(orden_metricas):
        val = metricas.get(key, None)
        if val is None:
            continue
        y = 0.80 - i * 0.12
        c = "#f0f3f4" if i % 2 == 0 else "white"
        ax_m.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.12,
                       transform=ax_m.transAxes, facecolor=c, zorder=0))
        ax_m.text(0.04, y + 0.02, key, transform=ax_m.transAxes,
                  fontsize=8.5, color=GRIS, fontweight="bold")
        ax_m.text(0.70, y + 0.02, f"{val:.4f}", transform=ax_m.transAxes,
                  fontsize=8.5, color=NEGRO, fontweight="bold")

    # Barra visual de F1
    ax_bar = fig.add_axes([0.03, 0.12, 0.94, 0.25])
    ax_bar.set_facecolor("white")
    claves = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]
    etiquetas = ["Accuracy", "Precision", "Recall", "F1-macro", "ROC-AUC"]
    vals = [metricas.get(k, 0) for k in claves]
    bars = ax_bar.barh(etiquetas, vals, color=AZUL, alpha=0.85, height=0.5)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_title("Metricas en conjunto de prueba", fontsize=10,
                     fontweight="bold", color=NEGRO, pad=6)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.tick_params(left=False)
    for bar, val in zip(bars, vals):
        ax_bar.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    out = EVIDENCIAS / f"mlflow_run_{idx:02d}_{nombre.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  Generada: {out.name}")
    return out


# ── Evidencia 3: Tabla comparativa de desempeno ───────────────────────────────
def tabla_comparativa(runs):
    datos = []
    for run in runs:
        nombre = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        m = run.data.metrics
        datos.append({
            "Modelo":          nombre,
            "Accuracy":        round(m.get("accuracy", 0), 4),
            "Precision":       round(m.get("precision_macro", 0), 4),
            "Recall":          round(m.get("recall_macro", 0), 4),
            "F1-macro":        round(m.get("f1_macro", 0), 4),
            "ROC-AUC":         round(m.get("roc_auc_ovr", 0), 4),
            "CV F1 mean":      round(m.get("cv_f1_macro_mean", 0), 4),
            "CV F1 std":       round(m.get("cv_f1_macro_std", 0), 4),
        })
    df = pd.DataFrame(datos)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.axis("off")

    fig.text(0.03, 0.93, "Comparativa de Modelos - Actividad5_Sentimientos",
             fontsize=13, fontweight="bold", color=NEGRO)
    fig.text(0.03, 0.86, "Metricas en conjunto de prueba (test size=20%, stratified, seed=42)",
             fontsize=9, color=GRIS)

    cols = list(df.columns)
    cell_data = df.values.tolist()
    col_widths = [0.18] + [0.115] * (len(cols) - 1)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d5d8dc")
        if r == 0:
            cell.set_facecolor(AZUL)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eaf4fb")
        else:
            cell.set_facecolor("white")
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold", color=AZUL)

    out = EVIDENCIAS / "tabla_comparativa_modelos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  Generada: {out.name}")
    return out


# ── Evidencia 4: Grafica comparativa de barras ───────────────────────────────
def grafica_comparativa(runs):
    metricas_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]
    etiquetas = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-macro", "ROC-AUC\n(OVR)"]
    colores = [AZUL, NARANJA]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)

    x = np.arange(len(metricas_keys))
    width = 0.35

    for i, run in enumerate(runs):
        nombre = run.data.tags.get("mlflow.runName", f"Modelo {i+1}")
        vals = [run.data.metrics.get(k, 0) for k in metricas_keys]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=nombre, color=colores[i], alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_ylim(0.85, 1.05)
    ax.set_ylabel("Valor", fontsize=10)
    ax.set_title("Comparativa de Desempeno - Regresion Logistica vs Random Forest",
                 fontsize=12, fontweight="bold", pad=12, color=NEGRO)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    out = EVIDENCIAS / "grafica_comparativa_barras.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  Generada: {out.name}")
    return out


# ── Evidencia 5: CV scores boxplot ───────────────────────────────────────────
def grafica_cv(runs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)

    nombres, medias, stds = [], [], []
    for run in runs:
        nombres.append(run.data.tags.get("mlflow.runName", "?"))
        medias.append(run.data.metrics.get("cv_f1_macro_mean", 0))
        stds.append(run.data.metrics.get("cv_f1_macro_std", 0))

    x = np.arange(len(nombres))
    bars = ax.bar(x, medias, yerr=stds, capsize=8, color=[AZUL, NARANJA],
                  alpha=0.88, error_kw=dict(elinewidth=2, ecolor=NEGRO), zorder=3)
    for bar, m, s in zip(bars, medias, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005,
                f"{m:.4f} ± {s:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0.85, 1.06)
    ax.set_ylabel("F1-macro (media ± std)", fontsize=10)
    ax.set_title("Validacion Cruzada Estratificada 5-Fold\n(scoring = f1_macro)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    out = EVIDENCIAS / "grafica_cv_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  Generada: {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando evidencias visuales del Punto 5...")
    runs = obtener_runs()
    if not runs:
        print("No hay runs. Ejecuta train.py primero.")
        exit(1)

    print(f"\nRuns encontrados: {len(runs)}")
    for r in runs:
        print(f"  - {r.data.tags.get('mlflow.runName')} | f1_macro={r.data.metrics.get('f1_macro', 0):.4f}")

    print("\nGenerando imagenes...")
    panel_experimentos(runs)
    for i, run in enumerate(runs):
        detalle_run(run, i + 1)
    tabla_comparativa(runs)
    grafica_comparativa(runs)
    grafica_cv(runs)

    print(f"\nEvidencias guardadas en: {EVIDENCIAS}")
    print(f"Archivos generados:")
    for f in sorted(EVIDENCIAS.glob("*.png")):
        print(f"  {f.name}")
