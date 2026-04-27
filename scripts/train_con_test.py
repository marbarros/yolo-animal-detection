"""
Script: train_con_test.py

Descripción:
Entrena modelos YOLO utilizando una partición aproximada:
70% entrenamiento / 20% validación / 10% test (externo).

Se evalúan múltiples modelos y se calculan métricas de desempeño.
"""

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ==========================================================
# SEMILLA (REPRODUCIBILIDAD)
# ==========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==========================================================
# RUTAS DEL DATASET
# ==========================================================

BASE_DIR = "C:/ruta/a/tu/carpeta/balanceado"

train_images = os.path.join(BASE_DIR, "images", "train")
val_images = os.path.join(BASE_DIR, "images", "val")

yaml_path = os.path.join(BASE_DIR, "dataset.yaml")

# ==========================================================
# CREACIÓN DEL ARCHIVO YAML PARA YOLO
# ==========================================================

yaml_content = f"""
path: {BASE_DIR}
train: images/train
val: images/val

nc: 1
names:
  0: animal
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("✅ YAML creado correctamente")

# ==========================================================
# FUNCIÓN DE ENTRENAMIENTO
# ==========================================================

def ejecutar_experimento(modelo_pt, alias):

    print(f"\n🚀 Entrenando modelo: {modelo_pt}")

    # Cargar modelo preentrenado
    model = YOLO(modelo_pt)

    # Entrenamiento
    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        batch=8,
        seed=SEED,
        project=f"runs_{alias}",
        name="train",
        exist_ok=True
    )

    # ======================================================
    # LECTURA DE MÉTRICAS
    # ======================================================

    csv_path = os.path.join(f"runs_{alias}", "train", "results.csv")

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        ultima = df.iloc[-1]

        # Métricas principales
        precision = ultima["metrics/precision(B)"]
        recall = ultima["metrics/recall(B)"]
        map50 = ultima["metrics/mAP50(B)"]
        map5095 = ultima["metrics/mAP50-95(B)"]

        # Cálculo F1-score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        print(f"\n📊 Resultados - {alias}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"mAP@50:    {map50:.4f}")
        print(f"mAP@50-95: {map5095:.4f}")

        # ==================================================
        # CURVA DE APRENDIZAJE
        # ==================================================

        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title(f"Evolución mAP - {alias}")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print("❌ No se encontró archivo de métricas")

# ==========================================================
# MODELOS A EVALUAR
# ==========================================================

experimentos = [
    ("yolov8s.pt", "v8s"),
    ("yolov8n.pt", "v8n"),
    ("yolo11s.pt", "v11s"),
    ("yolo11n.pt", "v11n")
]

# Ejecutar todos los experimentos
for modelo, alias in experimentos:
    ejecutar_experimento(modelo, alias)

print("\n✨ Experimento completo finalizado")
