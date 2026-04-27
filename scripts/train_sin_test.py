"""
Script: train_sin_test.py

Descripción:
Entrena modelos YOLO utilizando una partición:
80% entrenamiento / 20% validación.

No se utiliza conjunto de test.
"""

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

import os
import numpy as np
import random
from ultralytics import YOLO

# ==========================================================
# SEMILLA
# ==========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = "C:/ruta/a/tu/carpeta/balanceado"
yaml_path = os.path.join(BASE_DIR, "dataset.yaml")

# ==========================================================
# YAML
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

print("✅ YAML creado")

# ==========================================================
# FUNCIÓN DE ENTRENAMIENTO
# ==========================================================

def ejecutar_experimento(modelo_pt, alias):

    print(f"\n🚀 Entrenando {modelo_pt}")

    model = YOLO(modelo_pt)

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

    print(f"✅ {alias} finalizado")

# ==========================================================
# MODELOS
# ==========================================================

experimentos = [
    ("yolov8s.pt", "v8s"),
    ("yolov8n.pt", "v8n"),
    ("yolo11s.pt", "v11s"),
    ("yolo11n.pt", "v11n")
]

for modelo, alias in experimentos:
    ejecutar_experimento(modelo, alias)

print("\n✨ Entrenamiento finalizado")
