"""
Script: experimento_colab.py

Descripción:
Este script documenta pruebas iniciales realizadas en Google Colab para evaluar:

- Tiempo de entrenamiento
- Consumo de recursos
- Comportamiento de distintos modelos YOLO

Se utilizó principalmente para variar el número de épocas y observar su impacto.
"""

# ==========================================================
# ENTRENAMIENTO EN GOOGLE COLAB
# ==========================================================

# Este comando entrena un modelo YOLO directamente desde consola
# Se modificaba principalmente el número de épocas (epochs)

# Ejemplo: YOLOv8s con 50 épocas

# %time permite medir el tiempo total de ejecución
# (solo funciona en entorno tipo Jupyter / Colab)

%time !yolo task=detect mode=train \
    model=yolov8s.pt \
    data=/content/ppc2025.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8

# ==========================================================
# COMPRESIÓN DE RESULTADOS
# ==========================================================

# Se comprimía la carpeta de resultados para descargarla

import shutil

# Carpeta de resultados del entrenamiento
folder_path = "/content/runs/detect/train2"

# Nombre del archivo ZIP
zip_path = "/content/8s50con10.zip"

# Crear archivo comprimido
shutil.make_archive("/content/8s50con10", 'zip', folder_path)

# ==========================================================
# DESCARGA DEL ARCHIVO
# ==========================================================

from google.colab import files

# Descargar resultados al equipo local
files.download(zip_path)

# ==========================================================
# PREDICCIÓN (INFERENCIA)
# ==========================================================

# Se utiliza el modelo entrenado para generar predicciones
# sobre el conjunto de validación

!yolo task=detect mode=predict \
    model="/content/runs/detect/train3/weights/best.pt" \
    conf=0.30 \
    source="/content/PPC2025/images/val" \
    save=True \
    save_txt=True

