"""
Script: generar_labels_vacios.py

Descripción:
Este script genera archivos .txt vacíos para cada imagen ubicada en la carpeta "fondo".
En YOLO, cada imagen debe tener un archivo de etiquetas asociado, incluso si no contiene objetos.

Esto permite incluir imágenes negativas (sin animales) en el entrenamiento.
"""

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

import os                      # Manejo de rutas y archivos
from pathlib import Path       # Manejo de nombres de archivos

# ==========================================================
# DEFINICIÓN DE RUTAS
# ==========================================================

# ⚠️ IMPORTANTE: modificar según ubicación en tu computadora
BASE_PATH = "C:/ruta/a/tu/carpeta/balanceado"

# Carpeta con imágenes sin animales
FONDO_PATH = os.path.join(BASE_PATH, "fondo")

# Carpeta donde se guardarán los .txt vacíos
LABELS_F_PATH = os.path.join(BASE_PATH, "labelsF")

# Crear carpeta si no existe
os.makedirs(LABELS_F_PATH, exist_ok=True)

# ==========================================================
# PROCESO PRINCIPAL
# ==========================================================

# Filtrar solo imágenes
imagenes = [f for f in os.listdir(FONDO_PATH)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"📂 Imágenes encontradas: {len(imagenes)}")

creados = 0

# Recorrer cada imagen
for img in imagenes:

    # Obtener nombre sin extensión
    nombre_base = Path(img).stem

    # Crear ruta del archivo .txt
    txt_path = os.path.join(LABELS_F_PATH, nombre_base + ".txt")

    # Crear archivo vacío
    open(txt_path, "w").close()

    creados += 1

print(f"✅ Archivos .txt creados: {creados}")

# ==========================================================
# VERIFICACIÓN
# ==========================================================

labels_creados = [f for f in os.listdir(LABELS_F_PATH)
                  if f.endswith(".txt")]

print(f"📊 Total labels: {len(labels_creados)}")

# Chequeo de consistencia
if len(labels_creados) == len(imagenes):
    print("🎯 Todo correcto: cada imagen tiene su label")
else:
    print("⚠️ Inconsistencia detectada")
