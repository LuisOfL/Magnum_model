import os
import glob
import random
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN DEL PROYECTO Y CLASES
# ==========================================
CLASSES = {
    0: "Magnum clasica",
    1: "Oreo Mordizco"
}

EPOCHS = 100         # Se recomienda subir a 100 épocas para que el modelo asimile bien las variaciones
BATCH_SIZE = 16      # Reduce a 8 o 4 si te quedas sin VRAM
IMG_SIZE = 640       # Resolución de imagen
VAL_RATIO = 0.2      # 20% validación / 80% entrenamiento
MODEL_BASE = 'yolov8n.pt'

DIRECTORIO_RAIZ = Path(__file__).parent.resolve()
CARPETA_IMAGENES = DIRECTORIO_RAIZ / "images"

# ==========================================
# 2. OBTENER Y MEZCLAR LAS IMÁGENES
# ==========================================
extensiones = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
imagenes = []
for ext in extensiones:
    imagenes.extend(glob.glob(str(CARPETA_IMAGENES / ext)))

if len(imagenes) == 0:
    raise FileNotFoundError(f"No se encontraron imágenes dentro de '{CARPETA_IMAGENES}'.")

print(f"Total de imágenes encontradas: {len(imagenes)}")

random.seed(42)
random.shuffle(imagenes)

num_val = int(len(imagenes) * VAL_RATIO)
val_imgs = imagenes[:num_val]
train_imgs = imagenes[num_val:]

print(f"Imágenes para entrenamiento (Train): {len(train_imgs)}")
print(f"Imágenes para validación (Val): {len(val_imgs)}")

# ==========================================
# 3. GUARDAR LISTAS TRAIN.TXT Y VAL.TXT
# ==========================================
train_txt_path = DIRECTORIO_RAIZ / "train.txt"
val_txt_path = DIRECTORIO_RAIZ / "val.txt"

with open(train_txt_path, "w") as f:
    f.write("\n".join(train_imgs))

with open(val_txt_path, "w") as f:
    f.write("\n".join(val_imgs))

# ==========================================
# 4. CREAR ARCHIVO DATASET.YAML AUTOMÁTICO
# ==========================================
yaml_content = f"""path: {DIRECTORIO_RAIZ}
train: {train_txt_path}
val: {val_txt_path}

names:
"""
for idx, name in CLASSES.items():
    yaml_content += f"  {idx}: {name}\n"

yaml_path = DIRECTORIO_RAIZ / "dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Archivo de configuración generado en: {yaml_path}")

# ==========================================
# 5. INICIAR ENTRENAMIENTO CON AUGMENTATION
# ==========================================
if __name__ == '__main__':
    print("\nIniciando el entrenamiento con Data Augmentation...")
    
    model = YOLO(MODEL_BASE)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="entrenamiento_yolo",
        name="modelo_helados",
        exist_ok=True,

        # ----------------------------------------------------
        # AJUSTES DE LUZ, BRILLO Y COLOR (HSV)
        # ----------------------------------------------------
        hsv_v=0.6,        # Variación de Brillo/Luz (Value) (+/- 60%)
        hsv_s=0.7,        # Variación de Saturación de color (+/- 70%)
        hsv_h=0.02,       # Variación sutil de Tono (Hue)

        # ----------------------------------------------------
        # ROTACIONES Y TRANSFORMACIONES GEOMÉTRICAS
        # ----------------------------------------------------
        degrees=30.0,     # Rotación aleatoria entre -30° y +30°
        translate=0.1,    # Desplazamiento horizontal/vertical (+/- 10%)
        scale=0.5,        # Zoom/Escala aleatoria (+/- 50%)
        shear=10.0,       # Deformación por corte/inclinación (+/- 10°)
        perspective=0.001,# Cambio de perspectiva tridimensional
        fliplr=0.5,       # Volteo horizontal (50% de probabilidad)

        # ----------------------------------------------------
        # COMBINACIÓN DE IMÁGENES COMPUESTAS
        # ----------------------------------------------------
        mosaic=1.0,       # Combina 4 imágenes en 1 (100% probabilidad)
        mixup=0.15        # Superpone dos imágenes (15% probabilidad)
    )

    print("\n==========================================")
    print("¡Entrenamiento completado con éxito!")
    print(f"Modelo final guardado en: entrenamiento_yolo/modelo_helados/weights/best.pt")
    print("==========================================")