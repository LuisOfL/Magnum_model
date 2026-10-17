"""
Módulo de Entrenamiento Automatizado - YOLOv8 (Detector de Helados / MVP)
Autor: Luis Ángel Meléndez Chavero
Descripción: Configura el dataset, limpia caché anterior, define las 24 clases
exactas y ejecuta el pipeline de entrenamiento de Ultralytics YOLOv8.
"""

import os
import glob
import random
from pathlib import Path
from ultralytics import YOLO

# =====================================================================
# 1. CONSTANTES Y CONFIGURACIÓN GLOBAL (24 Clases exactas de Label Studio / notes.json)
# =====================================================================
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
VAL_RATIO = 0.2
MODEL_BASE = 'yolov8n.pt'

CLASSES = {
    0: "HOLANDA CHEMISSE PINA COCO 30X65ML",
    1: "HOLANDA CHOCO CREM CHOCOLATE 30X78ML",
    2: "HOLANDA CHOCO CREM FRESA 30X78ML",
    3: "HOLANDA CONEJO PALETA 25X70ML",
    4: "HOLANDA FLAMIN HOT MPK 8X4X65ML",
    5: "HOLANDA HEL MORDISKO CAFE 24X105ML",
    6: "HOLANDA HEL MORDISKO CLAS VAIN 24X105ML",
    7: "HOLANDA HELADO MORDISKO CAJETA 24X105ML",
    8: "HOLANDA MAGNUM ALMENDRAS 25X90ML",
    9: "HOLANDA MAGNUM AVELLANAS CARAM 25X90ML",
    10: "HOLANDA MAGNUM BLANCO ALMENDRAS 25X90ML",
    11: "HOLANDA MAGNUM CH COOKIE REMIX 25X85ML",
    12: "HOLANDA MAGNUM CLASICA 25X90ML",
    13: "HOLANDA MAGNUM DOUBLE RASPBERRY 25X93ML",
    14: "HOLANDA MAGNUM EUPHORIA CAFE VAI 25X90ML",
    15: "HOLANDA MORDISKO HEL CHOCOAVELL 24X105ML",
    16: "HOLANDA MORDISKO HEL OREO MPK 16X8X105ML",
    17: "HOLANDA PLTA TWISTER CH 24X70ML",
    18: "HOLANDA SOLERO FRESA 30X65ML",
    19: "HOLANDA SOLERO LIMON CITRUS MPK 8X4X65ML",
    20: "HOLANDA SOLERO MOUSSE LIMON 25X75ML",
    21: "HOLANDA SOLERO P FRUTOS ROJOS 25X94ML",
    22: "Oreo Mordizco",
    23: "Magnum clasica"
}

DIRECTORIO_RAIZ = Path(__file__).parent.resolve()
CARPETA_IMAGENES = DIRECTORIO_RAIZ / "images"

# =====================================================================
# 2. LIMPIEZA DE CACHÉ Y CARGA DE DATASET
# =====================================================================
def limpiar_cache_obsoleta():
    """Elimina archivos .cache previos para evitar conflictos de clases."""
    for cache_file in DIRECTORIO_RAIZ.glob("*.cache"):
        try:
            cache_file.unlink()
            print(f"[Cache Cleaned] Eliminado archivo obsoleto: {cache_file.name}")
        except Exception as e:
            print(f"[Warning] No se pudo eliminar {cache_file.name}: {e}")

def preparar_dataset():
    """Recopila imágenes, realiza división train/val y genera el dataset.yaml."""
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    imagenes = []
    for ext in extensiones:
        imagenes.extend(glob.glob(str(CARPETA_IMAGENES / ext)))

    if not imagenes:
        raise FileNotFoundError(f"❌ Error crítico: No se encontraron imágenes en '{CARPETA_IMAGENES}'.")

    print(f"[Dataset] Total de imágenes indexadas: {len(imagenes)}")

    random.seed(42)
    random.shuffle(imagenes)

    num_val = int(len(imagenes) * VAL_RATIO)
    val_imgs = imagenes[:num_val]
    train_imgs = imagenes[num_val:]

    print(f"[Dataset] Train split: {len(train_imgs)} imágenes")
    print(f"[Dataset] Val split: {len(val_imgs)} imágenes")

    train_txt_path = DIRECTORIO_RAIZ / "train.txt"
    val_txt_path = DIRECTORIO_RAIZ / "val.txt"

    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_imgs))

    with open(val_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_imgs))

    yaml_content = f"""path: {DIRECTORIO_RAIZ}
train: {train_txt_path}
val: {val_txt_path}

nc: {len(CLASSES)}
names:
"""
    for idx, name in CLASSES.items():
        yaml_content += f"  {idx}: '{name}'\n"

    yaml_path = DIRECTORIO_RAIZ / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[Dataset] Configuración generada exitosamente en: {yaml_path}")
    return yaml_path

# =====================================================================
# 3. PIPELINE DE ENTRENAMIENTO PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    print("🚀 Iniciando pipeline de entrenamiento YOLOv8 - MVP Helados...")
    
    limpiar_cache_obsoleta()
    yaml_path = preparar_dataset()

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="entrenamiento_yolo",
        name="modelo_helados",
        exist_ok=True,

        # Augmentation
        hsv_v=0.6,
        hsv_s=0.7,
        hsv_h=0.02,
        degrees=30.0,
        translate=0.1,
        scale=0.5,
        shear=10.0,
        perspective=0.001,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15
    )

    print("\n======================================================================")
    print("✨ ¡Entrenamiento completado con éxito!")
    print(f"📦 Pesos optimizados guardados en: entrenamiento_yolo/modelo_helados/weights/best.pt")
    print("======================================================================")