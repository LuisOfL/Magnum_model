import base64
import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from transformers import pipeline
from ultralytics import YOLO

app = FastAPI(title="Smart Freezer Audit API - Estimación por Profundidad")

BASE_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------
# Carga de Modelos YOLO y Depth Anything
# ---------------------------------------------------------
MODEL_PATH = BASE_DIR / "best.pt"
SEG_MODEL_PATH = BASE_DIR / "segmentation.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de detección: {MODEL_PATH}")

if not SEG_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de segmentación: {SEG_MODEL_PATH}")

USE_CUDA = torch.cuda.is_available()
PIPELINE_DEVICE = 0 if USE_CUDA else -1
YOLO_DEVICE = 0 if USE_CUDA else "cpu"

model = YOLO(str(MODEL_PATH))
seg_model = YOLO(str(SEG_MODEL_PATH))

# Estimador de profundidad monocular
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=PIPELINE_DEVICE,
)

# Nombres de clases
CLASSES_PATH = BASE_DIR / "classes.txt"
CLASS_NAMES: Dict[int, str] = {}
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = {i: line.strip() for i, line in enumerate(f) if line.strip()}

# ---------------------------------------------------------
# Configuración NLP y Reglas de Capacidades por SKU
# ---------------------------------------------------------
# Diccionario fácil de editar para definir la capacidad total según palabras clave presentes en la etiqueta
REGLAS_CAPACIDAD_TEXTO: Dict[str, int] = {
    "mordisko": 11,
    "mordizco": 11,
    "paleta": 12,
    "magnum": 12,
    "solero": 12,
    "cornetto": 10,
    "pint": 8,
    "DEFAULT": 12,  # De lo contrario caben 12 por defecto
}

# Paso de profundidad estimado por pieza (distancia relativa que ocupa 1 producto)
PASO_PROFUNDIDAD_POR_SKU: Dict[str, float] = {
    "mordisko": 0.045,
    "mordizco": 0.045,
    "paleta": 0.050,
    "magnum": 0.050,
    "solero": 0.050,
    "cornetto": 0.050,
    "pint": 0.060,
    "DEFAULT": 0.050,
}

REJILLA_KEYWORDS = (
    "rejilla", "canastilla", "separador", "carril", "parrilla", "estante"
)


def normalizar_texto(texto: str) -> str:
    """Limpia y normaliza el nombre del SKU para coincidencia de palabras clave."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()


def resolver_capacidad_sku(nombre_sku: str) -> int:
    """
    Usa NLP/coincidencia de texto para determinar la capacidad máxima.
    - Si el texto contiene 'mordisko' o 'mordizco' -> 11 piezas
    - Si contiene 'paleta' -> 12 piezas
    - De lo contrario -> 12 piezas (o el valor en DEFAULT)
    """
    texto_limpio = normalizar_texto(nombre_sku)
    for kw, capacidad in REGLAS_CAPACIDAD_TEXTO.items():
        if kw != "DEFAULT" and kw in texto_limpio:
            return capacidad
    return REGLAS_CAPACIDAD_TEXTO["DEFAULT"]


def resolver_paso_profundidad(nombre_sku: str) -> float:
    """Obtiene el paso de profundidad correspondiente al tipo de SKU."""
    texto_limpio = normalizar_texto(nombre_sku)
    for kw, paso in PASO_PROFUNDIDAD_POR_SKU.items():
        if kw != "DEFAULT" and kw in texto_limpio:
            return paso
    return PASO_PROFUNDIDAD_POR_SKU["DEFAULT"]


def calcular_mapa_profundidad(pil_img: Image.Image) -> np.ndarray:
    """Genera el mapa de profundidad con Depth Anything V2 y lo escala a la imagen original."""
    salida = depth_estimator(pil_img)
    predicted = salida.get("predicted_depth")

    if predicted is None:
        depth = np.asarray(salida["depth"], dtype=np.float32)
        depth_img = Image.fromarray(depth).resize(pil_img.size, Image.Resampling.BICUBIC)
        return np.asarray(depth_img, dtype=np.float32)

    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0).unsqueeze(0)
    elif predicted.ndim == 3:
        predicted = predicted.unsqueeze(1)

    predicted = F.interpolate(
        predicted.float(),
        size=(pil_img.height, pil_img.width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return predicted.detach().cpu().numpy().astype(np.float32)


def normalizar_mapa_profundidad(depth: np.ndarray) -> np.ndarray:
    """Normaliza el mapa de profundidad entre 0 y 1 ignorando valores atípicos."""
    validos = depth[np.isfinite(depth)]
    if validos.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    minimo, maximo = np.percentile(validos, [2, 98])
    rango = float(maximo - minimo)
    if rango < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - minimo) / rango, 0.0, 1.0).astype(np.float32)


def extraer_profundidad_bbox(depth: np.ndarray, item: Dict, factor_recorte: float = 0.50) -> Optional[float]:
    """Obtiene la mediana de profundidad en la región central de la bounding box."""
    if not item:
        return None
    
    alto, ancho = depth.shape
    cx = (item["x1"] + item["x2"]) / 2.0
    cy = (item["y1"] + item["y2"]) / 2.0
    bw = (item["x2"] - item["x1"]) * factor_recorte
    bh = (item["y2"] - item["y1"]) * factor_recorte

    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(ancho, int(cx + bw / 2))
    y2 = min(alto, int(cy + bh / 2))

    region = depth[y1:y2, x1:x2]
    validos = region[np.isfinite(region)]
    return float(np.median(validos)) if validos.size > 0 else None


def buscar_rejilla_mas_cercana(producto: Dict, rejillas: List[Dict], ancho: int, alto: int) -> Optional[Dict]:
    """Encuentra la rejilla más cercana a la bounding box del SKU mediante distancia euclidiana."""
    if not rejillas:
        return None

    def distancia(r):
        dx = (producto["x_centro"] - r["x_centro"]) / max(1, ancho)
        dy = (producto["y_centro"] - r["y_centro"]) / max(1, alto)
        return np.hypot(dx, dy)

    return min(rejillas, key=distancia)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        image_bytes = await file.read()
        pil_img = ImageOps.exif_transpose(
            Image.open(io.BytesIO(image_bytes))
        ).convert("RGB")
        ancho, alto = pil_img.size

        # 1. Detección de SKUs y Rejilla con YOLO Bounding Boxes
        resultado = model(pil_img, conf=0.25, device=YOLO_DEVICE)[0]

        # 2. Segmentación con YOLO Polígonos
        resultado_seg = seg_model(
            pil_img, 
            conf=0.25, 
            device=YOLO_DEVICE,
            retina_masks=True
        )[0]

        productos = []
        rejillas = []

        if resultado.boxes is not None:
            for indice, box in enumerate(resultado.boxes):
                clase_id = int(box.cls[0].item())
                nombre = CLASS_NAMES.get(
                    clase_id,
                    model.names.get(clase_id, f"Clase_{clase_id}"),
                )
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                deteccion = {
                    "id_deteccion": indice,
                    "nombre": nombre,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_centro": (x1 + x2) / 2.0,
                    "y_centro": (y1 + y2) / 2.0,
                    "confianza": float(box.conf[0].item()),
                }

                if any(k in nombre.lower() for k in REJILLA_KEYWORDS):
                    rejillas.append(deteccion)
                else:
                    productos.append(deteccion)

        # 3. Estimación de Mapa de Profundidad con Depth Anything V2
        depth_map = normalizar_mapa_profundidad(calcular_mapa_profundidad(pil_img))

        detalle_detecciones = []
        resumen_por_sku: Dict[str, Dict] = {}

        # 4. Cálculo de diferencias de profundidad e inventario por cada SKU
        for producto in productos:
            nombre = producto["nombre"]

            # Evaluación de NLP / Reglas para capacidad máxima y paso
            capacidad_max = resolver_capacidad_sku(nombre)
            paso_profundidad = resolver_paso_profundidad(nombre)

            # Rejilla de referencia asociada
            rejilla = buscar_rejilla_mas_cercana(producto, rejillas, ancho, alto)

            # Extraer profundidad de SKU y Rejilla
            profundidad_sku = extraer_profundidad_bbox(depth_map, producto)
            profundidad_rejilla = extraer_profundidad_bbox(depth_map, rejilla) if rejilla else None

            if profundidad_sku is None or profundidad_rejilla is None:
                faltantes = 0
                inventario_estimado = capacidad_max
                diferencia = 0.0
                estado = "sin_profundidad_o_rejilla"
            else:
                # Diferencia de profundidad entre la rejilla y el SKU
                diferencia = abs(profundidad_sku - profundidad_rejilla)

                # Si está al mismo nivel (diferencia cercana a 0), no faltan productos
                faltantes = int(round(diferencia / max(paso_profundidad, 1e-6)))
                faltantes = int(np.clip(faltantes, 0, capacidad_max))
                
                # Inventario Estimado = Capacidad Máxima - Faltantes
                inventario_estimado = capacidad_max - faltantes
                estado = "estimado_exitoso"

            detalle = {
                "id_deteccion": producto["id_deteccion"],
                "sku": nombre,
                "rejilla_asociada_id": rejilla["id_deteccion"] if rejilla else None,
                "profundidad_sku": round(profundidad_sku, 6) if profundidad_sku else None,
                "profundidad_rejilla": round(profundidad_rejilla, 6) if profundidad_rejilla else None,
                "diferencia_profundidad": round(diferencia, 6) if diferencia is not None else None,
                "capacidad_maxima": capacidad_max,
                "faltantes_calculados": faltantes,
                "inventario_estimado": inventario_estimado,
                "estado": estado,
            }
            detalle_detecciones.append(detalle)

            # Acumular resumen por categoría de SKU
            resumen = resumen_por_sku.setdefault(nombre, {
                "etiqueta": nombre,
                "visibles": 0,
                "stock_estimado": 0,
            })
            resumen["visibles"] += 1
            resumen["stock_estimado"] += inventario_estimado

        # 5. Generación de Imagen Anotada Bounding Boxes
        anotada_bgr = resultado.plot(boxes=True, masks=False)
        buffer_det = io.BytesIO()
        Image.fromarray(anotada_bgr[:, :, ::-1]).save(buffer_det, format="JPEG", quality=90)
        imagen_base64 = base64.b64encode(buffer_det.getvalue()).decode("utf-8")

        # 6. Generación de Imagen Anotada Segmentación
        anotada_seg_bgr = resultado_seg.plot(boxes=False, masks=True)
        buffer_seg = io.BytesIO()
        Image.fromarray(anotada_seg_bgr[:, :, ::-1]).save(buffer_seg, format="JPEG", quality=90)
        imagen_seg_base64 = base64.b64encode(buffer_seg.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "imagen_base64": imagen_base64,
            "imagen_seg_base64": imagen_seg_base64,
            "detecciones": list(resumen_por_sku.values()),
            "detalle_detecciones": detalle_detecciones,
            "total_skus_detectados": len(productos),
            "total_rejillas_detectadas": len(rejillas),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error en procesamiento de auditoría: {error}",
        ) from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
