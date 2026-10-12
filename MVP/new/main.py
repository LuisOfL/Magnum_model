import base64
import io
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from ultralytics import YOLO

app = FastAPI(title="YOLO Freezer Inventory Estimator API")

# Definición de rutas principales
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "best.pt"
CLASSES_PATH = BASE_DIR / "classes.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de pesos en {MODEL_PATH}")

# Carga del modelo YOLO
model = YOLO(str(MODEL_PATH))

# Cargar las clases desde classes.txt por su índice de línea (0 a N)
CLASS_NAMES: Dict[int, str] = {}
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = {i: line.strip() for i, line in enumerate(f) if line.strip()}

# -----------------------------------------------------------------------------
# DICCIONARIO DE CAPACIDADES MÁXIMAS POR CANAL/CARRIL Y TIPO DE SKU
# Define la cantidad de piezas estimadas que entran en la profundidad de un carril.
# -----------------------------------------------------------------------------
CAPACIDADES_POR_SKU: Dict[str, int] = {
    "Magnum": 12,
    "Mordisko": 11,
    "Cornetto": 10,
    "Solero": 10,
    "Pint": 8,
    "DEFAULT": 10  # Capacidad por defecto
}


def obtener_capacidad_sku(nombre_sku: str) -> int:
    """Busca la capacidad asignada al SKU o retorna la capacidad por defecto."""
    for key, cap in CAPACIDADES_POR_SKU.items():
        if key.lower() in nombre_sku.lower():
            return cap
    return CAPACIDADES_POR_SKU["DEFAULT"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        # 1. Leer imagen y corregir orientación según metadatos EXIF (Solución a fotos giradas)
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_width, img_height = pil_img.size

        # 2. Inferencia con YOLO
        results = model(pil_img, conf=0.25)
        result = results[0]

        # Generar imagen anotada codificada en Base64 para Flet
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 3. Extraer productos y rejillas/estantes
        productos = []
        rejillas = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())

                if cls_id in CLASS_NAMES:
                    nombre_clase = CLASS_NAMES[cls_id]
                elif hasattr(model, "names") and cls_id in model.names:
                    nombre_clase = model.names[cls_id]
                else:
                    nombre_clase = f"Clase_{cls_id}"

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_centro = (x1 + x2) / 2.0
                y_centro = (y1 + y2) / 2.0

                es_rejilla = any(
                    kw in nombre_clase.lower()
                    for kw in ["rejilla", "canastilla", "separador", "carril", "parrilla", "estante"]
                )

                if es_rejilla:
                    rejillas.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x_centro": x_centro, "y_centro": y_centro
                    })
                else:
                    productos.append({
                        "nombre": nombre_clase,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x_centro": x_centro, "y_centro": y_centro
                    })

        detecciones_filtradas = []

        if productos:
            # Conteo de detecciones visibles reales por etiqueta
            conteo_visibles: Dict[str, int] = {}
            for p in productos:
                nom = p["nombre"]
                conteo_visibles[nom] = conteo_visibles.get(nom, 0) + 1

            # =========================================================================
            # PASO 1: AGRUPACIÓN BI-DIMENSIONAL (NIVELES EN Y -> CARRILES EN X)
            # =========================================================================
            # Ordenar productos de arriba a abajo por su eje Y
            productos_ordenados_y = sorted(productos, key=lambda p: p["y_centro"])

            # 1.1 Segmentar en estantes horizontales (Niveles Y)
            estantes = []
            estante_actual = []
            y_ref = -1
            tolerancia_estante_y = img_height * 0.12  # Tolerancia vertical (~12% de la altura de la imagen)

            for p in productos_ordenados_y:
                if y_ref == -1:
                    estante_actual.append(p)
                    y_ref = p["y_centro"]
                elif abs(p["y_centro"] - y_ref) < tolerancia_estante_y:
                    estante_actual.append(p)
                    # Promedio móvil para adaptar variaciones en el eje Y
                    y_ref = sum(item["y_centro"] for item in estante_actual) / len(estante_actual)
                else:
                    estantes.append(estante_actual)
                    estante_actual = [p]
                    y_ref = p["y_centro"]
            if estante_actual:
                estantes.append(estante_actual)

            # 1.2 Dentro de cada estante, agrupar por carriles/columnas (Eje X)
            carriles_totales = []

            for estante in estantes:
                # Límites verticales del estante actual
                y_min_estante = min(p["y1"] for p in estante)
                y_max_estante = max(p["y2"] for p in estante)
                y_top_estante = max(0.0, y_min_estante - img_height * 0.02)
                y_bottom_estante = min(img_height, y_max_estante + img_height * 0.02)

                # Ordenar horizontalmente los productos de este estante
                estante_ordenado_x = sorted(estante, key=lambda p: p["x_centro"])
                col_carril = []
                x_ref = -1
                tolerancia_carril_x = img_width * 0.08  # Tolerancia horizontal (~8% del ancho)

                for p in estante_ordenado_x:
                    if x_ref == -1:
                        col_carril.append(p)
                        x_ref = p["x_centro"]
                    elif abs(p["x_centro"] - x_ref) < tolerancia_carril_x:
                        col_carril.append(p)
                    else:
                        carriles_totales.append({
                            "y_top": y_top_estante,
                            "y_bottom": y_bottom_estante,
                            "productos": col_carril
                        })
                        col_carril = [p]
                        x_ref = p["x_centro"]
                if col_carril:
                    carriles_totales.append({
                        "y_top": y_top_estante,
                        "y_bottom": y_bottom_estante,
                        "productos": col_carril
                    })

            # =========================================================================
            # PASOS 2, 3 Y 4: ESTIMACIÓN DE PROFUNDIDAD POR CARRIL DE CADA ESTANTE
            # =========================================================================
            stock_estimado_por_sku: Dict[str, int] = {}

            for carril in carriles_totales:
                prods_carril = carril["productos"]
                if not prods_carril:
                    continue

                # SKU superior/frontal en este carril del estante
                sku_superior = min(prods_carril, key=lambda p: p["y1"])
                nombre_sku = sku_superior["nombre"]

                y_top = carril["y_top"]
                y_bottom = carril["y_bottom"]
                y1_sku = sku_superior["y1"]

                altura_total_estante = max(1.0, y_bottom - y_top)
                altura_ocupada = max(0.0, y_bottom - y1_sku)

                # Proporción de llenado del carril dentro del estante
                ratio_llenado = max(0.0, min(1.0, altura_ocupada / altura_total_estante))

                capacidad_max = obtener_capacidad_sku(nombre_sku)
                piezas_estimadas_carril = int(round(ratio_llenado * capacidad_max))

                # Garantizar que el estimado por carril no sea inferior a las piezas visibles en dicho carril
                visibles_en_carril = len(prods_carril)
                stock_carril = max(visibles_en_carril, piezas_estimadas_carril)

                # Sumar la estimación de este carril al acumulado del tipo de SKU
                stock_estimado_por_sku[nombre_sku] = (
                    stock_estimado_por_sku.get(nombre_sku, 0) + stock_carril
                )

            # =========================================================================
            # PASO 5: CONSOLIDAR RESULTADOS PARA EL FRONTEND (FLET)
            # =========================================================================
            for nombre_sku, num_visibles in conteo_visibles.items():
                stock_est = stock_estimado_por_sku.get(nombre_sku, num_visibles)
                stock_est = max(num_visibles, stock_est)

                detecciones_filtradas.append({
                    "etiqueta": nombre_sku,
                    "visibles": num_visibles,
                    "frecuencia": stock_est
                })

        return {
            "status": "success",
            "imagen_base64": img_base64,
            "detecciones": detecciones_filtradas,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en inferencia y estimación de inventario: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)