import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO Detector API - Estimación de Inventario")

BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "best.pt"
CLASSES_PATH = BASE_DIR / "classes.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de pesos en {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Cargar las clases desde classes.txt por su índice de línea (0 a N)
CLASS_NAMES = {}
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = {i: line.strip() for i, line in enumerate(f) if line.strip()}

CAPACIDAD_MAXIMA_POR_CANAL = 12


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        # 1. Leer imagen
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_width, img_height = pil_img.size

        # 2. Inferencia con YOLO
        results = model(pil_img, conf=0.25)
        result = results[0]

        # 3. Generar imagen anotada
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 4. Extracción de cajas y mapeo de clases
        productos = []
        rejillas = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                
                # Mapeo prioritario con classes.txt
                if cls_id in CLASS_NAMES:
                    nombre_clase = CLASS_NAMES[cls_id]
                elif hasattr(model, "names") and cls_id in model.names:
                    nombre_clase = model.names[cls_id]
                else:
                    nombre_clase = f"Clase_{cls_id}"

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                if "rejilla" in nombre_clase.lower() or "canastilla" in nombre_clase.lower():
                    rejillas.append((x1, y1, x2, y2))
                else:
                    productos.append({
                        "nombre": nombre_clase, 
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "x_centro": (x1 + x2) / 2
                    })

        detecciones_filtradas = []
        
        if productos:
            # 4.1 Contar todas las detecciones visibles en la foto
            conteo_visibles = {}
            for p in productos:
                nom = p["nombre"]
                conteo_visibles[nom] = conteo_visibles.get(nom, 0) + 1

            # 4.2 Agrupar en columnas para estimación de profundidad
            productos.sort(key=lambda p: p["x_centro"])
            columnas = []
            col_actual = []
            x_ref = -1
            margen_columna = img_width * 0.10  
            
            for p in productos:
                if x_ref == -1:
                    col_actual.append(p)
                    x_ref = p["x_centro"]
                elif abs(p["x_centro"] - x_ref) < margen_columna:
                    col_actual.append(p)
                else:
                    columnas.append(col_actual)
                    col_actual = [p]
                    x_ref = p["x_centro"]
            if col_actual:
                columnas.append(col_actual)

            inventario_estimado = {}

            for col in columnas:
                top_prod = min(col, key=lambda p: p["y1"])
                nombre = top_prod["nombre"]
                
                y_top = img_height * 0.05
                y_bottom = img_height * 0.95
                
                for rx1, ry1, rx2, ry2 in rejillas:
                    if rx1 <= top_prod["x_centro"] <= rx2:
                        y_top = ry1
                        y_bottom = ry2
                        break
                
                altura_total = max(1.0, y_bottom - y_top)
                altura_ocupada = max(0.0, y_bottom - top_prod["y1"])
                
                ratio_llenado = max(0.0, min(1.0, altura_ocupada / altura_total))
                piezas_estimadas = int(round(ratio_llenado * CAPACIDAD_MAXIMA_POR_CANAL))
                
                inventario_estimado[nombre] = inventario_estimado.get(nombre, 0) + piezas_estimadas

            # 4.3 Consolidar todos los productos detectados
            for nombre, count_vis in conteo_visibles.items():
                stock_est = max(count_vis, inventario_estimado.get(nombre, 0))
                detecciones_filtradas.append({
                    "etiqueta": nombre,
                    "visibles": count_vis,
                    "frecuencia": stock_est
                })

        return {
            "status": "success",
            "imagen_base64": img_base64,
            "detecciones": detecciones_filtradas,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)