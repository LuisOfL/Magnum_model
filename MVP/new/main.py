import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO Detector API - Estimación de Inventario")

# Detectar la ruta local donde está 'best.pt'
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de pesos en {MODEL_PATH}")

# Cargar el modelo
model = YOLO(str(MODEL_PATH))

# Constante de inventario
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

        # 4. Algoritmo de Profundidad y Estimación
        productos = []
        rejillas = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                nombre_clase = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Separar si el modelo ya detecta rejillas, sino lo hace, usa los bordes de la foto
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
            # Agrupar productos en columnas (canales) basándose en su eje X
            productos.sort(key=lambda p: p["x_centro"])
            columnas = []
            col_actual = []
            x_ref = -1
            
            # Margen de 10% del ancho de la imagen para considerar que están en la misma columna
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

            # Diccionario para sumar el stock total de cada producto en todo el congelador
            inventario_total = {}

            # Calcular profundidad para cada columna detectada
            for col in columnas:
                # El producto de hasta arriba es el que tiene la coordenada Y1 más pequeña (más cerca del 0)
                top_prod = min(col, key=lambda p: p["y1"])
                nombre = top_prod["nombre"]
                
                # Determinar los límites Y del canal (si no hay rejilla etiquetada, asume el 5% y 95% de la foto)
                y_top = img_height * 0.05
                y_bottom = img_height * 0.95
                
                for rx1, ry1, rx2, ry2 in rejillas:
                    if rx1 <= top_prod["x_centro"] <= rx2:
                        y_top = ry1
                        y_bottom = ry2
                        break
                
                # Matemática de estimación (0 a 1)
                altura_total = max(1.0, y_bottom - y_top)
                altura_ocupada = max(0.0, y_bottom - top_prod["y1"])
                
                ratio_llenado = altura_ocupada / altura_total
                ratio_llenado = max(0.0, min(1.0, ratio_llenado)) # Asegurar que no pase de 100%
                
                # Calcular piezas reales y acumular
                piezas_estimadas = int(round(ratio_llenado * CAPACIDAD_MAXIMA_POR_CANAL))
                
                if nombre not in inventario_total:
                    inventario_total[nombre] = 0
                inventario_total[nombre] += piezas_estimadas

            # Formatear salida para Flet
            detecciones_filtradas = [
                {"etiqueta": k, "frecuencia": v}
                for k, v in inventario_total.items()
                if v > 0
            ]

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