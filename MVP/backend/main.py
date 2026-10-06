import base64
import io
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO Detector API")

# Detectar la ruta local donde está 'best.pt'
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo de pesos en {MODEL_PATH}")

# Cargar el modelo
model = YOLO(str(MODEL_PATH))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        # 1. Leer imagen enviada desde Flet
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Inferencia con YOLO
        results = model(pil_img, conf=0.25)
        result = results[0]

        # 3. Generar imagen anotada (convertir de BGR a RGB)
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        # 4. Codificar la imagen procesada a Base64
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 5. Extraer etiquetas y calcular frecuencias
        clases_detectadas = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                clases_detectadas.append(model.names[cls_id])

        conteo_total = Counter(clases_detectadas)

        # Filtrar: Únicamente frecuencias >= 2 (se omiten 0 y 1)
        MIN_FRECUENCIA = 1
        detecciones_filtradas = [
            {"etiqueta": k, "frecuencia": v}
            for k, v in conteo_total.items()
            if v >= MIN_FRECUENCIA
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