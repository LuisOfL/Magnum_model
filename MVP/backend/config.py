from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).parent.resolve()

MODEL_PATH = BASE_DIR / "best.pt"
SEG_MODEL_PATH = BASE_DIR / "segmentation.pt"
CLASSES_PATH = BASE_DIR / "classes.txt"

REGLAS_CAPACIDAD_TEXTO: Dict[str, int] = {
    "mordisko": 11,
    "mordizco": 11,
    "paleta": 12,
    "magnum": 12,
    "solero": 12,
    "cornetto": 10,
    "pint": 8,
    "DEFAULT": 12,
}

# PASO RECALIBRADO:
# Se reduce la escala del paso para reflejar correctamente 
# la diferencia de profundidad normalizada por unidad/paquete.
PASO_PROFUNDIDAD_POR_SKU: Dict[str, float] = {
    "mordisko": 0.012,
    "mordizco": 0.012,
    "paleta": 0.015,
    "magnum": 0.012,   # Con 0.0819 / 0.012 = ~7 faltantes calculados
    "solero": 0.015,
    "cornetto": 0.015,
    "pint": 0.020,
    "DEFAULT": 0.015,
}

REJILLA_KEYWORDS: Tuple[str, ...] = (
    "rejilla", "canastilla", "separador", "carril", "parrilla", "estante"
)