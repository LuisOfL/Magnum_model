import re
from config import REGLAS_CAPACIDAD_TEXTO, PASO_PROFUNDIDAD_POR_SKU


def normalizar_texto(texto: str) -> str:
    """Limpia y normaliza el nombre del SKU para coincidencia de palabras clave."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()


def resolver_capacidad_sku(nombre_sku: str) -> int:
    """Determina la capacidad máxima por coincidencia de palabras clave."""
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