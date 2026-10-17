from typing import Dict, List, Optional
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from models import depth_estimator


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
    """Obtiene la mediana de profundidad en la región central del Bounding Box del SKU."""
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


def extraer_coordenadas_rejillas(rejillas: List[Dict], ancho: int, alto: int) -> Optional[tuple[cKDTree, np.ndarray]]:
    """
    Extrae la lista de coordenadas (x, y) de todas las rejillas detectadas
    por YOLO Segmentación / Detección y crea la estructura k-NN (cKDTree).
    """
    coords_list = []

    for r in rejillas:
        if "mask" in r and r["mask"] is not None:
            mask = r["mask"]
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                coords = np.column_stack((xs, ys))
                coords_list.append(coords)
        else:
            x1, y1 = max(0, int(r["x1"])), max(0, int(r["y1"]))
            x2, y2 = min(ancho, int(r["x2"])), min(alto, int(r["y2"]))
            
            grid_y, grid_x = np.mgrid[y1:y2:3, x1:x2:3]
            if grid_y.size > 0:
                coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
                coords_list.append(coords)

    if not coords_list:
        return None

    todas_coords = np.vstack(coords_list)
    tree = cKDTree(todas_coords)
    return tree, todas_coords


def obtener_profundidad_rejilla_knn_pura(
    depth_map: np.ndarray, 
    producto: Dict, 
    knn_data: Optional[tuple[cKDTree, np.ndarray]], 
    k: int = 20
) -> Optional[float]:
    """
    Busca los K píxeles (x, y) de la rejilla más cercanos al centro de la paleta/SKU
    y retorna el promedio de sus valores en la matriz de profundidad.
    """
    if knn_data is None:
        return None

    tree, todas_coords = knn_data
    centro_paleta = np.array([producto["x_centro"], producto["y_centro"]])

    k_actual = min(k, len(todas_coords))
    _, indices = tree.query(centro_paleta, k=k_actual)

    if k_actual == 1:
        indices = [indices]

    pixeles_cercanos = todas_coords[indices]

    profundidades = [
        depth_map[py, px] 
        for px, py in pixeles_cercanos 
        if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1] and np.isfinite(depth_map[py, px])
    ]

    if profundidades:
        return float(np.mean(profundidades))

    return None