"""
Módulo de segmentación para imágenes médicas.

Implementa: Watershed con marcadores automáticos, Otsu, y segmentación
basada en distancia transform.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
from sklearn.cluster import KMeans


def umbralizar_otsu(imagen: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Aplica umbralización óptima de Otsu para segmentación automática.
    
    Otsu encuentra el umbral que minimiza la varianza intra-clase
    entre foreground y background.
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
    
    Retorna:
        Tuple[np.ndarray, float]: 
            - Imagen binarizada
            - Umbral óptimo encontrado
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    _, mascara = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    umbral = _
    
    return mascara, umbral


def umbralizar_adaptativo(imagen: np.ndarray, metodo: str = 'gaussian',
                          block_size: int = 11, c: float = 2.0) -> np.ndarray:
    """
    Aplica umbralización adaptativa local.
    
    Útil para imágenes con iluminación no uniforme.
    
    Parámetros:
        imagen: Imagen de entrada.
        metodo: 'gaussian' o 'mean'.
        block_size: Tamaño del vecindario (debe ser impar).
        c: Constante restada al umbral calculado.
    
    Retorna:
        np.ndarray: Imagen binarizada adaptativamente.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size debe ser impar.")
    
    if metodo == 'gaussian':
        umbral_tipo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif metodo == 'mean':
        umbral_tipo = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError("metodo debe ser 'gaussian' o 'mean'.")
    
    mascara = cv2.adaptiveThreshold(
        imagen,
        255,
        umbral_tipo,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    return mascara


def crear_marcadores_distancia(imagen: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    """
    Crea marcadores automáticos para Watershed usando transformada de distancia.
    
    Ideal para segmentar objetos separados que están tocándose.
    
    Parámetros:
        imagen: Imagen de entrada (binaria o grayscale).
        umbral: Umbral para binarización (0-1 si es float, 0-255 si imagen ya binaria).
    
    Retorna:
        np.ndarray: Mapa de marcadores para Watershed.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    if imagen.max() > 1:
        imagen_bin = (imagen > umbral * 255).astype(np.uint8)
    else:
        imagen_bin = (imagen > umbral).astype(np.uint8)
    
    distancia = cv2.distanceTransform(imagen_bin, cv2.DIST_L2, 5)
    
    _, marcadores = cv2.threshold(
        distancia,
        0.4 * distancia.max(),
        255,
        cv2.THRESH_BINARY
    )
    
    marcadores = marcadores.astype(np.int32)
    
    return marcadores


def segmentar_watershed(imagen: np.ndarray, marcadores: Optional[np.ndarray] = None,
                        umbral_auto: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta una imagen usando el algoritmo Watershed.
    
    Watershed trata la imagen como una superficie topográfica y
    encuentra líneas de división entre regiones.
    
    Parámetros:
        imagen: Imagen de entrada.
        marcadores: Marcadores predefinidos. Si None, se crean automáticamente.
        umbral_auto: Si True, usa Otsu para marcadores automáticos.
    
    Retorna:
        Tuple[np.ndarray, np.ndarray]:
            - Imagen segmentada con etiquetas de regiones
            - Contornos de Watershed
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen
    
    if marcadores is None:
        if umbral_auto:
            mascara_otsu, _ = umbralizar_otsu(imagen_gray)
            marcadores = crear_marcadores_distancia(mascara_otsu)
        else:
            marcadores = crear_marcadores_distancia(imagen_gray)
    
    imagen_rgb = cv2.cvtColor(imagen_gray, cv2.COLOR_GRAY2BGR)
    
    marcadores_watershed = cv2.watershed(imagen_rgb, marcadores)
    
    contornos = np.zeros_like(marcadores_watershed)
    contornos[marcadores_watershed == -1] = 255
    
    return marcadores_watershed, contornos


def segmentar_kmeans(imagen: np.ndarray, k: int = 3, 
                     iteraciones: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta una imagen usando K-Means clustering.
    
    Agrupa píxeles por intensidad en k clusters.
    
    Parámetros:
        imagen: Imagen de entrada.
        k: Número de clusters.
        iteraciones: Número de iteraciones del algoritmo.
    
    Retorna:
        Tuple[np.ndarray, np.ndarray]:
            - Imagen segmentada (etiquetas de clusters)
            - Centros de los clusters
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.flatten()
    
    imagen_flat = imagen_gray.reshape(-1, 1).astype(np.float32)
    
    kmeans = KMeans(
        n_clusters=k,
        max_iter=iteraciones,
        random_state=42,
        n_init='auto'
    )
    
    kmeans.fit(imagen_flat)
    
    etiquetas = kmeans.labels_.reshape(imagen.shape)
    
    centros = kmeans.cluster_centers_.flatten()
    
    return etiquetas.astype(np.uint8), centros


def extraer_componentes_conexos(mascara: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Extrae componentes conexos de una máscara binaria.
    
    Parámetros:
        mascara: Máscara binaria (0 y 255).
    
    Retorna:
        Tuple[np.ndarray, int]:
            - Mapa de componentes con etiquetas
            - Número de componentes encontrados
    """
    if mascara.max() > 1:
        mascara = (mascara > 0).astype(np.uint8)
    
    num_labels, labels = cv2.connectedComponents(mascara)
    
    return labels, num_labels


def filtrar_componentes_por_area(mascara: np.ndarray, 
                                  area_min: float = 10.0,
                                  area_max: float = 10000.0) -> np.ndarray:
    """
    Filtra componentes conexos por área mínima y máxima.
    
    Parámetros:
        mascara: Máscara binaria.
        area_min: Área mínima para mantener componente.
        area_max: Área máxima para mantener componente.
    
    Retorna:
        np.ndarray: Máscara filtrada.
    """
    labels, num_labels = extraer_componentes_conexos(mascara)
    
    mascara_filtrada = np.zeros_like(mascara)
    
    for label in range(1, num_labels):
        componente = (labels == label).astype(np.uint8)
        area = cv2.countNonZero(componente)
        
        if area_min <= area <= area_max:
            mascara_filtrada[labels == label] = 255
    
    return mascara_filtrada
