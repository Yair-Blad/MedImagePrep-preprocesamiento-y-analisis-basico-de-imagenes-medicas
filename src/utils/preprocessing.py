"""
Módulo de preprocesamiento para imágenes médicas.

Funciones: crop, resize, normalización, filtros (Gaussian, bilateral, mediano).
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def recortar(imagen: np.ndarray, x: int, y: int, 
             ancho: int, alto: int) -> np.ndarray:
    """
    Recorta una región de interés (ROI) de la imagen.
    
    Parámetros:
        imagen: Imagen de entrada (numpy array).
        x: Coordenada x del punto superior izquierdo.
        y: Coordenada y del punto superior izquierdo.
        ancho: Ancho de la región a recortar.
        alto: Alto de la región a recortar.
    
    Retorna:
        np.ndarray: Imagen recortada.
    
    Raises:
        ValueError: Si las coordenadas están fuera de los límites de la imagen.
    """
    alto_img, ancho_img = imagen.shape[:2]
    
    if x < 0 or y < 0 or x + ancho > ancho_img or y + alto > alto_img:
        raise ValueError(
            f"Las coordenadas de recorte están fuera de los límites. "
            f"Imagen: {ancho_img}x{alto_img}, Recorte: x={x}, y={y}, "
            f"ancho={ancho}, alto={alto}"
        )
    
    return imagen[y:y+alto, x:x+ancho]


def redimensionar(imagen: np.ndarray, ancho: int, alto: int, 
                  interpolacion: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Redimensiona la imagen a las dimensiones especificadas.
    
    Parámetros:
        imagen: Imagen de entrada.
        ancho: Nuevo ancho en píxeles.
        alto: Nuevo alto en píxeles.
        interpolacion: Método de interpolación de OpenCV.
            - cv2.INTER_AREA: Reducción (recomendado)
            - cv2.INTER_CUBIC: Ampliación suave
            - cv2.INTER_LINEAR: Ampliación rápida
    
    Retorna:
        np.ndarray: Imagen redimensionada.
    """
    return cv2.resize(imagen, (ancho, alto), interpolation=interpolacion)


def normalizar(imagen: np.ndarray, rango_min: int = 0, 
               rango_max: int = 255) -> np.ndarray:
    """
    Normaliza la imagen a un rango de intensidades especificado.
    
    Parámetros:
        imagen: Imagen de entrada.
        rango_min: Valor mínimo del rango de salida.
        rango_max: Valor máximo del rango de salida.
    
    Retorna:
        np.ndarray: Imagen normalizada.
    """
    if imagen.min() == imagen.max():
        return np.full_like(imagen, rango_min, dtype=np.uint8)
    
    imagen_normalizada = cv2.normalize(
        imagen,
        None,
        alpha=rango_min,
        beta=rango_max,
        norm_type=cv2.NORM_MINMAX
    )
    
    return imagen_normalizada.astype(np.uint8)


def filtro_gaussiano(imagen: np.ndarray, kernel: Tuple[int, int] = (5, 5), 
                     sigma: float = 0) -> np.ndarray:
    """
    Aplica un filtro Gaussiano para suavizar la imagen y reducir ruido.
    
    Parámetros:
        imagen: Imagen de entrada.
        kernel: Tamaño del kernel (debe ser impar, ej: (3,3), (5,5)).
        sigma: Desviación estándar del kernel. Si es 0, se calcula automáticamente.
    
    Retorna:
        np.ndarray: Imagen con filtro Gaussiano aplicado.
    """
    if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")
    
    return cv2.GaussianBlur(imagen, kernel, sigma)


def filtro_bilateral(imagen: np.ndarray, d: int = 9, 
                     sigmaColor: float = 75, 
                     sigmaSpace: float = 75) -> np.ndarray:
    """
    Aplica un filtro bilateral que suaviza preservando bordes.
    
    Ideal para imágenes médicas donde se quiere reducir ruido
    manteniendo los bordes de estructuras anatómicas.
    
    Parámetros:
        imagen: Imagen de entrada.
        d: Diámetro de cada vecindad de píxel.
        sigmaColor: Desviación estándar en espacio de color.
        sigmaSpace: Desviación estándar en espacio espacial.
    
    Retorna:
        np.ndarray: Imagen con filtro bilateral aplicado.
    """
    return cv2.bilateralFilter(imagen, d, sigmaColor, sigmaSpace)


def filtro_mediano(imagen: np.ndarray, kernel: int = 3) -> np.ndarray:
    """
    Aplica un filtro mediano para eliminar ruido "sal y pimienta".
    
    Parámetros:
        imagen: Imagen de entrada.
        kernel: Tamaño del kernel (debe ser impar).
    
    Retorna:
        np.ndarray: Imagen con filtro mediano aplicado.
    """
    if kernel % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")
    
    return cv2.medianBlur(imagen, kernel)


def filtro_promedio(imagen: np.ndarray, kernel: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Aplica un filtro de promedio (box filter) para suavizado básico.
    
    Parámetros:
        imagen: Imagen de entrada.
        kernel: Tamaño del kernel.
    
    Retorna:
        np.ndarray: Imagen con filtro de promedio aplicado.
    """
    return cv2.blur(imagen, kernel)


def aplicar_clahe(imagen: np.ndarray, clip_limit: float = 2.0, 
                  tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Mejora el contraste localmente, ideal para CT y radiografías.
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
        clip_limit: Límite de contraste (evita sobre-amplificación de ruido).
        tile_grid_size: Tamaño de la grilla para ecualización local.
    
    Retorna:
        np.ndarray: Imagen con CLAHE aplicado.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    return clahe.apply(imagen)
