"""
Módulo de histograma y ecualización para imágenes médicas.

Funciones: cálculo de histograma, ecualización, estadísticas de distribución.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def calcular_histograma(imagen: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calcula el histograma de intensidades de una imagen en escala de grises.
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
        bins: Número de intervalos del histograma.
    
    Retorna:
        np.ndarray: Histograma de frecuencias de intensidades.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    histograma = cv2.calcHist(
        [imagen],
        [0],
        None,
        [bins],
        [0, 256]
    )
    
    return histograma.flatten()


def ecualizar_histograma(imagen: np.ndarray) -> np.ndarray:
    """
    Ecualiza el histograma de una imagen para mejorar el contraste global.
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
    
    Retorna:
        np.ndarray: Imagen con histograma ecualizado.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    return cv2.equalizeHist(imagen)


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


def estadisticas_histograma(imagen: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticas descriptivas del histograma de intensidades.
    
    Parámetros:
        imagen: Imagen de entrada.
    
    Retorna:
        Dict[str, float]: Diccionario con media, desviación estándar,
                          mínimo, máximo, percentiles.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    media = np.mean(imagen)
    desviacion_std = np.std(imagen)
    minimo = np.min(imagen)
    maximo = np.max(imagen)
    
    percentil_25 = np.percentile(imagen, 25)
    percentil_50 = np.percentile(imagen, 50)  # mediana
    percentil_75 = np.percentile(imagen, 75)
    
    rango_intercuartil = percentil_75 - percentil_25
    
    return {
        'media': float(media),
        'desviacion_std': float(desviacion_std),
        'minimo': float(minimo),
        'maximo': float(maximo),
        'percentil_25': float(percentil_25),
        'mediana': float(percentil_50),
        'percentil_75': float(percentil_75),
        'rango_intercuartil': float(rango_intercuartil),
        'rango': float(maximo - minimo)
    }


def calcular_histograma_acumulado(imagen: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calcula el histograma acumulado de una imagen.
    
    Parámetros:
        imagen: Imagen de entrada.
        bins: Número de bins del histograma.
    
    Retorna:
        np.ndarray: Histograma acumulado normalizado [0, 1].
    """
    histograma = calcular_histograma(imagen, bins)
    
    histograma_acumulado = np.cumsum(histograma)
    
    histograma_normalizado = histograma_acumulado / histograma_acumulado[-1]
    
    return histograma_normalizado


def umbralizar_por_percentil(imagen: np.ndarray, percentil: float = 95.0) -> np.ndarray:
    """
    Calcula un umbral basado en un percentil del histograma.
    
    Útil para segmentar regiones de alta intensidad (ej: lesiones,
    opacidades en CT).
    
    Parámetros:
        imagen: Imagen de entrada.
        percentil: Percentil para calcular el umbral (0-100).
    
    Retorna:
        np.ndarray: Imagen binarizada con píxeles above el umbral.
    """
    umbral = np.percentile(imagen, percentil)
    
    _, mascara = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    
    return mascara


def analizar_distribucion(imagen: np.ndarray) -> Dict[str, any]:
    """
    Analiza la distribución de intensidades y sugiere parámetros
    de procesamiento.
    
    Parámetros:
        imagen: Imagen de entrada.
    
    Retorna:
        Dict[str, any]: Análisis completo de la distribución.
    """
    estadisticas = estadisticas_histograma(imagen)
    
    histograma = calcular_histograma(imagen)
    
    pico_principal = np.argmax(histograma)
    
    asimetria = (estadisticas['media'] - estadisticas['mediana']) / estadisticas['desviacion_std']
    
    sugerencias = {
        'contraste_bajo': estadisticas['rango'] < 100,
        'imagen_oscura': estadisticas['media'] < 80,
        'imagen_bright': estadisticas['media'] > 180,
        'ruido_alto': estadisticas['desviacion_std'] > 50
    }
    
    return {
        'estadisticas': estadisticas,
        'pico_principal': int(pico_principal),
        'asimetria': float(asimetria),
        'sugerencias': sugerencias
    }
