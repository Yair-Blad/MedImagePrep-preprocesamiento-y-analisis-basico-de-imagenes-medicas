"""
Módulo de cuantificación para imágenes médicas.

Calcula: porcentajes de regiones, estadísticas de textura, medidas
morfométricas de lesiones.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from skimage.feature import graycomatrix, graycoprops


def porcentaje_alta_intensidad(imagen: np.ndarray, umbral: float = 0.8) -> float:
    """
    Calcula el porcentaje de píxeles con alta intensidad en la imagen.
    
    Útil para cuantificar regiones de interés como:
    - Opacidades en vidrio esmerilado (CT pulmón COVID)
    - Lesiones hiperintensas
    - Calcificaciones
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
        umbral: Umbral relativo (0-1) o absoluto (0-255).
    
    Retorna:
        float: Porcentaje de píxeles above el umbral (0-100).
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    if umbral > 1:
        umbral_absoluto = umbral
    else:
        umbral_absoluto = umbral * 255
    
    pixeles_alta = np.sum(imagen >= umbral_absoluto)
    total_pixeles = imagen.size
    
    porcentaje = (pixeles_alta / total_pixeles) * 100
    
    return porcentaje


def medir_region(mascara: np.ndarray, imagen_original: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Mide propiedades morfométricas de una región segmentada.
    
    Parámetros:
        mascara: Máscara binaria de la región (0 y 255).
        imagen_original: Imagen original para calcular estadísticas de intensidad.
    
    Retorna:
        Dict[str, float]: Diccionario con área, perímetro, circularidad,
                          intensidad media, etc.
    """
    if mascara.max() > 1:
        mascara_bin = (mascara > 0).astype(np.uint8)
    else:
        mascara_bin = mascara.astype(np.uint8)
    
    contornos, _ = cv2.findContours(mascara_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        return {
            'area_pixeles': 0,
            'area_mm2': 0,
            'perimetro': 0,
            'circularidad': 0,
            'intensidad_media': 0,
            'intensidad_std': 0,
            'numero_componentes': 0
        }
    
    area_total = 0
    perimetro_total = 0
    numero_componentes = len(contornos)
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        
        area_total += area
        perimetro_total += perimetro
    
    if area_total > 0 and perimetro_total > 0:
        circularidad = 4 * np.pi * (area_total / (perimetro_total ** 2))
    else:
        circularidad = 0
    
    if imagen_original is not None:
        if len(imagen_original.shape) == 3:
            imagen_gray = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gray = imagen_original
        
        intensidad_media = np.mean(imagen_original[mascara_bin > 0])
        intensidad_std = np.std(imagen_original[mascara_bin > 0])
    else:
        intensidad_media = 0
        intensidad_std = 0
    
    hull_area = cv2.contourArea(cv2.convexHull(contornos[0])) if len(contornos) > 0 else 0
    solidity = float(area_total / hull_area) if hull_area > 0 else 0
    
    return {
        'area_pixeles': float(area_total),
        'area_mm2': float(area_total * 0.01),
        'perimetro': float(perimetro_total),
        'circularidad': float(circularidad),
        'intensidad_media': float(intensidad_media),
        'intensidad_std': float(intensidad_std),
        'numero_componentes': int(numero_componentes),
        'solidity': solidity
    }


def analisis_textura(imagen: np.ndarray, distancias: Tuple[int, ...] = (1, 2, 4),
                     angulos: Tuple[int, ...] = (0, 45, 90, 135)) -> Dict[str, float]:
    """
    Calcula medidas de textura usando GLCM (Gray-Level Co-occurrence Matrix).
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
        distancias: Distancias para calcular GLCM.
        angulos: Ángulos para calcular GLCM.
    
    Retorna:
        Dict[str, float]: Contraste, correlación, energía, homogeneidad.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    imagen_normalizada = imagen // 16
    
    glcm = graycomatrix(
        imagen_normalizada,
        distances=distancias,
        angles=angulos,
        symmetric=True,
        normed=True
    )
    
    contraste = graycoprops(glcm, 'contrast').mean()
    correlacion = graycoprops(glcm, 'correlation').mean()
    energia = graycoprops(glcm, 'energy').mean()
    homogeneidad = graycoprops(glcm, 'homogeneity').mean()
    
    return {
        'contraste': float(contraste),
        'correlacion': float(correlacion),
        'energia': float(energia),
        'homogeneidad': float(homogeneidad),
        'dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
        'asm': float(graycoprops(glcm, 'ASM').mean())
    }


def calcular_densidad_radiologica(imagen: np.ndarray, mascara: Optional[np.ndarray] = None,
                                   unidades_hounsfield: bool = False) -> Dict[str, float]:
    """
    Calcula densidad radiológica de una región (ej: unidades Hounsfield en CT).
    
    Parámetros:
        imagen: Imagen de entrada.
        mascara: Máscara de la región de interés. Si None, usa toda la imagen.
        unidades_hounsfield: Si True, asume que la imagen está en unidades HU.
    
    Retorna:
        Dict[str, float]: Densidad media, mínima, máxima, desviación.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    if mascara is not None:
        if mascara.max() > 1:
            mascara_bin = (mascara > 0).astype(np.uint8)
        else:
            mascara_bin = mascara.astype(np.uint8)
        
        region = imagen[mascara_bin > 0]
    else:
        region = imagen.flatten()
    
    if unidades_hounsfield:
        densidad_media = np.mean(region)
        densidad_min = np.min(region)
        densidad_max = np.max(region)
        densidad_std = np.std(region)
    else:
        densidad_media = np.mean(region)
        densidad_min = np.min(region)
        densidad_max = np.max(region)
        densidad_std = np.std(region)
    
    return {
        'densidad_media': float(densidad_media),
        'densidad_min': float(densidad_min),
        'densidad_max': float(densidad_max),
        'densidad_std': float(densidad_std),
        'rango': float(densidad_max - densidad_min)
    }


def indice_heterogeneidad(imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> float:
    """
    Calcula un índice de heterogeneidad de la imagen o región.
    
    Parámetros:
        imagen: Imagen de entrada.
        mascara: Máscara de región de interés.
    
    Retorna:
        float: Índice de heterogeneidad (coeficiente de variación).
    """
    if mascara is not None:
        if mascara.max() > 1:
            mascara_bin = (mascara > 0).astype(np.uint8)
        else:
            mascara_bin = mascara.astype(np.uint8)
        
        region = imagen[mascara_bin > 0]
    else:
        region = imagen.flatten()
    
    media = np.mean(region)
    desviacion = np.std(region)
    
    if media == 0:
        return 0
    
    indice = (desviacion / media) * 100
    
    return float(indice)


def volumen_lesion(mascara: np.ndarray, espesor_corte: float = 1.0,
                   area_pixel: float = 0.25) -> float:
    """
    Estima el volumen de una lesión a partir de una máscara 2D.
    
    Nota: Para volumen 3D real, se necesita stack de slices de CT.
    
    Parámetros:
        mascara: Máscara binaria de la lesión.
        espesor_corte: Espesor del corte en mm (default: 1.0 mm).
        area_pixel: Área de cada píxel en mm² (default: 0.25 mm² para 0.5x0.5 mm).
    
    Retorna:
        float: Volumen estimado en mm³.
    """
    area_pixeles = cv2.countNonZero(mascara)
    
    volumen = area_pixeles * area_pixel * espesor_corte
    
    return float(volumen)
