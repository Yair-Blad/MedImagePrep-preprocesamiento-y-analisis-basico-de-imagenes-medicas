"""
Módulo de detección de bordes para imágenes médicas.

Implementa: Sobel, Canny, Laplacian, y detección de bordes multidireccional.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def detectar_sobel(imagen: np.ndarray, dx: int = 1, dy: int = 0, 
                   kernel_size: int = 3, escala: float = 1.0) -> np.ndarray:
    """
    Calcula el gradiente de Sobel en dirección X, Y o combinada.
    
    Parámetros:
        imagen: Imagen de entrada (escala de grises).
        dx: Orden de derivada en dirección X (0, 1, o 2).
        dy: Orden de derivada en dirección Y (0, 1, o 2).
        kernel_size: Tamaño del kernel de Sobel (1, 3, 5, o 7).
        escala: Factor de escala para el resultado.
    
    Retorna:
        np.ndarray: Magnitud del gradiente de Sobel.
    """
    if dx == 0 and dy == 0:
        raise ValueError("Al menos una de dx o dy debe ser diferente de 0.")
    
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, dx, dy, ksize=kernel_size)
    
    magnitud = np.sqrt(sobel_x ** 2)
    
    magnitud_normalizada = cv2.normalize(
        magnitud,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )
    
    return magnitud_normalizada.astype(np.uint8)


def detectar_sobel_xy(imagen: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula gradientes de Sobel en X, Y y combina ambas direcciones.
    
    Parámetros:
        imagen: Imagen de entrada.
        kernel_size: Tamaño del kernel.
    
    Retorna:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Gradiente en dirección X
            - Gradiente en dirección Y
            - Magnitud combinada
    """
    grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    magnitud = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    magnitud_norm = cv2.normalize(
        magnitud,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    grad_x_norm = cv2.normalize(
        grad_x,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    grad_y_norm = cv2.normalize(
        grad_y,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    return grad_x_norm, grad_y_norm, magnitud_norm


def detectar_canny(imagen: np.ndarray, umbral1: int = 50, 
                   umbral2: int = 150, aperture: int = 3) -> np.ndarray:
    """
    Aplica el detector de bordes de Canny.
    
    Detector de bordes óptimo con supresión de no-máximos y seguimiento
    por histéresis.

    Parámetros:
        imagen: Imagen de entrada.
        umbral1: Umbral inferior para histéresis.
        umbral2: Umbral superior para histéresis.
        aperture: Apertura para operador Sobel interno.
    
    Retorna:
        np.ndarray: Mapa de bordes binario.
    """
    return cv2.Canny(imagen, umbral1, umbral2, apertureSize=aperture)


def detectar_laplacian(imagen: np.ndarray, kernel_size: int = 3, 
                       escala: float = 1.0) -> np.ndarray:
    """
    Calcula el Laplaciano de la imagen para detección de bordes.
    
    El Laplaciano es isótropo (no direccional) y detecta bordes
    en todas las direcciones simultáneamente.
    
    Parámetros:
        imagen: Imagen de entrada.
        kernel_size: Tamaño del kernel (debe ser impar).
        escala: Factor de escala.
    
    Retorna:
        np.ndarray: Laplaciano normalizado.
    """
    if kernel_size % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")
    
    laplaciano = cv2.Laplacian(imagen, cv2.CV_64F, ksize=kernel_size)
    
    laplaciano_abs = cv2.convertScaleAbs(laplaciano)
    
    return laplaciano_abs


def detectar_bordes_multidireccional(imagen: np.ndarray, 
                                     metodo: str = 'sobel') -> np.ndarray:
    """
    Detecta bordes combinando múltiples direcciones.
    
    Parámetros:
        imagen: Imagen de entrada.
        metodo: Método a usar ('sobel', 'canny', 'laplacian').
    
    Retorna:
        np.ndarray: Imagen con bordes realzados.
    """
    if metodo == 'sobel':
        _, _, magnitud = detectar_sobel_xy(imagen)
        return magnitud
    
    elif metodo == 'canny':
        return detectar_canny(imagen)
    
    elif metodo == 'laplacian':
        return detectar_laplacian(imagen)
    
    else:
        raise ValueError(f"Método no reconocido: {metodo}. Use 'sobel', 'canny' o 'laplacian'.")


def realzar_bordes(imagen: np.ndarray, peso_bordes: float = 0.5) -> np.ndarray:
    """
    Realza los bordes de una imagen sumando el mapa de bordes
    a la imagen original.
    
    Parámetros:
        imagen: Imagen original.
        peso_bordes: Peso de los bordes en la suma (0 a 1).
    
    Retorna:
        np.ndarray: Imagen con bordes realzados.
    """
    bordes = detectar_laplacian(imagen)
    
    imagen_realzada = cv2.addWeighted(
        imagen,
        1.0,
        bordes,
        peso_bordes,
        0
    )
    
    return imagen_realzada
