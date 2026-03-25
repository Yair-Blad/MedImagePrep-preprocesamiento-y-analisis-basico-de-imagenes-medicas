"""
MedImagePrep - Utilidades para procesamiento de imágenes médicas.
"""

from .io import cargar_imagen, cargar_dicom, detectar_formato
from .preprocessing import recortar, redimensionar, normalizar, filtro_gaussiano, filtro_bilateral, filtro_mediano
from .edges import detectar_sobel, detectar_canny, detectar_laplacian
from .histogram import calcular_histograma, ecualizar_histograma, estadisticas_histograma
from .segmentation import segmentar_watershed, crear_marcadores_distancia, umbralizar_otsu
from .quantification import porcentaje_alta_intensidad, medir_region, analisis_textura
from .visualization import mostrar_comparacion, guardar_imagen, plot_histograma, generar_reporte_pdf

__all__ = [
    'cargar_imagen',
    'cargar_dicom',
    'detectar_formato',
    'recortar',
    'redimensionar',
    'normalizar',
    'filtro_gaussiano',
    'filtro_bilateral',
    'filtro_mediano',
    'detectar_sobel',
    'detectar_canny',
    'detectar_laplacian',
    'calcular_histograma',
    'ecualizar_histograma',
    'estadisticas_histograma',
    'segmentar_watershed',
    'crear_marcadores_distancia',
    'umbralizar_otsu',
    'porcentaje_alta_intensidad',
    'medir_region',
    'analisis_textura',
    'mostrar_comparacion',
    'guardar_imagen',
    'plot_histograma',
    'generar_reporte_pdf'
]
