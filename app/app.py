"""
MedImagePrep - Aplicación Web Interactiva con Streamlit.

Permite cargar imágenes médicas, aplicar preprocesamiento,
visualizar resultados y descargar reportes.

Uso:
    streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tempfile
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import cargar_imagen, detectar_formato
from src.utils.preprocessing import (
    redimensionar,
    filtro_bilateral,
    filtro_gaussiano,
    filtro_mediano,
    aplicar_clahe
)
from src.utils.histogram import (
    ecualizar_histograma,
    estadisticas_histograma
)
from src.utils.edges import detectar_canny
from src.utils.segmentation import (
    segmentar_watershed,
    umbralizar_otsu,
    crear_marcadores_distancia
)
from src.utils.quantification import (
    porcentaje_alta_intensidad,
    medir_region,
    analisis_textura
)
from src.utils.visualization import (
    guardar_imagen,
    generar_reporte_pdf
)


def configurar_pagina():
    """Configura la página de Streamlit."""
    st.set_page_config(
        page_title="MedImagePrep - Análisis de Imágenes Médicas",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def mostrar_encabezado():
    """Muestra el encabezado de la aplicación."""
    st.title("🏥 MedImagePrep")
    st.markdown("### Preprocesamiento y Análisis de Imágenes Médicas")
    st.markdown("---")


def cargar_archivo_imagen():
    """Permite al usuario cargar una imagen médica."""
    st.sidebar.header("📁 Cargar Imagen")
    
    archivo = st.sidebar.file_uploader(
        "Seleccione una imagen médica",
        type=['png', 'jpg', 'jpeg', 'tiff', 'dcm'],
        help="Formatos soportados: PNG, JPG, TIFF, DICOM"
    )
    
    return archivo


def mostrar_controles_preprocesamiento():
    """Muestra controles deslizantes para parámetros de procesamiento."""
    st.sidebar.header("⚙️ Parámetros de Procesamiento")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        sigma_bilateral = st.slider(
            "Sigma Bilateral",
            min_value=10,
            max_value=150,
            value=75,
            help="Controla la preservación de bordes"
        )
        clip_limit = st.slider(
            "CLAHE Clip Limit",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
    
    with col2:
        kernel_gaussiano = st.slider(
            "Kernel Gaussiano",
            min_value=3,
            max_value=15,
            value=5,
            step=2
        )
        umbral_canny = st.slider(
            "Umbral Canny",
            min_value=30,
            max_value=200,
            value=100
        )
    
    umbral_cuantificacion = st.sidebar.slider(
        "Umbral para Cuantificación (%)",
        min_value=50,
        max_value=255,
        value=200,
        help="Píxeles above este valor se consideran alta intensidad"
    )
    
    return {
        'sigma_bilateral': sigma_bilateral,
        'clip_limit': clip_limit,
        'kernel_gaussiano': kernel_gaussiano,
        'umbral_canny': umbral_canny,
        'umbral_cuantificacion': umbral_cuantificacion / 255.0
    }


def procesar_imagen(imagen, params):
    """
    Procesa la imagen con los parámetros especificados.
    
    Retorna:
        dict: Imágenes procesadas y métricas.
    """
    imagen_suavizada = filtro_bilateral(
        imagen,
        d=9,
        sigmaColor=params['sigma_bilateral'],
        sigmaSpace=params['sigma_bilateral']
    )
    
    imagen_clahe = aplicar_clahe(
        imagen_suavizada,
        clip_limit=params['clip_limit'],
        tile_grid_size=(8, 8)
    )
    
    imagen_ecualizada = ecualizar_histograma(imagen_clahe)
    
    bordes = detectar_canny(
        imagen_ecualizada,
        umbral1=params['umbral_canny'] - 20,
        umbral2=params['umbral_canny'] + 50
    )
    
    mascara_otsu, umbral_otsu = umbralizar_otsu(imagen_ecualizada)
    marcadores = crear_marcadores_distancia(mascara_otsu)
    segmentacion, contornos = segmentar_watershed(imagen_ecualizada, marcadores)
    
    porcentaje = porcentaje_alta_intensidad(
        imagen_ecualizada,
        umbral=params['umbral_cuantificacion']
    )
    
    estadisticas = estadisticas_histograma(imagen_ecualizada)
    textura = analisis_textura(imagen_ecualizada)
    region_metrics = medir_region(mascara_otsu, imagen_ecualizada)
    
    metricas = {
        'porcentaje_alta_intensidad': porcentaje,
        'media_intensidad': estadisticas['media'],
        'desviacion_std': estadisticas['desviacion_std'],
        'mediana': estadisticas['mediana'],
        'contraste_textura': textura['contraste'],
        'homogeneidad': textura['homogeneidad'],
        'area_lesion_pixeles': region_metrics['area_pixeles'],
        'perimetro': region_metrics['perimetro'],
        'circularidad': region_metrics['circularidad'],
        'umbral_otsu': float(umbral_otsu)
    }
    
    return {
        'original': imagen,
        'suavizada': imagen_suavizada,
        'clahe': imagen_clahe,
        'ecualizada': imagen_ecualizada,
        'bordes': bordes,
        'segmentacion': contornos,
        'metricas': metricas
    }


def mostrar_metricas(metricas):
    """Muestra las métricas cuantitativas en la interfaz."""
    st.header("📊 Métricas Cuantitativas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Alta Intensidad",
            f"{metricas['porcentaje_alta_intensidad']:.2f}%"
        )
        st.metric(
            "Media",
            f"{metricas['media_intensidad']:.2f}"
        )
    
    with col2:
        st.metric(
            "Desviación Std",
            f"{metricas['desviacion_std']:.2f}"
        )
        st.metric(
            "Mediana",
            f"{metricas['mediana']:.2f}"
        )
    
    with col3:
        st.metric(
            "Contraste Textura",
            f"{metricas['contraste_textura']:.4f}"
        )
        st.metric(
            "Homogeneidad",
            f"{metricas['homogeneidad']:.4f}"
        )
    
    with col4:
        st.metric(
            "Área Lesión",
            f"{metricas['area_lesion_pixeles']:.0f} px"
        )
        st.metric(
            "Circularidad",
            f"{metricas['circularidad']:.4f}"
        )


def mostrar_imagenes(resultados):
    """Muestra las imágenes procesadas en grid."""
    st.header("🔍 Visualización de Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(resultados['original'], cmap='gray', width=300)
    
    with col2:
        st.subheader("Imagen Ecualizada")
        st.image(resultados['ecualizada'], cmap='gray', width=300)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Bordes (Canny)")
        st.image(resultados['bordes'], cmap='gray', width=300)
    
    with col4:
        st.subheader("Segmentación (Watershed)")
        st.image(resultados['segmentacion'], cmap='gray', width=300)


def generar_y_ofrecer_descarga(resultados, imagen_original):
    """Genera reporte PDF y ofrece descarga."""
    st.header("📄 Reporte PDF")
    
    if st.button("Generar Reporte PDF", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            ruta_original = guardar_imagen(
                resultados['original'],
                str(tmpdir_path / "original.png")
            )
            ruta_ecualizada = guardar_imagen(
                resultados['ecualizada'],
                str(tmpdir_path / "ecualizada.png")
            )
            ruta_bordes = guardar_imagen(
                resultados['bordes'],
                str(tmpdir_path / "bordes.png")
            )
            ruta_segmentacion = guardar_imagen(
                resultados['segmentacion'],
                str(tmpdir_path / "segmentacion.png")
            )
            
            imagenes_rutas = [
                ruta_original,
                ruta_ecualizada,
                ruta_bordes,
                ruta_segmentacion
            ]
            
            ruta_pdf = generar_reporte_pdf(
                ruta_salida=str(tmpdir_path / "reporte_medico.pdf"),
                metricas=resultados['metricas'],
                imagenes_rutas=imagenes_rutas,
                paciente_id="ANONIMO",
                comentario="Análisis generado con MedImagePrep"
            )
            
            with open(ruta_pdf, "rb") as pdf_file:
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_file.read(),
                    file_name=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )


def main():
    """Función principal de la aplicación Streamlit."""
    configurar_pagina()
    mostrar_encabezado()
    
    archivo = cargar_archivo_imagen()
    
    if archivo is not None:
        params = mostrar_controles_preprocesamiento()
        
        with st.spinner("Procesando imagen..."):
            imagen = cargar_imagen(str(archivo))
            resultados = procesar_imagen(imagen, params)
        
        st.success("¡Procesamiento completado!")
        
        mostrar_metricas(resultados['metricas'])
        mostrar_imagenes(resultados)
        generar_y_ofrecer_descarga(resultados, imagen)
        
        st.markdown("---")
        st.markdown("**Nota:** Esta herramienta es para investigación y desarrollo. "
                   "No usar para diagnóstico clínico sin validación apropiada.")
    else:
        st.info("👆 Por favor cargue una imagen médica para comenzar el análisis.")
        
        st.markdown("""
        ### Características de MedImagePrep:
        
        - ✅ Carga de PNG, JPG, TIFF y DICOM
        - ✅ Filtro bilateral para preservar bordes
        - ✅ CLAHE para mejora de contraste local
        - ✅ Ecualización de histograma
        - ✅ Detección de bordes Canny
        - ✅ Segmentación Watershed con marcadores automáticos
        - ✅ Cuantificación de regiones de alta intensidad
        - ✅ Análisis de textura GLCM
        - ✅ Generación de reportes PDF
        
        ### Casos de uso:
        
        - **CT de pulmón post-COVID**: Detección de opacidades en vidrio esmerilado
        - **Radiografías óseas**: Detección de lesiones y tumores
        - **Investigación médica**: Análisis cuantitativo de imágenes
        """)


if __name__ == "__main__":
    main()
