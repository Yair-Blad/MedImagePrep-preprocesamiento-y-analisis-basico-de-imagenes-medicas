"""
Módulo de visualización para imágenes médicas.

Funciones: mostrar comparaciones, guardar imágenes, plot de histogramas,
generación de reportes PDF.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import datetime


def mostrar_comparacion(imagen_original: np.ndarray, imagen_procesada: np.ndarray,
                        titulo_original: str = "Original",
                        titulo_procesada: str = "Procesada",
                        figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Muestra dos imágenes lado a lado para comparación visual.
    
    Parámetros:
        imagen_original: Imagen original sin procesar.
        imagen_procesada: Imagen después del procesamiento.
        titulo_original: Título para la imagen original.
        titulo_procesada: Título para la imagen procesada.
        figsize: Tamaño de la figura en pulgadas (ancho, alto).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(imagen_original, cmap='gray')
    axes[0].set_title(titulo_original)
    axes[0].axis('off')
    
    axes[1].imshow(imagen_procesada, cmap='gray')
    axes[1].set_title(titulo_procesada)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def mostrar_grid(imagenes: List[np.ndarray], titulos: List[str],
                 cols: int = 2, figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Muestra múltiples imágenes en un grid para comparación.
    
    Parámetros:
        imagenes: Lista de imágenes para mostrar.
        titulos: Lista de títulos para cada imagen.
        cols: Número de columnas en el grid.
        figsize: Tamaño de la figura.
    """
    rows = int(np.ceil(len(imagenes) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(imagenes):
            ax.imshow(imagenes[idx], cmap='gray')
            ax.set_title(titulos[idx])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def guardar_imagen(imagen: np.ndarray, ruta_salida: str,
                   formato: str = 'png', calidad: int = 95) -> str:
    """
    Guarda una imagen procesada en disco.
    
    Parámetros:
        imagen: Imagen a guardar.
        ruta_salida: Ruta de salida (archivo o directorio).
        formato: Formato de salida ('png', 'jpg', 'tiff').
        calidad: Calidad para JPG (1-100).
    
    Retorna:
        str: Ruta completa del archivo guardado.
    """
    ruta_path = Path(ruta_salida)
    
    if ruta_path.suffix == '' or ruta_path.is_dir():
        nombre_archivo = f"imagen_procesada_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{formato}"
        ruta_path = ruta_path / nombre_archivo
    
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    
    if formato.lower() in ['png', 'tiff']:
        cv2.imwrite(str(ruta_path), imagen)
    elif formato.lower() == 'jpg':
        cv2.imwrite(str(ruta_path), imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    
    return str(ruta_path)


def plot_histograma(imagen: np.ndarray, ruta_guardado: Optional[str] = None,
                    titulo: str = "Histograma de Intensidades",
                    mostrar_grid: bool = True) -> Figure:
    """
    Genera y muestra/graba el histograma de intensidades de una imagen.
    
    Parámetros:
        imagen: Imagen de entrada.
        ruta_guardado: Ruta para guardar la figura. Si None, solo muestra.
        titulo: Título del gráfico.
        mostrar_grid: Si True, muestra grilla en el plot.
    
    Retorna:
        Figure: Objeto matplotlib de la figura.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(histograma, color='blue', linewidth=2, label='Frecuencia')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Intensidad de píxeles (0-255)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_xlim([0, 256])
    ax.grid(mostrar_grid, alpha=0.3)
    ax.legend()
    
    media = np.mean(imagen)
    desviacion = np.std(imagen)
    
    stats_text = f'Media: {media:.2f}\nDesviación: {desviacion:.2f}'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if ruta_guardado:
        ruta_path = Path(ruta_guardado)
        ruta_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta_path, dpi=300, bbox_inches='tight')
    
    return fig


def generar_comparacion_histograma(imagen_original: np.ndarray,
                                    imagen_procesada: np.ndarray,
                                    ruta_guardado: Optional[str] = None) -> Figure:
    """
    Genera un gráfico comparativo de histogramas antes/después.
    
    Parámetros:
        imagen_original: Imagen sin procesar.
        imagen_procesada: Imagen procesada.
        ruta_guardado: Ruta para guardar la figura.
    
    Retorna:
        Figure: Objeto matplotlib de la figura.
    """
    hist_orig = cv2.calcHist([imagen_original], [0], None, [256], [0, 256])
    hist_proc = cv2.calcHist([imagen_procesada], [0], None, [256], [0, 256])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(hist_orig, color='blue', linewidth=2, label='Original', alpha=0.7)
    ax.plot(hist_proc, color='red', linewidth=2, label='Procesada', alpha=0.7)
    
    ax.set_title('Comparación de Histogramas: Antes vs Después', fontsize=14, fontweight='bold')
    ax.set_xlabel('Intensidad de píxeles (0-255)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_xlim([0, 256])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if ruta_guardado:
        ruta_path = Path(ruta_guardado)
        ruta_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta_path, dpi=300, bbox_inches='tight')
    
    return fig


def generar_reporte_pdf(ruta_salida: str, metricas: Dict[str, any],
                        imagenes_rutas: List[str],
                        titulo: str = "Reporte de Análisis de Imagen Médica",
                        paciente_id: str = "ANONIMO",
                        comentario: str = "") -> str:
    """
    Genera un reporte PDF profesional con imágenes, histogramas y métricas.
    
    Parámetros:
        ruta_salida: Ruta de salida para el archivo PDF.
        metricas: Diccionario con métricas calculadas (media, std, porcentajes, etc.).
        imagenes_rutas: Lista de rutas a imágenes para incluir en el reporte.
        titulo: Título principal del reporte.
        paciente_id: ID del paciente (anonimizado).
        comentario: Comentario o observaciones adicionales.
    
    Retorna:
        str: Ruta del PDF generado.
    """
    ruta_path = Path(ruta_salida)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        str(ruta_path),
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    elementos = []
    
    estilos = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        'TituloCustom',
        parent=estilos['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    estilo_subtitulo = ParagraphStyle(
        'SubtituloCustom',
        parent=estilos['Heading2'],
        fontSize=12,
        textColor=colors.darkgrey,
        spaceAfter=6
    )
    
    estilo_normal = ParagraphStyle(
        'NormalCustom',
        parent=estilos['Normal'],
        fontSize=10,
        leading=12
    )
    
    titulo_reporte = Paragraph(titulo, estilo_titulo)
    elementos.append(titulo_reporte)
    
    fecha_actual = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_metadata = Paragraph(f"<b>Fecha:</b> {fecha_actual} | <b>Paciente ID:</b> {paciente_id}", estilo_normal)
    elementos.append(info_metadata)
    elementos.append(Spacer(1, 0.2*inch))
    
    elementos.append(Paragraph("MÉTRICAS CUANTITATIVAS", estilo_subtitulo))
    
    datos_metricas = [['Métrica', 'Valor']]
    
    for clave, valor in metricas.items():
        clave_formateada = clave.replace('_', ' ').title()
        
        if isinstance(valor, float):
            valor_formateado = f"{valor:.4f}"
        else:
            valor_formateado = str(valor)
        
        datos_metricas.append([clave_formateada, valor_formateado])
    
    tabla_metricas = Table(datos_metricas, colWidths=[2.5*inch, 2.5*inch])
    tabla_metricas.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elementos.append(tabla_metricas)
    elementos.append(Spacer(1, 0.3*inch))
    
    if comentario:
        elementos.append(Paragraph("OBSERVACIONES", estilo_subtitulo))
        elementos.append(Paragraph(comentario, estilo_normal))
        elementos.append(Spacer(1, 0.2*inch))
    
    elementos.append(Paragraph("IMÁGENES DEL ANÁLISIS", estilo_subtitulo))
    
    for ruta_imagen in imagenes_rutas:
        if Path(ruta_imagen).exists():
            img = Image(str(ruta_imagen), width=4*inch, height=3*inch)
            elementos.append(img)
            elementos.append(Spacer(1, 0.1*inch))
            elementos.append(Paragraph(f"<i>{Path(ruta_imagen).name}</i>", estilo_normal))
            elementos.append(Spacer(1, 0.2*inch))
    
    elementos.append(Spacer(1, 0.3*inch))
    elementos.append(Paragraph("=" * 50, estilo_normal))
    elementos.append(Paragraph("<i>Reporte generado con MedImagePrep v1.0</i>", estilo_normal))
    
    doc.build(elementos)
    
    return str(ruta_path)


def crear_mosaico_comparacion(imagenes: List[np.ndarray],
                               titulos: List[str],
                               ruta_salida: str,
                               cols: int = 2) -> str:
    """
    Crea un mosaico de imágenes comparativas y lo guarda en disco.
    
    Parámetros:
        imagenes: Lista de imágenes para el mosaico.
        titulos: Lista de títulos para cada imagen.
        ruta_salida: Ruta de salida para el mosaico.
        cols: Número de columnas.
    
    Retorna:
        str: Ruta del archivo guardado.
    """
    rows = int(np.ceil(len(imagenes) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(imagenes):
            ax.imshow(imagenes[idx], cmap='gray')
            ax.set_title(titulos[idx], fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    ruta_path = Path(ruta_salida)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(str(ruta_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return str(ruta_path)
