"""
Módulo de entrada/salida para carga de imágenes médicas.

Soporta formatos: PNG, JPG, TIFF y DICOM (estándar médico).
"""

import cv2
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


def detectar_formato(ruta: str) -> str:
    """
    Detecta el formato de un archivo de imagen según su extensión.
    
    Parámetros:
        ruta: Ruta absoluta o relativa al archivo de imagen.
    
    Retorna:
        str: Formato detectado ('png', 'jpg', 'dcm', 'nii', 'desconocido').
    """
    extension = Path(ruta).suffix.lower()
    
    formatos = {
        '.png': 'png',
        '.jpg': 'jpg',
        '.jpeg': 'jpg',
        '.tiff': 'tiff',
        '.tif': 'tiff',
        '.dcm': 'dcm',
        '.dicom': 'dcm',
        '.nii': 'nii',
        '.nii.gz': 'nii'
    }
    
    return formatos.get(extension, 'desconocido')


def cargar_imagen(ruta: str, convertir_a_gris: bool = True) -> np.ndarray:
    """
    Carga una imagen en formato PNG, JPG o TIFF.
    
    Parámetros:
        ruta: Ruta al archivo de imagen.
        convertir_a_gris: Si True, convierte la imagen a escala de grises.
    
    Retorna:
        np.ndarray: Imagen cargada como array de numpy.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado.
    """
    ruta_path = Path(ruta)
    
    if not ruta_path.exists():
        raise FileNotFoundError(f"La imagen no existe en: {ruta}")
    
    formato = detectar_formato(ruta)
    
    if formato in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
        if convertir_a_gris:
            imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        else:
            imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        
        return imagen
    
    elif formato == 'dcm':
        return cargar_dicom(ruta)
    
    else:
        raise ValueError(f"Formato no soportado: {formato}. Use PNG, JPG, TIFF o DICOM.")


def cargar_dicom(ruta: str) -> np.ndarray:
    """
    Carga un archivo DICOM (.dcm) y extrae la imagen pixelada.
    
    Parámetros:
        ruta: Ruta al archivo DICOM.
    
    Retorna:
        np.ndarray: Imagen DICOM normalizada a [0, 255].
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si hay error al leer DICOM.
    """
    ruta_path = Path(ruta)
    
    if not ruta_path.exists():
        raise FileNotFoundError(f"El archivo DICOM no existe en: {ruta}")
    
    try:
        dicom_data = pydicom.dcmread(str(ruta))
        
        if hasattr(dicom_data, 'pixel_array'):
            imagen = dicom_data.pixel_array
        else:
            raise ValueError("El archivo DICOM no contiene datos de imagen (pixel_array).")
        
        imagen = normalizar_dicar(imagen)
        
        return imagen.astype(np.uint8)
    
    except Exception as e:
        raise Exception(f"Error al leer DICOM: {str(e)}")


def normalizar_dicar(imagen: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen DICOM a rango [0, 255].
    
    Parámetros:
        imagen: Array de numpy con datos DICOM (pueden ser int16 con valores negativos).
    
    Retorna:
        np.ndarray: Imagen normalizada a [0, 255].
    """
    if imagen.min() == imagen.max():
        return np.zeros_like(imagen, dtype=np.uint8)
    
    imagen_normalizada = cv2.normalize(
        imagen,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )
    
    return imagen_normalizada


def guardar_imagen_dicom(imagen: np.ndarray, ruta_salida: str, 
                         paciente_id: str = "ANONIMO",
                         estudio_desc: str = "MedImagePrep") -> None:
    """
    Guarda una imagen como archivo DICOM con metadatos básicos.
    
    Parámetros:
        imagen: Imagen a guardar (numpy array).
        ruta_salida: Ruta de salida para el archivo .dcm.
        paciente_id: ID del paciente (anonimizado por defectecto).
        estudio_desc: Descripción del estudio.
    """
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import datetime
    
    fecha_actual = datetime.datetime.now().strftime('%Y%m%d')
    
    dicom = FileDataset(ruta_salida, {}, 
                       file_meta_type=ExplicitVRLittleEndian,
                       preamble=b"\x00" * 128)
    
    dicom.PatientID = paciente_id
    dicom.StudyDescription = estudio_desc
    dicom.Modality = "OT"
    dicom.StudyDate = fecha_actual
    dicom.PixelData = imagen.tobytes()
    dicom.Rows, dicom.Columns = imagen.shape
    
    dicom.save_as(ruta_salida)
