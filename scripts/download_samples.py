"""
Script para descargar imágenes médicas de muestra desde datasets públicos.

Uso:
    python scripts/download_samples.py

Datasets incluidos:
- COVID-19 CT scans (MosMedData)
- Chest X-Ray samples (NIH, RSNA)
- Bone X-Ray samples (MURA)
"""

import requests
from pathlib import Path
import zipfile
import io

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def descargar_covid_ct_sample():
    """
    Descarga una muestra de CT de pulmón COVID-19.
    Fuente: MosMedData (https://mosmed.ai/)
    """
    print("\n[INFO] Descargando muestra COVID-19 CT...")
    
    url = "https://mosmed.ai/api/v1/covid19_1110/download"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        ruta_salida = OUTPUT_DIR / "covid_ct_sample.zip"
        
        with open(ruta_salida, 'wb') as f:
            f.write(response.content)
        
        print(f"[OK] Descargado: {ruta_salida}")
        
        with zipfile.ZipFile(ruta_salida, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        
        print("[OK] Extraído correctamente")
        
    except Exception as e:
        print(f"[ERROR] No se pudo descargar: {e}")


def descargar_chest_xray_sample():
    """
    Descarga una muestra de radiografía de tórax.
    Fuente: NIH Chest X-Ray Dataset samples
    """
    print("\n[INFO] Descargando muestra Chest X-Ray...")
    
    urls = [
        "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "https://www.rsna.org/education/ai-resources-and-training/ai-image-research-program/cxr-mnist"
    ]
    
    print("[INFO] Visite los siguientes enlaces para descargar manualmente:")
    for url in urls:
        print(f"  - {url}")


def descargar_muestras_kaggle():
    """
    Proporciona enlaces a datasets de Kaggle.
    Requiere cuenta de Kaggle y aceptación de términos.
    """
    print("\n[INFO] Datasets de Kaggle disponibles:")
    
    datasets = {
        "COVID-19 CT Lesion Segmentation":
            "https://www.kaggle.com/datasets/arnowang/covid19-ct-lesion-segmentation",
        "Bone Cancer Detection":
            "https://www.kaggle.com/datasets/todasdasd/bone-cancer-detection",
        "Chest X-Ray Pneumonia":
            "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "MURA Bone X-Ray":
            "https://www.kaggle.com/datasets/aryashah2k/mura-bone-xray-dataset"
    }
    
    print("\nPara descargar desde Kaggle (requiere cuenta):")
    print("  pip install kaggle")
    print("  kaggle datasets download -d arnowang/covid19-ct-lesion-segmentation")
    
    for nombre, url in datasets.items():
        print(f"\n  {nombre}:")
        print(f"    {url}")


def generar_imagen_prueba_sintetica():
    """
    Genera una imagen sintética de prueba para desarrollo.
    Simula una radiografía con estructuras anatómicas básicas.
    """
    print("\n[INFO] Generando imagen de prueba sintética...")
    
    import numpy as np
    import cv2
    
    imagen = np.zeros((512, 512), dtype=np.uint8)
    
    fondo = np.random.randint(20, 40, size=(512, 512), dtype=np.uint8)
    
    for i in range(5):
        centro_x = np.random.randint(100, 400)
        centro_y = np.random.randint(100, 400)
        radio = np.random.randint(20, 60)
        intensidad = np.random.randint(150, 255)
        
        cv2.circle(imagen, (centro_x, centro_y), radio, intensidad, -1)
        cv2.circle(imagen, (centro_x, centro_y), radio, intensidad - 50, 2)
    
    imagen = cv2.addWeighted(imagen, 0.7, fondo, 0.3, 0)
    
    imagen = cv2.GaussianBlur(imagen, (5, 5), 1.5)
    
    ruta_salida = OUTPUT_DIR / "test_sintetico.png"
    cv2.imwrite(str(ruta_salida), imagen)
    
    print(f"[OK] Imagen sintética guardada: {ruta_salida}")
    
    return ruta_salida


def main():
    """Función principal para descargar muestras."""
    print("=" * 60)
    print("MedImagePrep - Descargador de Imágenes de Muestra")
    print("=" * 60)
    
    print("\n[INFO] Directorio de salida:", OUTPUT_DIR)
    
    generar_imagen_prueba_sintetica()
    
    print("\n" + "=" * 60)
    print("RECURSOS PARA IMÁGENES MÉDICAS REALES")
    print("=" * 60)
    
    descargar_muestras_kaggle()
    
    print("\n" + "=" * 60)
    print("TCIA (The Cancer Imaging Archive)")
    print("=" * 60)
    
    tcia_collections = {
        "Lung collections": "https://www.cancerimagingarchive.net/collection/",
        "NSCLC-Radiomics": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
        "LIDC-IDRI (Lung CT)": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "Bone collections": "https://www.cancerimagingarchive.net/collection/"
    }
    
    for nombre, url in tcia_collections.items():
        print(f"\n  {nombre}: {url}")
    
    print("\n" + "=" * 60)
    print("¡Configuración completada!")
    print("=" * 60)
    print("\nPara ejecutar MedImagePrep con las imágenes de prueba:")
    print("  cd MedImagePrep")
    print("  python src/main.py --image data/samples/test_sintetico.png --mode cli --verbose")


if __name__ == "__main__":
    main()
