"""
Generador de imágenes médicas sintéticas para pruebas.

Crea imágenes que simulan:
- Radiografías de tórax
- CT de pulmón con opacidades
- Radiografías óseas con lesiones

Uso:
    python scripts/generate_test_images.py
"""

import numpy as np
import cv2
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generar_fondo_radiografia():
    """Genera fondo similar a radiografía de tórax."""
    fondo = np.zeros((512, 512), dtype=np.uint8)
    
    fondo[:] = np.random.randint(10, 30, size=(512, 512), dtype=np.uint8)
    
    for i in range(200):
        x = np.random.randint(0, 512)
        y = np.random.randint(0, 512)
        radio = np.random.randint(1, 3)
        intensidad = np.random.randint(30, 60)
        cv2.circle(fondo, (x, y), radio, intensidad, -1)
    
    return fondo


def generar_costillas(fondo):
    """Añade estructuras similares a costillas."""
    imagen = fondo.copy()
    
    for i in range(12):
        y_base = 100 + i * 35
        
        puntos = []
        for x in range(50, 460, 10):
            y = y_base + int(15 * np.sin(x / 50.0))
            puntos.append([x, y])
        
        puntos = np.array(puntos, dtype=np.int32)
        cv2.polylines(imagen, [puntos], False, 80, 2)
    
    return imagen


def generar_pulmones(fondo):
    """Genera campos pulmonares simulados."""
    imagen = fondo.copy()
    
    cv2.ellipse(imagen, (200, 280), (100, 180), 0, 0, 360, 50, -1)
    cv2.ellipse(imagen, (312, 280), (100, 180), 0, 0, 360, 50, -1)
    
    return imagen


def generar_opacidades_vidrio_esmerilado(imagen, num_opacidades=5):
    """
    Añade opacidades simulando vidrio esmerilado (COVID-19).
    """
    imagen_resultado = imagen.copy()
    
    for _ in range(num_opacidades):
        centro_x = np.random.randint(150, 360)
        centro_y = np.random.randint(200, 400)
        ejes = (np.random.randint(30, 80), np.random.randint(20, 50))
        angulo = np.random.randint(0, 90)
        
        overlay = imagen_resultado.copy()
        cv2.ellipse(overlay, (centro_x, centro_y), ejes, angulo, 0, 360, 120, -1)
        
        alpha = np.random.uniform(0.3, 0.6)
        cv2.addWeighted(overlay, alpha, imagen_resultado, 1 - alpha, 0, imagen_resultado)
    
    return imagen_resultado


def generar_lesion_osea(imagen, ubicacion='humero'):
    """
    Genera lesión ósea simulada (tumor/lesión).
    """
    imagen_resultado = imagen.copy()
    
    if ubicacion == 'humero':
        centro = (256, 150)
        radio = np.random.randint(15, 35)
    else:
        centro = (np.random.randint(150, 360), np.random.randint(200, 400))
        radio = np.random.randint(10, 30)
    
    overlay = imagen_resultado.copy()
    cv2.circle(overlay, centro, radio, 180, -1)
    
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, imagen_resultado, 1 - alpha, 0, imagen_resultado)
    
    cv2.circle(imagen_resultado, centro, radio, 200, 2)
    
    return imagen_resultado


def generar_corazon(fondo):
    """Añade silueta cardíaca simulada."""
    imagen = fondo.copy()
    
    puntos = []
    for angulo in np.linspace(0, 2*np.pi, 100):
        x = 256 + int(60 * np.cos(angulo))
        y = 300 + int(80 * np.sin(angulo))
        if y > 280:
            puntos.append([int(x), int(y)])
    
    puntos = np.array(puntos, dtype=np.int32)
    cv2.fillPoly(imagen, [puntos], 70)
    
    return imagen


def generar_radiografia_torax():
    """Genera radiografía de tórax sintética completa."""
    print("[INFO] Generando radiografía de tórax...")
    
    fondo = generar_fondo_radiografia()
    fondo = generar_costillas(fondo)
    fondo = generar_pulmones(fondo)
    fondo = generar_corazon(fondo)
    
    fondo = cv2.GaussianBlur(fondo, (7, 7), 2.0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fondo = clahe.apply(fondo)
    
    return fondo


def generar_ct_pulmon_covid():
    """Genera CT axial de pulmón con opacidades COVID."""
    print("[INFO] Generando CT pulmón COVID-19...")
    
    base = np.zeros((512, 512), dtype=np.uint8)
    base[:] = 40
    
    cv2.circle(base, (256, 256), 200, 60, -1)
    
    base = generar_opacidades_vidrio_esmerilado(base, num_opacidades=7)
    
    base = cv2.GaussianBlur(base, (5, 5), 1.5)
    
    return base


def generar_radiografia_osea():
    """Genera radiografía ósea con lesión simulada."""
    print("[INFO] Generando radiografía ósea...")
    
    base = np.zeros((512, 512), dtype=np.uint8)
    base[:] = np.random.randint(80, 120, size=(512, 512), dtype=np.uint8)
    
    cv2.rectangle(base, (200, 50), (312, 460), 140, -1)
    
    for i in range(50):
        x = np.random.randint(200, 312)
        y = np.random.randint(50, 460)
        radio = np.random.randint(1, 3)
        intensidad = np.random.randint(130, 160)
        cv2.circle(base, (x, y), radio, intensidad, -1)
    
    base = generar_lesion_osea(base, ubicacion='humero')
    
    base = cv2.GaussianBlur(base, (5, 5), 1.0)
    
    return base


def generar_fantasma_uniformidad():
    """Genera fantasma de uniformidad para calibración."""
    print("[INFO] Generando fantasma de uniformidad...")
    
    base = np.zeros((512, 512), dtype=np.uint8)
    base[:] = 128
    
    for i in range(5):
        radio = 40 - i * 8
        centro = 256
        intensidad = 100 + i * 20
        cv2.circle(base, (centro, centro), radio, intensidad, -1)
    
    return base


def main():
    """Genera todas las imágenes de prueba."""
    print("=" * 60)
    print("MedImagePrep - Generador de Imágenes de Prueba")
    print("=" * 60)
    
    print(f"\n[INFO] Directorio de salida: {OUTPUT_DIR}")
    
    imagenes = {
        'torax_normal.png': generar_radiografia_torax(),
        'ct_pulmon_covid.png': generar_ct_pulmon_covid(),
        'rx_osea_lesion.png': generar_radiografia_osea(),
        'fantasma_uniformidad.png': generar_fantasma_uniformidad()
    }
    
    print("\n[INFO] Guardando imágenes...")
    
    for nombre, imagen in imagenes.items():
        ruta = OUTPUT_DIR / nombre
        cv2.imwrite(str(ruta), imagen)
        print(f"  [OK] {nombre} - {imagen.shape[1]}x{imagen.shape[0]} píxeles")
    
    print("\n" + "=" * 60)
    print("Imágenes generadas exitosamente!")
    print("=" * 60)
    
    print("\nPara probar con las imágenes generadas:")
    print("\n  # Radiografía de tórax:")
    print("  python src/main.py --image data/samples/torax_normal.png --mode cli --verbose")
    
    print("\n  # CT pulmón COVID:")
    print("  python src/main.py --image data/samples/ct_pulmon_covid.png --mode cli --verbose")
    
    print("\n  # Radiografía ósea:")
    print("  python src/main.py --image data/samples/rx_osea_lesion.png --mode cli --verbose")


if __name__ == "__main__":
    main()
