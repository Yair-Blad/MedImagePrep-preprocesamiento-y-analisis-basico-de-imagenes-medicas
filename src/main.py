"""
MedImagePrep - Punto de entrada principal para CLI.

Uso:
    python src/main.py --image ruta/a/imagen.png --mode cli
    python src/main.py --image ruta/a/imagen.dcm --mode cli --verbose
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

from utils.io import cargar_imagen, detectar_formato
from utils.preprocessing import (
    redimensionar,
    filtro_gaussiano,
    filtro_bilateral,
    aplicar_clahe
)
from utils.edges import detectar_canny, detectar_sobel
from utils.histogram import (
    ecualizar_histograma,
    estadisticas_histograma,
    analizar_distribucion
)
from utils.segmentation import (
    segmentar_watershed,
    umbralizar_otsu,
    crear_marcadores_distancia
)
from utils.quantification import (
    porcentaje_alta_intensidad,
    medir_region,
    analisis_textura
)
from utils.visualization import (
    mostrar_comparacion,
    guardar_imagen,
    plot_histograma,
    generar_reporte_pdf,
    crear_mosaico_comparacion
)


def parsear_argumentos():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='MedImagePrep - Preprocesamiento y análisis de imágenes médicas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python src/main.py --image imagen_medica.png --mode cli
  python src/main.py --image ct_pulmon.dcm --mode cli --verbose
  python src/main.py --image rx_torax.png --mode cli --output reports/ --skip-report
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Ruta a la imagen médica (PNG, JPG, DICOM)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='cli',
        choices=['cli', 'gui'],
        help='Modo de ejecución: cli (línea de comandos) o gui (Streamlit)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reports/',
        help='Directorio de salida para resultados (default: reports/)'
    )
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='No generar reporte PDF'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada del procesamiento'
    )
    
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        metavar=('ANCHO', 'ALTO'),
        help='Redimensionar imagen a ANCHO x ALTO píxeles'
    )
    
    parser.add_argument(
        '--umbral',
        type=float,
        default=0.8,
        help='Umbral para cuantificación de alta intensidad (0-1, default: 0.8)'
    )
    
    return parser.parse_args()


def procesar_imagen_pipeline(ruta_imagen: str, output_dir: str,
                              umbral: float = 0.8,
                              redimensionar_dims: tuple = None,
                              verbose: bool = False) -> dict:
    """
    Ejecuta el pipeline completo de procesamiento de imagen médica.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MedImagePrep - Procesamiento de Imagen Médica")
    print("=" * 60)
    
    formato = detectar_formato(ruta_imagen)
    print(f"\n[INFO] Formato detectado: {formato.upper()}")
    print(f"[INFO] Archivo: {ruta_imagen}")
    
    imagen = cargar_imagen(ruta_imagen)
    print(f"[OK] Imagen cargada: {imagen.shape[1]}x{imagen.shape[0]} píxeles")
    
    if verbose:
        print(f"   Tipo de dato: {imagen.dtype}")
        print(f"   Rango de intensidades: [{imagen.min()}, {imagen.max()}]")
    
    imagen_original = imagen.copy()
    
    if redimensionar_dims:
        print(f"\n[INFO] Redimensionando a {redimensionar_dims[0]}x{redimensionar_dims[1]}...")
        imagen = redimensionar(imagen, redimensionar_dims[0], redimensionar_dims[1])
    
    print("\n[INFO] Aplicando preprocesamiento...")
    
    imagen_suavizada = filtro_bilateral(imagen, d=9, sigmaColor=75, sigmaSpace=75)
    print("   [OK] Filtro bilateral aplicado")
    
    imagen_clahe = aplicar_clahe(imagen_suavizada, clip_limit=2.0, tile_grid_size=(8, 8))
    print("   [OK] CLAHE aplicado")
    
    imagen_ecualizada = ecualizar_histograma(imagen_clahe)
    print("   [OK] Ecualización de histograma aplicada")
    
    print("\n[INFO] Detectando bordes...")
    bordes_canny = detectar_canny(imagen_ecualizada, umbral1=50, umbral2=150)
    print("   [OK] Bordes Canny detectados")
    
    print("\n[INFO] Segmentando con Watershed...")
    mascara_otsu, umbral_otsu = umbralizar_otsu(imagen_ecualizada)
    marcadores = crear_marcadores_distancia(mascara_otsu)
    segmentacion, contornos = segmentar_watershed(imagen_ecualizada, marcadores)
    print("   [OK] Segmentación Watershed completada")
    
    print("\n[INFO] Calculando métricas cuantitativas...")
    
    porcentaje = porcentaje_alta_intensidad(imagen_ecualizada, umbral=umbral)
    print(f"   [OK] Porcentaje de alta intensidad: {porcentaje:.2f}%")
    
    estadisticas = estadisticas_histograma(imagen_ecualizada)
    print(f"   [OK] Media: {estadisticas['media']:.2f}, Desviación: {estadisticas['desviacion_std']:.2f}")
    
    textura = analisis_textura(imagen_ecualizada)
    print(f"   [OK] Contraste de textura: {textura['contraste']:.4f}")
    
    region_metrics = medir_region(mascara_otsu, imagen_ecualizada)
    
    metricas = {
        'porcentaje_alta_intensidad': porcentaje,
        'media_intensidad': estadisticas['media'],
        'desviacion_std': estadisticas['desviacion_std'],
        'minimo': estadisticas['minimo'],
        'maximo': estadisticas['maximo'],
        'mediana': estadisticas['mediana'],
        'contraste_textura': textura['contraste'],
        'homogeneidad': textura['homogeneidad'],
        'area_lesion_pixeles': region_metrics['area_pixeles'],
        'perimetro': region_metrics['perimetro'],
        'circularidad': region_metrics['circularidad'],
        'umbral_otsu': float(umbral_otsu),
        'numero_componentes': region_metrics['numero_componentes']
    }
    
    print("\n[INFO] Guardando resultados...")
    
    ruta_original = guardar_imagen(imagen_original, str(output_path / "01_original.png"))
    ruta_ecualizada = guardar_imagen(imagen_ecualizada, str(output_path / "02_ecualizada.png"))
    ruta_bordes = guardar_imagen(bordes_canny, str(output_path / "03_bordes.png"))
    ruta_segmentacion = guardar_imagen(contornos, str(output_path / "04_segmentacion.png"))
    
    print(f"   [OK] Imágenes guardadas en: {output_path}")
    
    ruta_histograma = str(output_path / "histograma.png")
    plot_histograma(imagen_ecualizada, ruta_histograma)
    print("   [OK] Histograma generado")
    
    imagenes_rutas = [
        ruta_original,
        ruta_ecualizada,
        ruta_bordes,
        ruta_segmentacion,
        ruta_histograma
    ]
    
    return {
        'metricas': metricas,
        'imagenes': imagenes_rutas,
        'output_dir': str(output_path)
    }


def main():
    """Función principal de entrada CLI."""
    args = parsear_argumentos()
    
    if args.mode == 'gui':
        print("\n[INFO] Iniciando aplicación Streamlit GUI...")
        print("[INFO] Ejecutando: streamlit run app/app.py")
        import subprocess
        subprocess.run(["streamlit", "run", "app/app.py"])
        return
    
    try:
        resultados = procesar_imagen_pipeline(
            ruta_imagen=args.image,
            output_dir=args.output,
            umbral=args.umbral,
            redimensionar_dims=args.resize,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("RESULTADOS CUANTITATIVOS")
        print("=" * 60)
        
        metricas = resultados['metricas']
        print(f"\nPorcentaje de alta intensidad: {metricas['porcentaje_alta_intensidad']:.2f}%")
        print(f"Media de intensidad: {metricas['media_intensidad']:.2f}")
        print(f"Desviación estándar: {metricas['desviacion_std']:.2f}")
        print(f"Mediana: {metricas['mediana']:.2f}")
        print(f"Contraste de textura: {metricas['contraste_textura']:.4f}")
        print(f"Homogeneidad: {metricas['homogeneidad']:.4f}")
        print(f"Área de lesión (píxeles): {metricas['area_lesion_pixeles']:.2f}")
        print(f"Perímetro: {metricas['perimetro']:.2f}")
        print(f"Circularidad: {metricas['circularidad']:.4f}")
        print(f"Umbral de Otsu: {metricas['umbral_otsu']:.2f}")
        print(f"Número de componentes: {metricas['numero_componentes']}")
        
        if not args.skip_report:
            print("\n[INFO] Generando reporte PDF...")
            ruta_pdf = generar_reporte_pdf(
                ruta_salida=str(Path(args.output) / "reporte_analisis.pdf"),
                metricas=metricas,
                imagenes_rutas=resultados['imagenes'],
                paciente_id="ANONIMO",
                comentario="Análisis automatizado con MedImagePrep"
            )
            print(f"   [OK] Reporte PDF generado: {ruta_pdf}")
        
        print("\n" + "=" * 60)
        print("Procesamiento completado exitosamente!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error durante el procesamiento: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
