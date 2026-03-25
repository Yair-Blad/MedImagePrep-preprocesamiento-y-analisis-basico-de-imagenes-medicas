# MedImagePrep

Preprocesamiento y análisis básico de imágenes médicas para investigación y diagnóstico asistido.

De igual manera se agrega una pequeña App Web con Streamlit.
Pendiente de mejorar.

## Descripción

MedImagePrep es una herramienta modular en Python diseñada para el preprocesamiento y análisis de imágenes médicas, con enfoque en:

- **Tomografías computarizadas (CT) de pulmón post-COVID**: Mejora de visibilidad de opacidades en vidrio esmerilado y fibrosis pulmonar.
- **Radiografías y CT óseos**: Detección y medición de lesiones, tumores o anomalías óseas.

### Características principales

- ✅ Carga de imágenes en formatos PNG, JPG y **DICOM** (estándar médico)
- ✅ Preprocesamiento: crop, resize, normalización
- ✅ Filtros: Gaussian Blur, bilateral, mediano
- ✅ Detección de bordes: Sobel, Canny, Laplacian
- ✅ Histograma y ecualización de contraste
- ✅ Segmentación avanzada con **Watershed** y marcadores automáticos
- ✅ Cuantificación: porcentaje de regiones de alta intensidad, estadísticas del histograma
- ✅ Visualización comparativa antes/después
- ✅ Generación de **reportes PDF** profesionales con métricas
- ✅ Dos modos de ejecución: **CLI** y **aplicación web Streamlit**

---

## Instalación

### Requisitos previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- venv (módulo de entornos virtuales de Python)

### Pasos de instalación

#### 1. Clona o navega al directorio del proyecto:
```bash
cd MedImagePrep
```

#### 2. (Recomendado) Crear entorno virtual:

**En Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

Verás `(venv)` al inicio de tu terminal, indicando que el entorno está activo.

#### 3. Instala las dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instalará:
- `opencv-python` - Procesamiento de imágenes
- `numpy` - Operaciones numéricas
- `scipy` - Funciones científicas (Watershed)
- `matplotlib` - Visualización
- `pydicom` - Soporte DICOM
- `Pillow` - Manejo de imágenes
- `streamlit` - Aplicación web
- `reportlab` - Reportes PDF
- `scikit-image` - Análisis de textura (GLCM)
- `scikit-learn` - Machine learning (K-Means, clustering)

#### 4. Verifica la instalación:
```bash
python src/main.py --help
```

#### 5. (Opcional) Genera imágenes de prueba sintéticas:
```bash
python scripts/generate_test_images.py
```

### Desactivar el entorno virtual

Cuando termines de usar el proyecto:
```bash
deactivate
```

### Reactivar el entorno virtual

Para volver a trabajar en el proyecto:
```bash
cd MedImagePrep
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows
```

---

## Estructura del proyecto

```
MedImagePrep/
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py              # Punto de entrada CLI
│   └── utils/
│       ├── __init__.py
│       ├── io.py            # Carga PNG/JPG/DICOM
│       ├── preprocessing.py # Crop, resize, filtros
│       ├── edges.py         # Detección de bordes
│       ├── histogram.py     # Histograma y ecualización
│       ├── segmentation.py  # Watershed + marcadores
│       ├── quantification.py# Métricas y porcentajes
│       └── visualization.py # Gráficos y guardado
├── app/
│   ├── __init__.py
│   └── app.py               # Streamlit web app
├── notebooks/               # Jupyter notebooks experimentales
├── reports/                 # Reportes PDF generados
└── data/                    # Imágenes de prueba (no subir a Git)
```

---

## Uso

### Modo CLI (línea de comandos)

#### Ejecución básica:
```bash
python src/main.py --image ruta/a/imagen.png --mode cli
```

#### Opciones disponibles:
```
--image PATH       Ruta a la imagen médica (PNG, JPG o DICOM)
--mode MODE        Modo de ejecución: 'cli' o 'gui' (default: cli)
--output DIR       Directorio de salida para resultados (default: reports/)
--skip-report      No generar reporte PDF
--verbose          Mostrar información detallada del procesamiento
```

#### Ejemplos:

1. Procesar imagen PNG con todas las operaciones:
```bash
python src/main.py --image data/imagen_medica.png --mode cli --verbose
```

2. Procesar imagen DICOM:
```bash
python src/main.py --image data/ct_pulmon.dcm --mode cli --output reports/
```

3. Solo segmentación y cuantificación:
```bash
python src/main.py --image data/rx_torax.png --mode cli --skip-report
```

---

### Modo GUI (aplicación web Streamlit)

Inicia la aplicación web interactiva:
```bash
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

#### Características de la GUI:
- 📤 Arrastrar y soltar o seleccionar imagen
- 🎚️ Controles deslizantes para parámetros de filtros
- 👁️ Visualización lado a lado (antes/después)
- 📊 Histogramas interactivos
- 📥 Descarga de imágenes procesadas
- 📄 Generación y descarga de reporte PDF

---

## Funciones disponibles

### Módulo de carga (`utils/io.py`)
- `cargar_imagen(ruta)`: Carga PNG/JPG automáticamente
- `cargar_dicom(ruta)`: Carga archivos DICOM (.dcm)
- `detectar_formato(ruta)`: Detecta formato de archivo

### Módulo de preprocesamiento (`utils/preprocessing.py`)
- `recortar(imagen, x, y, ancho, alto)`: Recorta región de interés
- `redimensionar(imagen, ancho, alto)`: Cambia tamaño manteniendo proporción
- `normalizar(imagen)`: Normaliza intensidades a [0, 255]
- `filtro_gaussiano(imagen, kernel, sigma)`: Suavizado Gaussian
- `filtro_bilateral(imagen, d, sigmaColor, sigmaSpace)`: Filtro preservador de bordes
- `filtro_mediano(imagen, kernel)`: Eliminación de ruido sal-y-pimienta

### Módulo de bordes (`utils/edges.py`)
- `detectar_sobel(imagen, dx, dy, kernel_size)`: Gradiente Sobel
- `detectar_canny(imagen, umbral1, umbral2)`: Detector de bordes Canny
- `detectar_laplacian(imagen, kernel_size)`: Laplaciano para detección de bordes

### Módulo de histograma (`utils/histogram.py`)
- `calcular_histograma(imagen)`: Calcula histograma de intensidades
- `ecualizar_histograma(imagen)`: Ecualización de contraste
- `estadisticas_histograma(imagen)`: Media, desviación, percentiles

### Módulo de segmentación (`utils/segmentation.py`)
- `segmentar_watershed(imagen, marcadores=None)`: Segmentación Watershed
- `crear_marcadores_distancia(imagen)`: Marcadores automáticos con transformada de distancia
- `umbralizar_otsu(imagen)`: Umbral óptimo de Otsu

### Módulo de cuantificación (`utils/quantification.py`)
- `porcentaje_alta_intensidad(imagen, umbral)`: % de píxeles above umbral
- `medir_region(imagen, mascara)`: Área, perímetro, circularidad
- `analisis_textura(imagen)`: Contraste, correlación, energía, homogeneidad

### Módulo de visualización (`utils/visualization.py`)
- `mostrar_comparacion(original, procesada, titulo)`: Grid antes/después
- `guardar_imagen(imagen, ruta)`: Guarda imagen procesada
- `plot_histograma(imagen, ruta_guardado)`: Histograma con estadísticas
- `generar_reporte_pdf(ruta_salida, metricas, imagenes)`: Reporte profesional PDF

---

## Conjuntos de datos públicos recomendados

### COVID-19 y CT de pulmón

1. **COVID-19 CT Lesion Segmentation** (Kaggle)
   - https://www.kaggle.com/datasets/arnowang/covid19-ct-lesion-segmentation
   - Contiene CT con máscaras de segmentación de lesiones

2. **MosMedData: Chest CT scans of patients with COVID-19**
   - https://mosmed.ai/datasets/covid19_1110/
   - 1110 CT de pacientes con COVID-19, clasificados por severidad

3. **TCIA: Lung collections**
   - https://www.cancerimagingarchive.net/collection/
   - Colecciones públicas de imágenes de cáncer de pulmón

4. **LIDC-IDRI: Lung Image Database Consortium**
   - https://www.cancerimagingarchive.net/collection/lidc-idri/
   - 1018 CT de tórax con nódulos pulmonares anotados

### Cáncer de hueso y lesiones óseas

1. **Bone Cancer Detection** (Kaggle)
   - https://www.kaggle.com/datasets/todasdasd/bone-cancer-detection
   - Radiografías con tumores óseos

2. **MURA: Musculoskeletal Radiographs**
   - https://stanfordmlgroup.github.io/competitions/mura/
   - Dataset grande de radiografías musculoesqueléticas

3. **RSNA Bone Lesion Detection**
   - https://www.rsna.org/education/ai-resources-and-training/ai-image-research-program
   - Competencias de detección de lesiones óseas

4. **TCIA: Bone collections**
   - https://www.cancerimagingarchive.net/
   - Buscar "bone", "sarcoma", "osteosarcoma"

### Radiografías de Tórax

1. **Chest X-Ray Pneumonia Detection** (Kaggle)
   - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. **NIH Chest X-Ray Dataset**
   - https://nihcc.app.box.com/v/ChestXray-NIHCC
   - 112,120 imágenes con 14 patologías

3. **CXR-MNIST** (RSNA)
   - https://www.rsna.org/education/ai-resources-and-training/ai-image-research-program/cxr-mnist

---

## Descarga Automática de Muestras

El proyecto incluye scripts para descargar/generar imágenes de prueba:

### Generar imágenes sintéticas (no requiere descarga):
```bash
python scripts/generate_test_images.py
```

Genera:
- `torax_normal.png` - Radiografía de tórax simulada
- `ct_pulmon_covid.png` - CT con opacidades tipo COVID
- `rx_osea_lesion.png` - Radiografía ósea con lesión
- `fantasma_uniformidad.png` - Fantasma para calibración

### Descargar desde Kaggle (requiere cuenta):
```bash
pip install kaggle
kaggle datasets download -d arnowang/covid19-ct-lesion-segmentation
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

### Descargar desde TCIA:
```bash
# Requiere registro en https://www.cancerimagingarchive.net/
# Usar el cliente oficial de TCIA:
pip install pytcia
```

---

## Ejemplos de código

### Uso programático de las funciones:

```python
from src.utils.io import cargar_imagen
from src.utils.preprocessing import filtro_gaussiano, redimensionar
from src.utils.histogram import ecualizar_histograma
from src.utils.segmentation import segmentar_watershed
from src.utils.quantification import porcentaje_alta_intensidad

# Cargar imagen
imagen = cargar_imagen('data/ct_pulmon.png')

# Preprocesamiento
imagen_red = redimensionar(imagen, 512, 512)
imagen_suave = filtro_gaussiano(imagen_red, kernel=(5, 5), sigma=1.5)
imagen_ecualizada = ecualizar_histograma(imagen_suave)

# Segmentación
mascara = segmentar_watershed(imagen_ecualizada)

# Cuantificación
porcentaje = porcentaje_alta_intensidad(imagen_ecualizada, umbral=200)
print(f"Región de alta intensidad: {porcentaje:.2f}%")
```

### Generar reporte PDF:

```python
from src.utils.visualization import generar_reporte_pdf

metricas = {
    'media_intensidad': 145.3,
    'desviacion_std': 32.1,
    'porcentaje_alta_intensidad': 12.5,
    'area_lesion_mm2': 450.2
}

generar_reporte_pdf(
    ruta_salida='reports/reporte_ct.pdf',
    metricas=metricas,
    imagenes=['original.png', 'ecualizada.png', 'segmentacion.png']
)
```

---

## Flujo de trabajo recomendado

### Para CT de pulmón post-COVID:

1. Cargar imagen DICOM o PNG
2. Aplicar filtro bilateral (preserva bordes de opacidades)
3. Ecualizar histograma para mejorar contraste
4. Segmentar con Watershed usando marcadores automáticos
5. Cuantificar porcentaje de vidrio esmerilado
6. Generar reporte con métricas

### Para radiografías óseas:

1. Cargar imagen
2. Aplicar filtro mediano para reducir ruido
3. Detectar bordes con Canny
4. Umbralizar con Otsu
5. Medir área y perímetro de lesiones
6. Exportar resultados

---

## Contribuciones

Este proyecto es de código abierto y está diseñado como punto de partida para análisis de imágenes médicas. Sugerencias de mejora:

- Soporte para formatos NIfTI (.nii, .nii.gz) para neuroimágenes
- Integración con modelos de deep learning para segmentación automática
- Métricas adicionales de textura (GLCM, wavelets)
- Exportación a formatos DICOM secundarios (SR, SEG)

---

## Licencia

MIT License - Uso académico y de investigación. No usar para diagnóstico clínico sin validación apropiada.

---

## Contacto

Para preguntas o reportes de errores, abrir un issue en el repositorio GitHub.
