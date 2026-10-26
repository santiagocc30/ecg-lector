# Fase 1 y 2 · Investigación

Fecha: 2026-10-26

## 0. Límites del proyecto

- Usamos como referencia de funciones una app comercial de lectura de ECG (PMcardio, de Powerful Medical),
  estudiada solo a partir de su uso normal y de publicaciones públicas. No se analizó su código
  ni sus binarios.
- No copiamos sus modelos, código, textos, imágenes ni diseño. Nuestra app se construye desde cero
  sobre ECG-Digitiser (licencia BSD-2) y datos públicos.
- Este repositorio no incluye capturas de pantalla ni material de terceros.
- Resultado: un prototipo de investigación, **no un dispositivo médico**.

## 1. Equipo disponible

| Recurso | Valor | Consecuencia |
|---|---|---|
| GPU | AMD Radeon RX 6600 | Sin CUDA: nnU-Net correrá en CPU (lento: ~1–3 min por imagen, a medir) |
| RAM | 16 GB | Suficiente para inferencia, justa para entrenar |
| Python | 3.14 | ECG-Digitiser pide 3.11: hay que instalar un Python 3.11 aparte |

## 2. ECG-Digitiser (repositorio base)

Ubicación: `vendor/ECG-Digitiser` (excluido de git; 1,3 GB con pesos).
Ganador del PhysioNet Challenge 2024. Paper: arXiv:2410.14185.

### Cómo funciona (`src/run/digitize.py`)

```
foto PNG ──► 1. Rotación ──► 2. Segmentación ──► 3. Recorte por derivación ──► 4. Vectorización ──► WFDB (.dat/.hea)
             Canny + Hough     nnU-Net 2D           12 máscaras (I…V6)          píxel → mV / s     12 señales a 500 Hz
             (ángulo mediano   (modelo M3,          una por etiqueta            escala desde el
              de líneas        ~475 MB)                                         ancho de 2,5 s)
              horizontales)
```

1. **Rotación:** detecta bordes (Canny), busca líneas casi horizontales con la transformada de Hough
   y gira la imagen según el ángulo mediano.
2. **Segmentación:** nnU-Net clasifica cada píxel como fondo o como una de las 12 derivaciones (etiquetas 1–12).
3. **Recorte:** separa una máscara binaria por derivación.
4. **Vectorización:** para cada columna toma la posición media de los píxeles del trazo.
   La escala sale de suponer que cada derivación corta mide 2,5 s → mm por píxel →
   25 mm/s y 10 mm/mV. Remuestrea a 500 Hz (1250 muestras por derivación corta, 5000 la tira larga).

### Supuestos y limitaciones

- Formato fijo: 3 filas × 4 derivaciones de 2,5 s + tira de ritmo de 10 s, 25 mm/s, 10 mm/mV.
- Entrenado con imágenes **sintéticas** (ecg-image-kit + PTB-XL), rotación ≤ 5°.
  Fotos reales de móvil (perspectiva, sombras, papel curvado) pueden fallar.
- Solo digitaliza: **no da diagnóstico ni mediciones**.
- Usa `nnUNetv2_predict` por línea de comandos y carpetas temporales: hay que envolverlo para una API.
- Incluye un fork de nnU-Net (el oficial falla con PNG RGB).

## 3. App de referencia: funciones observadas

Fuentes: uso normal de la app y publicaciones públicas del fabricante (enlaces en §6).

### Flujo general

```
Foto del ECG ──► Digitalización ──► ECG redibujado ──► Modelos de IA ──► Informe ──► Calculadoras / asistente / PDF
```

### Funciones y equivalente propio

| Función de la app de referencia | Equivalente que podemos construir |
|---|---|
| Digitaliza la foto y vuelve a dibujar el ECG sobre cuadrícula | ECG-Digitiser + renderizado propio |
| Modelo de detección de infarto con oclusión (STEMI y equivalentes) | Clasificador entrenado con PTB-XL (opcional, difícil) |
| Estimación de fracción de eyección reducida | Fuera de alcance: requiere datos de ecocardiograma |
| Evaluación general en tres bloques: ritmo, conducción y otros (50+ hallazgos) | Reglas sobre mediciones (+ clasificador en v2) |
| Mediciones: FC, PR, QRS, QT/QTc, eje (en el plan de pago) | **Sí**: detección de ondas propia + NeuroKit2 |
| Calculadoras de riesgo cardiovascular (HEART, GRACE, CHA₂DS₂-VA, HAS-BLED, Wells, SCORE2…) | **Sí**: fórmulas clínicas publicadas |
| Asistente conversacional sobre el ECG | Opcional: LLM con el informe como contexto |
| Exportar informe | **Sí**: PDF |

### Datos públicos relevantes

- El fabricante publica una digitalización por deep learning con correlación > 0,91, SNR > 12,5 dB,
  < 7 s por ECG y 6,6 % de fallos, evaluada en su base pública PM-ECG-ID (6000 imágenes).
- Validación independiente (AMSTELHEART-1, 290 pacientes): sensibilidad 86 % y especificidad 92 %
  para cualquier anomalía mayor; más débil en isquemia.

## 4. Comparación inicial

| Capacidad | App de referencia | ECG-Digitiser | Hueco a cubrir |
|---|---|---|---|
| Foto real de móvil | Sí | Parcial (entrenado con sintéticas) | Preprocesado: corrección de perspectiva y recorte |
| Digitalización | Sí | Sí | — |
| Redibujar ECG | Sí | Solo gráfico de depuración | Renderizado sobre cuadrícula |
| Mediciones | Sí (de pago) | No | Módulo de mediciones |
| Diagnóstico | 50+ hallazgos + infarto con oclusión + FEVI | No | Reglas + clasificador |
| Calculadoras | Varias | No | Módulo de calculadoras |
| Informe PDF | Sí | No | Exportación |
| Velocidad | < 7 s | Minutos en CPU (a medir) | Aceptar o usar ONNX / DirectML |

## 5. Pendiente para la Fase 3 (plano)

1. Instalar Python 3.11 + ECG-Digitiser y medir tiempo y calidad en CPU con una imagen de prueba.
2. Decidir plataforma (propuesta: app web, con backend Python y frontend web usable desde el móvil).
3. Decidir alcance del diagnóstico (propuesta v1: mediciones + reglas de ritmo y conducción; sin STEMI ni FEVI).

## 6. Fuentes

- Krones F. et al. *Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts.* arXiv:2410.14185 (2024).
- Powerful Medical, página del producto: <https://powerfulmedical.com/pmcardio>
- Powerful Medical, *High Precision ECG Digitization Using Artificial Intelligence*: <https://www.powerfulmedical.com/research/high-precision-ecg-digitization-using-artificial-intelligence/>
- AMSTELHEART-1: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10282008/>

PMcardio y los nombres de sus funciones son marcas de Powerful Medical. Se mencionan solo para
identificar el producto de referencia; este proyecto no está afiliado ni respaldado por Powerful Medical.
