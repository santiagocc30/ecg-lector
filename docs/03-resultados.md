# Fase 5 · Pruebas y resultados

Fecha: 2026-10-26 · Equipo: CPU (AMD RX 6600 sin CUDA), 16 GB RAM.

## 1. Tests automáticos

`pytest tests` → **30 tests, todos pasan**: calculadoras, reglas de interpretación, mediciones sobre
un ECG sintético con intervalos conocidos (FC, PR, QRS, QT), recorte de la tira de ritmo,
preparación de la entrada de ECG-FM (reconstrucción de miembros, relleno con latidos, niveles),
renderizado y API.

## 2. Conjunto de evaluación

22 ECG reales de [PTB-XL](https://physionet.org/content/ptb-xl/) (500 Hz, 10 s), elegidos para cubrir
distintos diagnósticos. Cada señal se dibuja con nuestro renderizador en formato 3×4 + tira de ritmo
(`scripts/make_test_images.py`) y se digitaliza con la app.

| Diagnóstico PTB-XL | Registros |
|---|---|
| Normal | 1, 2, 3 |
| Bradicardia sinusal | 2, 12, 69 |
| Taquicardia sinusal | 103, 108, 263 |
| Fibrilación auricular | 17, 152, 351 |
| Bloqueo AV 1.er grado | 102, 182 |
| Bloqueo de rama izquierda / derecha | 180, 256 / 172, 195 |
| Hipertrofia ventricular izquierda | 30, 96 |
| Hemibloqueo anterior izquierdo | 41, 103, 172, 182 |
| Infarto inferior | 210, 257 |

> Las imágenes son limpias y generadas por ordenador. **No representan fotos de móvil**
> (perspectiva, sombras, papel arrugado), donde los resultados serán peores.

## 3. Digitalización (`scripts/evaluate_digitization.py`)

Comparación de cada derivación digitalizada con la señal original (264 derivaciones):

| Métrica | Media | Mediana | Peor registro |
|---|---|---|---|
| Correlación | **0,955** | 0,973 | 0,910 (fibrilación auricular 17) |
| SNR | **13,4 dB** | 12,7 dB | 8,7 dB (fibrilación auricular 17) |
| RMSE | 0,055 mV | — | 0,199 mV |
| Derivaciones detectadas | 12/12 en los 22 | | |
| Tiempo por ECG (CPU) | **90 s** | | 98 s |

Referencia publicada por el fabricante de la app comercial: correlación > 0,91 y SNR > 12,5 dB
en su base PM-ECG-ID (fotos reales), < 7 s por ECG en servidor. Nuestros números son comparables,
**pero sobre imágenes limpias**, así que no es una comparación justa.

Mejoras sobre el script original de ECG-Digitiser:

- 4× más rápido (modelo cargado una vez y sin test-time augmentation).
- Sin el re-escalado global que destruía las amplitudes cuando una derivación salía de rango.
- La tira de ritmo (II) se recorta al rango horizontal de las columnas: la segmentación a veces
  incluía la cola del pulso de calibración, lo que estiraba el eje de tiempo (ver §6). Afectó a 2
  de los 22 registros: el 180 pasó de 0,898 a 0,918 de correlación media y el 256 de 0,947 a 0,995.
- Opcional: `ECG_TILE_STEP=1.0` usa 4 parches en lugar de 9 (~40 s por ECG, con una pérdida pequeña
  de calidad).

## 4. Mediciones (`scripts/evaluate_measurements.py`)

Error absoluto frente a las mediciones del algoritmo **GE Marquette 12SL** incluidas en
[PTB-XL+](https://physionet.org/content/ptb-xl-plus/). Se mide sobre la señal original (prueba solo
el código de mediciones) y sobre la digitalizada (prueba toda la cadena).

| Medición | Señal original: media (mediana) | Digitalizada: media (mediana) |
|---|---|---|
| Frecuencia cardíaca | 1,3 lpm (0,4) | **1,3 lpm (0,3)** |
| PR | 17,7 ms (8,5) | 23,2 ms (11,0) |
| QRS | 10,9 ms (9,5) | 16,7 ms (11,0) |
| QT | 32,2 ms (28,9) | 31,1 ms (26,0) |
| Eje | 9,3° (5,8) | 11,7° (7,5) |

Con la corrección de la tira de ritmo, los errores de la señal digitalizada bajaron
(PR 30,0 → 23,2 ms; QRS 19,2 → 16,7 ms; QT 33,9 → 31,1 ms; eje 12,7 → 11,7°).

### Cómo se llegó a estos números

| Cambio | Efecto |
|---|---|
| QRS por pendiente del latido promedio (en lugar del DWT de NeuroKit2) | QRS de ~150 ms → valores realistas |
| Método de la tangente para el final de la T, buscando solo hasta el 60 % del RR | Se eliminan QT de ~600 ms (confundía la P siguiente con la T) |
| No descartar latidos por huecos de 1 píxel en el trazo | FC correcta en todos los registros |
| Intervalos **globales** (percentiles 25–75 sobre las 12 derivaciones), como 12SL | Sesgo del QT de −27 ms → ~−5 ms; detecta bloqueos de rama izquierda que en II se ven estrechos |

## 5. Detección de hallazgos (señal digitalizada)

| Hallazgo PTB-XL | Detectado |
|---|---|
| Bradicardia sinusal | 3/3 |
| Taquicardia sinusal | 2/3 |
| Fibrilación auricular | 3/3 |
| Bloqueo AV 1.er grado | 2/2 |
| Bloqueo de rama izquierda | 2/2 |
| Bloqueo de rama derecha | 2/2 |
| Hipertrofia ventricular izquierda (voltaje) | 1/2 |
| Hemibloqueo anterior izquierdo (eje) | 3/4 |
| Infarto inferior | No evaluado: no hay regla de ondas Q |

### Errores conocidos

- **Falsos positivos:** registro 3 (normal) con QTc 457 ms, justo por encima del límite;
  registro 263 (taquicardia) marcado como irregular; registro 96 con un QRS de 240 ms por una
  derivación mal digitalizada.
- 22 registros son pocos: los porcentajes tienen mucha incertidumbre y los umbrales se ajustaron
  sobre este mismo conjunto. Hay que validar con un conjunto distinto y más grande.

## 6. IA: ECG-FM (`scripts/evaluate_ecgfm.py`)

[ECG-FM](https://github.com/bowang-lab/ecg-fm) (código y pesos MIT) es un modelo de 90,9 M de
parámetros ajustado con MIMIC-IV-ECG para 17 diagnósticos. En CPU tarda **~0,3 s por ECG**.

### El problema del formato en papel

ECG-FM espera 5 s simultáneos de las 12 derivaciones; un ECG 3×4 solo muestra 2,5 s de cada una
(salvo la tira de ritmo II). Medimos tres entradas con los mismos 22 registros:
**completa** (señal original, límite superior), **papel** (original recortada como en el 3×4) y
**digitalizada** (lo que lee la app de la imagen).

| Cambio (entrada «papel», umbral 0,5) | Aciertos | Falsos positivos |
|---|---|---|
| Huecos a cero | 41/44 | 33 |
| + derivaciones de miembros reconstruidas (Einthoven/Goldberger) y bordes suavizados | 39/44 | 36 |
| + huecos rellenados con el **latido promedio del propio paciente** en el ritmo de la II | 36/44 | 25 |
| + **calibración** de probabilidades (deshacer el `pos_weight` del entrenamiento) | 33/44 | 7 |

El diagnóstico por partes mostró que el modelo depende sobre todo de las precordiales: recortarlas
duplicaba los falsos positivos y quitarlas del todo era aún peor.

La calibración resta `log(pos_weight)` a cada logit, con los pesos del propio archivo de
entrenamiento del modelo (`label_def.csv`). No se ajustó sobre nuestros datos. Sin ella, el modelo
marcaba extrasístoles en casi todos los ECG.

### Resultado (probabilidades calibradas)

| Entrada | Umbral 0,5 («probable») | Umbral 0,25 («posible») |
|---|---|---|
| Completa | 35/44, 3 FP | 37/44, 7 FP |
| Papel | 33/44, 7 FP | 36/44, 13 FP |
| Digitalizada | 30/44, 10 FP | 37/44, 27 FP |

La diferencia media de probabilidad entre «papel» y «completa» es de solo **0,04**: el relleno con
latidos del paciente recupera casi todo lo que pierde el formato 3×4. Con la señal digitalizada es
de 0,08: la mayor parte de la pérdida restante viene de la digitalización, no del formato.

Antes de corregir la tira de ritmo, la señal digitalizada daba 28/44 con 13 FP (umbral 0,5) y
35/44 con 30 FP (umbral 0,25).

### Fallo de digitalización encontrado gracias a ECG-FM

En los bloqueos de rama izquierda, ECG-FM acertaba con la señal en papel (99 %) pero fallaba con
la digitalizada (14–34 %). Causa: la máscara de la derivación II incluía la cola del pulso de
calibración, y como la vectorización reparte los 10 s sobre todo el ancho de la máscara, los latidos
se desplazaban hasta ~100 ms al final de la tira. Recortar la II al rango horizontal de las
columnas sube su correlación en el registro 180 de 0,33 a 0,57; el resto de la diferencia se debe a
que las ondas S de V3 invaden la franja de la tira de ritmo.

| Registro (bloqueo de rama izq.) | Probabilidad de bloqueo de rama izq. | Probabilidad de infarto (falsa) |
|---|---|---|
| 180, antes → después | 14 % → **71 %** | 71 % → 44 % |
| 256, antes → después | 34 % → **48 %** | 66 % → 41 % |

### En la app

Sección «Evaluación con IA · ECG-FM (experimental)»: diagnósticos con probabilidad calibrada
≥ 25 %, en dos niveles («probable» ≥ 50 %, «posible» 25–50 %), y un aviso si el modelo considera
que el registro es de mala calidad. También aparece en el PDF.

## 7. Comparación con la app de referencia

| Aspecto | App de referencia | ECG Lector |
|---|---|---|
| Digitalización | Sí, < 7 s (servidor) | Sí, ~90 s (CPU local) |
| Fotos reales | Validado | **No validado** |
| Mediciones | De pago | Gratis; error frente a 12SL en §4 |
| Ritmo / conducción | Modelo de IA (50+ hallazgos) | Reglas (10 hallazgos) + ECG-FM (17 diagnósticos) |
| Infarto | Sí, incluido infarto con oclusión | ECG-FM «Infarto» (sin distinguir oclusión aguda) |
| FEVI | Sí | No |
| Calculadoras | Varias | 4 (HEART, CHA₂DS₂-VA, HAS-BLED, Wells) |
| Informe PDF | Sí | Sí |
| Certificación médica | Sí | **No: prototipo de investigación** |

Pendiente: pasar las mismas imágenes por la app de referencia y anotar sus resultados en
`docs/comparacion.csv` para una comparación caso a caso.

## 8. Próximos pasos sugeridos

1. Probar con **fotos reales** y añadir corrección de perspectiva.
2. Acelerar la inferencia (ONNX Runtime + DirectML para la GPU AMD).
3. Regla de ondas Q patológicas (infarto antiguo) y criterios de ST.
4. Calculadoras GRACE y SCORE2.
5. Validar con 200+ registros del fold de test de PTB-XL.
