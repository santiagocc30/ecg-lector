# Fase 5 · Pruebas y resultados

Fecha: 2026-10-26 · Equipo: CPU (AMD RX 6600 sin CUDA), 16 GB RAM.

## 1. Tests automáticos

`pytest tests` → **24 tests, todos pasan**: calculadoras, reglas de interpretación, mediciones sobre
un ECG sintético con intervalos conocidos (FC, PR, QRS, QT), renderizado y API.

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
| Correlación | **0,952** | 0,973 | 0,898 (bloqueo de rama izq. 180) |
| SNR | **13,3 dB** | 12,7 dB | 8,7 dB (fibrilación auricular 17) |
| RMSE | 0,056 mV | — | 0,201 mV |
| Derivaciones detectadas | 12/12 en los 22 | | |
| Tiempo por ECG (CPU) | **91 s** | | 130 s |

Referencia publicada por el fabricante de la app comercial: correlación > 0,91 y SNR > 12,5 dB
en su base PM-ECG-ID (fotos reales), < 7 s por ECG en servidor. Nuestros números son comparables,
**pero sobre imágenes limpias**, así que no es una comparación justa.

Mejoras sobre el script original de ECG-Digitiser: 4× más rápido (modelo cargado una vez y sin
test-time augmentation) y sin el re-escalado global que destruía las amplitudes cuando una
derivación salía de rango.

## 4. Mediciones (`scripts/evaluate_measurements.py`)

Error absoluto frente a las mediciones del algoritmo **GE Marquette 12SL** incluidas en
[PTB-XL+](https://physionet.org/content/ptb-xl-plus/). Se mide sobre la señal original (prueba solo
el código de mediciones) y sobre la digitalizada (prueba toda la cadena).

| Medición | Señal original: media (mediana) | Digitalizada: media (mediana) |
|---|---|---|
| Frecuencia cardíaca | 1,3 lpm (0,4) | **1,4 lpm (0,5)** |
| PR | 17,7 ms (8,5) | 30,0 ms (11,5) |
| QRS | 10,9 ms (9,5) | 19,2 ms (11,8) |
| QT | 32,2 ms (28,9) | 33,9 ms (26,4) |
| Eje | 9,3° (5,8) | 12,7° (9,4) |

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

## 6. Comparación con la app de referencia

| Aspecto | App de referencia | ECG Lector |
|---|---|---|
| Digitalización | Sí, < 7 s (servidor) | Sí, ~90 s (CPU local) |
| Fotos reales | Validado | **No validado** |
| Mediciones | De pago | Gratis; error frente a 12SL en §4 |
| Ritmo / conducción | Modelo de IA (50+ hallazgos) | Reglas: 10 hallazgos |
| Infarto con oclusión / FEVI | Sí | No |
| Calculadoras | Varias | 4 (HEART, CHA₂DS₂-VA, HAS-BLED, Wells) |
| Informe PDF | Sí | Sí |
| Certificación médica | Sí | **No: prototipo de investigación** |

Pendiente: pasar las mismas imágenes por la app de referencia y anotar sus resultados en
`docs/comparacion.csv` para una comparación caso a caso.

## 7. Próximos pasos sugeridos

1. Probar con **fotos reales** y añadir corrección de perspectiva.
2. Acelerar la inferencia (ONNX Runtime + DirectML para la GPU AMD).
3. Regla de ondas Q patológicas (infarto antiguo) y criterios de ST.
4. Calculadoras GRACE y SCORE2.
5. Validar con 200+ registros del fold de test de PTB-XL.
