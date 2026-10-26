# Fase 3 · Plano de la app

Nombre provisional: **ECG Lector**. Prototipo de investigación, no es un dispositivo médico.

## 1. Alcance v1 (opción viable)

| Función | Incluido | Cómo |
|---|---|---|
| Subir foto o tomarla con la cámara del móvil | Sí | Formulario web con `capture="environment"` |
| Digitalización (12 derivaciones, 500 Hz) | Sí | ECG-Digitiser (nnU-Net M3) cargado una sola vez en memoria |
| ECG redibujado sobre cuadrícula | Sí | Renderizador propio (25 mm/s, 10 mm/mV, 3×4 + tira de ritmo II) |
| Mediciones: FC, RR, PR, QRS, QT, QTc, eje | Sí | Detección de ondas con NeuroKit2 + cálculo propio del eje |
| Interpretación por reglas (ritmo, conducción, otros) | Sí | Umbrales clínicos estándar (tabla en §4) |
| Calculadoras: HEART, CHA₂DS₂-VA, HAS-BLED, Wells EP | Sí | Fórmulas publicadas |
| Calculadoras: GRACE, SCORE2 (3 variantes) | v2 | Necesitan tablas de coeficientes por región |
| Informe PDF | Sí | Matplotlib (sin dependencias extra) |
| Detección de infarto con oclusión, estimación de FEVI, chat IA | No | Fuera de alcance de la v1 |

## 2. Arquitectura

```
Navegador (PC o móvil)
   │  HTML + JS estático (sin compilación)
   ▼
FastAPI (Python 3.11, uvicorn)  ── sirve /  y  /api/*
   │
   ├─ jobs.py         cola en un hilo: la inferencia en CPU tarda minutos
   │
   └─ pipeline.py     imagen ─► digitize ─► measure ─► interpret ─► render
          ├─ digitize.py     envoltorio de ECG-Digitiser (rotación Hough + nnU-Net + vectorización)
          ├─ measure.py      NeuroKit2: picos R, inicio/fin P, QRS, T → intervalos y eje
          ├─ interpret.py    reglas → lista de hallazgos con nivel (normal/atención/alterado)
          ├─ render.py       PNG del ECG redibujado
          ├─ calculators.py  calculadoras puras (sin estado)
          └─ report.py       PDF con ECG + mediciones + hallazgos + aviso legal
```

### Estructura de carpetas

```
electrocardiograma/
├─ ecgapp/            paquete Python (módulos de arriba + api.py)
├─ web/               index.html, app.js, styles.css
├─ tests/             pytest
├─ scripts/           evaluate.py (Fase 5), make_test_images.py
├─ docs/              investigación, plano, resultados
├─ vendor/            ECG-Digitiser (no versionado)
└─ data/              imágenes y resultados de pruebas (no versionado)
```

## 3. API

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/analyses` | Sube imagen → `{id, status:"queued"}` |
| GET | `/api/analyses/{id}` | Estado y, al terminar, resultado JSON |
| GET | `/api/analyses/{id}/ecg.png` | ECG redibujado |
| GET | `/api/analyses/{id}/report.pdf` | Informe |
| GET | `/api/calculators` | Lista y campos de cada calculadora |
| POST | `/api/calculators/{nombre}` | Calcula puntuación e interpretación |

Resultado de un análisis:

```json
{
  "leads": {"I": [...], "II": [...]},
  "fs": 500,
  "measurements": {"hr": 72, "rr_ms": 833, "pr_ms": 160, "qrs_ms": 92, "qt_ms": 380, "qtc_ms": 415, "axis_deg": 45},
  "findings": [{"category": "Ritmo", "text": "Ritmo sinusal", "level": "normal"}],
  "quality": {"leads_detected": 12, "warnings": []}
}
```

## 4. Reglas de interpretación v1

| Categoría | Regla | Hallazgo |
|---|---|---|
| Ritmo | RR regular (CV < 10 %) y FC 60–100 | Ritmo sinusal* |
| Ritmo | FC < 60 / > 100 | Bradicardia / Taquicardia |
| Ritmo | RR muy irregular (CV > 15 %) | Ritmo irregular: valorar fibrilación auricular |
| Conducción | PR > 200 ms | Bloqueo AV de 1.er grado |
| Conducción | PR < 120 ms | PR corto |
| Conducción | QRS ≥ 120 ms | QRS ancho (bloqueo de rama) |
| Conducción | QRS 110–119 ms | Retraso de conducción intraventricular |
| Otro | QTc > 460 ms (mujer) / > 450 ms (hombre o sin dato) | QT largo |
| Otro | QTc < 340 ms | QT corto |
| Otro | Eje < −30° / > 90° | Desviación del eje a la izquierda / derecha |
| Otro | Sokolow-Lyon SV1 + RV5/6 > 3,5 mV | Criterio de voltaje de HVI |
| Otro | QRS < 0,5 mV en todas las de miembros | Bajo voltaje |

\* Sin detección fiable de onda P, "sinusal" se reporta como "probable".

## 5. Pruebas (Fase 5)

1. **Unitarias:** calculadoras (casos de los artículos originales), reglas, mediciones sobre señales sintéticas.
2. **Digitalización:** registros reales de PTB-XL → imagen generada con nuestro renderizador →
   digitalizar → comparar con la señal original (correlación, SNR, RMSE).
3. **Mediciones:** comparar FC, PR, QRS y QT con PTB-XL+ (mediciones del algoritmo GE 12SL).
4. **Comparación con la app de referencia:** las mismas imágenes en las dos apps; las fotos se pasan por la app de referencia
   y se anota su resultado en `docs/comparacion.csv`.

## 6. Riesgos

| Riesgo | Mitigación |
|---|---|
| CPU lenta (minutos por ECG) | Cola asíncrona + barra de progreso; v2: ONNX / DirectML para la AMD |
| Fotos reales muy distintas a las sintéticas | Aviso de calidad; v2: corrección de perspectiva |
| Errores de medición por mala digitalización | Mostrar calidad por derivación y marcar valores dudosos |
| Uso clínico indebido | Aviso visible en la app y en el PDF |
