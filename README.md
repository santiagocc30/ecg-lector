# ECG Lector

App web que lee la foto de un ECG de 12 derivaciones, la digitaliza con
[ECG-Digitiser](https://github.com/felixkrones/ECG-Digitiser) y devuelve:

- el ECG redibujado sobre cuadrícula,
- mediciones (FC, PR, QRS, QT, QTc, eje, Sokolow-Lyon),
- interpretación por reglas (ritmo, conducción, otros),
- evaluación con IA ([ECG-FM](https://github.com/bowang-lab/ecg-fm), experimental): 17 diagnósticos
  con probabilidad calibrada, incluido infarto, flutter, extrasístoles y marcapasos,
- calculadoras clínicas (HEART, CHA₂DS₂-VA, HAS-BLED, Wells EP),
- un informe PDF.

> **Prototipo de investigación. No es un dispositivo médico** y no debe usarse
> para diagnóstico ni decisiones clínicas.

Documentación del proyecto en [`docs/`](docs): investigación (fases 1–2), plano (fase 3) y resultados (fase 5).

## Instalación (Windows)

Requiere Python 3.11 y git-lfs (los pesos del modelo pesan ~1,4 GB).

```bash
git clone --depth 1 https://github.com/felixkrones/ECG-Digitiser.git vendor/ECG-Digitiser
py -3.11 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -c constraints.txt -r requirements.txt
.venv\Scripts\pip install -c constraints.txt vendor\ECG-Digitiser\nnUNet
```

### IA opcional: ECG-FM (~1 GB)

`fairseq-signals` se importa desde su carpeta; no hace falta instalarlo ni compilar nada.
Si falta el modelo, la app funciona igual, sin la sección de IA.

```bash
git clone --depth 1 https://github.com/Jwoo5/fairseq-signals.git vendor/fairseq-signals
git clone --depth 1 https://github.com/bowang-lab/ecg-fm.git vendor/ecg-fm
.venv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download('wanglab/ecg-fm', 'mimic_iv_ecg_finetuned.pt', local_dir='vendor/ecg-fm/ckpts')"
```

## Uso

```bash
.venv\Scripts\uvicorn ecgapp.api:app --port 8000
```

Abre <http://localhost:8000>. Para usarla desde el móvil en la misma red Wi-Fi,
arranca con `--host 0.0.0.0` y abre `http://<IP-del-PC>:8000`.

En CPU cada análisis tarda unos 90 segundos.

## Tests y evaluación

```bash
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python -m scripts.make_test_images --ids 1 2 3
.venv\Scripts\python -m scripts.evaluate_digitization
```

## Estructura

| Carpeta | Contenido |
|---|---|
| `ecgapp/` | digitalización, mediciones, reglas, calculadoras, render, PDF y API |
| `web/` | interfaz (HTML + JS sin compilación) |
| `tests/` | pytest |
| `scripts/` | generación de imágenes de prueba y evaluación |
| `docs/` | investigación, plano y resultados |
| `vendor/` | ECG-Digitiser, ECG-FM y fairseq-signals (no versionados) |

## Créditos y licencias de terceros

- **ECG-Digitiser** (Krones et al., BSD 2-Clause): se descarga aparte en `vendor/` y no se
  redistribuye aquí. `ecgapp/digitize.py` importa sus funciones y usa sus pesos preentrenados.
  Paper: [arXiv:2410.14185](https://arxiv.org/abs/2410.14185).
- **nnU-Net** (Isensee et al., Apache 2.0), a través del fork incluido en ECG-Digitiser.
- **PTB-XL** y **PTB-XL+** (PhysioNet, CC BY 4.0): solo se descargan para las pruebas; no se
  incluyen en el repositorio.
- **NeuroKit2** (MIT) para la detección de picos R y ondas P.
- **ECG-FM** (McKeen et al., código y pesos MIT; [arXiv:2408.05178](https://arxiv.org/abs/2408.05178)),
  ajustado con MIMIC-IV-ECG, y **fairseq-signals** (MIT). Se descargan aparte en `vendor/`.
- Las calculadoras implementan puntuaciones clínicas publicadas (HEART, CHA₂DS₂-VA, HAS-BLED, Wells).

PMcardio es una marca de Powerful Medical. Se menciona en `docs/` solo como producto de
referencia; este proyecto no está afiliado ni respaldado por Powerful Medical y no contiene
material suyo.
