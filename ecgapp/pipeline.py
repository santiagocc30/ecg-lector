"""Image -> digitised leads -> measurements -> findings -> rendered ECG."""
import numpy as np

from ecgapp import ecgfm
from ecgapp.digitize import digitize
from ecgapp.interpret import interpret
from ecgapp.measure import measure
from ecgapp.render import render_ecg

DISCLAIMER = ("Prototipo de investigación. No es un dispositivo médico y no debe usarse "
              "para diagnóstico ni decisiones clínicas.")


def _round(v, nd=0):
    return None if v is None else round(float(v), nd)


def analyze(image, sex=None, progress=lambda step: None):
    progress("Digitalizando (puede tardar 1–2 minutos en CPU)")
    dig = digitize(image)
    leads, fs = dig["leads"], dig["fs"]

    progress("Midiendo intervalos")
    m = measure(leads, fs)

    progress("Interpretando")
    findings = interpret(m, sex=sex)

    ai = None
    if ecgfm.available():
        progress("Evaluación con IA (ECG-FM)")
        try:
            probs = ecgfm.predict(leads, fs, rpeaks=m.get("rpeaks"))
            ai = {"model": "ECG-FM (bowang-lab, MIMIC-IV-ECG)", "findings": ecgfm.summarize(probs),
                  "probabilities": {k: round(v, 3) for k, v in probs.items()},
                  "poor_quality": probs.get("Poor data quality", 0) >= ecgfm.PROBABLE}
        except Exception as exc:  # the AI section is optional; never fail the analysis for it
            m.setdefault("warnings", []).append(f"ECG-FM no pudo evaluar el ECG: {exc}")

    progress("Dibujando ECG")
    png = render_ecg(leads, fs, header="ECG digitalizado", compact=True)

    detected = [n for n, s in leads.items() if s is not None]
    warnings = list(m.pop("warnings", []))
    if len(detected) < 12:
        missing = [n for n, s in leads.items() if s is None]
        warnings.append(f"Derivaciones no detectadas: {', '.join(missing)}")
    if abs(dig["rotation_deg"]) > 5:
        warnings.append(f"Imagen girada {dig['rotation_deg']:.1f}°: la precisión puede ser menor.")
    if ai and ai["poor_quality"]:
        warnings.append("La IA considera que la calidad del registro es mala: interpreta los resultados con cautela.")

    measurements = {
        "hr": _round(m["hr"]), "rr_ms": _round(m["rr_ms"]), "pr_ms": _round(m["pr_ms"]),
        "qrs_ms": _round(m["qrs_ms"]), "qt_ms": _round(m["qt_ms"]), "qtc_ms": _round(m["qtc_ms"]),
        "qtc_fridericia_ms": _round(m["qtc_fridericia_ms"]), "axis_deg": _round(m["axis_deg"]),
        "sokolow_lyon_mv": _round(m["sokolow_lyon_mv"], 2),
    }
    result = {
        "fs": fs,
        # NaN (not plotted) becomes null in the JSON.
        "leads": {n: (None if s is None else [None if np.isnan(v) else round(float(v), 4) for v in s])
                  for n, s in leads.items()},
        "measurements": measurements,
        "findings": findings,
        "ai": ai,
        "quality": {"leads_detected": len(detected), "rotation_deg": round(dig["rotation_deg"], 2), "warnings": warnings},
        "disclaimer": DISCLAIMER,
    }
    return result, png
