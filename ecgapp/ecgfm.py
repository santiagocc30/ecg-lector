"""ECG-FM (bowang-lab/ecg-fm, MIT) multi-label classifier.

Loads the MIMIC-IV-ECG finetuned checkpoint through fairseq-signals, imported
from vendor/fairseq-signals (not pip-installed: its C++ extensions are only
needed for training). Preprocessing follows notebooks/infer_quickstart.ipynb:
12 leads in standard order, 500 Hz, per-lead standardisation, non-overlapping
5 s segments, then a per-label mean/max over the segments.
"""
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
FAIRSEQ_DIR = ROOT / "vendor" / "fairseq-signals"
ECGFM_DIR = ROOT / "vendor" / "ecg-fm"
CKPT = ECGFM_DIR / "ckpts" / "mimic_iv_ecg_finetuned.pt"
LABEL_DEF = ECGFM_DIR / "data" / "mimic_iv_ecg" / "labels" / "label_def.csv"

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
SEGMENT = 5 * FS

# Same-sample aggregation from the official notebook: rhythm-wide labels are
# averaged, intermittent events (ectopy, pacing, noise) take the max.
AGG_MAX = {"Poor data quality", "Premature ventricular contraction", "Ventricular tachycardia",
           "Supraventricular tachycardia with aberrancy", "Electronic pacemaker"}

LABELS_ES = {
    "Poor data quality": "Mala calidad del registro",
    "Sinus rhythm": "Ritmo sinusal",
    "Premature ventricular contraction": "Extrasístoles ventriculares",
    "Tachycardia": "Taquicardia",
    "Ventricular tachycardia": "Taquicardia ventricular",
    "Supraventricular tachycardia with aberrancy": "Taquicardia supraventricular con aberrancia",
    "Atrial fibrillation": "Fibrilación auricular",
    "Atrial flutter": "Flutter auricular",
    "Bradycardia": "Bradicardia",
    "Accessory pathway conduction": "Conducción por vía accesoria",
    "Atrioventricular block": "Bloqueo auriculoventricular",
    "1st degree atrioventricular block": "Bloqueo AV de 1.er grado",
    "Bifascicular block": "Bloqueo bifascicular",
    "Right bundle branch block": "Bloqueo de rama derecha",
    "Left bundle branch block": "Bloqueo de rama izquierda",
    "Infarction": "Infarto",
    "Electronic pacemaker": "Marcapasos",
}

# Calibrated-probability bands (chosen on 22 PTB-XL records, see
# docs/03-resultados.md): >= PROBABLE has few false positives, POSSIBLE
# trades more alerts for sensitivity.
PROBABLE, POSSIBLE = 0.5, 0.25

_model = None
_labels = None
_lock = threading.Lock()


def _pos_weights(labels):
    weights = pd.read_csv(LABEL_DEF, index_col="name")["pos_weight_train"]
    return [float(weights[lab]) for lab in labels]


def available():
    return CKPT.exists() and FAIRSEQ_DIR.exists()


def _load():
    global _model, _labels
    with _lock:
        if _model is None:
            if str(FAIRSEQ_DIR) not in sys.path:
                sys.path.insert(0, str(FAIRSEQ_DIR))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from fairseq_signals.models import build_model_from_checkpoint

                model = build_model_from_checkpoint(checkpoint_path=str(CKPT))
            model.eval()
            _labels = pd.read_csv(LABEL_DEF, index_col="name").index.to_list()
            _model = model
    return _model, _labels


def complete_limb_leads(leads):
    """Fill gaps in the six limb leads from lead II plus any other limb lead.

    Limb leads span a 2-D space (Einthoven/Goldberger: III = II - I,
    aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2). With lead II as the
    10 s rhythm strip, each 2.5 s column of a 3x4 printout gives lead I for
    its stretch, so all six limb leads become available wherever one of
    them was printed: the whole first 5 s of a standard layout.
    """
    ii = leads.get("II")
    if ii is None:
        return leads
    ii = np.asarray(ii, dtype=float)
    lead_i = np.full(len(ii), np.nan)
    estimates = {
        "I": lambda x: x,
        "III": lambda x: ii - x,
        "aVR": lambda x: -2 * x - ii,
        "aVL": lambda x: x + ii / 2,
        "aVF": lambda x: 2 * (ii - x),
    }
    for name, to_i in estimates.items():
        sig = leads.get(name)
        if sig is None:
            continue
        est = to_i(np.asarray(sig, dtype=float)[: len(ii)])
        fill = np.isnan(lead_i[: len(est)]) & np.isfinite(est)
        lead_i[: len(est)][fill] = est[fill]
    derived = {"I": lead_i, "III": ii - lead_i, "aVR": -(lead_i + ii) / 2,
               "aVL": lead_i - ii / 2, "aVF": ii - lead_i / 2}
    out = dict(leads)
    for name, sig in derived.items():
        orig = leads.get(name)
        if orig is None:
            out[name] = sig
        else:
            orig = np.asarray(orig, dtype=float)
            out[name] = np.where(np.isfinite(orig[: len(sig)]), orig[: len(sig)], sig)
    return out


def fill_with_beat_templates(leads, fs, rpeaks):
    """Fill each lead's unplotted stretches with its own median beat.

    A 3x4 printout shows 2-3 beats per lead. Their median beat is pasted at
    every R peak of the 10 s rhythm strip (lead II) where the lead was not
    printed, so the model sees the patient's own QRS-T morphology in all
    leads, with the real rhythm. Beat-to-beat changes outside the printed
    stretch (e.g. an ectopic beat) are not recoverable this way.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return leads
    rr = int(np.median(np.diff(rpeaks)))
    pre, post = int(0.4 * rr), rr - int(0.4 * rr)
    out = dict(leads)
    for name, sig in leads.items():
        if sig is None or name == "II":
            continue
        sig = np.asarray(sig, dtype=float)
        ok = np.isfinite(sig)
        if ok.all() or not ok.any():
            continue
        beats = [sig[r - pre : r + post] for r in rpeaks
                 if r - pre >= 0 and r + post <= len(sig) and ok[r - pre : r + post].all()]
        if not beats:
            continue
        template = np.median(np.vstack(beats), axis=0)
        # Remove the baseline ramp so consecutive pasted beats join at 0.
        start, end = np.median(template[:10]), np.median(template[-10:])
        template -= np.linspace(start, end, len(template))
        filled = sig.copy()
        base = np.nanmedian(sig)
        for r in rpeaks:
            a, b = r - pre, r + post
            ta, tb = max(-a, 0), pre + post - max(b - len(sig), 0)
            seg = slice(max(a, 0), min(b, len(sig)))
            gap = ~ok[seg]
            filled[seg][gap] = (template[ta:tb] + base)[gap]
        out[name] = filled
    return out


def _taper_edges(z, ok, fs, ramp_s=0.04):
    """Fade standardised signal to 0 over `ramp_s` at each plotted/unplotted edge."""
    ramp = int(ramp_s * fs)
    if ramp < 2 or ok.all():
        return z
    w = np.ones(len(z))
    edges = np.flatnonzero(np.diff(ok.astype(int)))
    for e in edges:
        if ok[e]:  # plotted -> unplotted: fade out before e
            a = max(e - ramp + 1, 0)
            w[a : e + 1] = np.minimum(w[a : e + 1], np.linspace(1, 0, e + 1 - a))
        else:  # unplotted -> plotted: fade in after e
            b = min(e + 1 + ramp, len(z))
            w[e + 1 : b] = np.minimum(w[e + 1 : b], np.linspace(0, 1, b - e - 1))
    return z * w


def prepare(leads, fs):
    """Stack leads into a standardised (12, n) array at 500 Hz.

    Limb-lead gaps are first filled from lead II (complete_limb_leads).
    Remaining unplotted samples (NaN, e.g. the chest leads outside their
    2.5 s column) and missing leads become 0 after standardisation, i.e. the
    lead mean, with short fades at the edges so the cuts do not look like
    beats.
    """
    leads = complete_limb_leads(leads)
    n = None
    rows = []
    for name in LEAD_ORDER:
        sig = leads.get(name)
        sig = None if sig is None else np.asarray(sig, dtype=float)
        if sig is not None and fs != FS:
            t_old = np.arange(len(sig)) / fs
            t_new = np.arange(int(len(sig) * FS / fs)) / FS
            ok = np.isfinite(sig)
            sig = np.interp(t_new, t_old[ok], sig[ok]) if ok.sum() > 1 else None
        rows.append(sig)
        if sig is not None:
            n = len(sig) if n is None else min(n, len(sig))
    if n is None:
        raise ValueError("No hay derivaciones para ECG-FM.")
    out = np.zeros((12, n), dtype=np.float32)
    for i, sig in enumerate(rows):
        if sig is None:
            continue
        sig = sig[:n]
        ok = np.isfinite(sig)
        if ok.sum() < 10 or np.nanstd(sig) < 1e-6:
            continue  # constant or empty lead -> zeros
        z = (sig - np.nanmean(sig)) / np.nanstd(sig)
        out[i] = _taper_edges(np.where(ok, z, 0.0), ok, FS)
    return out


def _rpeaks(leads, fs):
    from ecgapp.measure import measure

    return measure(leads, fs).get("rpeaks") or []


def summarize(probs):
    """Findings for the report: labels at or above POSSIBLE, most likely first."""
    findings = []
    for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        if p < POSSIBLE or label == "Poor data quality":
            continue
        if label == "Sinus rhythm":
            level = "normal"
        else:
            level = "alterado" if p >= PROBABLE else "atencion"
        findings.append({"label": label, "text": LABELS_ES.get(label, label),
                         "probability": round(p, 3), "level": level,
                         "band": "probable" if p >= PROBABLE else "posible"})
    return findings


def predict(leads, fs, rpeaks=None, fill_templates=True, calibrate=True):
    """Return {label: probability} aggregated over the 5 s segments.

    `rpeaks` (sample indices in lead II) are computed when not given.
    `calibrate` undoes the finetuning loss's pos_weight: weighted BCE pushes
    a label's optimal logit up by log(pos_weight), which inflates rare labels
    (e.g. PVC, weight ~14.5); subtracting it gives prevalence-calibrated
    probabilities.
    """
    model, labels = _load()
    if fill_templates and any(np.isnan(np.asarray(s, dtype=float)).any() for s in leads.values() if s is not None):
        leads = complete_limb_leads(leads)
        leads = fill_with_beat_templates(leads, fs, _rpeaks(leads, fs) if rpeaks is None else rpeaks)
    x = prepare(leads, fs)
    n_seg = max(x.shape[1] // SEGMENT, 1)
    if x.shape[1] < SEGMENT:
        x = np.pad(x, ((0, 0), (0, SEGMENT - x.shape[1])))
    segments = np.stack([x[:, i * SEGMENT : (i + 1) * SEGMENT] for i in range(n_seg)])
    with torch.no_grad():
        logits = model(source=torch.from_numpy(segments))["out"]
    if calibrate:
        logits = logits - torch.log(torch.tensor(_pos_weights(labels), dtype=logits.dtype))
    probs = torch.sigmoid(logits).numpy()
    return {
        label: float(probs[:, j].max() if label in AGG_MAX else probs[:, j].mean())
        for j, label in enumerate(labels)
    }
