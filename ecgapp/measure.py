"""Interval, rate, axis and voltage measurements from digitised leads.

Timing comes from the 10 s lead II rhythm strip (NeuroKit2 R-peak detection
and DWT delineation). The other leads only exist for 2.5 s each, so they are
used for amplitude features around the R peaks that fall inside their window.
"""
import warnings

import neurokit2 as nk
import numpy as np

QRS_HALF_WINDOW_S = 0.06


def _fill_nan(x):
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None, ok
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x, ok


def _within_plotted(ok):
    """True for samples between the first and last plotted sample."""
    inside = np.zeros_like(ok)
    idx = np.flatnonzero(ok)
    if idx.size:
        inside[idx[0] : idx[-1] + 1] = True
    return inside


def _median_ms(values, fs):
    values = [v for v in values if np.isfinite(v) and v > 0]
    return float(np.median(values) / fs * 1000) if values else None


def _median_beat(sig, rpeaks, fs):
    """Median beat from 250 ms before R to 90 % of the RR interval after it."""
    pre = int(0.25 * fs)
    post = int(min(0.6, 0.9 * np.median(np.diff(rpeaks)) / fs) * fs)
    beats = [sig[r - pre : r + post] for r in rpeaks if r - pre >= 0 and r + post <= len(sig)]
    if len(beats) < 2:
        return None, pre
    return np.median(np.vstack(beats), axis=0), pre


def _t_end_tangent(template, pre, qrs_off_rel, fs):
    """T-wave end (relative to R) by the tangent method, or None."""
    start = pre + qrs_off_rel + int(0.08 * fs)
    if start >= len(template) - 5:
        return None
    baseline = np.median(template[max(pre - int(0.06 * fs) - int(0.03 * fs), 0) : pre - int(0.06 * fs)])
    seg = template[start:] - baseline
    peak = int(np.argmax(np.abs(seg)))
    slope = np.gradient(seg)
    after = slope[peak:]
    if len(after) < 3:
        return None
    # Steepest slope on the descending limb, towards the baseline.
    k = peak + int(np.argmax(-np.sign(seg[peak]) * after))
    if slope[k] == 0:
        return None
    t_end = k - seg[k] / slope[k]
    if not (peak < t_end < len(seg)):
        return None
    return start + t_end - pre


def _qrs_bounds(template, pre, fs):
    """QRS onset/offset relative to the R peak (samples), from the median beat.

    NeuroKit2's DWT R onsets/offsets run wide (~150+ ms on normal beats), so
    the QRS edges are taken where the slope of the median beat drops to a
    fraction of its peak slope.
    """
    slope = np.abs(np.gradient(template))
    # Slope beyond this fraction of the steepest QRS slope counts as QRS activity.
    thresh = 0.08 * slope[pre - int(0.06 * fs) : pre + int(0.06 * fs)].max()
    limit = int(0.12 * fs)
    onset = pre
    while onset > pre - limit and slope[onset - 1] > thresh:
        onset -= 1
    offset = pre
    while offset < pre + limit and slope[offset + 1] > thresh:
        offset += 1
    # Short flat runs inside the QRS (e.g. at the R tip) stop the scan early;
    # continue while activity resumes within 10 ms.
    gap = int(0.01 * fs)
    while offset < pre + limit and slope[offset + 1 : offset + 1 + gap].max() > thresh:
        offset += 1
    while onset > pre - limit and slope[onset - gap : onset].max() > thresh:
        onset -= 1
    return onset - pre, offset - pre


def _intervals(sig, rpeaks, fs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs, method="dwt")
    n = len(rpeaks)
    p_on = np.asarray(waves.get("ECG_P_Onsets", [np.nan] * n), dtype=float)[:n]
    template, pre = _median_beat(sig, rpeaks, fs)
    if template is None:
        return {"pr_ms": None, "qrs_ms": None, "qt_ms": None, "p_waves_found": 0}
    q_rel, s_rel = _qrs_bounds(template, pre, fs)
    t_end_rel = _t_end_tangent(template, pre, s_rel, fs)
    return {
        "pr_ms": _median_ms(rpeaks + q_rel - p_on, fs),
        "qrs_ms": (s_rel - q_rel) / fs * 1000,
        "qt_ms": None if t_end_rel is None else (t_end_rel - q_rel) / fs * 1000,
        "p_waves_found": int(np.isfinite(p_on).sum()),
    }


def _net_qrs(sig, rpeaks, fs):
    """Median net QRS deflection (positive peak minus |negative peak|) in mV."""
    half = int(QRS_HALF_WINDOW_S * fs)
    nets, r_amp, s_amp, p2p = [], [], [], []
    for r in rpeaks:
        seg = sig[max(r - half, 0) : r + half]
        if len(seg) < half or not np.all(np.isfinite(seg)):
            continue
        base = np.median(sig[max(r - 3 * half, 0) : max(r - 2 * half, 1)]) if r > 2 * half else np.median(seg)
        seg = seg - base
        nets.append(seg.max() + seg.min())
        r_amp.append(max(seg.max(), 0.0))
        s_amp.append(max(-seg.min(), 0.0))
        p2p.append(seg.max() - seg.min())
    if not nets:
        return None
    return {"net": float(np.median(nets)), "r": float(np.median(r_amp)), "s": float(np.median(s_amp)), "p2p": float(np.median(p2p))}


def measure(leads, fs):
    """Return a dict of measurements; values are None when not measurable."""
    out = {k: None for k in ("hr", "rr_ms", "rr_cv", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms", "qtc_fridericia_ms",
                             "axis_deg", "sokolow_lyon_mv", "limb_qrs_max_mv")}
    out["warnings"] = []

    rhythm, ok = _fill_nan(leads.get("II")) if leads.get("II") is not None else (None, None)
    if rhythm is None or ok.sum() < 3 * fs:
        out["warnings"].append("La tira de ritmo (II) no se pudo leer: no hay mediciones de tiempo.")
        return out

    clean = nk.ecg_clean(rhythm, sampling_rate=fs)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    # Steep QRS strokes often leave single-pixel gaps in the digitised trace,
    # so only drop peaks that sit in a mostly missing stretch.
    half = int(0.05 * fs)
    rpeaks = np.array([r for r in rpeaks if ok[max(r - half, 0) : r + half].mean() > 0.5], dtype=int)
    if len(rpeaks) < 3:
        out["warnings"].append("Menos de 3 latidos detectados en II.")
        return out

    rr = np.diff(rpeaks) / fs
    out["rr_ms"] = float(np.median(rr) * 1000)
    out["hr"] = float(60 / np.median(rr))
    out["rr_cv"] = float(np.std(rr) / np.mean(rr))

    try:
        out.update(_intervals(clean, rpeaks, fs))
    except Exception as exc:  # NeuroKit2 can fail on very noisy strips
        out["warnings"].append(f"Delineación de ondas fallida: {exc}")

    if out["qt_ms"]:
        rr_s = np.median(rr)
        out["qtc_ms"] = out["qt_ms"] / np.sqrt(rr_s)
        out["qtc_fridericia_ms"] = out["qt_ms"] / np.cbrt(rr_s)

    amps = {}
    for name, sig in leads.items():
        if sig is None or name == "II":
            continue
        sig, sig_ok = _fill_nan(sig)
        if sig is None:
            continue
        # Only beats inside the lead's own 2.5 s window, with the gaps filled.
        inside = np.array([r for r in rpeaks if sig_ok[max(r - half, 0) : r + half].mean() > 0.5], dtype=int)
        # Interpolation also extends edge values across the unplotted columns.
        sig = np.where(sig_ok | _within_plotted(sig_ok), sig, np.nan)
        amps[name] = _net_qrs(sig, inside, fs)
    amps["II"] = _net_qrs(rhythm, rpeaks, fs)

    if amps.get("I") and amps.get("aVF"):
        out["axis_deg"] = float(np.degrees(np.arctan2(amps["aVF"]["net"], amps["I"]["net"])))
    if amps.get("V1") and (amps.get("V5") or amps.get("V6")):
        r_lat = max(a["r"] for a in (amps.get("V5"), amps.get("V6")) if a)
        out["sokolow_lyon_mv"] = amps["V1"]["s"] + r_lat
    limb = [amps[n]["p2p"] for n in ("I", "II", "III", "aVR", "aVL", "aVF") if amps.get(n)]
    if limb:
        out["limb_qrs_max_mv"] = float(max(limb))

    out["rpeaks"] = rpeaks.tolist()
    return out
