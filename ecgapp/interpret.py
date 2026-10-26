"""Rule-based interpretation of the measurements (see docs/02-plano.md §4)."""

NORMAL, ATTENTION, ABNORMAL = "normal", "atencion", "alterado"


def _f(category, text, level):
    return {"category": category, "text": text, "level": level}


def interpret(m, sex=None):
    """Turn a measure() dict into findings grouped as Ritmo / Conducción / Otro."""
    findings = []
    hr, cv = m.get("hr"), m.get("rr_cv")
    irregular = False

    # Rhythm
    if hr is None:
        findings.append(_f("Ritmo", "No se pudo determinar el ritmo", ATTENTION))
    else:
        # Tuned on PTB-XL: both AF records exceed 0.10 on both metrics, sinus
        # rhythms (including sinus arrhythmia) stay below on at least one.
        rmssd = m.get("rr_rmssd_rel")
        irregular = cv is not None and cv > 0.10 and (rmssd is None or rmssd > 0.10)
        has_p = (m.get("p_waves_found") or 0) >= 3 and m.get("pr_ms") is not None
        if irregular:
            findings.append(_f("Ritmo", "Ritmo irregular: valorar fibrilación auricular", ABNORMAL))
        elif hr < 60:
            findings.append(_f("Ritmo", f"Bradicardia{' sinusal' if has_p else ''} ({hr:.0f} lpm)", ATTENTION))
        elif hr > 100:
            findings.append(_f("Ritmo", f"Taquicardia{' sinusal' if has_p else ''} ({hr:.0f} lpm)", ATTENTION))
        else:
            text = "Ritmo sinusal" if has_p else "Ritmo regular, probablemente sinusal"
            findings.append(_f("Ritmo", f"{text} ({hr:.0f} lpm)", NORMAL))

    # Conduction
    conduction = []
    # Without organised atrial activity there is no PR interval to judge.
    pr = None if irregular else m.get("pr_ms")
    qrs = m.get("qrs_ms")
    if pr is not None and pr > 200:
        conduction.append(_f("Conducción", f"Bloqueo AV de 1.er grado (PR {pr:.0f} ms)", ABNORMAL))
    elif pr is not None and pr < 120:
        conduction.append(_f("Conducción", f"PR corto ({pr:.0f} ms)", ATTENTION))
    if qrs is not None and qrs >= 120:
        conduction.append(_f("Conducción", f"QRS ancho ({qrs:.0f} ms): posible bloqueo de rama", ABNORMAL))
    elif qrs is not None and qrs >= 110:
        conduction.append(_f("Conducción", f"Retraso inespecífico de la conducción intraventricular (QRS {qrs:.0f} ms)", ATTENTION))
    findings += conduction or [_f("Conducción", "No hay signos de alteraciones de la conducción", NORMAL)]

    # Other
    other = []
    qtc = m.get("qtc_ms")
    long_qt = 460 if sex == "F" else 450
    if qtc is not None and qtc > long_qt:
        other.append(_f("Otro", f"QT largo (QTc {qtc:.0f} ms)", ABNORMAL))
    elif qtc is not None and qtc < 340:
        other.append(_f("Otro", f"QT corto (QTc {qtc:.0f} ms)", ATTENTION))
    axis = m.get("axis_deg")
    if axis is not None and -90 <= axis < -30:
        other.append(_f("Otro", f"Desviación del eje a la izquierda ({axis:.0f}°)", ATTENTION))
    elif axis is not None and 90 < axis <= 180:
        other.append(_f("Otro", f"Desviación del eje a la derecha ({axis:.0f}°)", ATTENTION))
    elif axis is not None and -180 <= axis < -90:
        other.append(_f("Otro", f"Eje extremo ({axis:.0f}°)", ABNORMAL))
    sl = m.get("sokolow_lyon_mv")
    if sl is not None and sl > 3.5:
        other.append(_f("Otro", f"Criterio de voltaje de hipertrofia ventricular izquierda (Sokolow-Lyon {sl:.1f} mV)", ATTENTION))
    limb = m.get("limb_qrs_max_mv")
    if limb is not None and limb < 0.5:
        other.append(_f("Otro", "Bajo voltaje en derivaciones de miembros", ATTENTION))
    findings += other or [_f("Otro", "No hay signos de otras alteraciones evaluadas", NORMAL)]

    return findings
