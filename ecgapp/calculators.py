"""Clinical risk scores, defined as data so the UI can build their forms.

Each field is either a yes/no item worth `points`, or a choice whose options
carry their own points. compute() adds them up and maps the total to a band.
"""


def _yes(id_, label, points=1):
    return {"id": id_, "label": label, "type": "bool", "points": points}


def _choice(id_, label, options):
    return {"id": id_, "label": label, "type": "choice",
            "options": [{"value": v, "label": lab, "points": p} for v, lab, p in options]}


CALCULATORS = {
    "heart": {
        "name": "HEART Score",
        "description": "Riesgo de evento cardíaco mayor a 6 semanas en dolor torácico",
        "fields": [
            _choice("history", "Historia", [(0, "Poco sospechosa", 0), (1, "Moderadamente sospechosa", 1), (2, "Muy sospechosa", 2)]),
            _choice("ecg", "ECG", [(0, "Normal", 0), (1, "Alteración inespecífica de la repolarización", 1), (2, "Depresión ST significativa", 2)]),
            _choice("age", "Edad", [(0, "< 45 años", 0), (1, "45–64 años", 1), (2, "≥ 65 años", 2)]),
            _choice("risk", "Factores de riesgo", [(0, "Ninguno", 0), (1, "1–2 factores", 1), (2, "≥ 3 o enfermedad aterosclerótica", 2)]),
            _choice("troponin", "Troponina", [(0, "≤ límite normal", 0), (1, "1–3 × límite normal", 1), (2, "> 3 × límite normal", 2)]),
        ],
        "bands": [(3, "Riesgo bajo (MACE 0,9–1,7 %)", "normal"), (6, "Riesgo moderado (MACE 12–16,6 %)", "atencion"),
                  (10, "Riesgo alto (MACE 50–65 %)", "alterado")],
    },
    "cha2ds2va": {
        "name": "CHA₂DS₂-VA Score",
        "description": "Riesgo de ictus en fibrilación auricular (ESC 2024)",
        "fields": [
            _yes("chf", "Insuficiencia cardíaca"),
            _yes("htn", "Hipertensión"),
            _choice("age", "Edad", [(0, "< 65 años", 0), (1, "65–74 años", 1), (2, "≥ 75 años", 2)]),
            _yes("dm", "Diabetes"),
            _yes("stroke", "Ictus, AIT o tromboembolismo previo", 2),
            _yes("vascular", "Enfermedad vascular (IAM, enfermedad arterial periférica, placa aórtica)"),
        ],
        "bands": [(0, "Riesgo bajo: no se recomienda anticoagulación", "normal"),
                  (1, "Considerar anticoagulación oral", "atencion"),
                  (8, "Anticoagulación oral recomendada", "alterado")],
    },
    "hasbled": {
        "name": "HAS-BLED Score",
        "description": "Riesgo de hemorragia mayor con anticoagulación",
        "fields": [
            _yes("htn", "Hipertensión no controlada (PAS > 160 mmHg)"),
            _yes("renal", "Función renal alterada"),
            _yes("liver", "Función hepática alterada"),
            _yes("stroke", "Ictus previo"),
            _yes("bleeding", "Hemorragia previa o predisposición"),
            _yes("inr", "INR lábil"),
            _yes("elderly", "Edad > 65 años"),
            _yes("drugs", "Antiagregantes o AINE"),
            _yes("alcohol", "Consumo de alcohol (≥ 8 bebidas/semana)"),
        ],
        "bands": [(2, "Riesgo de sangrado bajo–moderado", "normal"),
                  (9, "Riesgo de sangrado alto: corregir factores modificables", "alterado")],
    },
    "wells_pe": {
        "name": "Wells Score (EP)",
        "description": "Probabilidad clínica de embolia pulmonar",
        "fields": [
            _yes("dvt_signs", "Signos clínicos de TVP", 3),
            _yes("pe_likely", "EP es el diagnóstico más probable", 3),
            _yes("hr100", "Frecuencia cardíaca > 100 lpm", 1.5),
            _yes("immobile", "Inmovilización ≥ 3 días o cirugía en 4 semanas", 1.5),
            _yes("previous", "TVP o EP previa", 1.5),
            _yes("hemoptysis", "Hemoptisis", 1),
            _yes("cancer", "Cáncer activo", 1),
        ],
        "bands": [(4, "EP improbable (≤ 4): considerar dímero D", "normal"),
                  (12.5, "EP probable (> 4): considerar angio-TC", "alterado")],
    },
}


def public_definitions():
    return {k: {kk: vv for kk, vv in v.items() if kk != "bands"} for k, v in CALCULATORS.items()}


def compute(name, answers):
    """Score `answers` ({field_id: bool | option value}) for calculator `name`."""
    calc = CALCULATORS.get(name)
    if calc is None:
        raise KeyError(name)
    total = 0
    for field in calc["fields"]:
        value = answers.get(field["id"])
        if field["type"] == "bool":
            if value in (True, "true", 1, "1"):
                total += field["points"]
        else:
            if value is None:
                raise ValueError(f"Falta el campo: {field['label']}")
            option = next((o for o in field["options"] if str(o["value"]) == str(value)), None)
            if option is None:
                raise ValueError(f"Valor no válido para {field['label']}: {value}")
            total += option["points"]
    text, level = next((t, lv) for limit, t, lv in calc["bands"] if total <= limit)
    return {"name": calc["name"], "score": total, "interpretation": text, "level": level}
