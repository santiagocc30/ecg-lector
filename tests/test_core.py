import neurokit2 as nk
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecgapp import calculators
from ecgapp.interpret import interpret
from ecgapp.measure import measure
from ecgapp.render import render_ecg

FS = 500


QRS_MS, PR_MS, QT_MS = 100, 160, 380


def synthetic_beat_train(hr, seconds=10):
    """ECG with known intervals: P wave, triangular 100 ms QRS, T wave.

    PR (P onset -> QRS onset) = 160 ms, QT (QRS onset -> T end) = 380 ms.
    """
    t = np.arange(int(seconds * FS)) / FS
    sig = np.zeros_like(t)
    half = QRS_MS / 2000
    for r in np.arange(0.4, seconds - 0.5, 60 / hr):
        d = t - r
        sig += 0.15 * np.exp(-0.5 * ((d + half + PR_MS / 1000 - 0.05) / 0.0125) ** 2)  # P: centre 50 ms after onset
        tri = np.clip(1 - np.abs(d) / half, 0, None)
        sig += 1.2 * tri - 0.15 * np.clip(1 - np.abs(d + half * 0.8) / (half * 0.2), 0, None)
        t_end = -half + QT_MS / 1000
        sig += 0.3 * np.exp(-0.5 * ((d - (t_end - 0.08)) / 0.03) ** 2)  # T: ends ~80 ms after its peak
    return sig + np.random.default_rng(0).normal(0, 0.005, len(sig))


def synthetic_leads(hr):
    ecg = synthetic_beat_train(hr)
    return {name: ecg.copy() for name in
            ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")}


# ---------- calculators ----------

def test_heart_score_bands():
    low = calculators.compute("heart", {"history": 0, "ecg": 0, "age": 1, "risk": 1, "troponin": 0})
    high = calculators.compute("heart", {"history": 2, "ecg": 2, "age": 2, "risk": 2, "troponin": 1})
    assert (low["score"], low["level"]) == (2, "normal")
    assert (high["score"], high["level"]) == (9, "alterado")


def test_cha2ds2va_stroke_and_age_75():
    out = calculators.compute("cha2ds2va", {"age": 2, "stroke": True})
    assert out["score"] == 4
    assert "recomendada" in out["interpretation"]


def test_cha2ds2va_zero():
    assert calculators.compute("cha2ds2va", {"age": 0})["level"] == "normal"


def test_hasbled_threshold():
    two = calculators.compute("hasbled", {"htn": True, "elderly": True})
    three = calculators.compute("hasbled", {"htn": True, "elderly": True, "drugs": True})
    assert two["level"] == "normal" and three["level"] == "alterado"


def test_wells_half_points():
    out = calculators.compute("wells_pe", {"hr100": True, "previous": True, "hemoptysis": True})
    assert out["score"] == 4.0 and "improbable" in out["interpretation"]


def test_missing_choice_raises():
    with pytest.raises(ValueError):
        calculators.compute("heart", {"history": 0})


# ---------- interpretation rules ----------

BASE = {"hr": 75, "rr_cv": 0.03, "p_waves_found": 8, "pr_ms": 160, "qrs_ms": 90, "qtc_ms": 410,
        "axis_deg": 45, "sokolow_lyon_mv": 2.0, "limb_qrs_max_mv": 1.2}


def texts(m, **kw):
    return [f["text"] for f in interpret({**BASE, **m}, **kw)]


def test_normal_ecg():
    findings = interpret(BASE)
    assert all(f["level"] == "normal" for f in findings)
    assert findings[0]["text"].startswith("Ritmo sinusal")


@pytest.mark.parametrize("change, expected", [
    ({"hr": 45}, "Bradicardia"),
    ({"hr": 120}, "Taquicardia"),
    ({"rr_cv": 0.25}, "fibrilación auricular"),
    ({"pr_ms": 240}, "Bloqueo AV de 1.er grado"),
    ({"qrs_ms": 140}, "QRS ancho"),
    ({"qtc_ms": 480}, "QT largo"),
    ({"axis_deg": -60}, "a la izquierda"),
    ({"axis_deg": 120}, "a la derecha"),
    ({"sokolow_lyon_mv": 4.0}, "hipertrofia"),
    ({"limb_qrs_max_mv": 0.3}, "Bajo voltaje"),
])
def test_rules(change, expected):
    assert any(expected in t for t in texts(change))


def test_qt_limit_depends_on_sex():
    assert any("QT largo" in t for t in texts({"qtc_ms": 455}, sex="M"))
    assert not any("QT largo" in t for t in texts({"qtc_ms": 455}, sex="F"))


# ---------- measurements ----------

@pytest.mark.parametrize("hr", [55, 75, 110])
def test_heart_rate_on_synthetic_ecg(hr):
    m = measure(synthetic_leads(hr), FS)
    assert m["hr"] == pytest.approx(hr, abs=5)
    assert m["rr_cv"] < 0.1
    assert m["qrs_ms"] == pytest.approx(QRS_MS, abs=20)
    assert m["pr_ms"] == pytest.approx(PR_MS, abs=40)
    assert m["qt_ms"] == pytest.approx(QT_MS, abs=50)


def test_missing_rhythm_lead_warns():
    leads = synthetic_leads(70)
    leads["II"] = None
    m = measure(leads, FS)
    assert m["hr"] is None and m["warnings"]


# ---------- digitisation helpers ----------

def test_trim_rhythm_strip_drops_calibration_pulse_pixels():
    import torch

    from ecgapp.digitize import trim_rhythm_strip

    mask = torch.zeros((1, 400, 1000), dtype=torch.uint8)
    mask[0, 50:60, 100:300] = 1    # I, first column
    mask[0, 50:60, 700:900] = 10   # V4, last column
    mask[0, 300:320, 80:900] = 2   # II rhythm strip, starting inside the pulse
    out = trim_rhythm_strip(mask)
    assert (out[0, 300:320, 80:100] == 0).all()
    assert (out[0, 300:320, 100:900] == 2).all()
    assert (out[0, 50:60, 100:300] == 1).all()


# ---------- rendering and API ----------

def test_render_returns_png():
    png = render_ecg(synthetic_leads(70), FS)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_calculators_and_bad_upload():
    from ecgapp.api import app

    client = TestClient(app)
    assert "heart" in client.get("/api/calculators").json()
    r = client.post("/api/calculators/cha2ds2va", json={"age": 1, "htn": True})
    assert r.json()["score"] == 2
    assert client.post("/api/calculators/nope", json={}).status_code == 404
    bad = client.post("/api/analyses", files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    assert client.get("/api/analyses/doesnotexist").status_code == 404
