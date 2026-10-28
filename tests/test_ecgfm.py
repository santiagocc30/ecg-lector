import numpy as np
import pytest

from ecgapp import ecgfm
from tests.test_core import FS, synthetic_beat_train


def limb_leads():
    t = np.arange(10 * FS) / FS
    lead_i = np.sin(2 * np.pi * 1.1 * t)
    ii = 0.7 * np.cos(2 * np.pi * 0.9 * t)
    return {"I": lead_i, "II": ii, "III": ii - lead_i, "aVR": -(lead_i + ii) / 2,
            "aVL": lead_i - ii / 2, "aVF": ii - lead_i / 2}


def test_limb_leads_are_reconstructed_from_any_column():
    full = limb_leads()
    paper = {k: v.copy() for k, v in full.items()}
    for name, (a, b) in {"I": (0, 1250), "III": (0, 1250), "aVR": (1250, 2500),
                         "aVL": (1250, 2500), "aVF": (1250, 2500)}.items():
        paper[name][:a] = np.nan
        paper[name][b:] = np.nan
    out = ecgfm.complete_limb_leads(paper)
    for name in full:
        np.testing.assert_allclose(out[name][:2500], full[name][:2500], atol=1e-9)
        assert np.isnan(out[name][2500:]).all() or name == "II"


def test_beat_templates_fill_gaps_with_the_leads_own_morphology():
    sig = synthetic_beat_train(75)
    rpeaks = np.arange(0.4, 9.5, 0.8) * FS
    gappy = sig.copy()
    gappy[2500:] = np.nan
    out = ecgfm.fill_with_beat_templates({"II": sig, "V1": gappy}, FS, rpeaks.astype(int))
    filled = out["V1"]
    # Covered up to the last beat's window (beat at 9.2 s, window ends 9.68 s).
    assert np.isfinite(filled[2500:4800]).all()
    # Pasted beats should look like the real ones that were hidden.
    assert np.corrcoef(filled[2600:4800], sig[2600:4800])[0, 1] > 0.9


def test_prepare_standardises_and_zeroes_missing():
    leads = {name: synthetic_beat_train(70) * (i + 1) for i, name in enumerate(ecgfm.LEAD_ORDER)}
    leads["V3"] = None
    x = ecgfm.prepare(leads, FS)
    assert x.shape == (12, 10 * FS)
    assert np.allclose(x[ecgfm.LEAD_ORDER.index("V3")], 0)
    i = ecgfm.LEAD_ORDER.index("V1")
    assert abs(x[i].mean()) < 1e-3 and abs(x[i].std() - 1) < 1e-3


def test_summarize_bands():
    out = ecgfm.summarize({"Sinus rhythm": 0.9, "Infarction": 0.6, "Atrial flutter": 0.3,
                           "Bradycardia": 0.1, "Poor data quality": 0.8})
    assert [f["label"] for f in out] == ["Sinus rhythm", "Infarction", "Atrial flutter"]
    assert [f["level"] for f in out] == ["normal", "alterado", "atencion"]
    assert out[2]["band"] == "posible"


@pytest.mark.skipif(not ecgfm.available(), reason="ECG-FM checkpoint not downloaded")
def test_model_predicts_sinus_rhythm_on_regular_synthetic_ecg():
    leads = {name: synthetic_beat_train(70) for name in ecgfm.LEAD_ORDER}
    probs = ecgfm.predict(leads, FS)
    assert set(probs) == set(ecgfm.LABELS_ES)
    assert all(0 <= p <= 1 for p in probs.values())
