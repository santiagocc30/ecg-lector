"""Compare our measurements and findings with PTB-XL+ (GE 12SL) and PTB-XL labels.

Measures each record twice: on the original signal (tests the measurement
code alone) and on the digitised one (tests the whole pipeline).

    python -m scripts.evaluate_measurements
"""
import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ecgapp.interpret import interpret
from ecgapp.measure import measure

warnings.filterwarnings("ignore")

REF_COLUMNS = {"hr": "HR_Ventr_Global", "pr_ms": "PR_Int_Global", "qrs_ms": "QRS_Dur_Global",
               "qt_ms": "QT_Int_Global", "axis_deg": "R_AxisFrontal_Global"}
LABELS = {"hr": "FC (lpm)", "pr_ms": "PR (ms)", "qrs_ms": "QRS (ms)", "qt_ms": "QT (ms)", "axis_deg": "Eje (°)"}

# PTB-XL statement -> substring our findings must contain to count as detected.
CONDITIONS = {
    "SBRAD": "Bradicardia", "STACH": "Taquicardia", "AFIB": "fibrilación auricular",
    "1AVB": "Bloqueo AV", "CLBBB": "QRS ancho", "CRBBB": "QRS ancho",
    "LVH": "hipertrofia", "LAFB": "a la izquierda",
}


def load_leads(path):
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "fs"}, int(z["fs"])


def axis_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default="data/ground_truth")
    parser.add_argument("--digitised", default="data/digitised")
    parser.add_argument("--ref", default="data/12sl_features.csv")
    parser.add_argument("--db", default="data/ptbxl_database.csv")
    parser.add_argument("--out", default="data/eval_measurements.json")
    args = parser.parse_args()

    ref = pd.read_csv(args.ref, on_bad_lines="skip").dropna(subset=["ecg_id"])
    ref = ref.set_index(ref.ecg_id.astype(int))
    db = pd.read_csv(args.db, index_col="ecg_id", on_bad_lines="skip")

    rows = []
    for gt_path in sorted(Path(args.truth).glob("*.npz")):
        ecg_id = int(gt_path.stem.split("_")[1])
        dig_path = Path(args.digitised) / gt_path.name
        if ecg_id not in ref.index or not dig_path.exists():
            continue
        row = {"ecg_id": ecg_id, "codes": sorted(ast.literal_eval(db.loc[ecg_id, "scp_codes"]))}
        for source, path in (("orig", gt_path), ("dig", dig_path)):
            leads, fs = load_leads(path)
            m = measure(leads, fs)
            row[source] = {k: m.get(k) for k in REF_COLUMNS}
            row[f"{source}_findings"] = [f["text"] for f in interpret(m) if f["level"] != "normal"]
        row["ref"] = {k: float(ref.loc[ecg_id, col]) for k, col in REF_COLUMNS.items()}
        rows.append(row)

    print(f"{len(rows)} registros\n")
    print("Error absoluto medio frente a GE 12SL (mediana entre paréntesis):")
    print(f"{'':10s} {'señal original':>22s} {'digitalizada':>22s}")
    summary = {}
    for key, label in LABELS.items():
        line = f"{label:10s}"
        for source in ("orig", "dig"):
            errs = []
            for r in rows:
                ours, theirs = r[source][key], r["ref"][key]
                if ours is None or not np.isfinite(theirs):
                    continue
                errs.append(axis_error(ours, theirs) if key == "axis_deg" else abs(ours - theirs))
            found = f"{len(errs)}/{len(rows)}"
            mae = np.mean(errs) if errs else float("nan")
            med = np.median(errs) if errs else float("nan")
            summary.setdefault(key, {})[source] = {"mae": mae, "median": med, "n": len(errs)}
            line += f" {mae:8.1f} ({med:5.1f}) n={found:>5s}"
        print(line)

    print("\nDetección de hallazgos (PTB-XL) con la señal digitalizada:")
    detection = {}
    for code, needle in CONDITIONS.items():
        with_code = [r for r in rows if code in r["codes"]]
        if not with_code:
            continue
        hits = sum(any(needle in f for f in r["dig_findings"]) for r in with_code)
        detection[code] = {"detected": hits, "total": len(with_code)}
        print(f"  {code:6s} {hits}/{len(with_code)}")
    normals = [r for r in rows if r["codes"] == ["NORM"] or r["codes"] == ["NORM", "SR"]]
    if normals:
        clean = sum(not r["dig_findings"] for r in normals)
        detection["NORM_sin_hallazgos"] = {"detected": clean, "total": len(normals)}
        print(f"  NORM sin falsas alarmas: {clean}/{len(normals)}")

    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary, "detection": detection},
                                         indent=2, default=float))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
