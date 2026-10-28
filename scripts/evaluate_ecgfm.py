"""How much does ECG-FM lose on paper ECGs?

Runs ECG-FM on each PTB-XL record three ways:
  full      original 10 s, 12 simultaneous leads (upper bound)
  paper     original, but each lead kept only in its 2.5 s column of the
            3x4 layout (lead II kept whole, as the rhythm strip)
  digitised what the app reads back from the rendered image
and scores the predictions against the PTB-XL statements.

    python -m scripts.evaluate_ecgfm
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgapp import ecgfm
from ecgapp.render import ROWS, SEGMENT_S

# PTB-XL statement(s) -> ECG-FM label
PTBXL_TO_ECGFM = {
    "Sinus rhythm": {"SR", "SARRH", "SBRAD", "STACH"},
    "Atrial fibrillation": {"AFIB"},
    "Atrial flutter": {"AFLT"},
    "Tachycardia": {"STACH", "SVTAC", "PSVT"},
    "Bradycardia": {"SBRAD"},
    "1st degree atrioventricular block": {"1AVB"},
    "Atrioventricular block": {"1AVB", "2AVB", "3AVB"},
    "Left bundle branch block": {"CLBBB"},
    "Right bundle branch block": {"CRBBB"},
    "Infarction": {"IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "IPMI", "IPLMI", "PMI", "INJAS", "INJAL", "INJIN"},
    "Premature ventricular contraction": {"PVC"},
    "Accessory pathway conduction": {"WPW"},
    "Electronic pacemaker": {"PACE"},
}
THRESHOLD = 0.5


def paper_mask(leads, fs):
    out = {}
    for row in ROWS:
        for col, name in enumerate(row):
            sig = np.full(len(leads[name]), np.nan)
            a, b = int(col * SEGMENT_S * fs), int((col + 1) * SEGMENT_S * fs)
            sig[a:b] = leads[name][a:b]
            out[name] = sig
    out["II"] = leads["II"]
    return out


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "fs"}, int(z["fs"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default="data/ground_truth")
    parser.add_argument("--digitised", default="data/digitised")
    parser.add_argument("--db", default="data/ptbxl_database.csv")
    parser.add_argument("--out", default="data/eval_ecgfm.json")
    parser.add_argument("--no-calibrate", action="store_true", help="Keep the pos_weight-inflated probabilities")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    opts = {"calibrate": not args.no_calibrate}
    threshold = args.threshold

    db = pd.read_csv(args.db, index_col="ecg_id")
    rows = []
    for gt in sorted(Path(args.truth).glob("*.npz")):
        dig = Path(args.digitised) / gt.name
        if not dig.exists():
            continue
        ecg_id = int(gt.stem.split("_")[1])
        codes = set(ast.literal_eval(db.loc[ecg_id, "scp_codes"]))
        leads, fs = load(gt)
        dleads, dfs = load(dig)
        rows.append({
            "ecg_id": ecg_id,
            "codes": sorted(codes),
            "truth": {lab: bool(codes & src) for lab, src in PTBXL_TO_ECGFM.items()},
            "full": ecgfm.predict(leads, fs, **opts),
            "paper": ecgfm.predict(paper_mask(leads, fs), fs, **opts),
            "digitised": ecgfm.predict(dleads, dfs, **opts),
        })

    modes = ("full", "paper", "digitised")
    print(f"{len(rows)} registros · umbral {threshold}\n")
    print(f"{'Diagnóstico':36s} {'n':>3s}  " + "  ".join(f"{m:>16s}" for m in modes))
    summary = {}
    for lab in PTBXL_TO_ECGFM:
        pos = [r for r in rows if r["truth"][lab]]
        neg = [r for r in rows if not r["truth"][lab]]
        if not pos:
            continue
        cells = []
        for m in modes:
            tp = sum(r[m][lab] >= threshold for r in pos)
            fp = sum(r[m][lab] >= threshold for r in neg)
            summary.setdefault(lab, {})[m] = {"tp": tp, "pos": len(pos), "fp": fp, "neg": len(neg)}
            cells.append(f"{tp}/{len(pos)} acierto, {fp} FP")
        print(f"{lab:36s} {len(pos):3d}  " + "  ".join(f"{c:>16s}" for c in cells))

    totals = {m: [0, 0, 0] for m in modes}  # correct, total positives, false positives
    for lab, per in summary.items():
        for m in modes:
            totals[m][0] += per[m]["tp"]
            totals[m][1] += per[m]["pos"]
            totals[m][2] += per[m]["fp"]
    print("\nTotal: " + "   ".join(f"{m}: {c}/{t} aciertos, {fp} FP" for m, (c, t, fp) in totals.items()))
    for m in ("paper", "digitised"):
        diff = np.mean([abs(r[m][lab] - r["full"][lab]) for r in rows for lab in r["full"]])
        print(f"Diferencia media de probabilidad {m} vs full: {diff:.3f}")

    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary, "totals": totals}, indent=2))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
