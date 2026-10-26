"""Download PTB-XL records from PhysioNet and render them as ECG images.

Each record produces data/test_images/<id>.png (the input for digitisation)
and data/ground_truth/<id>.npz (the original signals, for scoring).

    python -m scripts.make_test_images --ids 1 2 3
"""
import argparse
from pathlib import Path

import numpy as np
import wfdb

from ecgapp.render import render_ecg

PTBXL = "ptb-xl/1.0.3"


def record_path(ecg_id):
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    return f"{PTBXL}/records500/{folder}", f"{ecg_id:05d}_hr"


def fetch(ecg_id):
    pn_dir, name = record_path(ecg_id)
    rec = wfdb.rdrecord(name, pn_dir=pn_dir)
    leads = {n: rec.p_signal[:, i] for i, n in enumerate(rec.sig_name)}
    # PTB-XL spells the augmented leads in upper case.
    for upper, proper in (("AVR", "aVR"), ("AVL", "aVL"), ("AVF", "aVF")):
        if upper in leads:
            leads[proper] = leads.pop(upper)
    return leads, rec.fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    img_dir = Path(args.out) / "test_images"
    gt_dir = Path(args.out) / "ground_truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for ecg_id in args.ids:
        leads, fs = fetch(ecg_id)
        name = f"ptbxl_{ecg_id:05d}"
        render_ecg(leads, fs, img_dir / f"{name}.png", header=f"PTB-XL {ecg_id}")
        np.savez(gt_dir / f"{name}.npz", fs=fs, **leads)
        print(f"{name}: {len(leads)} leads at {fs} Hz")


if __name__ == "__main__":
    main()
