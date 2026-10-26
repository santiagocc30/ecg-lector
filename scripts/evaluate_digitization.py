"""Score digitisation against the original PTB-XL signals.

    python -m scripts.evaluate_digitization --images data/test_images --truth data/ground_truth
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ecgapp.digitize import LEAD_NAMES, digitize, load_image


def score_lead(pred, true):
    """Correlation, SNR (dB) and RMSE (mV) over the samples that were digitised."""
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 50:
        return None
    p, t = pred[ok], true[ok]
    # Digitisation cannot recover the absolute DC level, so compare without it.
    p, t = p - p.mean(), t - t.mean()
    noise = p - t
    snr = 10 * np.log10(np.sum(t**2) / max(np.sum(noise**2), 1e-12))
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else 0.0
    return {"corr": corr, "snr_db": float(snr), "rmse_mv": float(np.sqrt(np.mean(noise**2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/test_images")
    parser.add_argument("--truth", default="data/ground_truth")
    parser.add_argument("--out", default="data/eval_digitization.json")
    parser.add_argument("--save", default="data/digitised", help="Where to store the digitised leads (.npz)")
    args = parser.parse_args()

    results = {}
    for img_path in sorted(Path(args.images).glob("*.png")):
        truth = np.load(Path(args.truth) / f"{img_path.stem}.npz")
        t0 = time.perf_counter()
        out = digitize(load_image(img_path))
        seconds = time.perf_counter() - t0
        Path(args.save).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.save) / f"{img_path.stem}.npz", fs=out["fs"],
                 **{k: v for k, v in out["leads"].items() if v is not None})
        n = len(next(v for v in out["leads"].values() if v is not None))
        per_lead = {}
        for name in LEAD_NAMES:
            pred = out["leads"][name]
            per_lead[name] = None if pred is None else score_lead(pred, truth[name][:n])
        found = [s for s in per_lead.values() if s]
        results[img_path.stem] = {"seconds": seconds, "leads_found": len(found), "per_lead": per_lead}
        mean = lambda k: np.mean([s[k] for s in found]) if found else float("nan")  # noqa: E731
        print(f"{img_path.stem}: {seconds:5.1f}s  leads={len(found):2d}  "
              f"corr={mean('corr'):.3f}  snr={mean('snr_db'):5.2f} dB  rmse={mean('rmse_mv'):.3f} mV")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
