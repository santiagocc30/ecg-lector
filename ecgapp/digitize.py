"""ECG image -> 12-lead signals, using ECG-Digitiser (vendor/ECG-Digitiser).

Differences from vendor's src/run/digitize.py:
- the nnU-Net model is loaded once and run in-process (no CLI, no temp dirs);
- test-time mirroring is off by default (several times faster on CPU);
- out-of-range leads are baseline-corrected one by one instead of rescaling
  every lead to [-1, 1], which destroyed the amplitudes.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import rotate

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "ECG-Digitiser"
MODEL_DIR = VENDOR / "models" / "M3" / "nnUNet_results" / "Dataset500_Signals" / "nnUNetTrainer__nnUNetPlans__2d"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from config import FREQUENCY, LEAD_LABEL_MAPPING, LONG_SIGNAL_LENGTH_SEC, Y_SHIFT_RATIO  # noqa: E402
from src.run.digitize import cut_binary, get_rotation_angle, vectorise  # noqa: E402

LEAD_NAMES = list(LEAD_LABEL_MAPPING.keys())
MAX_ABS_MV = 5.0

_predictor = None
_lock = threading.Lock()


def get_predictor(use_mirroring=False):
    global _predictor
    with _lock:
        if _predictor is None:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            p = nnUNetPredictor(
                tile_step_size=0.5,
                use_mirroring=use_mirroring,
                perform_everything_on_device=device.type == "cuda",
                device=device,
                verbose=False,
                allow_tqdm=False,
            )
            p.initialize_from_trained_model_folder(str(MODEL_DIR), use_folds=("all",), checkpoint_name="checkpoint_final.pth")
            _predictor = p
    return _predictor


def load_image(path_or_file):
    """Read an image as a uint8 (3, H, W) tensor."""
    img = Image.open(path_or_file).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()


def segment(image):
    """Run nnU-Net on a (3, H, W) uint8 tensor; returns a (1, H, W) label mask."""
    # Same array layout as nnU-Net's NaturalImage2DIO reader (see vendor fork).
    arr = image.numpy()[None].astype(np.float32)
    seg = get_predictor().predict_single_npy_array(arr, {"spacing": (999, 1, 1)}, None, None, False)
    seg = np.asarray(seg)
    while seg.ndim > 2:
        seg = seg[0]
    return torch.from_numpy(seg.astype(np.uint8))[None]


def _fix_baseline(signal):
    """Remove a constant offset when a lead lands far from 0 mV."""
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return signal
    if np.nanmax(np.abs(finite)) > MAX_ABS_MV:
        signal = signal - np.median(finite)
    return signal


def digitize(image):
    """Digitise a (3, H, W) uint8 tensor.

    Returns {"fs", "leads": {name: np.ndarray in mV, NaN where not plotted},
    "rotation_deg", "mask"}.
    """
    image = image[:3]
    angle = get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
    angle = 0.0 if np.isnan(angle) else float(angle)
    image_rot = rotate(image, angle)

    mask = segment(image_rot)
    masks, positions, _ = cut_binary(mask, image_rot)

    widths = [m.shape[2] for m in masks.values() if m is not None]
    if not widths:
        raise ValueError("No se detectó ninguna derivación en la imagen.")
    median_w = np.median(widths)
    short_w = np.mean([w for w in widths if w < 2 * median_w])
    mm_per_pixel = 25 * (2.5 / short_w)
    sec_per_pixel = mm_per_pixel / 25
    mv_per_pixel = mm_per_pixel / 10

    n_total = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
    leads = {}
    for name in LEAD_NAMES:
        m = masks[name]
        if m is None:
            leads[name] = None
            continue
        sig = vectorise(image_rot, m, positions[name]["y1"], sec_per_pixel, mv_per_pixel, Y_SHIFT_RATIO, name).numpy()
        full = np.full(n_total, np.nan)
        if len(sig) >= n_total:
            full[:] = sig[:n_total]
        else:
            # Short leads sit in their 2.5 s column of the 3x4 layout.
            col = next(i for i, row in enumerate(_COLUMNS) if name in row)
            start = int(col * 2.5 * FREQUENCY)
            full[start : start + len(sig)] = sig
        leads[name] = _fix_baseline(full)

    return {"fs": FREQUENCY, "leads": leads, "rotation_deg": angle, "mask": mask}


_COLUMNS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
