"""Draw a 12-lead ECG on standard paper (25 mm/s, 10 mm/mV).

Layout matches what ECG-Digitiser was trained on: three rows of four 2.5 s
leads plus a 10 s lead II rhythm strip, on a US-letter landscape page. The
row baselines come from Y_SHIFT_RATIO in ECG-Digitiser's config.py.
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
MM_PER_S, MM_PER_MV = 25.0, 10.0
X0_MM = 15.0  # left margin before the first trace
ROWS = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
ROW_BASELINE_MM = [126.0, 90.0, 54.0]
RHYTHM_LEAD, RHYTHM_BASELINE_MM = "II", 21.0
SEGMENT_S = 2.5

GRID_MINOR = "#f4c7c3"
GRID_MAJOR = "#e8857c"


def _draw_grid(ax):
    for mm in np.arange(0, PAGE_W_MM + 0.01, 1.0):
        major = round(mm) % 5 == 0
        ax.axvline(mm, color=GRID_MAJOR if major else GRID_MINOR, lw=0.6 if major else 0.3, zorder=0)
    for mm in np.arange(0, ax.get_ylim()[1] + 0.01, 1.0):
        major = round(mm) % 5 == 0
        ax.axhline(mm, color=GRID_MAJOR if major else GRID_MINOR, lw=0.6 if major else 0.3, zorder=0)


def _trace(ax, signal, fs, t0, t1, x_mm, y_mm):
    seg = np.asarray(signal[int(t0 * fs) : int(t1 * fs)], dtype=float)
    seg = np.nan_to_num(seg)
    x = x_mm + np.arange(len(seg)) / fs * MM_PER_S
    ax.plot(x, y_mm + seg * MM_PER_MV, color="black", lw=0.9, zorder=2)


def _calibration_pulse(ax, y_mm):
    x = X0_MM - 9.0
    ax.plot([x, x + 1, x + 1, x + 6, x + 6, x + 7], [y_mm, y_mm, y_mm + 10, y_mm + 10, y_mm, y_mm],
             color="black", lw=0.9, zorder=2)


COMPACT_H_MM = 150.0  # just above the top row; drops the blank header band


def render_ecg(leads, fs, dest=None, dpi=200, header=None, grid=True, compact=False):
    """Render `leads` ({name: 1-D array in mV}) to PNG.

    Returns the PNG bytes when `dest` is None, otherwise writes to `dest`.
    Missing leads are left blank. `compact` crops the empty top of the page
    for display; keep the full page for images fed to the digitiser.
    """
    page_h = COMPACT_H_MM if compact else PAGE_H_MM
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, page_h / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, page_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    if grid:
        _draw_grid(ax)

    for row, baseline in zip(ROWS, ROW_BASELINE_MM):
        _calibration_pulse(ax, baseline)
        for col, name in enumerate(row):
            x_mm = X0_MM + col * SEGMENT_S * MM_PER_S
            ax.text(x_mm + 1, baseline - 6, name, fontsize=9, zorder=3)
            if leads.get(name) is not None:
                _trace(ax, leads[name], fs, col * SEGMENT_S, (col + 1) * SEGMENT_S, x_mm, baseline)
            # ecg-image-kit marks lead boundaries with a short bar; the
            # segmentation model relies on it to split the leads in a row.
            if col < len(row) - 1:
                x_sep = x_mm + SEGMENT_S * MM_PER_S
                ax.plot([x_sep, x_sep], [baseline - 4, baseline + 4], color="black", lw=2.0, zorder=2)

    _calibration_pulse(ax, RHYTHM_BASELINE_MM)
    ax.text(X0_MM + 1, RHYTHM_BASELINE_MM - 6, RHYTHM_LEAD, fontsize=9, zorder=3)
    if leads.get(RHYTHM_LEAD) is not None:
        _trace(ax, leads[RHYTHM_LEAD], fs, 0, 4 * SEGMENT_S, X0_MM, RHYTHM_BASELINE_MM)

    ax.text(X0_MM, page_h - 8, header or "", fontsize=9, zorder=3)
    ax.text(PAGE_W_MM - 60, 4, "25 mm/s   10 mm/mV", fontsize=8, zorder=3)

    if dest is None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        return buf.getvalue()
    fig.savefig(dest, format="png", dpi=dpi)
    plt.close(fig)
    return None
