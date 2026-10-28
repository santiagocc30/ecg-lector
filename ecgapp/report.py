"""Two-page PDF report: the redrawn ECG, then measurements and findings."""
import io
import textwrap
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LABELS = [
    ("hr", "Frecuencia cardíaca", "lpm"), ("rr_ms", "RR", "ms"), ("pr_ms", "PR", "ms"),
    ("qrs_ms", "QRS", "ms"), ("qt_ms", "QT", "ms"), ("qtc_ms", "QTc (Bazett)", "ms"),
    ("qtc_fridericia_ms", "QTc (Fridericia)", "ms"), ("axis_deg", "Eje QRS", "°"),
    ("sokolow_lyon_mv", "Sokolow-Lyon", "mV"),
]
LEVEL_COLOR = {"normal": "#1a7f37", "atencion": "#b35900", "alterado": "#c62828"}


def build_report(result, ecg_png):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
        ax.imshow(mpimg.imread(io.BytesIO(ecg_png), format="png"))
        ax.axis("off")
        fig.text(0.02, 0.96, "ECG Lector · ECG digitalizado", fontsize=14, weight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        fig.text(0.08, y, "Informe de ECG", fontsize=18, weight="bold")
        y -= 0.03
        fig.text(0.08, y, datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=9, color="#555")

        y -= 0.05
        fig.text(0.08, y, "Mediciones", fontsize=13, weight="bold")
        for key, label, unit in LABELS:
            y -= 0.028
            value = result["measurements"].get(key)
            fig.text(0.10, y, label, fontsize=10)
            fig.text(0.50, y, "—" if value is None else f"{value:g} {unit}", fontsize=10)

        y -= 0.05
        fig.text(0.08, y, "Interpretación automática", fontsize=13, weight="bold")
        for f in result["findings"]:
            for i, line in enumerate(textwrap.wrap(f"{f['category']}: {f['text']}", 85)):
                y -= 0.025
                fig.text(0.10, y, ("• " if i == 0 else "  ") + line, fontsize=10, color=LEVEL_COLOR.get(f["level"], "black"))

        ai = result.get("ai")
        if ai:
            y -= 0.05
            fig.text(0.08, y, "Evaluación con IA (ECG-FM, experimental)", fontsize=13, weight="bold")
            if not ai["findings"]:
                y -= 0.025
                fig.text(0.10, y, "Sin hallazgos por encima del 25 %", fontsize=10)
            for f in ai["findings"]:
                y -= 0.025
                fig.text(0.10, y, f"• {f['text']}", fontsize=10, color=LEVEL_COLOR.get(f["level"], "black"))
                fig.text(0.60, y, f"{f['band']} · {round(f['probability'] * 100)} %", fontsize=10, color="#555")

        warnings = result["quality"]["warnings"]
        if warnings:
            y -= 0.05
            fig.text(0.08, y, "Avisos de calidad", fontsize=13, weight="bold")
            for w in warnings:
                for line in textwrap.wrap(w, 85):
                    y -= 0.025
                    fig.text(0.10, y, line, fontsize=9)

        for i, line in enumerate(textwrap.wrap(result["disclaimer"], 95)):
            fig.text(0.08, 0.06 - i * 0.018, line, fontsize=8, color="#555")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
