"""FastAPI app: analysis queue, calculators and the static web UI.

    .venv\\Scripts\\uvicorn ecgapp.api:app --port 8000
"""
import io
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from ecgapp import calculators
from ecgapp.digitize import load_image

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_JOBS = 50

app = FastAPI(title="ECG Lector")
# One worker: the model is large and the CPU is the bottleneck anyway.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs = {}
_jobs_lock = threading.Lock()


def _run(job_id, image, sex):
    from ecgapp.pipeline import analyze
    from ecgapp.report import build_report

    job = _jobs[job_id]

    def progress(step):
        job["step"] = step

    job["status"] = "running"
    try:
        result, png = analyze(image, sex=sex, progress=progress)
        job["png"] = png
        job["pdf"] = build_report(result, png)
        job["result"] = result
        job["status"] = "done"
    except Exception as exc:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/analyses")
async def create_analysis(image: UploadFile = File(...), sex: str = Form(None)):
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "La imagen supera 20 MB.")
    try:
        tensor = load_image(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "El archivo no es una imagen válida.")
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        if len(_jobs) >= MAX_JOBS:
            oldest = next(iter(_jobs))
            _jobs.pop(oldest)
        _jobs[job_id] = {"status": "queued", "step": "En cola"}
    _executor.submit(_run, job_id, tensor, sex if sex in ("M", "F") else None)
    return {"id": job_id, "status": "queued"}


def _job(job_id):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Análisis no encontrado.")
    return job


@app.get("/api/analyses/{job_id}")
def get_analysis(job_id: str):
    job = _job(job_id)
    out = {"id": job_id, "status": job["status"], "step": job.get("step")}
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return out


@app.get("/api/analyses/{job_id}/ecg.png")
def get_png(job_id: str):
    job = _job(job_id)
    if "png" not in job:
        raise HTTPException(409, "El análisis aún no ha terminado.")
    return Response(job["png"], media_type="image/png")


@app.get("/api/analyses/{job_id}/report.pdf")
def get_pdf(job_id: str):
    job = _job(job_id)
    if "pdf" not in job:
        raise HTTPException(409, "El análisis aún no ha terminado.")
    return Response(job["pdf"], media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="ecg-{job_id}.pdf"'})


@app.get("/api/calculators")
def list_calculators():
    return calculators.public_definitions()


@app.post("/api/calculators/{name}")
def run_calculator(name: str, answers: dict):
    try:
        return calculators.compute(name, answers)
    except KeyError:
        raise HTTPException(404, "Calculadora no encontrada.")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
