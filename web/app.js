const $ = (sel) => document.querySelector(sel);

const METRICS = [
  ["hr", "Frecuencia", "lpm"], ["pr_ms", "PR", "ms"], ["qrs_ms", "QRS", "ms"],
  ["qt_ms", "QT", "ms"], ["qtc_ms", "QTc (Bazett)", "ms"], ["qtc_fridericia_ms", "QTc (Fridericia)", "ms"],
  ["rr_ms", "RR", "ms"], ["axis_deg", "Eje", "°"], ["sokolow_lyon_mv", "Sokolow-Lyon", "mV"],
];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function show(id, visible) { $(id).hidden = !visible; }

// ---------- Analysis ----------

$("#image").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("#file-label").textContent = file.name;
  const preview = $("#preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(e.target);
  $("#analyze-btn").disabled = true;
  show("#error-card", false);
  show("#results", false);
  show("#progress-card", true);
  $("#progress-text").textContent = "Subiendo imagen…";
  try {
    const res = await fetch("/api/analyses", { method: "POST", body: data });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "No se pudo subir la imagen.");
    await poll(body.id);
  } catch (err) {
    showError(err.message);
  } finally {
    $("#analyze-btn").disabled = false;
  }
});

async function poll(id) {
  const started = Date.now();
  for (;;) {
    const res = await fetch(`/api/analyses/${id}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || "Error consultando el análisis.");
    $("#progress-text").textContent = job.step || "Procesando…";
    $("#progress-time").textContent = `${Math.round((Date.now() - started) / 1000)} s`;
    if (job.status === "done") return showResult(id, job.result);
    if (job.status === "error") throw new Error(job.error);
    await new Promise((r) => setTimeout(r, 1500));
  }
}

function showError(msg) {
  show("#progress-card", false);
  $("#error-card").textContent = `No se pudo analizar el ECG: ${msg}`;
  show("#error-card", true);
}

function showResult(id, result) {
  show("#progress-card", false);
  show("#upload-card", false);
  $("#ecg-img").src = `/api/analyses/${id}/ecg.png`;
  $("#pdf-link").href = `/api/analyses/${id}/report.pdf`;

  const findings = $("#findings");
  findings.replaceChildren();
  const byCategory = {};
  for (const f of result.findings) (byCategory[f.category] ||= []).push(f);
  for (const [cat, items] of Object.entries(byCategory)) {
    const row = el("div", { class: "row" }, el("div", { class: "label" }, cat));
    for (const f of items) row.append(el("div", { class: `value ${f.level}` }, f.text));
    findings.append(row);
  }

  const grid = $("#measurements");
  grid.replaceChildren();
  for (const [key, label, unit] of METRICS) {
    const v = result.measurements[key];
    grid.append(el("div", { class: "metric" },
      el("div", { class: "label" }, label),
      el("div", { class: "value" }, v == null ? "—" : `${Number(v)} ${unit}`)));
  }

  const warnings = result.quality.warnings;
  const warnBox = $("#warnings");
  warnBox.replaceChildren();
  if (warnings.length) {
    warnBox.append(el("strong", {}, "Avisos de calidad"), el("ul", {}, ...warnings.map((w) => el("li", {}, w))));
  }
  warnBox.hidden = !warnings.length;
  show("#results", true);
}

$("#new-btn").addEventListener("click", () => {
  show("#results", false);
  show("#upload-card", true);
  $("#upload-form").reset();
  $("#preview").hidden = true;
  $("#file-label").textContent = "Elegir foto o tomarla con la cámara";
});

// ---------- Calculators ----------

let calculators = {};
let currentCalc = null;

async function loadCalculators() {
  const res = await fetch("/api/calculators");
  calculators = await res.json();
  const list = $("#calculators");
  list.replaceChildren();
  for (const [key, c] of Object.entries(calculators)) {
    const btn = el("button", { class: "calc-row row", type: "button" },
      el("div", {}, el("div", { class: "value" }, c.name), el("div", { class: "label" }, c.description)),
      el("span", { class: "chev" }, "›"));
    btn.addEventListener("click", () => openCalculator(key));
    list.append(btn);
  }
}

function openCalculator(key) {
  currentCalc = key;
  const c = calculators[key];
  $("#calc-title").textContent = c.name;
  $("#calc-desc").textContent = c.description;
  const fields = $("#calc-fields");
  fields.replaceChildren();
  for (const f of c.fields) {
    if (f.type === "bool") {
      fields.append(el("label", { class: "check" },
        el("input", { type: "checkbox", name: f.id }), el("span", {}, `${f.label} (+${f.points})`)));
    } else {
      const select = el("select", { name: f.id });
      for (const o of f.options) select.append(el("option", { value: o.value }, `${o.label} (${o.points})`));
      fields.append(el("label", { class: "field" }, f.label, select));
    }
  }
  show("#calc-result", false);
  $("#calc-dialog").showModal();
}

$("#calc-run").addEventListener("click", async () => {
  const answers = {};
  for (const input of $("#calc-fields").querySelectorAll("input, select")) {
    answers[input.name] = input.type === "checkbox" ? input.checked : input.value;
  }
  const res = await fetch(`/api/calculators/${currentCalc}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(answers),
  });
  const out = await res.json();
  const box = $("#calc-result");
  box.replaceChildren();
  if (!res.ok) {
    box.append(el("div", { class: "value alterado" }, out.detail || "Error"));
  } else {
    box.append(el("div", { class: "score" }, `${out.score} ${out.score === 1 ? "punto" : "puntos"}`),
      el("div", { class: `row value ${out.level}` }, out.interpretation));
  }
  box.hidden = false;
});

loadCalculators().catch(() => {
  $("#calculators").textContent = "No se pudieron cargar las calculadoras.";
});
