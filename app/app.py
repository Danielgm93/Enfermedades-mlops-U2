from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, conint, confloat
from model.rules import predict_status

app = FastAPI(title="Rare Disease Triage API", version="1.0.0")


class InputPayload(BaseModel):
    fever: confloat(ge=0, le=10) = Field(..., description="Fiebre (0-10)")
    pain: confloat(ge=0, le=10) = Field(..., description="Dolor (0-10)")
    days: confloat(ge=0) = Field(..., description="Duración en días (>=0)")
    comorbidity: confloat(ge=0, le=5) | None = 0
    age: conint(ge=0, le=120) | None = 40


@app.get("/health", response_class=JSONResponse)
def health() -> Dict[str, Any]:
    """
    Nueva funcionalidad: endpoint de healthcheck.
    Sirve para verificar que el servicio está vivo y conocer versión básica.
    """
    return {
        "status": "ok",
        "service": "rare-disease-triage-api",
        "version": "1.0.0",
        "description": "Servicio de triage para enfermedad rara (demo MLOps).",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Rare Disease Triage</title></head>
      <body style="font-family:sans-serif; max-width:640px; margin:2rem auto;">
        <h2>Rare Disease Triage — Demo</h2>
        <p>Ingrese valores y presione <b>Predecir</b>.</p>
        <form id="form">
          <label>Fiebre (0-10): <input name="fever" type="number" step="0.1" min="0" max="10" value="2.5"/></label><br/>
          <label>Dolor (0-10): <input name="pain" type="number" step="0.1" min="0" max="10" value="3.0"/></label><br/>
          <label>Días de síntomas: <input name="days" type="number" step="1" min="0" value="3"/></label><br/>
          <label>Comorbilidad (0-5): <input name="comorbidity" type="number" step="1" min="0" max="5" value="0"/></label><br/>
          <label>Edad (0-120): <input name="age" type="number" step="1" min="0" max="120" value="40"/></label><br/>
          <button type="submit">Predecir</button>
        </form>
        <pre id="out" style="background:#f5f5f5; padding:1rem; white-space:pre-wrap; margin-top:1rem;"></pre>
        <script>
          const form = document.getElementById('form');
          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = Object.fromEntries(new FormData(form).entries());
            for (const k in data) {
              data[k] = Number(data[k]);
            }
            const r = await fetch('/predict', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(data)
            });
            const j = await r.json();
            document.getElementById('out').textContent = JSON.stringify(j, null, 2);
          });
        </script>
      </body>
    </html>
    """


@app.post("/predict")
def predict(payload: InputPayload):
    result = predict_status(payload.dict())
    return JSONResponse({"status": result, "input": payload.dict()})
