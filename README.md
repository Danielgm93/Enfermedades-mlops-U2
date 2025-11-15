# Rare Disease Triage — Demo MLOps con FastAPI y Docker
- **Autor:** Daniel Alejandro García Mendez
- **Curso:** MLOps – Semanas 3–4: Repo en GitHub + GitHub Actions  
- **Repositorio:** [mlops-rare-disease](https://github.com/Danielgm93/mlops-rare-disease)


Este repositorio implementa un **servicio de triage para una enfermedad rara** usando FastAPI pensado como ejercicio de **MLOps** para practicar:

- Estructura de servicio y “modelo” desacoplado (`model/rules.py`).
- Contenerización con Docker.
- Pruebas unitarias con `pytest`.
- Automatización con **GitHub Actions** (CI para PR y CI/CD para `develop`).
- Publicación de imagen en **GitHub Container Registry (GHCR)**.
- Evolución del servicio mediante **nuevos requerimientos** (nueva predicción y nueva funcionalidad).

> El “modelo” NO es un modelo de ML real, sino una función determinística de reglas.  
> Esto permite concentrarse en el pipeline, el contrato de entrada/salida y el despliegue.

---

## 1. Alcance por actividad

### Actividad 1: servicio base + contenedor + CI inicial

En la **Actividad 1** se implementó:

- **Servicio FastAPI** con:
  - Página HTML en `/` con formulario para ingresar síntomas.
  - Endpoint `POST /predict` que llama al “modelo” y retorna un estado clínico:
    - `NO ENFERMO`
    - `ENFERMEDAD LEVE`
    - `ENFERMEDAD AGUDA`
    - `ENFERMEDAD CRÓNICA`
- **Modelo basado en reglas** en `model/rules.py`:
  - Recibe un diccionario con:
    - `fever` (0–10), `pain` (0–10), `days` (>=0),
    - `comorbidity` (0–5, opcional),
    - `age` (0–120, opcional).
  - Aplica reglas determinísticas para producir una de las 4 etiquetas.
- **Pruebas unitarias** en `tests/test_rules.py`:
  - Verifican que la función `predict_status` cubre correctamente los 4 estados.
- **Contenerización con Docker**:
  - `Dockerfile` para construir la imagen del servicio.
  - Ejecución local del contenedor exponiendo el puerto 8000.
- **Documentación inicial** del pipeline en `pipeline.md`.

### Actividad 2: nuevos requerimientos + CI/CD

En la **Actividad 2** se extendió el proyecto con:

1. **Nueva lógica de predicción** (nuevo requerimiento de modelo)
   - Se actualizó `predict_status` en `model/rules.py` para considerar de forma especial a:
     - Pacientes **mayores o iguales a 65 años**,
     - Con **síntomas prolongados (>= 7 días)**,
   - Ajustando el `score` de severidad para reflejar mayor riesgo.
   - Se añadió un nuevo caso de prueba en `tests/test_rules.py` para validar esta lógica.

2. **Nueva funcionalidad en la API**
   - Se agregó el endpoint:
     - `GET /health`
   - Devuelve un JSON con:
     - `status: "ok"`,
     - `service`, `version` y una breve descripción.
   - Sirve como **healthcheck** para monitoreo y orquestadores (K8s, etc.).

3. **GitHub Actions para CI (PR) y CI/CD (develop)**
   - Workflow **PR CI – Comentarios y pruebas** (`.github/workflows/pr-ci.yml`):
     - Se ejecuta en cada `pull_request`.
     - Instala dependencias, corre `pytest` y registra mensajes de inicio/fin en los logs.
   - Workflow **Develop CI/CD – Tests y Docker** (`.github/workflows/develop-cicd.yml`):
     - Se ejecuta en `push` a la rama `develop`.
     - Corre `pytest` (verifica el modelo y las reglas).
     - Construye la imagen Docker y la publica en **GHCR** con tags:
       - `ghcr.io/danielgm93/mlops-rare-disease:latest`
       - `ghcr.io/danielgm93/mlops-rare-disease:<commit_sha>`

4. **Flujo de trabajo con ramas y PR**
   - Rama principal: `main` (protegida, integración vía PR).
   - Ramas de trabajo usadas, por ejemplo:
     - `actividad-1` → setup inicial.
     - `fix-pr-test` → corrección de pruebas y paths.
     - `develop` → CI/CD con Docker y GHCR.
     - `feature-nueva-prediccion` → nueva lógica de predicción.
     - `feature-nueva-funcionalidad` → nuevo endpoint `/health`.
   - Cada rama se integra a `main` mediante **pull requests** revisados por el pipeline de CI.

---

## 2. Requisitos

- Docker 20+ (para ejecutar el servicio contenedorizado).
- Python 3.11+ (para desarrollo local y ejecución de pruebas).

---

## 3. Uso local

### 3.1. Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate     # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2. Ejecutar la API en local

Asumiendo que el módulo de la app es `app.app`:

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

La API quedará disponible en:

- Formulario HTML: `http://localhost:8000/`
- Endpoint JSON: `POST http://localhost:8000/predict`
- Healthcheck: `GET http://localhost:8000/health`

### 3.3. Ejecutar pruebas unitarias

```bash
python -m pytest
```

---

## 4. Uso con Docker

### 4.1. Construir la imagen

```bash
docker build -t rare-disease:latest .
```

### 4.2. Ejecutar el contenedor

```bash
docker run --rm -p 8000:8000 rare-disease:latest
```

La API quedará disponible en: `http://localhost:8000/`

- Formulario: `GET /`
- Predicción: `POST /predict`
- Healthcheck: `GET /health`

---

## 5. Ejemplos de uso (`curl`)

### 5.1. Llamar al endpoint `/predict`

```bash
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"fever":2.5,"pain":3,"days":3,"comorbidity":0,"age":40}'
```

### 5.2. Cobertura de los 4 estados (ejemplos)

```bash
# NO ENFERMO
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"fever":0.5,"pain":1.0,"days":0}'

# ENFERMEDAD LEVE
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"fever":2.0,"pain":3.0,"days":3}'

# ENFERMEDAD AGUDA
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"fever":6.0,"pain":6.0,"days":10,"comorbidity":1}'

# ENFERMEDAD CRÓNICA
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"fever":4.0,"pain":4.0,"days":45}'
```

### 5.3. Llamar al endpoint `/health`

```bash
curl -s http://localhost:8000/health
```

Salida esperada (similar):

```json
{
  "status": "ok",
  "service": "rare-disease-triage-api",
  "version": "1.0.0",
  "description": "Servicio de triage para enfermedad rara (demo MLOps)."
}
```

---

## 6. Estructura del proyecto

```text
/mlops-rare-disease
├─ pipeline.md                  # Descripción y diagrama del pipeline MLOps
├─ Dockerfile
├─ .dockerignore
├─ requirements.txt
├─ app/
│  └─ app.py                    # API FastAPI (/, /predict, /health)
├─ model/
│  └─ rules.py                  # "Modelo" basado en reglas determinísticas
├─ tests/
│  └─ test_rules.py             # Pruebas unitarias del modelo
└─ .github/
   └─ workflows/
      ├─ pr-ci.yml              # CI para PR: pytest
      └─ develop-cicd.yml       # CI/CD en develop: pytest + Docker + GHCR
```

---

## 7. Notas y extensiones

- Para un escenario real, se puede reemplazar `model/rules.py` por:
  - Un modelo de ML en Sklearn/PyTorch/ONNX, respetando el **contrato** de entrada/salida.
- El endpoint `/health` facilita la integración con:
  - Orquestadores (Kubernetes),
  - Sistemas de monitoreo (Prometheus, etc.),
  - Chequeos de disponibilidad (load balancers / API gateways).
- La configuración de GitHub Actions se puede extender con:
  - Análisis estático (flake8, mypy),
  - Escaneo de seguridad de la imagen Docker,
  - Despliegue automático a un entorno de staging o producción.
