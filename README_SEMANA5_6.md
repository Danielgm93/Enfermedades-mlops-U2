# Rare Disease Triage – Pipeline MLOps (Reestructuración Semanas 5–6)

- **Autor:** D W
- **Curso:** MLOps – Semanas 5–6: Pipeline de MLOps (Reestructuración)
- **Repositorio:** https://github.com/Danielgm93/mlops-rare-disease

Este repositorio describe un **pipeline MLOps end-to-end** para un sistema de **triage de enfermedades** (comunes y huérfanas) a partir de síntomas y datos clínicos de pacientes.

La reestructuración de Semanas 5–6 tiene como propósito:

- Retomar la propuesta inicial de las Semanas 1–2.
- Incorporar los conceptos de MLOps vistos durante el curso.
- Definir un proceso **end-to-end** suficientemente detallado para que un equipo de ML pueda implementarlo.
- Alinear el diseño con:
  - El código actual del repositorio (FastAPI + Docker + CI/CD).
  - Los requisitos del **proyecto final** (modelos ONNX, descarga desde bucket, despliegue automático en dev/prod).

---

## 1. Problema y alcance

### 1.1 Descripción del problema

En el contexto médico:

- Para **enfermedades comunes**, suelen existir grandes volúmenes de datos históricos (registros EHR/HCE).
- Para **enfermedades huérfanas** o de baja prevalencia, los datos son escasos y desbalanceados.

Se requiere un sistema que, dado un vector de síntomas y variables clínicas básicas de un paciente, estime si:

- No presenta indicios relevantes de enfermedad.
- Puede estar cursando una enfermedad de tipo **leve/aguda/crónica**.
- Puede estar ante un cuadro compatible con una **enfermedad huérfana**, donde la prioridad es no dejar pasar casos de riesgo (alta sensibilidad).

El sistema se concibe como **herramienta de triage y priorización**, no como sustituto del diagnóstico médico.

### 1.2 Objetivo del pipeline MLOps

El pipeline MLOps propuesto debe permitir:

1. Ir desde **datos clínicos crudos** hasta modelos entrenados, versionados, auditables y exportados a **ONNX**.
2. Desplegar estos modelos como **servicios web** accesibles para los médicos:
   - En un **computador local** (contenedor ligero, sin dependencia de la nube si los recursos lo permiten).
   - En un **servidor o servicio en la nube**, accesible vía HTTP/HTTPS.
3. Incorporar prácticas MLOps para:
   - Versionado de **datos**, **modelos** y **código**.
   - Integración y despliegue continuo (**CI/CD** con GitHub Actions).
   - Monitoreo del servicio y del modelo, y ciclo de **reentrenamiento**.

---

## 2. Suposiciones y restricciones

### 2.1 Suposiciones

- Existe una **Historia Clínica Electrónica (HCE/EHR)** con acceso controlado a datos clínicos anonimizados.
- El entrenamiento del modelo se realiza en un entorno **offline**.
- El modelo se ejecuta como servicio dentro de un **contenedor Docker**.
- Los modelos entrenados se almacenan en un **bucket externo**, no dentro del repositorio.
- FastAPI es la tecnología de inferencia.

### 2.2 Restricciones

- Requisitos de **privacidad y ética médica**.
- Manejo de **desbalance extremo** por enfermedades huérfanas.
- Soporte para ejecución **local o en la nube**.

### 2.3 Implicaciones

- Separación clara entre pipeline offline (entrenamiento) y online (serving).
- Uso obligatorio de modelos **ONNX**.
- Todos los artefactos versionados fuera del repositorio.

---

## 3. Arquitectura general

```mermaid
flowchart LR
    subgraph Datos
        A[EHR / HCE] --> B[Ingesta batch<br/>Python + Airflow]
        B --> C[Data Lake / DW<br/>Parquet + SQL]
    end

    subgraph Offline (ML)
        C --> D[Preprocesamiento<br/>pandas + sklearn]
        D --> E[Entrenamiento & tuning<br/>Modelos tabulares]
        E --> F[Evaluación<br/>ROC-AUC, F1, PR-AUC]
        F --> G[Registro de modelo<br/>MLflow Model Registry]
        G --> H[Exportación a ONNX<br/>subida a bucket]
    end

    subgraph Online (Serving)
        H --> I[Inferencia<br/>FastAPI + Docker + ONNXRuntime]
        I --> J[(Médico / HCE)]
    end

    subgraph Ops (MLOps)
        K[GitHub Actions<br/>CI: tests<br/>CD: build + deploy] --> I
        I --> L[Monitoring<br/>logs + métricas]
        L --> M[Retraining<br/>Airflow / Scripts]
        M --> D
    end
```

---

## 4. Etapas del pipeline end-to-end

### 4.1 Ingesta y gobierno de datos

Incluye extracción desde EHR, anonimización, validación, carga a Data Lake y DW.  
Tecnologías: **Airflow**, **pandas**, **Parquet**, **SQL**, **S3/GCS/Azure**.

### 4.2 Preprocesamiento y feature engineering

Incluye validación de esquema, manejo de outliers, encoding, escalado, features clínicas.  
Tecnologías: `sklearn.Pipeline`, `ColumnTransformer`.

### 4.3 Manejo de desbalance y modelado

Técnicas: oversampling, class weights, enfoque de dos etapas, ajuste de umbrales.  
Modelos típicos: Logistic Regression, RandomForest, XGBoost.

### 4.4 Entrenamiento y validación

Validación: StratifiedKFold, temporal split, evaluación por subgrupos.  
Métricas: ROC-AUC, F1, PR-AUC, recall para clases raras.

### 4.5 Registro y exportación a ONNX

- MLflow Tracking + Model Registry.
- Exportación a ONNX (`skl2onnx`).
- Almacenamiento en bucket externo.

### 4.6 Servicio de inferencia (FastAPI + Docker)

- Descarga de modelo ONNX desde bucket.
- Inferencia con `ONNXRuntime`.
- Logging obligatorio de predicciones en TXT:
  - `predicciones_dev.txt`
  - `predicciones_prod.txt`

### 4.7 CI/CD y MLOps

- GitHub Actions:

  - **Test stage**:
    - descarga de modelo ONNX
    - descarga de datos de prueba
    - pruebas: inferencia + métrica mínima
  - **Build/Promote stage**:
    - construcción de imagen Docker
    - push a GHCR
    - despliegue a dev o prod

- Monitoreo:
  - Logs de predicciones
  - Detección de drift
  - Ciclo de reentrenamiento

---

## 5. Implementación actual en el repositorio

- API FastAPI (`app/app.py`)
- Modelo placeholder con reglas (`model/rules.py`)
- Dockerfile para contenerización
- GitHub Actions (CI/CD)
- Pruebas unitarias (`tests/test_rules.py`)

---

## 6. CHANGELOG — Propuesta inicial vs reestructurada

### Propuesta inicial (Semanas 1–2)

- Descripción general sin detalle técnico profundo.
- Mención del problema y tipos de datos.
- Pipeline conceptual (diseño, desarrollo, producción).

### Propuesta reestructurada (Semanas 5–6)

- Detalle completo de:
  - ingestión
  - validación
  - preprocesamiento
  - modelado
  - exportación ONNX
  - despliegue
  - CI/CD
  - monitoreo y reentrenamiento
- Tecnologías justificadas por etapa.
- Integración explícita con proyecto final (dev/prod + modelos ONNX + buckets).

---

## 7. Estructura del repositorio

```
/mlops-rare-disease
├─ README.md
├─ pipeline.md
├─ CHANGELOG.md
├─ Dockerfile
├─ .dockerignore
├─ requirements.txt
├─ app/
│  └─ app.py
├─ model/
│  └─ rules.py
├─ tests/
│  └─ test_rules.py
└─ .github/
   └─ workflows/
      ├─ pr-ci.yml
      └─ develop-cicd.yml
```
