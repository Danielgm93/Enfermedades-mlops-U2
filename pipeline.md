# Pipeline MLOps — Detección de Enfermedades (Comunes y Huérfanas)

> Panorama general end‑to‑end para un sistema que a partir de **síntomas** de un paciente, predice el estado:  
> **NO ENFERMO · ENFERMEDAD LEVE · ENFERMEDAD AGUDA · ENFERMEDAD CRÓNICA**

## 1) Diseño (restricciones y tipos de datos)

- **Tipos de datos**: síntomas numéricos/ordinales (intensidad de dolor, fiebre), categóricos (antecedentes), texto corto (síntomas libres), señales/imagen (posible expansión).
- **Privacidad & Ética**: PHI/PII protegida, consentimiento, minimización de datos, trazabilidad de acceso.
- **Desbalance y escasez**: enfermedades huérfanas → **pocos datos**. Considerar: _few-shot_, _transfer learning_, _data augmentation_, calibración y umbrales conservadores.
- **No objetivos**: el sistema **no es diagnóstico médico**; asiste la priorización y derivación.
- **Métricas**: para comunes → AUROC/F1/Sensitivity/Specificity; para huérfanas → **recall/NPV**, _PR-AUC_, _calibration error_.

## 2) Desarrollo (fuentes, manejo, modelos, validación)

### Fuentes y manejo

- **Ingesta** desde EHRs/CSV/APIs.
- **Validación de datos**: esquema, tipos, rangos, valores faltantes, _leakage_ (p. ej., etiqueta en features).
- **Feature store**: definiciones consistentes (train/serve), _time-travel_.

### Modelos candidatos

- Baseline interpretable (LogReg/GBM) para comunes.
- Para huérfanas: **transfer learning**, **meta-learning** o **ensembles** con calibración (Platt/Isotonic).
- Umbrales y triage: salida “**no seguro**” → ruta a especialista.

### Validación / Test

- **Validación estratificada** por prevalencia y centro.
- **Holdout temporal** si hay series.
- **Evaluación por subgrupos** (edad/sexo/centro) para equidad.
- **Pruebas unitarias** (código), **tests de datos** (drift/calidad), **tests de desempeño** (degradación máxima permitida).

## 3) Producción (despliegue, monitoreo, re‑entrenamiento)

- **Despliegue**: servicio HTTP _as-a-service_ (API), contenedores (Docker), opción _edge_ para clínicas desconectadas.
- **Monitoreo**:
  - _Servicio_: latencia, errores, disponibilidad.
  - _Datos_: _schema_, rangos, _missingness_, **drift** (PSI/KL).
  - _Modelo_: AUROC/F1/Recall, calibración, _alertas_ cuando se degrada.
- **Ciclo de mejora**: _feedback loop_ (etiquetas posteriores), **re‑entrenamiento** programado o condicional (cuando hay datos nuevos o _drift_).
- **Gobierno**: versionado de **datos/modelos/código**, _model registry_, auditoría y _model cards_.

## 4) Diagrama general

```mermaid
flowchart LR
  subgraph Ingesta
    A[EHR/CSV/APIs] --> B[Validación de datos
(esquema/rangos)]
    B --> C[Feature Store
 + Time-travel]
  end

  subgraph Entrenamiento
    C --> D[Preparación de datos]
    D --> E[Entrenamiento
(baseline / transfer / meta)]
    E --> F[Evaluación & Calibración]
    F -->|OK| G[Registro de Modelo
(Model Registry)]
  end

  subgraph Despliegue
    G --> H[Imagen Docker
(Serving API)]
    H --> I[Orquestación (K8s/Edge)]
  end

  subgraph Operación
    I --> J[Predicciones]
    J --> K[Monitoreo servicio]
    J --> L[Monitoreo datos]
    J --> M[Monitoreo modelo]
    L --> N{Drift?}
    M --> N
    N -->|Sí| O[Re-entrenar]
    O --> E
    N -->|No| P[Operación continua]
  end
```

## 5) Notas de implementación mínima (para la práctica)

- _Este repositorio incluye_ una API de ejemplo con una función determinística que devuelve uno de los 4 estados.
- Sustituir por un modelo real cuando exista, manteniendo **contratos** de entrada/salida y pruebas.
