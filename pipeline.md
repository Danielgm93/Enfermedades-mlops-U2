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

  subgraph DIS ["Diseno del sistema"]
    D1["Definir problema\nEnfermedades comunes y huerfanas"]
    D2["Restricciones y limitaciones\nPocos datos, privacidad, no es diagnostico"]
    D3["Definir tipos de datos\nSintomas numericos, categoricos, texto"]
    D4["Definir salidas del modelo\nNO ENFERMO / LEVE / AGUDA / CRONICA"]
    D5["Definir metricas y objetivos\nRecall en huerfanas, precision global"]

    D1 --> D2 --> D3 --> D4 --> D5
  end

  subgraph DEV ["Desarrollo del modelo"]
    S1["Fuentes de datos\nEHR, historiales, CSV, estudios"]
    S2["Ingesta y validacion\nEsquema, rangos, faltantes, outliers"]
    S3["Preprocesamiento\nNormalizacion, codificacion, train/val/test"]
    S4["Modelado para comunes\nModelos clasicos bien entrenados"]
    S5["Modelado para huerfanas\nFew-shot, transfer learning, reglas"]
    S6["Evaluacion y seleccion\nMetricas por clase y tipo de enfermedad"]
    S7["Empaquetar logica de prediccion\nFuncion predict_status"]

    S1 --> S2 --> S3 --> S4
    S3 --> S5
    S4 --> S6
    S5 --> S6 --> S7
  end

  subgraph PROD ["Produccion y MLOps"]
    P1["Definir contrato de entrada/salida\nJSON con al menos 3 sintomas"]
    P2["Implementar servicio web\nFastAPI/Flask con endpoint /predict"]
    P3["Pruebas unitarias\nCasos que cubren los 4 estados"]
    P4["Dockerfile\nInstalar dependencias y exponer servicio"]
    P5["Construir imagen Docker\ndocker build ..."]
    P6["Medico ejecuta contenedor\ndocker run -p ... rare-disease"]
    P7["Uso en consulta\nMedico envia sintomas y recibe estado"]
    P8["Monitoreo basico\nLogs, errores, patrones de uso"]
    P9["Recoleccion de nuevos datos\nPara reentrenar y mejorar el modelo"]

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9
  end

  D5 --> S1
  S7 --> P1
 ```

## 5) Notas de implementación mínima (para la práctica)

- _Este repositorio incluye_ una API de ejemplo con una función determinística que devuelve uno de los 4 estados.
- Sustituir por un modelo real cuando exista, manteniendo **contratos** de entrada/salida y pruebas.
