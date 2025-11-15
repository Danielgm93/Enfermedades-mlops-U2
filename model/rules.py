
from typing import Dict, Any

LABELS = [
    "NO ENFERMO",
    "ENFERMEDAD LEVE",
    "ENFERMEDAD AGUDA",
    "ENFERMEDAD CRÓNICA",
]

def predict_status(features: Dict[str, Any]) -> str:
    """
    Reglas simples y determinísticas para demostrar el flujo.
    Espera como mínimo tres campos numéricos:
      - fever (0-10)
      - pain (0-10)
      - days (duración de síntomas, entero >=0)
    Campos opcionales: comorbidity (0-5), age (años).
    """
    fever = float(features.get("fever", 0))
    pain = float(features.get("pain", 0))
    days = float(features.get("days", 0))
    comorb = float(features.get("comorbidity", 0))
    age = float(features.get("age", 40))

    # Normaliza a rangos razonables
    fever = max(0.0, min(10.0, fever))
    pain = max(0.0, min(10.0, pain))
    days = max(0.0, days)
    comorb = max(0.0, min(5.0, comorb))
    age = max(0.0, min(120.0, age))

    # Puntaje compuesto (ejemplo simple)
    score = 0.5*fever + 0.4*pain + 0.1*min(days, 30) + 0.3*comorb
    # Factor crónico si síntomas muy prolongados
    chronic_boost = 2.0 if days >= 30 else 0.0
    score += chronic_boost

    # Reglas para cubrir los 4 estados
    if score < 3.0 and days < 2:
        return LABELS[0]  # NO ENFERMO
    if score < 6.0 and days < 7:
        return LABELS[1]  # LEVE
    if score < 9.5 and days < 30:
        return LABELS[2]  # AGUDA
    if age >= 65 and days >= 7:
        score += 1.5  # Riesgo adicional en mayores
    return LABELS[3]      # CRÓNICA


if __name__ == "__main__":
    # Pequeña demo
    examples = [
        {"fever": 0.5, "pain": 1, "days": 0},            # NO ENFERMO
        {"fever": 2.0, "pain": 3, "days": 3},            # LEVE
        {"fever": 6.0, "pain": 6, "days": 10, "comorbidity": 1},  # AGUDA
        {"fever": 4.0, "pain": 4, "days": 45},           # CRÓNICA
    ]
    for x in examples:
        print(x, "->", predict_status(x))
