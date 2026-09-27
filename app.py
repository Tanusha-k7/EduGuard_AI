
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Load the trained model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "eduguard_model.joblib"

model = joblib.load(MODEL_PATH)


# Create the FastAPI application
app = FastAPI(
    title="EduGuard AI API",
    description="Student Performance Prediction and Early Warning System",
    version="1.0.0"
)


# Input schema: 18 features expected by the trained model
class StudentInput(BaseModel):
    code_module: str
    code_presentation: str
    date_registration_num: float
    early_assessment_count: float
    early_mean_score: float
    early_max_score: float
    early_min_score: float
    early_total_clicks: float
    early_active_days: float
    early_unique_sites: float
    gender: str
    region: str
    highest_education: str
    imd_band: str
    age_band: str
    num_of_prev_attempts: float
    studied_credits: float
    disability: str


@app.get("/")
def home():
    return {
        "message": "EduGuard AI API is running",
        "model_loaded": True
    }


@app.post("/predict")
def predict_student_outcome(student: StudentInput):

    # Convert submitted data into a one-row DataFrame
    input_data = pd.DataFrame([student.model_dump()])

    # Predict student outcome and class probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Map each class to its probability
    probability_by_class = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    # Combined model-estimated probability of Fail or Withdrawn
    risk_probability = (
        probability_by_class.get("Fail", 0.0)
        + probability_by_class.get("Withdrawn", 0.0)
    )

    return {
        "predicted_outcome": str(prediction),
        "class_probabilities": {
            name: round(probability, 4)
            for name, probability in probability_by_class.items()
        },
        "risk_probability": round(risk_probability, 4),
        "note": (
            "Risk probability is the model-estimated combined probability "
            "of Fail or Withdrawn. It is not a guaranteed outcome."
        )
    }
