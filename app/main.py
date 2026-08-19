from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import pickle


# --------------------------------
# Paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Data_Analysis" / "linear_model.pkl"

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------
# Load model
# --------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------
# FastAPI application
# --------------------------------

app = FastAPI(
    title="Student Math Score Prediction API",
    description="API for predicting student's mathematics score",
    version="1.0.0"
)


# --------------------------------
# Serve static files
# --------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------
# Input schema
# --------------------------------

class StudentData(BaseModel):

    gender: str

    race_ethnicity: str

    parental_level_of_education: str

    lunch: str

    test_preparation_course: str

    reading_score: float

    writing_score: float


# --------------------------------
# Home page
# --------------------------------

@app.get("/")
def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(data: StudentData):

    input_data = pd.DataFrame([{
        "gender": data.gender,
        "race_ethnicity": data.race_ethnicity,
        "parental_level_of_education": data.parental_level_of_education,
        "lunch": data.lunch,
        "test_preparation_course": data.test_preparation_course,
        "reading_score": data.reading_score,
        "writing_score": data.writing_score
    }])

    # Same encoding used during model training
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ],
        drop_first=False,
        dtype=int
    )

    # Make sure input columns are exactly the same as training columns
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    return {
        "predicted_math_score": round(float(prediction[0]), 2)
    }