#code for API, use predict in production lecture solution as template
import pandas as pd

from stroke.logic.model import load_model

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 💡 Preload the model to accelerate the predictions
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = load_model()

#http://0.0.0.0:8000/predict?id=4567&gender=Other&age=45&hypertension=0&heart_disease=1&ever_married=Yes&work_type=Private&Residence_type=Urban&avg_glucose_level=112.4&bmi=21&smoking_status=formerly%20smoked


#gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status

@app.get("/predict")
def predict(
        gender: str,
        age: float,
        hypertension: int,
        heart_disease: int,
        ever_married: str,
        work_type: str,
        Residence_type: str,
        avg_glucose_level: float,
        bmi: float,
        smoking_status: str
    ):      # 1
    """
    Make a single course prediction.
    """

    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    X_pred = pd.DataFrame(locals(), index=[0])


    model = app.state.model

    y_pred = model.predict(X_pred)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(stroke=int(y_pred))


@app.get("/")
def root():
    return dict(greeting="Hello")
