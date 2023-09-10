import os
import numpy as np

##################  VARIABLES  ##################
GCP_PROJECT = "wagon-bootcamp-392717"
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "Downloads")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "Downloads")

COLUMN_NAMES_RAW = ['id','gender','age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type','avg_glucose_level','bmi','smoking_status']

DTYPES_RAW = {
    "id":"int16",
    "gender":"object",
    "age": "int16",
    "hypertension":"int16",
    "heart_disease":"int16",
    "ever_married":"object",
    "work_type":"object",
    "Residence_type":"object",
    "bmi": "float32",
    "avg_glucose_level": "float32",
    "smoking_status":"object"
}

DTYPES_PROCESSED = np.float32

# "/Users/jmadu1/.lewagon/mlops/data/raw/query_2009-01-01_2015-01-01_1k.csv"
# "/Users/jmadu1/.lewagon/mlops/data/processed/train_processed_1k.csv"
# "/Users/jmadu1/Documents/healthcare-dataset-stroke-data.csv"
