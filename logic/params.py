import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "500k"
CHUNK_SIZE = 100000
GCP_PROJECT = "wagon-bootcamp-392717"
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "Downloads")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "Downloads")

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

DTYPES_RAW = {
    "age": "int16",
    "bmi": "float32",
    "avg_glucose_level": "float32"
}

DTYPES_PROCESSED = np.float32

# "/Users/jmadu1/.lewagon/mlops/data/raw/query_2009-01-01_2015-01-01_1k.csv"
# "/Users/jmadu1/.lewagon/mlops/data/processed/train_processed_1k.csv"
# "/Users/jmadu1/Documents/healthcare-dataset-stroke-data.csv"

"""
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   id                 5110 non-null   int64
 1   gender             5110 non-null   object
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64
 4   heart_disease      5110 non-null   int64
 5   ever_married       5110 non-null   object
 6   work_type          5110 non-null   object
 7   Residence_type     5110 non-null   object
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object
 11  stroke             5110 non-null   int64
"""
