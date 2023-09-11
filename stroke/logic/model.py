import numpy as np
import pandas as pd
import time
import pickle

from colorama import Fore, Style

# Timing the TF import
print(Fore.BLUE + "\nLoading KNClassifier..." + Style.RESET_ALL)
start = time.perf_counter()

from sklearn.neighbors import KNeighborsClassifier

end = time.perf_counter()
print(f"\n✅ KNClassifier loaded ({round(end - start, 2)}s)")

def load_model():

    pipeline = pickle.load(open("/Users/jmadu1/code/joannesmadu/chance-of-stroke/pipeline.pkl", "rb"))

    return pipeline


def pipeline_predict(new_data):

    pipeline = pickle.load(open("/Users/jmadu1/code/joannesmadu/chance-of-stroke/pipeline.pkl", "rb"))

    predicted_class = pipeline.predict(new_data.loc[:,'gender':'smoking_status'])

    return predicted_class


def initialize_model(model):

    model = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)

    print("✅ Model initialized")

    return model

def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray
        ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(X)
    model.transform(X)

    print(f"✅ Model trained on {len(X)} rows with {5} neighbors")

    return model
