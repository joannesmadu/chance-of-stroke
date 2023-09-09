import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple
from keras import Model

# Timing the TF import
print(Fore.BLUE + "\nLoading KNClassifier..." + Style.RESET_ALL)
start = time.perf_counter()

from sklearn.neighbors import KNeighborsClassifier

end = time.perf_counter()
print(f"\n✅ KNClassifier loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple) -> Model:

    model = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)

    print("✅ Model initialized")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray
        ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(X)
    model.transform(X)

    print(f"✅ Model trained on {len(X)} rows with {5} neighbors")

    return model
