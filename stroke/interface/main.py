import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from stroke.logic.params import *
from stroke.logic.model import initialize_model, train_model
from stroke.logic.preprocess import preprocess_features

def preprocess_and_train() -> None:
    """
    - Retrieve dataset from Kaggle
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a KNClassifier model on it
    - Save the model
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)


    # Retrieve dataset from Kaggle or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", "healthcare-dataset-stroke-data.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")
        data = pd.read_csv(data_query_cache_path)

    else:
        print("Loading data from Kaggle server...")

        #import and set up Kaggle and API
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        #Download Kaggle dataset
        kaggle.api.dataset_download_file("fedesoriano/stroke-prediction-dataset", "healthcare-dataset-stroke-data.csv", path=None, force=False, quiet=True)

        # Save it locally to accelerate the next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    # Create (X_train, y_train) without data leaks
    split_ratio = 0.2

    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1)

    X_train = data_train.drop("stroke")
    y_train = data_train[["stroke"]]


    # Create X_train_processed using `preprocessor.py`
    X_train_processed = preprocess_features(X_train)

    # Train a model on the training set, using `model.py`
    model = None

    model = initialize_model(input_shape=X_train_processed.shape[1:])

    model = train_model(
        model, X_train_processed, y_train)

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental training on the (already preprocessed) dataset, stored locally

    - Load data
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunk, and final model weights on the local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: train in batches" + Style.RESET_ALL)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store the val_mae of each chunk

    # Iterate in chunks and partially fit on each chunk
    chunks = pd.read_csv(
        data_processed_path,
        chunksize=CHUNK_SIZE,
        header=None,
        dtype=DTYPES_PROCESSED
    )

    for chunk_id, chunk in enumerate(chunks):
        print(f"Training on preprocessed chunk n°{chunk_id}")

        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val_mae of each chunk in `metrics_val_list`
        # $CODE_BEGIN
        if model is None:
            model = initialize_model(input_shape=X_train_chunk.shape[1:])

        model = compile_model(model, learning_rate)

        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            validation_data=(X_val_chunk, y_val_chunk)
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)

        print(metrics_val_chunk)
        # $CODE_END

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

    # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        # preprocess()
        # train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
