import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("/Users/jmadu1/Documents/healthcare-dataset-stroke-data.csv")

from colorama import Fore, Style

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """

        # AGE PIPE
        age_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        # BMI PIPE
        bmi_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # GLUCOSE PIPE
        glucose_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        #CATEGORICAL PIPE
        categ_pipe = Pipeline([('cat_transformer', OneHotEncoder(handle_unknown='ignore'))])

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer([
                ("age_pipe", age_pipe, ["age"]),
                ("bmi_pipe", bmi_pipe, ["bmi"]),
                ("glucose_pipe", glucose_pipe, ["avg_glucose_level"]),
                ("categoricals", categ_pipe, ["ever_married","work_type","Residence_type","smoking_status","gender"])],
                remainder='passthrough', n_jobs=-1)
        return final_preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
