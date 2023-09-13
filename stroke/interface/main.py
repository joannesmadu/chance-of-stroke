import numpy as np
import pandas as pd

from colorama import Fore, Style

from stroke.logic.model import pipeline_predict
from stroke.logic.preprocess import clean_data

def preprocess():

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    clean_data()

    print("✅ preprocess_and_train() done")



def pred() -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    pipeline_predict()

    print(f"✅ pred() done")


if __name__ == '__main__':
    try:
        preprocess()
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
