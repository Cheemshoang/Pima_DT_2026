import pandas as pd
from pathlib import Path

def get_data():
    DATA_DIR = Path.cwd() / 'data'
    TRAIN_DIR = DATA_DIR / 'train.csv'
    VAL_DIR = DATA_DIR / 'val.csv'
    TEST_DIR = DATA_DIR / 'test.csv' 
    TEST_TARGET_DIR = DATA_DIR / 'y_test.csv'

    X_train = pd.read_csv(TRAIN_DIR)
    X_val = pd.read_csv(VAL_DIR)
    X_test = pd.read_csv(TEST_DIR)
    y_test = pd.read_csv(TEST_TARGET_DIR)

    y_train = X_train["quality"]
    X_train = X_train.drop(columns=["quality"])

    return X_train, y_train, X_val, X_test, y_test