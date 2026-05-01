# data_loader.py - Loads train data and model for the Optima DSS

import pandas as pd
import pickle
from config import TRAIN_DATA_FILE, MODEL_FILE

def load_train_data():
    """Loads and returns the training dataset."""
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    print(f"train_data loaded: {len(train_data):,} rows")
    return train_data


def load_model():
    """Loads and returns the trained XGBoost model."""
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded")
    return model


def get_product_avg_price(train_data):
    """
    Computes average price per product from training data.
    Used for Price_Relative computation in what-if functions.
    """
    return train_data.groupby('Product ID')['Avg_Price'].mean()


def get_pr_range(train_data):
    """
    Returns the 1st and 99th percentile of Price_Relative from training data.
    Used by validate_scenario to flag out-of-range inputs.
    """
    pr_min = float(train_data['Price_Relative'].quantile(0.01))
    pr_max = float(train_data['Price_Relative'].quantile(0.99))
    return pr_min, pr_max


def load_all():
    """
    Loads everything needed by the dashboard.
    Returns: train_data, model, product_avg_price, pr_min, pr_max
    """
    train_data        = load_train_data()
    model             = load_model()
    product_avg_price = get_product_avg_price(train_data)
    pr_min, pr_max    = get_pr_range(train_data)

    return train_data, model, product_avg_price, pr_min, pr_max