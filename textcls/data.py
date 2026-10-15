"""Data loading and output helpers."""

from pathlib import Path

import pandas as pd


def load_data(train_csv="train_data.csv", test_csv="test_data_unlabeled.csv"):
    """Load labeled training data and unlabeled test data.

    pandas.read_csv is required because the text column contains quoted
    multiline fields with embedded commas and quotes.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    X = train_df["text"].astype(str).tolist()
    y = train_df["target"].values
    X_test = test_df["text"].astype(str).tolist()
    return X, y, X_test


def write_predictions(predictions, output_path="predictions.csv"):
    """Write one prediction per line, no header and no index."""
    output_path = Path(output_path)
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)
    return output_path

