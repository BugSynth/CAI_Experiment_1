"""Evaluation helpers."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def score_predictions(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
    }


def results_frame(rows):
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False).reset_index(drop=True)

