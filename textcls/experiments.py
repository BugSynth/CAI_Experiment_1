"""End-to-end experiment workflows used by the CLI and compatibility scripts."""

import time
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .config import FEATURE_CONFIGS, PLOT_DIR, RANDOM_STATE, TEST_SIZE
from .data import load_data, write_predictions
from .evaluation import results_frame, score_predictions
from .features import extract_features
from .models import canonical_model_name, candidate_grid, default_params, run_model, short_model_name
from .plotting import HAS_MATPLOTLIB, plot_epoch_accuracy, plot_representation_comparison, record_epoch_accuracy
from .preprocessing import preprocess_many


def _prepare_splits(train_csv, test_csv, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X, y, X_test = load_data(train_csv, test_csv)
    X = preprocess_many(X)
    X_test = preprocess_many(X_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X, y, X_test, X_train, X_val, y_train, y_val


def feature_kwargs(method):
    return dict(FEATURE_CONFIGS[method]["kwargs"])


def tune_models(
    train_csv="train_data.csv",
    test_csv="test_data_unlabeled.csv",
    feature_method="tfidf",
    output_csv="model_comparison.csv",
    plot_curves=False,
    plot_dir=PLOT_DIR,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    _, _, _, X_train, X_val, y_train, y_val = _prepare_splits(train_csv, test_csv, test_size, random_state)
    X_train_vec, X_val_vec, _ = extract_features(
        X_train, X_eval=X_val, method=feature_method, **feature_kwargs(feature_method)
    )

    rows = []
    for model, params in candidate_grid():
        print(f"Running {canonical_model_name(model)} {params} ...", flush=True)
        predictions, _ = run_model(model, X_train_vec, y_train, X_val_vec, random_state=random_state, **params)
        rows.append(
            {
                "model": canonical_model_name(model),
                "params": params,
                **score_predictions(y_val, predictions),
            }
        )

    df = results_frame(rows)
    df.to_csv(output_csv, index=False)
    print(df.to_string(index=False), flush=True)

    if plot_curves:
        for model, params in candidate_grid():
            epochs, accs = record_epoch_accuracy(model, params, X_train_vec, y_train, X_val_vec, y_val)
            path = plot_epoch_accuracy(model, params, epochs, accs, out_dir=plot_dir)
            if path:
                print(f"saved {path}", flush=True)
    return df


def compare_representations(
    train_csv="train_data.csv",
    test_csv="test_data_unlabeled.csv",
    output_csv="representation_comparison.csv",
    pivot_csv="representation_comparison_pivot.csv",
    plot_dir=PLOT_DIR,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    _, _, _, X_train, X_val, y_train, y_val = _prepare_splits(train_csv, test_csv, test_size, random_state)
    model_specs = [
        ("NB", "nb", {"alpha": 0.5}, True),
        ("LR", "lr", {"C": 10.0, "max_iter": 2000}, False),
        ("SVM", "svm", {"kernel": "linear", "C": 10.0}, False),
        ("MLP", "mlp", {"hidden_layer_sizes": (50,), "alpha": 0.0001, "max_iter": 200}, False),
    ]
    rows = []
    configs = list(FEATURE_CONFIGS.values())

    for config in configs:
        print(f"\n=== {config['label']} ({config['method']}) ===", flush=True)
        start = time.time()
        X_train_vec, X_val_vec, _ = extract_features(
            X_train, X_eval=X_val, method=config["method"], **dict(config["kwargs"])
        )
        extract_sec = time.time() - start
        for display_name, model, params, non_negative_only in model_specs:
            if non_negative_only and config["method"] == "lsa":
                rows.append(
                    {
                        "method": config["method"],
                        "label": config["label"],
                        "model": display_name,
                        "accuracy": None,
                        "macro_f1": None,
                        "n_features": X_train_vec.shape[1],
                        "extract_sec": extract_sec,
                        "fit_sec": None,
                    }
                )
                print(f"  {display_name}: skipped (LSA contains negative values)", flush=True)
                continue

            start = time.time()
            predictions, _ = run_model(model, X_train_vec, y_train, X_val_vec, random_state=random_state, **params)
            fit_sec = time.time() - start
            scores = score_predictions(y_val, predictions)
            rows.append(
                {
                    "method": config["method"],
                    "label": config["label"],
                    "model": display_name,
                    **scores,
                    "n_features": X_train_vec.shape[1],
                    "extract_sec": extract_sec,
                    "fit_sec": fit_sec,
                }
            )
            print(
                f"  {display_name}: accuracy={scores['accuracy']:.4f} macro-F1={scores['macro_f1']:.4f} "
                f"features={X_train_vec.shape[1]} fit={fit_sec:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    df.pivot(index="label", columns="model", values=["accuracy", "macro_f1"]).to_csv(pivot_csv)
    plot_path = plot_representation_comparison(df, configs, [spec[0] for spec in model_specs], out_dir=plot_dir)
    if plot_path:
        print(f"saved {plot_path}", flush=True)
    return df


def train_final(
    train_csv="train_data.csv",
    test_csv="test_data_unlabeled.csv",
    output_csv="predictions.csv",
    feature_method="hashing",
    model="mlp",
    model_params=None,
    validate=True,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    X, y, X_test, X_train, X_val, y_train, y_val = _prepare_splits(train_csv, test_csv, test_size, random_state)
    params = default_params(model) if model_params is None else dict(model_params)

    if validate:
        X_train_vec, X_val_vec, _ = extract_features(
            X_train, X_eval=X_val, method=feature_method, **feature_kwargs(feature_method)
        )
        val_predictions, _ = run_model(model, X_train_vec, y_train, X_val_vec, random_state=random_state, **params)
        scores = score_predictions(y_val, val_predictions)
        print(
            f"Validation: {canonical_model_name(model)} + {feature_method} "
            f"accuracy={scores['accuracy']:.4f} macro-F1={scores['macro_f1']:.4f}",
            flush=True,
        )

    X_all_vec, _, X_test_vec = extract_features(
        X, X_test=X_test, method=feature_method, **feature_kwargs(feature_method)
    )
    predictions, _ = run_model(model, X_all_vec, y, X_test_vec, random_state=random_state, **params)
    output_path = write_predictions(predictions, output_csv)
    print(
        f"Saved {len(predictions)} predictions to {output_path} "
        f"using {canonical_model_name(model)} + {feature_method} ({params})",
        flush=True,
    )
    return output_path


def plot_best_training_curve(
    train_csv="train_data.csv",
    test_csv="test_data_unlabeled.csv",
    output_png="pic/best_hashing_mlp_loss_accuracy.png",
    n_epochs=80,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """Plot loss/accuracy for the final Hashing + MLP configuration."""
    if not HAS_MATPLOTLIB:
        print("matplotlib unavailable; skipped best training curve.", flush=True)
        return None

    import os

    import matplotlib.pyplot as plt

    _, _, _, X_train, X_val, y_train, y_val = _prepare_splits(train_csv, test_csv, test_size, random_state)
    X_train_vec, X_val_vec, _ = extract_features(
        X_train, X_eval=X_val, method="hashing", **feature_kwargs("hashing")
    )

    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        alpha=0.0001,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )

    epochs, losses, accuracies = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(1, n_epochs + 1):
            model.fit(X_train_vec, y_train)
            epochs.append(epoch)
            losses.append(model.loss_)
            accuracies.append(accuracy_score(y_val, model.predict(X_val_vec)))

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax1.plot(epochs, losses, color="#d62728", label="Training loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, color="#1f77b4", label="Validation accuracy", linewidth=2)
    ax2.set_ylabel("Validation accuracy", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0.70, 0.95)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")
    fig.suptitle("Hashing + MLP Training Curve")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Saved best training curve to {output_png} "
        f"(best accuracy={max(accuracies):.4f}, final accuracy={accuracies[-1]:.4f})",
        flush=True,
    )
    return output_png

