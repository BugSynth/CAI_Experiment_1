"""Plotting helpers for experiment figures."""

import os

from sklearn.metrics import accuracy_score

from .config import PLOT_DIR, RANDOM_STATE
from .models import build_model, canonical_model_name

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def params_label(model, params):
    name = canonical_model_name(model)
    if name == "MultinomialNB":
        return f"alpha={params.get('alpha', 1.0)}"
    if name == "SVC":
        return f"kernel=linear,C={params.get('C', 1.0)}"
    if name == "LogisticRegression":
        return f"C={params.get('C', 1.0)},max_iter={params.get('max_iter', 1000)}"
    if name == "MLPClassifier":
        hidden = "x".join(str(h) for h in params.get("hidden_layer_sizes", (100,)))
        return f"hidden={hidden},alpha={params.get('alpha', 0.0001)}"
    return ""


def plot_filename(model, params):
    name = canonical_model_name(model)
    if name == "MultinomialNB":
        key = f"alpha{params.get('alpha', 1.0)}"
    elif name == "SVC":
        key = f"linear_C{params.get('C', 1.0)}"
    elif name == "LogisticRegression":
        key = f"C{params.get('C', 1.0)}_iter{params.get('max_iter', 1000)}"
    elif name == "MLPClassifier":
        hidden = "_".join(str(h) for h in params.get("hidden_layer_sizes", (100,)))
        key = f"hidden{hidden}_alpha{params.get('alpha', 0.0001)}"
    else:
        key = "unknown"
    return f"{name}_{key}.png"


def record_epoch_accuracy(model, params, X_train, y_train, X_val, y_val, n_epochs=20):
    name = canonical_model_name(model)
    epochs = list(range(1, n_epochs + 1))

    if name in ("MultinomialNB", "SVC"):
        estimator = build_model(model, random_state=RANDOM_STATE, **params)
        estimator.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, estimator.predict(X_val))
        return epochs, [accuracy] * n_epochs

    if name == "LogisticRegression":
        params = {**params, "max_iter": 1}
    elif name == "MLPClassifier":
        params = {**params, "max_iter": 1}
    estimator = build_model(model, random_state=RANDOM_STATE, **params)
    estimator.set_params(warm_start=True)

    accuracies = []
    for _ in epochs:
        estimator.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_val, estimator.predict(X_val)))
    return epochs, accuracies


def plot_epoch_accuracy(model, params, epochs, accuracies, out_dir=PLOT_DIR):
    if not HAS_MATPLOTLIB:
        return None
    os.makedirs(out_dir, exist_ok=True)
    name = canonical_model_name(model)
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(epochs, accuracies, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.title(f"{name} ({params_label(model, params)})")
    plt.tight_layout()
    path = os.path.join(out_dir, plot_filename(model, params))
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_representation_comparison(df, feature_configs, model_names, out_dir=PLOT_DIR):
    if not HAS_MATPLOTLIB:
        return None
    os.makedirs(out_dir, exist_ok=True)
    labels = [cfg["label"] for cfg in feature_configs]
    x = range(len(labels))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for metric, ax in zip(["accuracy", "macro_f1"], axes):
        for i, model in enumerate(model_names):
            values = []
            for cfg in feature_configs:
                row = df[(df["method"] == cfg["method"]) & (df["model"] == model)]
                values.append(row[metric].iloc[0] if not row.empty else None)
            ax.bar([xi + (i - len(model_names) / 2 + 0.5) * width for xi in x], values, width=width, label=model)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0.70, 1.02)
        ax.set_title("Validation " + ("Accuracy" if metric == "accuracy" else "Macro-F1"))
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "representation_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

