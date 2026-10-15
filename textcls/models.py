"""Classifier construction, wrappers, and candidate grids."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import DEFAULT_MODEL_PARAMS, MODEL_ALIASES, RANDOM_STATE


def canonical_model_name(model):
    key = model.lower()
    return MODEL_ALIASES.get(key, model)


def short_model_name(model):
    canonical = canonical_model_name(model)
    for short, full in MODEL_ALIASES.items():
        if full == canonical:
            return short
    raise ValueError(f"Unknown model {model!r}. Choose from {sorted(MODEL_ALIASES)}")


def build_model(model, random_state=RANDOM_STATE, **params):
    name = canonical_model_name(model)
    if name == "MultinomialNB":
        return MultinomialNB(alpha=params.get("alpha", 1.0))
    if name == "SVC":
        return SVC(
            kernel=params.get("kernel", "linear"),
            C=params.get("C", 1.0),
            random_state=random_state,
        )
    if name == "LogisticRegression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "liblinear"),
            random_state=random_state,
        )
    if name == "MLPClassifier":
        return MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
            alpha=params.get("alpha", 0.0001),
            max_iter=params.get("max_iter", 200),
            random_state=random_state,
        )
    raise ValueError(f"Unknown model {model!r}. Choose from {sorted(MODEL_ALIASES)}")


def run_model(model, X_train, y_train, X_test, random_state=RANDOM_STATE, **params):
    estimator = build_model(model, random_state=random_state, **params)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    return predictions, estimator


def run_nb(X_train, y_train, X_test, alpha=1.0, y_test=None):
    predictions, estimator = run_model("nb", X_train, y_train, X_test, alpha=alpha)
    return (predictions, estimator) if y_test is not None else predictions


def run_svm(X_train, y_train, X_test, kernel="linear", C=1.0, random_state=RANDOM_STATE, y_test=None):
    predictions, estimator = run_model(
        "svm", X_train, y_train, X_test, kernel=kernel, C=C, random_state=random_state
    )
    return (predictions, estimator) if y_test is not None else predictions


def run_lr(X_train, y_train, X_test, C=1.0, max_iter=1000, random_state=RANDOM_STATE, y_test=None):
    predictions, estimator = run_model(
        "lr", X_train, y_train, X_test, C=C, max_iter=max_iter, random_state=random_state
    )
    return (predictions, estimator) if y_test is not None else predictions


def run_mlp(
    X_train,
    y_train,
    X_test,
    hidden_layer_sizes=(100,),
    alpha=0.0001,
    max_iter=200,
    random_state=RANDOM_STATE,
    y_test=None,
):
    predictions, estimator = run_model(
        "mlp",
        X_train,
        y_train,
        X_test,
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )
    return (predictions, estimator) if y_test is not None else predictions


def candidate_grid():
    candidates = []
    for alpha in [0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
        candidates.append(("nb", {"alpha": alpha}))
    for C in [0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]:
        candidates.append(("svm", {"kernel": "linear", "C": C}))
    for C in [0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]:
        for max_iter in [1000, 2000, 3000]:
            candidates.append(("lr", {"C": C, "max_iter": max_iter}))
    for hidden in [(30,), (50,), (100,), (50, 25)]:
        for alpha in [0.00001, 0.0001, 0.001]:
            candidates.append(("mlp", {"hidden_layer_sizes": hidden, "alpha": alpha, "max_iter": 200}))
    return candidates


def default_params(model):
    return dict(DEFAULT_MODEL_PARAMS[short_model_name(model)])

