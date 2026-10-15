"""Text representation methods behind one leak-free interface."""

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline

from .config import RANDOM_STATE

_METHOD_DEFAULTS = {
    "bow": {},
    "tfidf": {},
    "tfidf_char": {"analyzer": "char_wb", "ngram_range": (2, 5)},
    "hashing": {"n_features": 2 ** 18, "alternate_sign": False},
    "lsa": {"n_components": 200},
}


def available_methods():
    return sorted(_METHOD_DEFAULTS)


def make_vectorizer(method="tfidf", **kwargs):
    if method not in _METHOD_DEFAULTS:
        raise ValueError(f"Unknown feature method {method!r}. Choose from {available_methods()}")

    defaults = dict(_METHOD_DEFAULTS[method])
    if method == "bow":
        return CountVectorizer(**{**defaults, **kwargs})
    if method in ("tfidf", "tfidf_char"):
        return TfidfVectorizer(**{**defaults, **kwargs})
    if method == "hashing":
        return HashingVectorizer(**{**defaults, **kwargs})
    if method == "lsa":
        n_components = kwargs.pop("n_components", defaults["n_components"])
        return make_pipeline(
            TfidfVectorizer(**kwargs),
            TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
        )
    raise AssertionError("unreachable")


def _add_bias(X):
    if X is None:
        return None
    if sparse.issparse(X):
        bias = sparse.csr_matrix(np.ones((X.shape[0], 1), dtype=X.dtype))
        return sparse.hstack([X, bias]).tocsr()
    bias = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, bias])


def extract_features(X_train, X_eval=None, X_test=None, method="tfidf", **kwargs):
    """Fit on training text only and transform validation/test text."""
    vectorizer = make_vectorizer(method, **kwargs)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_eval_vec = vectorizer.transform(X_eval) if X_eval is not None else None
    X_test_vec = vectorizer.transform(X_test) if X_test is not None else None
    return _add_bias(X_train_vec), _add_bias(X_eval_vec), _add_bias(X_test_vec)

