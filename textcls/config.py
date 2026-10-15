"""Shared experiment configuration."""

RANDOM_STATE = 42
TEST_SIZE = 0.2
PLOT_DIR = "pic"

DEFAULT_FEATURE_METHOD = "hashing"
DEFAULT_MODEL = "mlp"

COMMON_WORD_KWARGS = {
    "lowercase": True,
    "stop_words": "english",
    "min_df": 2,
    "max_df": 0.9,
    "ngram_range": (1, 1),
}

FEATURE_CONFIGS = {
    "bow": {
        "label": "BoW",
        "method": "bow",
        "kwargs": {**COMMON_WORD_KWARGS, "max_features": 10000},
    },
    "tfidf": {
        "label": "TF-IDF",
        "method": "tfidf",
        "kwargs": {**COMMON_WORD_KWARGS, "max_features": 10000, "sublinear_tf": True},
    },
    "tfidf_char": {
        "label": "TF-IDF-char",
        "method": "tfidf_char",
        "kwargs": {
            "analyzer": "char_wb",
            "ngram_range": (2, 5),
            "lowercase": True,
            "min_df": 2,
            "max_df": 0.9,
            "max_features": 10000,
            "sublinear_tf": True,
        },
    },
    "hashing": {
        "label": "Hashing",
        "method": "hashing",
        "kwargs": {
            "n_features": 2 ** 15,
            "alternate_sign": False,
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 1),
        },
    },
    "lsa": {
        "label": "LSA",
        "method": "lsa",
        "kwargs": {**COMMON_WORD_KWARGS, "max_features": 10000, "sublinear_tf": True, "n_components": 200},
    },
}

MODEL_ALIASES = {
    "nb": "MultinomialNB",
    "svm": "SVC",
    "lr": "LogisticRegression",
    "mlp": "MLPClassifier",
}

DEFAULT_MODEL_PARAMS = {
    "nb": {"alpha": 0.5},
    "svm": {"kernel": "linear", "C": 10.0},
    "lr": {"C": 10.0, "max_iter": 2000},
    "mlp": {"hidden_layer_sizes": (50,), "alpha": 0.0001, "max_iter": 200},
}

