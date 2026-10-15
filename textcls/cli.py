"""Command line interface for the text classification experiment."""

import argparse

from .config import DEFAULT_FEATURE_METHOD, DEFAULT_MODEL, FEATURE_CONFIGS
from .experiments import compare_representations, plot_best_training_curve, train_final, tune_models
from .models import default_params


def _hidden_layers(value):
    return tuple(int(part) for part in value.replace("x", ",").split(",") if part)


def build_parser():
    parser = argparse.ArgumentParser(description="Run Experiment 1 news text classification.")
    parser.add_argument("--mode", choices=["final", "tune", "representations", "all"], default="all")
    parser.add_argument("--train-csv", default="train_data.csv")
    parser.add_argument("--test-csv", default="test_data_unlabeled.csv")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--feature-method", choices=sorted(FEATURE_CONFIGS), default=DEFAULT_FEATURE_METHOD)
    parser.add_argument("--model", choices=["nb", "svm", "lr", "mlp"], default=DEFAULT_MODEL)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--hidden-layer-sizes", type=_hidden_layers, default=None, help="Example: 50 or 100,50")
    parser.add_argument("--no-validate", action="store_true", help="Skip the validation printout in final mode.")
    parser.add_argument("--plot-curves", action="store_true", help="Generate epoch-accuracy curves in tune mode.")
    parser.add_argument("--model-results", default="model_comparison.csv")
    parser.add_argument("--representation-results", default="representation_comparison.csv")
    parser.add_argument("--representation-pivot", default="representation_comparison_pivot.csv")
    return parser


def _selected_model_params(args):
    params = default_params(args.model)
    if args.alpha is not None:
        params["alpha"] = args.alpha
    if args.C is not None:
        params["C"] = args.C
    if args.kernel is not None:
        params["kernel"] = args.kernel
    if args.max_iter is not None:
        params["max_iter"] = args.max_iter
    if args.hidden_layer_sizes is not None:
        params["hidden_layer_sizes"] = args.hidden_layer_sizes
    return params


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode in ("tune", "all"):
        tune_models(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            feature_method=args.feature_method,
            output_csv=args.model_results,
            plot_curves=args.plot_curves,
        )
    if args.mode in ("representations", "all"):
        compare_representations(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_csv=args.representation_results,
            pivot_csv=args.representation_pivot,
        )
    if args.mode == "all":
        plot_best_training_curve(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_png="pic/best_hashing_mlp_loss_accuracy.png",
        )
    if args.mode in ("final", "all"):
        final_feature_method = "hashing" if args.mode == "all" else args.feature_method
        final_model = "mlp" if args.mode == "all" else args.model
        final_params = (
            {"hidden_layer_sizes": (50,), "alpha": 0.0001, "max_iter": 200}
            if args.mode == "all"
            else _selected_model_params(args)
        )
        train_final(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_csv=args.output,
            feature_method=final_feature_method,
            model=final_model,
            model_params=final_params,
            validate=not args.no_validate,
        )


if __name__ == "__main__":
    main()

