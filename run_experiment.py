"""Unified entry point for Experiment 1.

Running without arguments reproduces the complete experiment workflow:
    1. TF-IDF model hyperparameter search with the expanded grid.
    2. Text representation ablation.
    3. Hashing + MLP loss/accuracy training curve.
    4. Final Hashing + MLP prediction file.
"""

from textcls.cli import _selected_model_params, build_parser
from textcls.experiments import (
    compare_representations,
    plot_best_training_curve,
    train_final,
    tune_models,
)


def main(argv=None):
    parser = build_parser()
    parser.set_defaults(mode="all", feature_method="tfidf", plot_curves=False)
    args = parser.parse_args(argv)

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
