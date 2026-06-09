import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import cast
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

try:
    from .dataset import LABELS, load_data_splits
    from .model import build_model
except ImportError:
    from dataset import LABELS, load_data_splits
    from model import build_model

if hasattr(sys.stdout, "reconfigure"):
    cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "random_forest_tuning_results.json"


def default_parameter_grid():
    return [
        {
            "max_features": 20000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "n_estimators": 120,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 20000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": 60,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 20000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 20000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 30000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 50000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 30000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": 80,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 30000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": 0.8,
            "rf_max_features": "sqrt",
        },
        {
            "max_features": 30000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": None,
            "bootstrap": True,
            "max_samples": None,
            "rf_max_features": "sqrt",
        },
    ]


def json_ready(params):
    clean = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            clean[key] = list(value)
        else:
            clean[key] = value
    return clean


def print_train_command(params):
    max_depth = "None" if params["max_depth"] is None else str(params["max_depth"])
    max_samples = "None" if params["max_samples"] is None else str(params["max_samples"])
    class_weight = "None" if params["class_weight"] is None else params["class_weight"]
    rf_max_features = "None" if params["rf_max_features"] is None else params["rf_max_features"]
    ngram_min, ngram_max = params["ngram_range"]

    print("\nTrain command:")
    print(
        "python algorithm/random_forest/train.py "
        f"--max-features {params['max_features']} "
        f"--ngram-min {ngram_min} --ngram-max {ngram_max} "
        f"--min-df {params['min_df']} --max-df {params['max_df']} "
        f"--n-estimators {params['n_estimators']} "
        f"--max-depth {max_depth} "
        f"--min-samples-leaf {params['min_samples_leaf']} "
        f"--min-samples-split {params['min_samples_split']} "
        f"--class-weight {class_weight} "
        f"--max-samples {max_samples} "
        f"--rf-max-features {rf_max_features}"
    )


def run_tuning(
    data_dir=None,
    sample_pct=0.1,
    random_state=42,
    results_path=DEFAULT_RESULTS_PATH,
):
    print("Loading data splits...")
    x_train_full, x_valid, _, y_train_full, y_valid, _, stats = load_data_splits(data_dir=data_dir)

    print(f"Original train size: {len(x_train_full)}")
    print(f"Validation size: {len(x_valid)}")

    # Downsample training set if sample_pct < 1.0 to speed up tuning
    if sample_pct < 1.0:
        x_train, _, y_train, _ = train_test_split(
            x_train_full,
            y_train_full,
            train_size=sample_pct,
            random_state=random_state,
            stratify=y_train_full,
        )
        print(f"Downsampled train size for tuning: {len(x_train)} ({sample_pct * 100:.1f}%)")
    else:
        x_train, y_train = x_train_full, y_train_full

    parameter_grid = default_parameter_grid()

    results = []
    print("\nStarting hyperparameter grid search...")
    print("-" * 132)
    print(
        f"{'#':<3} | {'TFIDF':<18} | {'Trees':<5} | {'Depth':<6} | {'Leaf':<4} | "
        f"{'Weight':<18} | {'MaxSamp':<7} | {'Val Acc':<8} | {'Val MacroF1':<11} | {'Time (s)':<8}"
    )
    print("-" * 132)

    for idx, params in enumerate(parameter_grid, start=1):
        start_time = time.time()

        model = build_model(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            min_df=params["min_df"],
            max_df=params["max_df"],
            sublinear_tf=params["sublinear_tf"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            class_weight=params["class_weight"],
            bootstrap=params["bootstrap"],
            max_samples=params["max_samples"],
            rf_max_features=params["rf_max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_valid)
        val_acc = accuracy_score(y_valid, y_pred)
        val_f1 = f1_score(y_valid, y_pred, average="macro")

        elapsed = time.time() - start_time

        tfidf_str = f"{params['max_features']}/{params['min_df']}/{params['max_df']}"
        depth_str = "None" if params["max_depth"] is None else str(params["max_depth"])
        max_samples_str = "None" if params["max_samples"] is None else str(params["max_samples"])
        weight_str = str(params["class_weight"])
        print(
            f"{idx:<3} | {tfidf_str:<18} | {params['n_estimators']:<5} | {depth_str:<6} | "
            f"{params['min_samples_leaf']:<4} | {weight_str:<18} | {max_samples_str:<7} | "
            f"{val_acc:.4f}  | {val_f1:.4f}     | {elapsed:.1f}s"
        )

        results.append({
            "params": params,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "classification_report": classification_report(
                y_valid,
                y_pred,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            ),
            "time": elapsed,
        })

    best_run = max(results, key=lambda x: x["val_macro_f1"])

    results_path = Path(results_path)
    results_path.parent.mkdir(exist_ok=True)
    payload = {
        "data": {
            "data_dir": str(stats["data_dir"]),
            "train_size_full": len(x_train_full),
            "train_size_tuning": len(x_train),
            "valid_size": len(x_valid),
            "sample_pct": sample_pct,
            "random_state": random_state,
        },
        "best_params": json_ready(best_run["params"]),
        "best_validation_accuracy": best_run["val_acc"],
        "best_validation_macro_f1": best_run["val_macro_f1"],
        "results": [
            {
                **run,
                "params": json_ready(run["params"]),
            }
            for run in results
        ],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 132)
    print("\n===== TUNING SUMMARY =====")
    print("Best Parameters:")
    for k, v in best_run["params"].items():
        print(f"  {k}: {v}")
    print(f"Best Validation Accuracy: {best_run['val_acc']:.4f}")
    print(f"Best Validation Macro F1: {best_run['val_macro_f1']:.4f}")
    print(f"Saved tuning results: {results_path}")
    print_train_command(best_run["params"])

    return best_run["params"]


def main():
    parser = argparse.ArgumentParser(description="Tune Random Forest hyperparameters.")
    parser.add_argument(
        "--data-dir",
        "--data",
        dest="data_dir",
        help="Directory containing train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=0.1,
        help="Percentage of training data to use for search speed (default: 0.1)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-out", default=str(DEFAULT_RESULTS_PATH))
    args = parser.parse_args()

    run_tuning(
        data_dir=args.data_dir,
        sample_pct=args.sample_pct,
        random_state=args.random_state,
        results_path=args.results_out,
    )


if __name__ == "__main__":
    main()
