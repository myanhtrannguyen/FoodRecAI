import argparse
import io
from pathlib import Path
import sys
from typing import cast

import joblib

try:
    from .dataset import load_data_splits
    from .model import build_model
except ImportError:
    from dataset import load_data_splits
    from model import build_model


OUTPUT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "random_forest_model.joblib"

if hasattr(sys.stdout, "reconfigure"):
    cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")


def train_model(
    data_dir=None,
    model_path=DEFAULT_MODEL_PATH,
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True,
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=2,
    class_weight="balanced",
    bootstrap=True,
    max_samples=None,
    rf_max_features="sqrt",
    random_state=42,
    n_jobs=-1,
):
    x_train, _, _, y_train, _, _, stats = load_data_splits(data_dir=data_dir)

    model = build_model(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        bootstrap=bootstrap,
        max_samples=max_samples,
        rf_max_features=rf_max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)

    stats.update({
        "model_path": model_path,
        "max_features": max_features,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "sublinear_tf": sublinear_tf,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "class_weight": class_weight,
        "bootstrap": bootstrap,
        "max_samples": max_samples,
        "rf_max_features": rf_max_features,
    })
    return model, stats


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest sentiment model.")
    parser.add_argument(
        "--data-dir",
        "--data",
        dest="data_dir",
        help="Directory containing train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--sublinear-tf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", default="None")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument(
        "--class-weight",
        default="balanced_subsample",
        choices=["balanced", "balanced_subsample", "None"],
    )
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", default="None")
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    class_weight = args.class_weight
    if class_weight == "None":
        class_weight = None
    max_depth = None if args.max_depth == "None" else int(args.max_depth)
    max_samples = None if args.max_samples == "None" else float(args.max_samples)
    rf_max_features = None if args.rf_max_features == "None" else args.rf_max_features

    _, stats = train_model(
        data_dir=args.data_dir,
        model_path=args.model_out,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        class_weight=class_weight,
        bootstrap=args.bootstrap,
        max_samples=max_samples,
        rf_max_features=rf_max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    print("===== TRAIN =====")
    print(f"Data dir: {stats['data_dir']}")
    print(f"Train file: {stats['train_path']}")
    print(f"Valid file: {stats['valid_path']}")
    print(f"Test file: {stats['test_path']}")
    print(f"Total used: {stats['total_used']}")
    print(f"Train size: {stats['train_size']}")
    print(f"Valid size: {stats['valid_size']}")
    print(f"Test size: {stats['test_size']}")
    print(f"Skipped invalid rows: {stats['skipped_invalid']}")
    print(f"Params: max_features={stats['max_features']}, ngram_range={stats['ngram_range']}, min_df={stats['min_df']}, max_df={stats['max_df']}, sublinear_tf={stats['sublinear_tf']}")
    print(f"RF params: n_estimators={stats['n_estimators']}, max_depth={stats['max_depth']}, min_samples_leaf={stats['min_samples_leaf']}, min_samples_split={stats['min_samples_split']}, class_weight={stats['class_weight']}, bootstrap={stats['bootstrap']}, max_samples={stats['max_samples']}, rf_max_features={stats['rf_max_features']}")
    print(f"Saved model: {stats['model_path']}")


if __name__ == "__main__":
    main()
