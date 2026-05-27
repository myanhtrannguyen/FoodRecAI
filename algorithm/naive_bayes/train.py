import argparse
from pathlib import Path
import sys

import joblib

try:
    from .dataset import load_data_splits
    from .model import build_model
except ImportError:
    from dataset import load_data_splits
    from model import build_model


OUTPUT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "naive_bayes_model.joblib"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def train_model(
    data_dir=None,
    model_path=DEFAULT_MODEL_PATH,
    max_features=50000,
    alpha=1.0,
):
    x_train, _, _, y_train, _, _, stats = load_data_splits(data_dir=data_dir)

    model = build_model(max_features=max_features, alpha=alpha)
    model.fit(x_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)

    stats.update({
        "model_path": model_path,
        "max_features": max_features,
        "alpha": alpha,
    })
    return model, stats


def main():
    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment model.")
    parser.add_argument(
        "--data-dir",
        "--data",
        dest="data_dir",
        help="Directory containing train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    _, stats = train_model(
        data_dir=args.data_dir,
        model_path=args.model_out,
        max_features=args.max_features,
        alpha=args.alpha,
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
    print(f"Saved model: {stats['model_path']}")


if __name__ == "__main__":
    main()
