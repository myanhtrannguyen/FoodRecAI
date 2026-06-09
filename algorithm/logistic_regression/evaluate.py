import argparse
import json
from html import escape
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from algorithm.logistic_regression.dataset import LABELS, load_data_splits
from algorithm.logistic_regression.train import DEFAULT_MODEL_PATH, train_model, OUTPUT_DIR as TRAIN_OUTPUT_DIR


OUTPUT_DIR = Path(__file__).resolve().parent / "result"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def write_report(output_path, lines):
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(data_dir=None, model_path=DEFAULT_MODEL_PATH, train_if_missing=False):
    model_path = Path(model_path)
    best_params = "Loaded from existing model"

    if not model_path.exists():
        if not train_if_missing:
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run train.py first or pass --train-if-missing."
            )
        print("Model not found. Starting hyperparameter tuning...")
        _, stats = train_model(
            data_dir=data_dir, 
            model_path=model_path,
            c_list=[0.1, 1.0, 10.0],
            max_features_list=[10000, 30000, 50000],
            ngram_max_list=[1, 2]
        )
        best_params = stats.get("best_params", "Unknown")
    else:
        metadata_path = model_path.parent / "tuning_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                best_params = meta.get("best_params", best_params)

    _, _, x_test, _, _, y_test, stats = load_data_splits(data_dir=data_dir)
    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    
    return metrics, report, matrix, stats, best_params


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tuned Logistic Regression model.")
    parser.add_argument(
        "--data-dir",
        "--data",
        dest="data_dir",
        help="Directory containing train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = Path(args.model)
    
    metrics, report, matrix, stats, best_params = evaluate_model(
        data_dir=args.data_dir,
        model_path=model_path,
        train_if_missing=args.train_if_missing,
    )

    report_lines = [
        "===== MODEL CONFIGURATION =====",
        f"Model file: {model_path}",
        f"Best Parameters Found: {best_params}",
        "",
        "===== DATA STATS =====",
        f"Data dir: {stats.get('data_dir', 'N/A')}",
        f"Test size: {stats.get('test_size', len(matrix.sum(axis=1)))}",
        "",
        "===== TEST SET METRICS =====",
        *[f"{name}: {value:.4f}" for name, value in metrics.items()],
        "",
        "===== CLASSIFICATION REPORT =====",
        report,
        "===== CONFUSION MATRIX =====",
        str(matrix),
    ]

    report_path = OUTPUT_DIR / "logistic_regression_report.txt"
    matrix_path = OUTPUT_DIR / "confusion_matrix_tuned.svg"

    write_report(report_path, report_lines)
    matrix_path.write_text(
        matrix_to_svg(matrix, "Logistic Regression Sentiment (Tuned)"),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nSaved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")

if __name__ == "__main__":
    main()