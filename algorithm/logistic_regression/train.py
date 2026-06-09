import argparse
import csv
import json
import itertools
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from algorithm.logistic_regression.dataset import LABELS, load_data_splits
from algorithm.logistic_regression.model import build_logistic_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "best_logistic_model.joblib"


def evaluate_split(model, x, y_true, labels):
    y_pred = model.predict(x)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_model(
    data_dir=None,
    output_dir=None,
    model_path=None,
    c_list=[0.1, 1.0, 10.0],
    max_features_list=[10000, 30000, 50000],
    ngram_max_list=[1, 2],
    penalty="l2",
    solver="saga",
    max_iter=1000,
    random_state=42,
):
    x_train, x_valid, x_test, y_train, y_valid, y_test, data_stats = load_data_splits(data_dir=data_dir)
    class_names = [label for label in LABELS if label in set(y_train + y_valid + y_test)]
    
    param_grid = list(itertools.product(c_list, max_features_list, ngram_max_list))
    
    best_val_f1 = -1.0
    best_model = None
    best_params = None

    print(f"Starting Grid Search over {len(param_grid)} combinations...")

    for c_val, m_feat, n_gram in param_grid:
        print(f"Testing: C={c_val}, max_features={m_feat}, ngrams=(1,{n_gram})...", end=" ")

        model = build_logistic_pipeline(
            num_classes=len(set(y_train)),
            max_features=m_feat,
            ngram_range=(1, n_gram),
            min_df=2,
            C=c_val,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

        model.fit(x_train, y_train)

        val_res = evaluate_split(model, x_valid, y_valid, labels=class_names)
        current_f1 = val_res["macro_f1"]
        print(f"Val F1: {current_f1:.4f}")

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model = model
            best_params = {
                "C": c_val,
                "max_features": m_feat,
                "ngram_max": n_gram,
                "penalty": penalty,
                "solver": solver
            }

    print("\n" + "="*40)
    print(f"TUNING COMPLETE")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Best Params: {best_params}")
    print("="*40 + "\n")

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = evaluate_split(best_model, x_test, y_test, labels=class_names)
    valid_metrics_final = evaluate_split(best_model, x_valid, y_valid, labels=class_names)

    model_path = Path(model_path) if model_path else output_dir / DEFAULT_MODEL_PATH.name
    joblib.dump(best_model, model_path)

    metadata = {
        "model_type": "Logistic Regression",
        "best_params": best_params,
        "data_stats": data_stats,
        "train_label_distribution": dict(Counter(y_train)),
    }

    with open(output_dir / "tuning_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_valid": valid_metrics_final, "final_test": test_metrics}, f, indent=2)

    return best_model, {
        "best_params": best_params,
        "valid": valid_metrics_final,
        "test": test_metrics,
        "model_path": str(model_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Tuned Logistic Regression Trainer")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))

    parser.add_argument("--c-values", type=str, default="0.01,0.1,1.0,10.0", help="C values to test")
    parser.add_argument("--features", type=str, default="10000,30000,50000", help="TFIDF max features to test")
    parser.add_argument("--ngrams", type=str, default="1,2", help="Max ngram range to test (e.g. 1 or 2)")

    parser.add_argument("--penalty", default="l2", choices=["l1", "l2"])
    parser.add_argument("--solver", default="saga")
    parser.add_argument("--max-iter", type=int, default=1000)

    args = parser.parse_args()

    c_list = [float(x) for x in args.c_values.split(",")]
    f_list = [int(x) for x in args.features.split(",")]
    n_list = [int(x) for x in args.ngrams.split(",")]

    _, stats = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        c_list=c_list,
        max_features_list=f_list,
        ngram_max_list=n_list,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter
    )

    print(f"Best Parameters: {stats['best_params']}")
    print(f"Final Test Accuracy: {stats['test']['accuracy']:.4f}")
    print(f"Final Test Macro F1: {stats['test']['macro_f1']:.4f}")
    print(f"Model saved to: {stats['model_path']}")


if __name__ == "__main__":
    main()