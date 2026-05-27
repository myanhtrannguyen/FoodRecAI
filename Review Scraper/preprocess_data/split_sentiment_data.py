import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "data_for_sentiment_analysis" / "reviews_clean.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "data_for_sentiment_analysis"
LABELS = ["negative", "neutral", "positive"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_rows_by_label(input_path):
    rows_by_label = defaultdict(list)
    invalid = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            label = item.get("rating_label")
            if label not in LABELS:
                invalid += 1
                continue

            rows_by_label[label].append(item)

    return rows_by_label, invalid


def split_rows(rows_by_label, train_ratio, valid_ratio, random_state):
    rng = random.Random(random_state)
    splits = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for label in LABELS:
        rows = rows_by_label[label]
        rng.shuffle(rows)

        total = len(rows)
        train_count = int(total * train_ratio)
        valid_count = int(total * valid_ratio)

        splits["train"].extend(rows[:train_count])
        splits["valid"].extend(rows[train_count:train_count + valid_count])
        splits["test"].extend(rows[train_count + valid_count:])

    for rows in splits.values():
        rng.shuffle(rows)

    return splits


def write_splits(splits, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for item in rows:
                file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def print_summary(input_path, output_dir, splits, invalid):
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Skipped invalid rows: {invalid}")

    for split_name in ["train", "valid", "test"]:
        rows = splits[split_name]
        counts = Counter(item["rating_label"] for item in rows)
        print(f"{split_name}: total={len(rows)} labels={dict(counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Split sentiment reviews into train/valid/test jsonl files by label."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to reviews_clean.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for split files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.train_ratio <= 0 or args.valid_ratio <= 0:
        raise ValueError("train-ratio and valid-ratio must be greater than 0.")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train-ratio + valid-ratio must be less than 1.")

    rows_by_label, invalid = read_rows_by_label(input_path)
    splits = split_rows(rows_by_label, args.train_ratio, args.valid_ratio, args.random_state)
    write_splits(splits, output_dir)
    print_summary(input_path, output_dir, splits, invalid)


if __name__ == "__main__":
    main()
