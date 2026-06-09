# Random Forest Sentiment Model

This folder contains the TF-IDF + Random Forest pipeline for restaurant review sentiment classification.

The model predicts one of three labels:

- `negative`
- `neutral`
- `positive`

## Files

| File | Purpose |
| --- | --- |
| `dataset.py` | Loads `train.jsonl`, `valid.jsonl`, and `test.jsonl` from the sentiment dataset. |
| `model.py` | Builds the `TfidfVectorizer` + `RandomForestClassifier` pipeline. |
| `tune.py` | Runs a validation-based hyperparameter search. |
| `train.py` | Trains the final Random Forest model and saves it as a Joblib file. |
| `evaluate.py` | Evaluates a saved model on the test split and writes the report/SVG confusion matrix. |
| `result/` | Stores the trained model, tuning output, evaluation report, and confusion matrix. |

## Data

By default, the scripts read data from:

```text
data/data_for_sentiment_analysis/
```

Expected files:

```text
train.jsonl
valid.jsonl
test.jsonl
```

Each row should contain review text fields such as `title` and `content`, plus a `rating_label` value in `negative`, `neutral`, or `positive`. If `rating_label` is missing, the loader falls back to the Foody rating thresholds used by preprocessing: `rating >= 8.0` is `positive`, `6.0 <= rating < 8.0` is `neutral`, and `rating < 6.0` is `negative`.

To use a different dataset folder, pass `--data-dir`:

```bash
python algorithm/random_forest/train.py --data-dir path/to/data
python algorithm/random_forest/evaluate.py --data-dir path/to/data
python algorithm/random_forest/tune.py --data-dir path/to/data
```

## Train

Recommended command for the currently selected Random Forest model:

```bash
python algorithm/random_forest/train.py \
  --max-features 20000 \
  --ngram-min 1 \
  --ngram-max 2 \
  --min-df 1 \
  --max-df 1.0 \
  --sublinear-tf \
  --n-estimators 300 \
  --max-depth None \
  --min-samples-leaf 2 \
  --min-samples-split 2 \
  --class-weight balanced \
  --bootstrap \
  --max-samples None \
  --rf-max-features sqrt
```

Output model:

```text
algorithm/random_forest/result/random_forest_model.joblib
```

On Windows PowerShell, the same command can be run on one line:

```powershell
python algorithm\random_forest\train.py --max-features 20000 --ngram-min 1 --ngram-max 2 --min-df 1 --max-df 1.0 --sublinear-tf --n-estimators 300 --max-depth None --min-samples-leaf 2 --min-samples-split 2 --class-weight balanced --bootstrap --max-samples None --rf-max-features sqrt
```

## Evaluate

Run evaluation after training:

```bash
python algorithm/random_forest/evaluate.py
```

This writes:

```text
algorithm/random_forest/result/random_forest_report.txt
algorithm/random_forest/result/confusion_matrix_2_labels.svg
```

To evaluate a custom model path:

```bash
python algorithm/random_forest/evaluate.py --model path/to/model.joblib
```

## Hyperparameter Tuning

Run the built-in focused grid search:

```bash
python algorithm/random_forest/tune.py
```

Default tuning uses `10%` of the training split for speed and evaluates each candidate on the full validation split. The output is saved to:

```text
algorithm/random_forest/result/random_forest_tuning_results.json
```

You can change the tuning sample size:

```bash
python algorithm/random_forest/tune.py --sample-pct 0.2
python algorithm/random_forest/tune.py --sample-pct 1.0
```

Use `--sample-pct 1.0` for a full-data tuning run. It is much slower because Random Forest is expensive on high-dimensional sparse TF-IDF text features.

The tuning grid is defined in `default_parameter_grid()` inside `tune.py`. Important parameters to tune:

| Parameter | Meaning | Notes |
| --- | --- | --- |
| `max_features` | Maximum TF-IDF vocabulary size. | Larger values keep more words/ngrams but increase memory and training time. |
| `ngram_range` | Token n-grams used by TF-IDF. | `(1, 2)` uses unigrams and bigrams. |
| `min_df` / `max_df` | Filters rare or overly common terms. | Useful for reducing noisy vocabulary. |
| `sublinear_tf` | Applies logarithmic TF scaling. | Usually helpful for text classification. |
| `n_estimators` | Number of trees. | More trees are usually more stable but slower and produce larger model files. |
| `max_depth` | Maximum tree depth. | Tune this on the validation split. The final selected model uses `None` to avoid the underfitting seen with the previous shallow `20` depth. |
| `min_samples_leaf` | Minimum samples per leaf node. | Higher values regularize trees; `2` worked best in the final model. |
| `class_weight` | Class balancing strategy. | `balanced` improved minority-class handling in this dataset. |
| `rf_max_features` | Number of features considered per tree split. | `sqrt` keeps split search cheaper for sparse text features. |
| `max_samples` | Bootstrap sample fraction. | Can speed up training but may reduce performance. |

After tuning, `tune.py` prints a suggested `train.py` command. Use that command to train a full model, then run `evaluate.py` on the test set.

## Current Best Performance

Evaluation split:

| Split | Size |
| --- | ---: |
| Train | 86,015 |
| Validation | 10,751 |
| Test | 10,755 |

Final model parameters:

```text
tfidf__max_features: 20000
tfidf__ngram_range: (1, 2)
tfidf__min_df: 1
tfidf__max_df: 1.0
tfidf__sublinear_tf: True
clf__n_estimators: 300
clf__max_depth: None
clf__min_samples_leaf: 2
clf__min_samples_split: 2
clf__class_weight: balanced
clf__bootstrap: True
clf__max_samples: None
clf__max_features: sqrt
```

Test metrics:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.7251 |
| Balanced accuracy | 0.7172 |
| Macro precision | 0.7592 |
| Macro recall | 0.7172 |
| Macro F1 | 0.7309 |
| Weighted precision | 0.7382 |
| Weighted recall | 0.7251 |
| Weighted F1 | 0.7237 |

Per-class test report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.82 | 0.70 | 0.76 | 1,375 |
| neutral | 0.67 | 0.82 | 0.74 | 4,813 |
| positive | 0.79 | 0.63 | 0.70 | 4,567 |

Confusion matrix:

```text
[[ 961  339   75]
 [ 153 3957  703]
 [  54 1633 2880]]
```

Compared with the earlier shallow-depth baseline report, Macro F1 improved from `0.7146` to `0.7309`, and Weighted F1 improved from `0.7039` to `0.7237`. Hyperparameters should be selected with validation results; the test metrics above are the final held-out evaluation for the selected configuration.

## Notes

Random Forest can work for this task, but sparse TF-IDF text features are usually better suited to linear classifiers such as Logistic Regression or Linear SVM. In this project, Random Forest is useful as a non-linear baseline and comparison point.
