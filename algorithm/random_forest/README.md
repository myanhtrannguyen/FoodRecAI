# Random Forest Sentiment Classification

## Hyperparameter Tuning

Run grid search to find optimal parameters:
```bash
python algorithm/random_forest/tune.py
```

**Key parameters:**
- `max_features`: TF-IDF vocabulary size
- `n_estimators`: Number of trees (default: 300)
- `min_samples_leaf`: Regularization to prevent overfitting
- `class_weight`: "balanced" for imbalanced classes
- `ngram_range`: (1, 2) for unigrams and bigrams

**Tuning on subset of data (faster):**
```bash
python algorithm/random_forest/tune.py --sample-pct 0.2
```

**Full data tuning (more accurate, slower):**
```bash
python algorithm/random_forest/tune.py --sample-pct 1.0
```

Results saved to: `algorithm/random_forest/result/random_forest_tuning_results.json`

## Performance Results

**Best Model Test Metrics:**
- Overall Accuracy: 72.5%
- Macro F1: 0.731

**Per-class Performance:**

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Negative | 0.82 | 0.70 | 0.76 |
| Neutral | 0.67 | 0.82 | 0.74 |
| Positive | 0.79 | 0.63 | 0.70 |

**Dataset Split:**
- Train: 86,015 samples
- Validation: 10,751 samples
- Test: 10,755 samples
