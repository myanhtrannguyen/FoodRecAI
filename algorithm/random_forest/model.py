from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_model(
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
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
                lowercase=True,
                strip_accents=None,
            ),
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                bootstrap=bootstrap,
                max_samples=max_samples,
                max_features=rf_max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
    ])
