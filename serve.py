"""
serve.py
--------
Loads the packaged model bundle (artifacts/model_bundle.joblib) and shows how
to use it for real predictions -- this is your "demoable on real data" step.

Run:
    python serve.py
"""

import joblib
import pandas as pd

from data import load_dataset

BUNDLE_PATH = "artifacts/model_bundle.joblib"


def load_model():
    return joblib.load(BUNDLE_PATH)


def predict(bundle, X: pd.DataFrame):
    """
    X must contain the same feature columns the model was trained on.
    Returns calibrated probabilities and the decision at the documented
    operating threshold.
    """
    probs = bundle["model"].predict_proba(X[bundle["feature_cols"]])[:, 1]
    decisions = (probs >= bundle["threshold"]).astype(int)
    return probs, decisions


if __name__ == "__main__":
    bundle = load_model()
    print(f"Loaded model. Operating threshold = {bundle['threshold']:.2f}")
    print(f"Documented test AUC = {bundle['metadata']['test_performance']['roc_auc']}")

    # Demo: run on a handful of real, unseen-looking rows from the dataset
    df = load_dataset()
    sample = df.sample(5, random_state=1)
    probs, decisions = predict(bundle, sample)

    for i, (idx, row) in enumerate(sample.iterrows()):
        print(
            f"Row {idx}: true={int(row['target'])}  "
            f"prob={probs[i]:.3f}  decision={decisions[i]}  segment={row['segment']}"
        )
