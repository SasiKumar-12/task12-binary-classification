"""
data.py
-------
Loads the dataset used for this task.

By default we use the Breast Cancer Wisconsin dataset (built into scikit-learn,
no internet required, real data). It's a binary classification problem:
malignant (1) vs benign (0).

We also engineer a synthetic "segment" column (tumor size: small/large) purely
so we have something to slice metrics by in the fairness/stability check
(step 4 of the task brief). Swap this out for a real demographic/segment
column if you plug in your own dataset later.

If you want to use your OWN csv instead, set DATA_PATH below and make sure it
has a binary "target" column.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_PATH = None  # e.g. "my_data.csv" -- if set, load_dataset() will use this instead


def load_dataset():
    if DATA_PATH:
        df = pd.read_csv(DATA_PATH)
        assert "target" in df.columns, "CSV must have a 'target' column (0/1)"
        return df

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns={"target": "target"})

    # sklearn encodes 0=malignant, 1=benign -- flip so 1 = malignant (the "positive"/
    # riskier class we care about catching), which is more intuitive for thresholding.
    df["target"] = 1 - df["target"]

    # Synthetic segment for the fairness/stability check: split on tumor size.
    median_radius = df["mean radius"].median()
    df["segment"] = df["mean radius"].apply(lambda x: "large_tumor" if x >= median_radius else "small_tumor")

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df["target"].value_counts(normalize=True))
    print(df["segment"].value_counts())
