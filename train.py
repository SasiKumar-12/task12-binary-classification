"""
train.py
--------
End-to-end pipeline for Task 12: Binary Classification.

Covers every step in the task brief:
  1. Train and tune the classifier on the full pipeline.
  2. Calibrate probabilities and verify with a calibration curve.
  3. Pick the cost-optimal threshold.
  4. Evaluate across folds and key segments for stability/fairness.
  5. Document the operating point and expected error rates.
  6. Package the model for serving.

Run:
    python train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_dataset

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Cost matrix: how expensive is each type of mistake?
# In this dataset (1 = malignant), missing a malignant case (false negative) is
# far worse than a false alarm on a benign case (false positive).
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0


def main():
    # -----------------------------------------------------------------------
    # 0. Load data, split into train/val/test (never touch test until the end)
    # -----------------------------------------------------------------------
    df = load_dataset()
    feature_cols = [c for c in df.columns if c not in ("target", "segment")]
    X, y, seg = df[feature_cols], df["target"], df["segment"]

    X_train, X_temp, y_train, y_temp, seg_train, seg_temp = train_test_split(
        X, y, seg, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, seg_val, seg_test = train_test_split(
        X_temp, y_temp, seg_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # -----------------------------------------------------------------------
    # 1. Train and tune the classifier on the full pipeline
    # -----------------------------------------------------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=SEED)),
    ])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 8, None],
        "clf__min_samples_leaf": [1, 3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    # -----------------------------------------------------------------------
    # 2. Calibrate probabilities and verify with a calibration curve
    # -----------------------------------------------------------------------
    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    val_probs_raw = best_pipe.predict_proba(X_val)[:, 1]
    val_probs_cal = calibrated.predict_proba(X_val)[:, 1]

    brier_raw = brier_score_loss(y_val, val_probs_raw)
    brier_cal = brier_score_loss(y_val, val_probs_cal)
    print(f"Brier score  -  raw: {brier_raw:.4f}  calibrated: {brier_cal:.4f}")

    frac_pos_raw, mean_pred_raw = calibration_curve(y_val, val_probs_raw, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_val, val_probs_cal, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred_raw, frac_pos_raw, "o-", label=f"Raw (Brier={brier_raw:.3f})")
    plt.plot(mean_pred_cal, frac_pos_cal, "o-", label=f"Calibrated (Brier={brier_cal:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Validation Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "calibration_curve.png", dpi=150)
    plt.close()
    print("Saved calibration_curve.png")

    # Use whichever model is actually better calibrated downstream
    final_model = calibrated if brier_cal <= brier_raw else best_pipe
    used_calibration = brier_cal <= brier_raw
    print(f"Using {'calibrated' if used_calibration else 'raw'} model going forward")

    # -----------------------------------------------------------------------
    # 3. Pick the cost-optimal threshold (on validation set only)
    # -----------------------------------------------------------------------
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        preds = (val_probs_cal >= t).astype(int) if used_calibration else (val_probs_raw >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])
    print(f"Cost-optimal threshold: {best_threshold:.2f} (expected cost={costs[best_idx]:.1f})")

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, costs)
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Chosen t={best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel(f"Expected cost (FN x{COST_FALSE_NEGATIVE}, FP x{COST_FALSE_POSITIVE})")
    plt.title("Cost vs Threshold (Validation Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "threshold_cost_curve.png", dpi=150)
    plt.close()
    print("Saved threshold_cost_curve.png")

    # -----------------------------------------------------------------------
    # 4. Evaluate across folds and key segments for stability/fairness
    # -----------------------------------------------------------------------
    # Cross-validated ROC-AUC stability (already tuned; re-check spread across folds)
    fold_aucs = []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        fold_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=SEED, **{k.replace("clf__", ""): v
                                                                    for k, v in search.best_params_.items()})),
        ])
        fold_pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_probs = fold_pipe.predict_proba(X_train.iloc[te_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train.iloc[te_idx], fold_probs))
    print(f"Fold ROC-AUCs: {[round(a, 4) for a in fold_aucs]}")
    print(f"Fold AUC mean={np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

    # Held-out TEST set: final, untouched-until-now evaluation
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)
    test_f1 = f1_score(y_test, test_preds)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    print(f"TEST  AUC={test_auc:.4f}  F1={test_f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Segment breakdown on the test set (fairness / stability check)
    segment_report = {}
    for seg_name in seg_test.unique():
        mask = (seg_test == seg_name).values
        if mask.sum() < 5:
            continue
        seg_probs = test_probs[mask]
        seg_true = y_test[mask]
        seg_preds = (seg_probs >= best_threshold).astype(int)
        seg_auc = roc_auc_score(seg_true, seg_probs) if len(set(seg_true)) > 1 else float("nan")
        stn, sfp, sfn, stp = confusion_matrix(seg_true, seg_preds, labels=[0, 1]).ravel()
        segment_report[seg_name] = {
            "n": int(mask.sum()),
            "auc": round(float(seg_auc), 4),
            "recall_positive": round(float(stp / (stp + sfn)) if (stp + sfn) else float("nan"), 4),
            "false_positive_rate": round(float(sfp / (sfp + stn)) if (sfp + stn) else float("nan"), 4),
        }
    print("Segment report:", json.dumps(segment_report, indent=2))

    # -----------------------------------------------------------------------
    # 5. Document the operating point and expected error rates
    # -----------------------------------------------------------------------
    operating_point = {
        "model": "RandomForestClassifier" + (" + isotonic calibration" if used_calibration else ""),
        "best_params": search.best_params_,
        "calibration_used": used_calibration,
        "brier_score_validation": {"raw": round(brier_raw, 4), "calibrated": round(brier_cal, 4)},
        "chosen_threshold": round(best_threshold, 4),
        "cost_assumptions": {
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "cost_false_positive": COST_FALSE_POSITIVE,
        },
        "cv_stability": {
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "mean": round(float(np.mean(fold_aucs)), 4),
            "std": round(float(np.std(fold_aucs)), 4),
        },
        "test_performance": {
            "roc_auc": round(float(test_auc), 4),
            "f1": round(float(test_f1), 4),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
            "expected_false_negative_rate": round(float(fn / (fn + tp)), 4),
            "expected_false_positive_rate": round(float(fp / (fp + tn)), 4),
        },
        "segment_report": segment_report,
    }
    with open(ARTIFACT_DIR / "operating_point.json", "w") as f:
        json.dump(operating_point, f, indent=2)
    print("Saved operating_point.json")

    # ROC curve for the doc/demo
    RocCurveDisplay.from_predictions(y_test, test_probs)
    plt.title(f"ROC Curve (Test) — AUC={test_auc:.3f}")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 6. Package the model for serving
    # -----------------------------------------------------------------------
    bundle = {
        "model": final_model,
        "threshold": best_threshold,
        "feature_cols": feature_cols,
        "metadata": operating_point,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model_bundle.joblib")
    print(f"Saved model_bundle.joblib -> ready for serve.py")


if __name__ == "__main__":
    main()
