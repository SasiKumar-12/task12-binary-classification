# Task 12 — Binary Classification (PlaceMux Phase 1)

A calibrated, threshold-justified binary classifier, packaged for serving,
using the built-in Breast Cancer Wisconsin dataset (real data, no download needed).

## Project structure
```
task12_binary_classification/
├── data.py          # loads & preps the dataset (+ a "segment" column for fairness checks)
├── train.py         # full pipeline: tune → calibrate → threshold → segment eval → document → package
├── serve.py         # loads the packaged model and runs predictions (the "demo" step)
├── requirements.txt
└── artifacts/        # created after you run train.py
    ├── calibration_curve.png
    ├── threshold_cost_curve.png
    ├── roc_curve.png
    ├── operating_point.json
    └── model_bundle.joblib
```

## Step 1 — Open the project in VS Code
1. Unzip/copy this folder somewhere on your machine.
2. In VS Code: `File → Open Folder...` and select `task12_binary_classification`.

## Step 2 — Create a virtual environment
Open the VS Code integrated terminal (`` Ctrl+` `` / `` Cmd+` ``) and run:

```bash
python3 -m venv .venv
```

Activate it:
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

VS Code will usually pop up a prompt asking "select this as your interpreter" —
click **Yes**. If it doesn't, open the Command Palette (`Ctrl+Shift+P`) →
**Python: Select Interpreter** → choose the one inside `.venv`.

## Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 4 — Sanity-check the data loader
```bash
python data.py
```
You should see the dataset shape, class balance, and segment counts printed.

## Step 5 — Run the full pipeline
```bash
python train.py
```
This does everything the task brief asks for in one pass:
1. **Trains and tunes** a `RandomForestClassifier` inside a `Pipeline` (scaler + model),
   using `GridSearchCV` with stratified 5-fold CV, scored on ROC-AUC.
2. **Calibrates probabilities** with `CalibratedClassifierCV` (isotonic), and
   compares Brier scores raw vs. calibrated — saves `calibration_curve.png` so
   you can *see* the calibration, not just assume it.
3. **Picks a cost-optimal threshold** by sweeping thresholds against a cost
   matrix (missing a malignant case is treated as 10x worse than a false
   alarm) — saves `threshold_cost_curve.png`.
4. **Checks stability/fairness** two ways: (a) spread of ROC-AUC across CV
   folds, and (b) a breakdown of AUC / recall / false-positive-rate by
   `segment` (small vs. large tumor size, standing in for a real demographic
   or business segment) on the held-out test set.
5. **Documents the operating point** — every number above gets written to
   `artifacts/operating_point.json`: chosen threshold, cost assumptions,
   fold stability, test-set confusion matrix, expected error rates, segment
   report.
6. **Packages the model** into `artifacts/model_bundle.joblib` — contains the
   fitted model, the chosen threshold, feature column order, and the full
   metadata, so it's self-contained for serving.

## Step 6 — Demo it on real data
```bash
python serve.py
```
Loads the packaged bundle and scores a handful of real rows, printing the
predicted probability and the decision at the documented threshold — this is
your "demonstrable live on real data" checkbox from the Definition of Done.

## Step 7 — Swap in your own data (optional)
Open `data.py` and either:
- point `DATA_PATH` at your own CSV (must include a binary `target` column
  and ideally a `segment`-like column for the fairness check), or
- replace `load_dataset()` entirely with your own loading logic.

Everything downstream in `train.py`/`serve.py` just needs a dataframe with a
`target` column and the rest as features, so no other changes are required.

## Where each Definition-of-Done item is covered
| Requirement | Where |
|---|---|
| Calibrated, threshold-justified classifier | `train.py` steps 1–3, `operating_point.json` |
| Stable, segment-checked metrics | `train.py` step 4, `segment_report` in `operating_point.json` |
| Packaged for serving | `artifacts/model_bundle.joblib` + `serve.py` |
| Demoable on real data | `python serve.py` |

## Pitfalls this avoids (per the brief)
- **Uncalibrated probabilities used as if exact** → raw vs. calibrated Brier
  scores are compared explicitly; the better one is used, and the curve is
  saved as evidence either way.
- **Hidden per-segment failure** → `segment_report` in `operating_point.json`
  surfaces per-segment AUC, recall, and false-positive rate on the test set.
- **No documented operating point** → `operating_point.json` is the single
  source of truth for the threshold, cost assumptions, and expected error rates.
