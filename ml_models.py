"""
EcoComp AI - Part 2 ML evaluation  (fixed for Mac/Windows)
Reads CSV from the SAME FOLDER as this script.
Run simulate.py FIRST, then run this file.
"""
import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                             mean_absolute_error, confusion_matrix)

# ── reads from same folder as this file ──
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RNG_SEED = 7

def stage_label(progress):
    if progress < 10:   return 0
    elif progress < 60: return 1
    elif progress < 85: return 2
    else:               return 3

STAGE_NAMES = ["Initial", "Active/Thermophilic", "Cooling", "Maturing/Mature"]

def build_features(df):
    df = df.sort_values(["run_id", "hour"]).copy()
    for c in ["temp", "moisture", "o2", "ch4"]:
        df[f"{c}_roll_mean"] = df.groupby("run_id")[c].transform(
            lambda s: s.rolling(6, min_periods=1).mean())
        df[f"{c}_roll_std"] = df.groupby("run_id")[c].transform(
            lambda s: s.rolling(6, min_periods=1).std().fillna(0))
    df["stage"] = df["progress"].apply(stage_label)
    max_day = df.groupby("run_id")["day"].transform("max")
    df["days_remaining"] = (max_day - df["day"]).clip(lower=0)
    return df

def main():
    csv_path = os.path.join(DATA_DIR, "all_runs_timeseries.csv")
    if not os.path.exists(csv_path):
        print("ERROR: all_runs_timeseries.csv not found.")
        print("Run simulate.py FIRST, then run this file.")
        return

    df = pd.read_csv(csv_path)
    df = build_features(df)

    feature_cols = ["temp","moisture","o2","ch4","fan","hour",
                    "temp_roll_mean","temp_roll_std",
                    "moisture_roll_mean","moisture_roll_std",
                    "o2_roll_mean","o2_roll_std",
                    "ch4_roll_mean","ch4_roll_std"]

    rng = np.random.default_rng(RNG_SEED)
    runs = df["run_id"].unique()
    rng.shuffle(runs)
    n_test = max(1, int(len(runs) * 0.25))
    test_runs = set(runs[:n_test])
    train_df = df[~df["run_id"].isin(test_runs)]
    test_df  = df[ df["run_id"].isin(test_runs)]

    print(f"Train: {train_df['run_id'].nunique()} batches | "
          f"Test: {test_df['run_id'].nunique()} batches (held-out)")

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RNG_SEED)
    clf.fit(train_df[feature_cols], train_df["stage"])
    pred_stage = clf.predict(test_df[feature_cols])
    acc = accuracy_score(test_df["stage"], pred_stage)
    f1  = f1_score(test_df["stage"], pred_stage, average="macro")
    cm  = confusion_matrix(test_df["stage"], pred_stage, labels=[0,1,2,3])

    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RNG_SEED)
    reg.fit(train_df[feature_cols], train_df["days_remaining"])
    pred_rem = reg.predict(test_df[feature_cols])
    r2  = r2_score(test_df["days_remaining"], pred_rem)
    mae = mean_absolute_error(test_df["days_remaining"], pred_rem)

    print(f"\n=== ML RESULTS ===")
    print(f"Stage Classifier  — Accuracy: {acc:.3f} ({acc*100:.1f}%)  |  Macro F1: {f1:.3f}")
    print(f"Days Regressor    — R2: {r2:.3f}  |  MAE: {mae:.2f} days")
    print(f"\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print(f"{'':22s} {'Initial':>8} {'Active':>8} {'Cooling':>8} {'Mature':>8}")
    for i, row in enumerate(cm):
        print(f"{STAGE_NAMES[i]:22s} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    results = {
        "n_train_runs": int(train_df["run_id"].nunique()),
        "n_test_runs":  int(test_df["run_id"].nunique()),
        "stage_classifier_accuracy": float(acc),
        "stage_classifier_macro_f1": float(f1),
        "stage_confusion_matrix": cm.tolist(),
        "stage_names": STAGE_NAMES,
        "days_remaining_r2": float(r2),
        "days_remaining_mae_days": float(mae)
    }
    out_path = os.path.join(DATA_DIR, "ml_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
