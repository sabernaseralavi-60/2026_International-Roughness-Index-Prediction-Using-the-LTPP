"""
Train and compare IRI prediction models for one pavement family (flexible or
rigid), following the benchmarking design of Song et al. (2022): an
MEPDG-style linear regression benchmark, a Random Forest, an ANN (MLP), and
a gradient-boosted ensemble (LightGBM, playing the role of ThunderGBM, which
has no maintained public Python wheel) explained with SHAP.

Usage: python 04_model.py <flexible|rigid>

Outputs:
    tables/<family>_model_metrics.csv
    plots/<family>_shap_summary.png
    plots/<family>_pred_vs_actual.png
    plots/<family>_feature_importance.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
PLOTS = os.path.join(ROOT, "plots")
TABLES = os.path.join(ROOT, "tables")
RANDOM_STATE = 42

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

FLEX_FEATURES = ["AGE_YR", "SITE_FACTOR", "BOUND_THICKNESS_CM", "FREEZE_INDEX_YR",
                  "TOTAL_ANN_PRECIP", "MEAN_ANN_TEMP_AVG",
                  "MEPDG_TRANS_CRACK_LENGTH_AC", "MEPDG_LONG_CRACK_LENGTH_AC",
                  "HPMS16_CRACKING_PERCENT_AC", "PATCH_A", "MAX_MEAN_DEPTH_WIRE_REF"]

RIGID_ZERO_FILL = ["AVG_WHEELPATH_FAULT", "TRANS_SPALLING_L", "TRANS_CRACK_NO",
                    "PATCH_RIGID_A", "MEPDG_PUNCHOUTS_CRCP", "PUNCHOUTS_NO",
                    "LONG_SPALLING_L"]
RIGID_FEATURES = ["AGE_YR", "BOUND_THICKNESS_CM", "FREEZE_INDEX_YR",
                   "TOTAL_ANN_PRECIP", "MEAN_ANN_TEMP_AVG", "IS_CRCP"] + RIGID_ZERO_FILL


def load_data(family):
    df = pd.read_csv(os.path.join(PROC, f"{family}_iri_clean.csv"))
    if family == "flexible":
        feats = FLEX_FEATURES
    else:
        df["IS_CRCP"] = df["PAVEMENT_FAMILY"].astype(str).str.startswith("CRC").astype(int)
        # A JPCP row has no punchouts by construction (CRCP-only distress) and
        # vice versa for faulting: these are structural zeros, not missing data.
        for c in RIGID_ZERO_FILL:
            df[c] = df[c].fillna(0)
        feats = RIGID_FEATURES
    df = df.dropna(subset=feats + ["MEAN_IRI"])
    return df, feats


def mepdg_style_benchmark(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    return lr, pred


def evaluate(name, y_test, pred, timing_s):
    return {
        "Model": name,
        "R2": r2_score(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "Train_time_s": round(timing_s, 4),
    }


def run(family):
    import time
    df, feats = load_data(family)
    X = df[feats].values
    y = df["MEAN_IRI"].values
    groups = df["SHRP_ID"].values

    # Group split by SHRP_ID (physical LTPP section), not by row: a naive
    # random row split would place repeat visits of the *same* section in
    # both train and test (median 4-9 repeat visits/section here), leaking
    # section-specific information and inflating R^2. This is the standard
    # fix for panel/longitudinal LTPP data and is the single most important
    # methodological correction from the reviewer pass.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = []

    t0 = time.time()
    lr, pred = mepdg_style_benchmark(X_train, y_train, X_test, y_test)
    results.append(evaluate("MEPDG-style Linear Regression", y_test, pred, time.time() - t0))

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, pred_rf, time.time() - t0))

    t0 = time.time()
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE, early_stopping=True)
    ann.fit(Xtr_s, y_train)
    pred_ann = ann.predict(Xte_s)
    results.append(evaluate("ANN (MLP)", y_test, pred_ann, time.time() - t0))

    t0 = time.time()
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                             num_leaves=15, min_child_samples=15,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=0.5,
                             random_state=RANDOM_STATE, verbosity=-1)
    gbm.fit(X_train, y_train)
    pred_gbm = gbm.predict(X_test)
    gbm_time = time.time() - t0
    results.append(evaluate("Gradient-Boosted Ensemble (LightGBM)", y_test, pred_gbm, gbm_time))

    # 5-fold group cross-validation (grouped by SHRP_ID) for a more robust
    # accuracy estimate than the single held-out split, given the modest N.
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                           num_leaves=15, min_child_samples=15,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.5,
                           random_state=RANDOM_STATE, verbosity=-1),
        X, y, groups=groups, cv=gkf, scoring="r2")
    print(f"5-fold group CV R2 (LightGBM): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    metrics = pd.DataFrame(results)
    metrics["CV_R2_mean"] = np.nan
    metrics["CV_R2_std"] = np.nan
    metrics.loc[metrics.Model.str.contains("LightGBM"), "CV_R2_mean"] = cv_scores.mean()
    metrics.loc[metrics.Model.str.contains("LightGBM"), "CV_R2_std"] = cv_scores.std()
    metrics["Speedup_vs_ANN"] = metrics.loc[metrics.Model == "ANN (MLP)", "Train_time_s"].values[0] / metrics["Train_time_s"]
    metrics["Speedup_vs_RF"] = metrics.loc[metrics.Model == "Random Forest", "Train_time_s"].values[0] / metrics["Train_time_s"]
    metrics.to_csv(os.path.join(TABLES, f"{family}_model_metrics.csv"), index=False)
    print(f"\n=== {family} ===")
    print(metrics.round(4).to_string(index=False))

    # SHAP explanation of the winning ensemble model
    explainer = shap.TreeExplainer(gbm)
    X_test_df = pd.DataFrame(X_test, columns=feats)
    shap_values = explainer.shap_values(X_test_df)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None)
    plt.title(f"SHAP Feature Impact on IRI — {family.title()} Pavements")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{family}_shap_summary.png"), bbox_inches="tight")
    plt.close()

    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#3B7EA1" if family == "flexible" else "#B54A4A")
    ax.set_xlabel("Mean |SHAP value| (impact on predicted IRI, m/km)")
    ax.set_title(f"Feature Importance — {family.title()} Pavements")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{family}_feature_importance.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred_gbm, alpha=0.5, s=18,
               color="#3B7EA1" if family == "flexible" else "#B54A4A")
    lims = [min(y_test.min(), pred_gbm.min()), max(y_test.max(), pred_gbm.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Measured IRI (m/km)")
    ax.set_ylabel("Predicted IRI (m/km)")
    r2 = metrics.loc[metrics.Model.str.contains("LightGBM"), "R2"].values[0]
    ax.set_title(f"Predicted vs. Measured IRI — {family.title()} (R²={r2:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{family}_pred_vs_actual.png"), bbox_inches="tight")
    plt.close(fig)

    imp.to_csv(os.path.join(TABLES, f"{family}_shap_importance.csv"))
    return metrics, imp


if __name__ == "__main__":
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    family = sys.argv[1] if len(sys.argv) > 1 else "flexible"
    run(family)
