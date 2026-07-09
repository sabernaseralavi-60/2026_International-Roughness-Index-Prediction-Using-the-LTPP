"""
Train and compare IRI prediction models, addressing the Q1-reviewer critique
pass in full:

  - Traffic (AADTT, KESAL) and FWD load-transfer efficiency are no longer
    dropped for missingness; they are imputed with an IterativeImputer
    (MICE-style) fitted *only on the training fold* inside a sklearn
    Pipeline, so imputation cannot leak test-fold information.
  - LightGBM and the ANN are tuned with Optuna against group-aware
    cross-validated R^2, and the tuned hyperparameters are reported.
  - Rigid pavements are modeled three ways: a unified JPCP+JRCP+CRCP model
    (IS_CRCP indicator, as before), plus separate JPCP/JRCP-only and
    CRCP-only submodels, so a reviewer can see whether pooling helps or
    hurts.
  - A second "operational" variant adds PREV_IRI / YEARS_SINCE_PREV (the
    section's own most recent prior IRI reading) alongside the primary
    "structural" variant (distress/climate/structure only, no lagged IRI).
    Both are reported side by side, since they answer different questions.
  - An ablation model without traffic/LTE quantifies their marginal
    contribution (the "sensitivity analysis" requested in review) on the
    *same* rows as the full model, rather than confounding the comparison
    with a different, traffic-complete-only sample.

Usage: python 04_model.py <flexible|rigid|rigid_jpcc|rigid_crcp>
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import shap
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
PLOTS = os.path.join(ROOT, "plots")
TABLES = os.path.join(ROOT, "tables")
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 40

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

# Structural predictors: never imputed, required complete at the data-prep
# stage (see code/02_clean_and_merge.py FLEX_CORE / RIGID_CORE).
FLEX_STRUCT = ["AGE_YR", "SITE_FACTOR", "BOUND_THICKNESS_CM", "FREEZE_INDEX_YR",
               "TOTAL_ANN_PRECIP", "MEAN_ANN_TEMP_AVG",
               "MEPDG_TRANS_CRACK_LENGTH_AC", "MEPDG_LONG_CRACK_LENGTH_AC",
               "HPMS16_CRACKING_PERCENT_AC", "PATCH_A", "MAX_MEAN_DEPTH_WIRE_REF"]
FLEX_IMPUTED = ["AADTT_LTPPLN", "KESAL_YEAR"]

RIGID_STRUCT_COMMON = ["AGE_YR", "BOUND_THICKNESS_CM", "FREEZE_INDEX_YR",
                        "TOTAL_ANN_PRECIP", "MEAN_ANN_TEMP_AVG"]
RIGID_ZERO_FILL_JPCC = ["AVG_WHEELPATH_FAULT", "TRANS_SPALLING_L", "TRANS_CRACK_NO", "PATCH_RIGID_A"]
RIGID_ZERO_FILL_CRCP = ["MEPDG_PUNCHOUTS_CRCP", "PUNCHOUTS_NO", "LONG_SPALLING_L"]
RIGID_IMPUTED = ["AADTT_LTPPLN", "KESAL_YEAR", "LOAD_TRANSFER_EFFICIENCY"]

LAG_FEATURES = ["PREV_IRI", "YEARS_SINCE_PREV"]


def load_data(family):
    """family in {flexible, rigid, rigid_jpcc, rigid_crcp}."""
    if family == "flexible":
        df = pd.read_csv(os.path.join(PROC, "flexible_iri_clean.csv"))
        struct = list(FLEX_STRUCT)
        zero_fill = []
    else:
        df = pd.read_csv(os.path.join(PROC, "rigid_iri_clean.csv"))
        fam = df["PAVEMENT_FAMILY"].astype(str)
        is_jpcc = fam.str.startswith(("JPC", "JRC"))
        if family == "rigid_jpcc":
            df = df[is_jpcc].copy()
            struct = list(RIGID_STRUCT_COMMON)
            zero_fill = list(RIGID_ZERO_FILL_JPCC)
        elif family == "rigid_crcp":
            df = df[~is_jpcc].copy()
            struct = list(RIGID_STRUCT_COMMON)
            zero_fill = list(RIGID_ZERO_FILL_CRCP)
        else:  # unified rigid
            df["IS_CRCP"] = (~is_jpcc).astype(int)
            struct = list(RIGID_STRUCT_COMMON) + ["IS_CRCP"]
            zero_fill = RIGID_ZERO_FILL_JPCC + RIGID_ZERO_FILL_CRCP

    for c in zero_fill:
        df[c] = df[c].fillna(0)
    struct = struct + zero_fill
    imputed = FLEX_IMPUTED if family == "flexible" else RIGID_IMPUTED
    df = df.dropna(subset=struct + ["MEAN_IRI"])
    return df, struct, imputed


def build_pipeline(model):
    """IterativeImputer fitted only on the training fold via Pipeline.fit()
    (never on test data). Columns with no missing values pass through
    unchanged -- MICE only fills actual gaps -- so structural predictors
    that are already complete are unaffected."""
    return Pipeline([
        ("impute", IterativeImputer(random_state=RANDOM_STATE, max_iter=15, sample_posterior=False)),
        ("model", model),
    ])


def optuna_tune_lgbm(X_train, y_train, groups_train):
    gkf = GroupKFold(n_splits=4)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        )
        model = build_pipeline(
            lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, **params))
        scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=gkf, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def optuna_tune_ann(X_train, y_train, groups_train):
    gkf = GroupKFold(n_splits=4)
    arch_choices = [(32,), (64,), (64, 32), (32, 16), (64, 32, 16)]

    def objective(trial):
        arch = arch_choices[trial.suggest_int("arch_idx", 0, len(arch_choices) - 1)]
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        model = Pipeline([
            ("impute", IterativeImputer(random_state=RANDOM_STATE, max_iter=15)),
            ("scale", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=arch, alpha=alpha, learning_rate_init=lr,
                                    max_iter=3000, early_stopping=True, n_iter_no_change=25,
                                    random_state=RANDOM_STATE)),
        ])
        scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=gkf, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = dict(study.best_params)
    best["hidden_layer_sizes"] = arch_choices[best.pop("arch_idx")]
    return best, study.best_value


def evaluate(name, y_test, pred, timing_s, extra=None):
    row = {
        "Model": name,
        "R2": r2_score(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "Train_time_s": round(timing_s, 4),
    }
    if extra:
        row.update(extra)
    return row


def fit_eval_split(X, y, groups, feats):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], groups[train_idx]


def run_variant(df, struct, imputed, family, variant_label, use_lag=False):
    feats = list(struct) + list(imputed) + (LAG_FEATURES if use_lag else [])
    d = df.dropna(subset=["MEAN_IRI"] + (LAG_FEATURES if use_lag else []))
    X = d[feats].values
    y = d["MEAN_IRI"].values
    groups = d["SHRP_ID"].values
    if len(d) < 40 or d["SHRP_ID"].nunique() < 10:
        print(f"  [skip] {variant_label}: too few rows/sections ({len(d)}/{d['SHRP_ID'].nunique()})")
        return None

    X_train, X_test, y_train, y_test, groups_train = fit_eval_split(X, y, groups, feats)
    results = []

    t0 = time.time()
    lr = build_pipeline(LinearRegression())
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results.append(evaluate("MEPDG-style Linear Regression", y_test, pred, time.time() - t0))

    t0 = time.time()
    rf = build_pipeline(RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, pred_rf, time.time() - t0))

    ann_best, ann_cv = optuna_tune_ann(X_train, y_train, groups_train)
    t0 = time.time()
    ann = Pipeline([
        ("impute", IterativeImputer(random_state=RANDOM_STATE, max_iter=15)),
        ("scale", StandardScaler()),
        ("model", MLPRegressor(max_iter=3000, early_stopping=True, n_iter_no_change=25,
                                random_state=RANDOM_STATE, **ann_best)),
    ])
    ann.fit(X_train, y_train)
    pred_ann = ann.predict(X_test)
    results.append(evaluate("ANN (MLP, Optuna-tuned)", y_test, pred_ann, time.time() - t0,
                             {"CV_R2_mean": ann_cv}))

    gbm_best, gbm_cv = optuna_tune_lgbm(X_train, y_train, groups_train)
    t0 = time.time()
    gbm = build_pipeline(lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, **gbm_best))
    gbm.fit(X_train, y_train)
    pred_gbm = gbm.predict(X_test)
    gbm_time = time.time() - t0

    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        build_pipeline(lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, **gbm_best)),
        X, y, groups=groups, cv=gkf, scoring="r2")

    results.append(evaluate("Gradient-Boosted Ensemble (LightGBM, Optuna-tuned)", y_test, pred_gbm, gbm_time,
                             {"CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std()}))

    metrics = pd.DataFrame(results)
    rf_time = metrics.loc[metrics.Model == "Random Forest", "Train_time_s"].values[0]
    ann_time = metrics.loc[metrics.Model.str.contains("ANN"), "Train_time_s"].values[0]
    metrics["Speedup_vs_RF"] = rf_time / metrics["Train_time_s"]
    metrics["Speedup_vs_ANN"] = ann_time / metrics["Train_time_s"]
    metrics["Variant"] = variant_label
    metrics["N_train"] = len(X_train)
    metrics["N_test"] = len(X_test)

    tag = f"{family}_{variant_label}"
    metrics.to_csv(os.path.join(TABLES, f"{tag}_model_metrics.csv"), index=False)
    print(f"\n=== {tag} (best LightGBM params: {gbm_best}) ===")
    print(metrics[["Model", "R2", "RMSE", "MAE", "Train_time_s", "Speedup_vs_RF"]].round(4).to_string(index=False))

    return dict(metrics=metrics, gbm=gbm, X_test=X_test, y_test=y_test, pred_gbm=pred_gbm,
                feats=feats, gbm_best=gbm_best, cv_scores=cv_scores)


def make_shap_plots(result, family, variant_label):
    gbm, X_test, feats = result["gbm"], result["X_test"], result["feats"]
    X_test_imputed = gbm.named_steps["impute"].transform(X_test)
    explainer = shap.TreeExplainer(gbm.named_steps["model"])
    X_test_df = pd.DataFrame(X_test_imputed, columns=feats)
    shap_values = explainer.shap_values(X_test_df)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None)
    plt.title(f"SHAP Feature Impact on IRI — {family} ({variant_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"{family}_{variant_label}_shap_summary.png"), bbox_inches="tight")
    plt.close()

    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values()
    imp.to_csv(os.path.join(TABLES, f"{family}_{variant_label}_shap_importance.csv"))

    fig, ax = plt.subplots(figsize=(6, 6))
    y_test, pred_gbm = result["y_test"], result["pred_gbm"]
    ax.scatter(y_test, pred_gbm, alpha=0.5, s=18, color="#3B7EA1")
    lims = [min(y_test.min(), pred_gbm.min()), max(y_test.max(), pred_gbm.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Measured IRI (m/km)")
    ax.set_ylabel("Predicted IRI (m/km)")
    r2 = result["metrics"].loc[result["metrics"].Model.str.contains("LightGBM"), "R2"].values[0]
    ax.set_title(f"Predicted vs. Measured IRI — {family} {variant_label} (R²={r2:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{family}_{variant_label}_pred_vs_actual.png"), bbox_inches="tight")
    plt.close(fig)
    return imp


def run(family):
    df, struct, imputed = load_data(family)
    print(f"{family}: {len(df)} rows, {df.SHRP_ID.nunique()} sections")

    all_metrics = []

    r_struct = run_variant(df, struct, imputed, family, "structural")
    if r_struct:
        all_metrics.append(r_struct["metrics"])
        make_shap_plots(r_struct, family, "structural")

    r_ablate = run_variant(df, struct, [], family, "no_traffic_lte")
    if r_ablate:
        all_metrics.append(r_ablate["metrics"])

    if "PREV_IRI" in df.columns:
        r_lag = run_variant(df, struct, imputed, family, "operational_lagged", use_lag=True)
        if r_lag:
            all_metrics.append(r_lag["metrics"])
            make_shap_plots(r_lag, family, "operational_lagged")

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(os.path.join(TABLES, f"{family}_all_variants_metrics.csv"), index=False)

    return all_metrics


if __name__ == "__main__":
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    family = sys.argv[1] if len(sys.argv) > 1 else "flexible"
    run(family)
