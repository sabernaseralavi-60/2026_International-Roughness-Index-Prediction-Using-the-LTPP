"""Consolidate the per-family leakage/age/imputation/extended-features
sensitivity checks and the XGBoost/CatBoost comparison into single tables
for the manuscript."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")

FAMILIES = ["flexible", "rigid", "rigid_jpcc", "rigid_crcp"]
FAMILY_LABEL = {
    "flexible": "Flexible", "rigid": "Rigid, pooled",
    "rigid_jpcc": "Rigid, JPCP/JRCP only", "rigid_crcp": "Rigid, CRCP only",
}

# --- Leakage comparison (naive vs. group-aware, same tuned hyperparameters) ---
rows = []
for fam in FAMILIES:
    path = os.path.join(TABLES, f"{fam}_leakage_comparison.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    naive = df.loc[df.Split.str.startswith("Naive"), "R2"].values[0]
    grouped = df.loc[df.Split.str.startswith("Group"), "R2"].values[0]
    rows.append({"Pavement family": FAMILY_LABEL[fam], "Naive-split R2": round(naive, 3),
                 "Group-aware R2": round(grouped, 3), "Inflation": round(naive - grouped, 3)})
leakage = pd.DataFrame(rows)
leakage.to_csv(os.path.join(TABLES, "leakage_comparison_all_families.csv"), index=False)
print(leakage.to_string(index=False))

# --- Age / imputation / extended-features sensitivity ---
rows = []
for fam in FAMILIES:
    headline = pd.read_csv(os.path.join(TABLES, f"{fam}_structural_model_metrics.csv"))
    headline_r2 = headline.loc[headline.Model.str.contains("LightGBM"), "R2"].values[0]
    row = {"Pavement family": FAMILY_LABEL[fam], "Headline R2": round(headline_r2, 3)}

    age_path = os.path.join(TABLES, f"{fam}_age_sensitivity.csv")
    if os.path.exists(age_path):
        age = pd.read_csv(age_path)
        row["Age-verified-only R2"] = round(age["R2"].values[0], 3)
        row["Age-verified N sections"] = int(age["Sections"].values[0])

    imp_path = os.path.join(TABLES, f"{fam}_imputation_sensitivity.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path)
        mi = imp[imp.Method.str.startswith("Multiple")]
        row["Multi-imputation R2 mean"] = round(mi["R2_mean"].values[0], 3)
        row["Multi-imputation R2 std"] = round(mi["R2_std"].values[0], 3)

    ext_path = os.path.join(TABLES, f"{fam}_extended_features_check.csv")
    if os.path.exists(ext_path):
        ext = pd.read_csv(ext_path)
        row["Extended-features R2"] = round(ext.loc[ext["Feature set"] == "Extended structural", "R2"].values[0], 3)

    rows.append(row)
sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(TABLES, "sensitivity_checks_all_families.csv"), index=False)
print()
print(sens.to_string(index=False))

# --- XGBoost / CatBoost comparison (structural variant) ---
rows = []
for fam in FAMILIES:
    df = pd.read_csv(os.path.join(TABLES, f"{fam}_structural_model_metrics.csv"))
    row = {"Pavement family": FAMILY_LABEL[fam]}
    for model_key, col in [("LightGBM", "LightGBM R2"), ("XGBoost", "XGBoost R2"), ("CatBoost", "CatBoost R2")]:
        match = df[df.Model.str.contains(model_key)]
        row[col] = round(match["R2"].values[0], 3) if len(match) else None
    rows.append(row)
gbm_compare = pd.DataFrame(rows)
gbm_compare.to_csv(os.path.join(TABLES, "gbm_implementation_comparison.csv"), index=False)
print()
print(gbm_compare.to_string(index=False))
