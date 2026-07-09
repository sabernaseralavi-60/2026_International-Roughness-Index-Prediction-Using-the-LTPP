"""Build a single master comparison table (LightGBM only, across every
family x variant combination) for the paper's main results table, and a
second table isolating the pooled-vs-split rigid comparison."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")

FAMILIES = ["flexible", "rigid", "rigid_jpcc", "rigid_crcp"]
VARIANTS = ["structural", "no_traffic_lte", "operational_lagged"]

FAMILY_LABEL = {
    "flexible": "Flexible (GPS-1/2)",
    "rigid": "Rigid, pooled (JPCP+JRCP+CRCP)",
    "rigid_jpcc": "Rigid, JPCP/JRCP only",
    "rigid_crcp": "Rigid, CRCP only",
}
VARIANT_LABEL = {
    "structural": "Structural (no traffic/LTE)".replace("no traffic/LTE", "full"),
    "no_traffic_lte": "Structural, traffic+LTE ablated",
    "operational_lagged": "Operational (+ lagged IRI)",
}
VARIANT_LABEL["structural"] = "Structural (full)"

rows = []
for fam in FAMILIES:
    for var in VARIANTS:
        path = os.path.join(TABLES, f"{fam}_{var}_model_metrics.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        gbm = df[df.Model.str.contains("LightGBM")].iloc[0]
        rows.append({
            "Pavement family": FAMILY_LABEL[fam],
            "Variant": VARIANT_LABEL[var],
            "N_train": int(gbm.get("N_train", pd.NA)) if "N_train" in gbm else pd.NA,
            "N_test": int(gbm.get("N_test", pd.NA)) if "N_test" in gbm else pd.NA,
            "R2 (held-out)": round(gbm["R2"], 3),
            "RMSE": round(gbm["RMSE"], 3),
            "CV R2 mean": round(gbm.get("CV_R2_mean", float("nan")), 3) if pd.notna(gbm.get("CV_R2_mean", pd.NA)) else pd.NA,
            "CV R2 std": round(gbm.get("CV_R2_std", float("nan")), 3) if pd.notna(gbm.get("CV_R2_std", pd.NA)) else pd.NA,
        })

master = pd.DataFrame(rows)
master.to_csv(os.path.join(TABLES, "master_lightgbm_comparison.csv"), index=False)
print(master.to_string(index=False))
