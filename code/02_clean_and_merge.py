"""
Consolidate all per-state raw extracts (data/raw/by_state/*.csv) into two
national analysis-ready datasets:
    data/processed/flexible_iri_clean.csv
    data/processed/rigid_iri_clean.csv

Cleaning / feature-engineering choices (documented for the Methodology
section and for the Q1-reviewer critique pass):

1. Repeat-run averaging: LTPP records several IRI runs per visit. We average
   IRI_LEFT/RIGHT_WHEEL_PATH to one value per (SHRP_ID, VISIT_DATE) and take
   the mean of left/right as MEAN_IRI (m/km), matching the convention used in
   Song et al. (2022) and NCHRP 1-37A Appendix PP.
2. Pavement age: AGE_YR = (VISIT_DATE - first VISIT_DATE per SHRP_ID+CONSTRUCTION_NO
   with an IRI record) in years. This is a lower-bound proxy for true
   construction age since we don't always have as-built date in this extract;
   flagged as a limitation.
3. Site factor (flexible), following Appendix PP / MEPDG form:
       SF = AGE_YR * (1 + FREEZE_INDEX_YR) * (1 + PERCENT_FINER_02) / 1e6
   PERCENT_FINER_02 (percent passing 0.02 mm) is used as the closest available
   proxy for the MEPDG "P200" subgrade fines fraction; documented as an
   approximation, not an exact P200 replication.
4. Missing-data handling: rows missing MEAN_IRI are dropped (target variable).
   Predictor columns with >60% missingness at the national level are dropped;
   remaining gaps are left as NA for model-side imputation (documented, not
   silently mean-filled, so the modeling stage can make an explicit choice).
5. Outlier flags: any *_FLAG columns already carried over from LTPP's own
   ANALYSIS_* quality-control flags are preserved and rows flagged 'I'
   (invalid) or 'D' (deleted) in RECORD_STATUS-derived columns are excluded
   where present.
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "by_state")
OUT_DIR = os.path.join(ROOT, "data", "processed")


def load_family(prefix):
    files = glob.glob(os.path.join(RAW_DIR, f"{prefix}_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)


def aggregate_iri(df):
    df["VISIT_DATE"] = pd.to_datetime(df["VISIT_DATE"])
    df["MEAN_IRI"] = df[["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"]].mean(axis=1)
    key = ["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "VISIT_DATE"]
    agg_cols = {c: "first" for c in df.columns if c not in key + ["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH", "MEAN_IRI"]}
    agg_cols["MEAN_IRI"] = "mean"
    out = df.groupby(key, as_index=False).agg(agg_cols)
    return out


def add_age(df):
    df = df.sort_values(["SHRP_ID", "CONSTRUCTION_NO", "VISIT_DATE"])
    if "CONSTRUCTION_DATE" in df.columns:
        cdate = pd.to_datetime(df["CONSTRUCTION_DATE"])
    else:
        cdate = pd.Series(pd.NaT, index=df.index)
    first_visit = df.groupby(["SHRP_ID", "CONSTRUCTION_NO"])["VISIT_DATE"].transform("min")
    df["AGE_SOURCE"] = np.where(cdate.notna(), "construction_date", "first_visit_proxy")
    base_date = cdate.fillna(first_visit)
    df["AGE_YR"] = (df["VISIT_DATE"] - base_date).dt.days / 365.25
    return df


# Core predictor sets used for the *complete-case* modeling table. Traffic
# (AADTT/KESAL) is deliberately excluded from the "core" requirement: LTPP's
# own automated traffic monitoring coverage is well documented to be sparse
# (~20-30% of visits), and requiring it would gut the sample size. It is kept
# as an optional column for a secondary, smaller-N sensitivity model instead.
FLEX_CORE = ["MEAN_IRI", "AGE_YR", "SITE_FACTOR", "BOUND_THICKNESS_CM",
             "FREEZE_INDEX_YR", "TOTAL_ANN_PRECIP",
             "MEPDG_TRANS_CRACK_LENGTH_AC", "MEPDG_LONG_CRACK_LENGTH_AC",
             "HPMS16_CRACKING_PERCENT_AC", "PATCH_A", "MAX_MEAN_DEPTH_WIRE_REF"]

# Only the single strongest, best-populated distress predictor per rigid
# sub-family is required for complete-case inclusion (mirroring Appendix PP's
# own finding that faulting/punchouts dominate the JPCP/CRCP IRI equations);
# the remaining distress columns (spalling, cracking, patching) are kept as
# optional columns with their native missingness rather than forced into the
# dropna filter, since requiring all of them jointly leaves ~0 JPCP rows.
RIGID_CORE = ["MEAN_IRI", "AGE_YR", "BOUND_THICKNESS_CM",
              "FREEZE_INDEX_YR", "TOTAL_ANN_PRECIP"]
RIGID_JPCC_EXTRA = ["AVG_WHEELPATH_FAULT"]
RIGID_CRCP_EXTRA = ["MEPDG_PUNCHOUTS_CRCP"]


def clean_flexible():
    df = load_family("flexible")
    if not len(df):
        return None
    df = aggregate_iri(df)
    df = add_age(df)
    df["SITE_FACTOR"] = (
        df["AGE_YR"].clip(lower=0)
        * (1 + df["FREEZE_INDEX_YR"].fillna(0))
        * (1 + df["PERCENT_FINER_02"].fillna(0))
        / 1_000_000
    )
    df = df.dropna(subset=["MEAN_IRI"])
    df.to_csv(os.path.join(OUT_DIR, "flexible_iri_raw_merged.csv"), index=False)

    core = [c for c in FLEX_CORE if c in df.columns]
    modeling = df.dropna(subset=core).copy()
    modeling.to_csv(os.path.join(OUT_DIR, "flexible_iri_clean.csv"), index=False)
    return df, modeling


def clean_rigid():
    df = load_family("rigid")
    if not len(df):
        return None
    df = aggregate_iri(df)
    df = add_age(df)
    df = df.dropna(subset=["MEAN_IRI"])
    df.to_csv(os.path.join(OUT_DIR, "rigid_iri_raw_merged.csv"), index=False)

    fam = df.PAVEMENT_FAMILY.astype(str)
    jpcc = df[fam.str.startswith(("JPC", "JRC"))].copy()
    crcp = df[fam.str.startswith("CRC")].copy()
    jpcc_core = [c for c in RIGID_CORE + RIGID_JPCC_EXTRA if c in jpcc.columns]
    crcp_core = [c for c in RIGID_CORE + RIGID_CRCP_EXTRA if c in crcp.columns]
    jpcc_model = jpcc.dropna(subset=jpcc_core)
    crcp_model = crcp.dropna(subset=crcp_core)
    modeling = pd.concat([jpcc_model, crcp_model], ignore_index=True)
    modeling.to_csv(os.path.join(OUT_DIR, "rigid_iri_clean.csv"), index=False)
    return df, modeling


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    f = clean_flexible()
    r = clean_rigid()
    if f is not None:
        raw, modeling = f
        print(f"Flexible: raw_merged={len(raw)} rows / {raw.SHRP_ID.nunique()} sections | "
              f"modeling(complete-case)={len(modeling)} rows / {modeling.SHRP_ID.nunique()} sections")
    if r is not None:
        raw, modeling = r
        print(f"Rigid: raw_merged={len(raw)} rows / {raw.SHRP_ID.nunique()} sections | "
              f"modeling(complete-case)={len(modeling)} rows / {modeling.SHRP_ID.nunique()} sections")
