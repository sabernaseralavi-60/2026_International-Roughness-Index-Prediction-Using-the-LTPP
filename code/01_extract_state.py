"""
Extract flexible (GPS-1/GPS-2) and rigid (GPS-3/GPS-5) pavement IRI records
plus matched distress, structure, traffic, and climate covariates from one
LTPP Standard Data Release (SDR) Access database (SDR_xx_<ST>_Primary_Data.accdb).

Usage: python 01_extract_state.py <path_to_accdb> <state_abbr> <out_dir>

Design notes (documented for the methodology section):
- GPS-1 = AC surface on granular base, GPS-2 = AC surface on bound/stabilized base
  -> both are "flexible" (asphalt) pavement experiments, matching Song et al. (2022).
- GPS-3 = JPCP (jointed plain concrete), GPS-5 = JPCP w/ AC overlay -> rigid family JPCC.
  CRCP experiments are GPS-4/GPS-6/GPS-9 in some states; we pull CRCP distress
  wherever PAVEMENT_FAMILY = 'CRCP' shows up in ANALYSIS_IRI regardless of exact
  GPS number, since CRCP-specific distress is only meaningful for that family.
- IRI is paired to the nearest distress survey (by SHRP_ID + CONSTRUCTION_NO)
  within +/- 365 days. LTPP manual distress surveys run on a slower cadence
  (roughly biennial) than automated profile/IRI runs (roughly annual), so a
  tight +/-60-day window discards most legitimate same-condition-cycle pairs;
  +/-365 days is the widest window that still guarantees at most one
  distress cycle separates the paired records, consistent with the
  "on-cycle" merge tolerance discussed in NCHRP 1-37A Appendix PP.
- Structural layer thickness is aggregated to a single "AC_THICKNESS" /
  "PCC_THICKNESS" (sum of bound layer REPR_THICKNESS) per CONSTRUCTION_NO.
- Traffic (AADTT, KESAL) and climate (freeze index, precip) are matched by
  calendar year of the IRI visit.
- True pavement age (AGE_YR) uses INV_AGE.CONSTRUCTION_DATE (falling back to
  TRAFFIC_OPEN_DATE) rather than "years since first LTPP visit", since a
  section is frequently monitored for years before its earliest IRI record.
"""
import sys
import pyodbc
import pandas as pd

FLEX_SQL = """
SELECT
    e.STATE_CODE, e.SHRP_ID, e.CONSTRUCTION_NO, e.GPS_SPS, e.EXPERIMENT_NO,
    i.VISIT_DATE, i.IRI_LEFT_WHEEL_PATH, i.IRI_RIGHT_WHEEL_PATH, i.PAVEMENT_FAMILY
FROM EXPERIMENT_SECTION e
INNER JOIN ANALYSIS_IRI i
    ON e.STATE_CODE = i.STATE_CODE AND e.SHRP_ID = i.SHRP_ID
    AND e.CONSTRUCTION_NO = i.CONSTRUCTION_NO
WHERE e.GPS_SPS = 'G' AND e.EXPERIMENT_NO IN ('1','2')
"""

RIGID_SQL = """
SELECT
    e.STATE_CODE, e.SHRP_ID, e.CONSTRUCTION_NO, e.GPS_SPS, e.EXPERIMENT_NO,
    i.VISIT_DATE, i.IRI_LEFT_WHEEL_PATH, i.IRI_RIGHT_WHEEL_PATH, i.PAVEMENT_FAMILY
FROM EXPERIMENT_SECTION e
INNER JOIN ANALYSIS_IRI i
    ON e.STATE_CODE = i.STATE_CODE AND e.SHRP_ID = i.SHRP_ID
    AND e.CONSTRUCTION_NO = i.CONSTRUCTION_NO
WHERE e.GPS_SPS = 'G' AND e.EXPERIMENT_NO IN ('3','4','5')
"""
# GPS-3 = JPCP (new, on granular/treated base), GPS-4 = JRCP, GPS-5 = CRCP (new).
# These are the "new rigid construction" experiments, the rigid-family analogue
# of GPS-1/2 for flexible. GPS-6/7/9 are excluded: they are AC-over-PCC or
# PCC-over-PCC *overlay* experiments (mixed/rehabilitated structures), not a
# clean rigid baseline, mirroring why GPS-1/2 (not SPS rehab experiments) were
# used for the flexible family.
# NB: PAVEMENT_FAMILY as recorded in ANALYSIS_IRI is a granular structural code
# (e.g. 'JPCTB' = JPCP on treated base, 'JPCUB' = JPCP on unbound base, 'JRCP',
# 'CRCP') rather than a clean 'JPCP'/'CRCP' label, so classification here is
# done via EXPERIMENT_NO, and PAVEMENT_FAMILY is kept only as a descriptive tag.


def q(cn, sql):
    return pd.read_sql(sql, cn)


def extract(db_path, state_abbr, out_dir):
    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + db_path
    )
    cn = pyodbc.connect(conn_str)

    flex = q(cn, FLEX_SQL)
    rigid = q(cn, RIGID_SQL)

    if len(flex):
        dis_ac = q(cn, "SELECT * FROM ANALYSIS_DIS_AC")
        rut = q(cn, "SELECT * FROM ANALYSIS_RUTTING")
        flex = merge_nearest(flex, dis_ac, "SURVEY_DATE",
                              ["HPMS16_CRACKING_PERCENT_AC", "MEPDG_CRACKING_PERCENT_AC",
                               "MEPDG_TRANS_CRACK_LENGTH_AC", "MEPDG_LONG_CRACK_LENGTH_AC",
                               "PATCH_A", "POTHOLES_A", "BLEEDING", "RAVELING"])
        flex = merge_nearest(flex, rut, "SURVEY_DATE",
                              ["MAX_MEAN_DEPTH_WIRE_REF", "LLH_DEPTH_WIRE_REF_MEAN",
                               "RLH_DEPTH_WIRE_REF_MEAN"])
        flex = add_structure(cn, flex, layer_types=("AC",))
        flex = add_traffic_climate(cn, flex)
        flex = add_construction_date(cn, flex)
        flex["STATE_ABBR"] = state_abbr
        flex.to_csv(f"{out_dir}/flexible_{state_abbr}.csv", index=False)

    if len(rigid):
        dis_j = q(cn, "SELECT * FROM ANALYSIS_DIS_JPCC")
        dis_c = q(cn, "SELECT * FROM ANALYSIS_DIS_CRCP")
        fam = rigid.PAVEMENT_FAMILY.astype(str)
        rigid_j = rigid[fam.str.startswith(("JPC", "JRC"))]
        rigid_c = rigid[fam.str.startswith("CRC")]
        rigid_j = merge_nearest(rigid_j, dis_j, "SURVEY_DATE",
                                 ["HPMS16_CRACKING_PERCENT_JPCC", "MEPDG_CRACKING_PERCENT_JPCC",
                                  "AVG_WHEELPATH_FAULT", "TRANS_SPALLING_L", "TRANS_CRACK_NO",
                                  "PATCH_RIGID_A"])
        rigid_c = merge_nearest(rigid_c, dis_c, "SURVEY_DATE",
                                 ["HPMS16_CRACKING_PERCENT_CRCP", "MEPDG_PUNCHOUTS_CRCP",
                                  "PUNCHOUTS_NO", "LONG_SPALLING_L", "PATCH_RIGID_A"])
        rigid_all = pd.concat([rigid_j, rigid_c], ignore_index=True)
        rigid_all = add_structure(cn, rigid_all, layer_types=("PC", "AC"))
        rigid_all = add_traffic_climate(cn, rigid_all)
        rigid_all = add_construction_date(cn, rigid_all)
        rigid_all["STATE_ABBR"] = state_abbr
        rigid_all.to_csv(f"{out_dir}/rigid_{state_abbr}.csv", index=False)

    cn.close()
    return len(flex), len(rigid)


def add_construction_date(cn, df):
    try:
        age = q(cn, "SELECT SHRP_ID, STATE_CODE, CONSTRUCTION_NO, CONSTRUCTION_DATE, TRAFFIC_OPEN_DATE FROM INV_AGE")
    except Exception:
        df["CONSTRUCTION_DATE"] = pd.NaT
        return df
    age["CONSTRUCTION_DATE"] = pd.to_datetime(age["CONSTRUCTION_DATE"])
    age["TRAFFIC_OPEN_DATE"] = pd.to_datetime(age["TRAFFIC_OPEN_DATE"])
    age["CONSTRUCTION_DATE"] = age["CONSTRUCTION_DATE"].fillna(age["TRAFFIC_OPEN_DATE"])
    age = age[["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO", "CONSTRUCTION_DATE"]]
    return df.merge(age, on=["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"], how="left")


def merge_nearest(base, dis, date_col, keep_cols, tol_days=365):
    if not len(dis):
        for c in keep_cols:
            base[c] = pd.NA
        return base
    dis = dis.copy()
    dis[date_col] = pd.to_datetime(dis[date_col])
    base = base.copy()
    base["VISIT_DATE"] = pd.to_datetime(base["VISIT_DATE"])
    out_rows = []
    for (sid, cno), grp in base.groupby(["SHRP_ID", "CONSTRUCTION_NO"]):
        cand = dis[(dis.SHRP_ID == sid) & (dis.CONSTRUCTION_NO == cno)]
        for _, r in grp.iterrows():
            row = r.to_dict()
            if len(cand):
                diffs = (cand[date_col] - r["VISIT_DATE"]).abs()
                idx = diffs.idxmin()
                if diffs.loc[idx].days <= tol_days:
                    for c in keep_cols:
                        row[c] = cand.loc[idx, c]
                else:
                    for c in keep_cols:
                        row[c] = pd.NA
            else:
                for c in keep_cols:
                    row[c] = pd.NA
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def add_structure(cn, df, layer_types):
    layers = q(cn, "SELECT SHRP_ID, STATE_CODE, CONSTRUCTION_NO, LAYER_TYPE, REPR_THICKNESS FROM SECTION_LAYER_STRUCTURE")
    layers = layers[layers.LAYER_TYPE.isin(layer_types)]
    thick = layers.groupby(["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"])["REPR_THICKNESS"].sum().reset_index()
    thick = thick.rename(columns={"REPR_THICKNESS": "BOUND_THICKNESS_CM"})
    subg = q(cn, "SELECT SHRP_ID, STATE_CODE, CONSTRUCTION_NO, PLASTICITY_INDEX, PERCENT_FINER_02 FROM INV_SUBGRADE")
    subg = subg.groupby(["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"]).mean(numeric_only=True).reset_index()
    df = df.merge(thick, on=["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"], how="left")
    df = df.merge(subg, on=["SHRP_ID", "STATE_CODE", "CONSTRUCTION_NO"], how="left")
    return df


def add_traffic_climate(cn, df):
    df["YEAR"] = df["VISIT_DATE"].dt.year
    aadtt = q(cn, "SELECT STATE_CODE, SHRP_ID, YEAR, AADTT_LTPPLN FROM TRF_MEPDG_AADTT_LTPP_LN")
    esal = q(cn, "SELECT STATE_CODE, SHRP_ID, YEAR, KESAL_YEAR FROM TRF_ESAL_COMPUTED")
    df = df.merge(aadtt, on=["STATE_CODE", "SHRP_ID", "YEAR"], how="left")
    df = df.merge(esal, on=["STATE_CODE", "SHRP_ID", "YEAR"], how="left")

    try:
        link = q(cn, "SELECT STATE_CODE, SHRP_ID, VWS_ID FROM CLM_SITE_VWS_LINK")
        temp = q(cn, "SELECT VWS_ID, YEAR, FREEZE_INDEX_YR, MEAN_ANN_TEMP_AVG FROM CLM_VWS_TEMP_ANNUAL")
        precip = q(cn, "SELECT VWS_ID, YEAR, TOTAL_ANN_PRECIP FROM CLM_VWS_PRECIP_ANNUAL")
        clim = temp.merge(precip, on=["VWS_ID", "YEAR"], how="outer")
        df = df.merge(link, on=["STATE_CODE", "SHRP_ID"], how="left")
        df = df.merge(clim, on=["VWS_ID", "YEAR"], how="left")
    except Exception:
        df["FREEZE_INDEX_YR"] = pd.NA
        df["MEAN_ANN_TEMP_AVG"] = pd.NA
        df["TOTAL_ANN_PRECIP"] = pd.NA
    return df


if __name__ == "__main__":
    db_path, state_abbr, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_flex, n_rigid = extract(db_path, state_abbr, out_dir)
    print(f"{state_abbr}: flexible={n_flex} rigid={n_rigid}")
