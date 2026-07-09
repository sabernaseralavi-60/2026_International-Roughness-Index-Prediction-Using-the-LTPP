import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
PLOTS = os.path.join(ROOT, "plots")
TABLES = os.path.join(ROOT, "tables")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

flex = pd.read_csv(os.path.join(PROC, "flexible_iri_clean.csv"))
rigid = pd.read_csv(os.path.join(PROC, "rigid_iri_clean.csv"))

summary_rows = []
for name, df in [("Flexible (GPS-1/2)", flex), ("Rigid (GPS-3/4/5/6/9)", rigid)]:
    summary_rows.append({
        "Dataset": name, "N_records": len(df), "N_sections": df.SHRP_ID.nunique(),
        "N_states": df.STATE_ABBR.nunique(),
        "IRI_mean": df.MEAN_IRI.mean(), "IRI_sd": df.MEAN_IRI.std(),
        "IRI_min": df.MEAN_IRI.min(), "IRI_max": df.MEAN_IRI.max(),
        "Age_mean_yr": df.AGE_YR.mean(),
    })
summary = pd.DataFrame(summary_rows).round(2)
summary.to_csv(os.path.join(TABLES, "dataset_summary.csv"), index=False)
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (name, df, color) in zip(axes, [("Flexible", flex, "#3B7EA1"), ("Rigid", rigid, "#B54A4A")]):
    ax.hist(df.MEAN_IRI, bins=25, color=color, edgecolor="white")
    ax.set_title(f"{name} pavements (n={len(df)})")
    ax.set_xlabel("Mean IRI (m/km)")
    ax.set_ylabel("Count")
fig.suptitle("Distribution of International Roughness Index by Pavement Family")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "iri_distribution.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(flex.AGE_YR, flex.MEAN_IRI, s=14, alpha=0.5, color="#3B7EA1", label="Flexible")
ax.scatter(rigid.AGE_YR, rigid.MEAN_IRI, s=14, alpha=0.5, color="#B54A4A", label="Rigid")
ax.set_xlabel("Pavement age at survey (years)")
ax.set_ylabel("Mean IRI (m/km)")
ax.set_title("IRI vs. Pavement Age Across Families")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "iri_vs_age.png"), bbox_inches="tight")
plt.close(fig)

print("\nMissingness (flexible modeling set):")
print(flex.isna().mean().round(3).to_string())
print("\nMissingness (rigid modeling set):")
print(rigid.isna().mean().round(3).to_string())
