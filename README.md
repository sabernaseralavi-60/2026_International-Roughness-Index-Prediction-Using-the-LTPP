# Beyond Asphalt: Cross-Pavement-Family IRI Prediction Using the LTPP Database

A reproducible machine-learning pipeline that predicts International Roughness Index (IRI) for **both**
flexible (asphalt) and rigid (JPCP/JRCP/CRCP) pavements from the FHWA Long-Term Pavement Performance (LTPP)
database, using an explainable gradient-boosted ensemble (LightGBM + SHAP) benchmarked against an
MEPDG-style regression, a random forest, and a neural network.

## 📥 Downloads

| File | View online | Direct download |
|---|---|---|
| 📄 Manuscript — English (PDF) | — | **[⬇ Download PDF](https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP/raw/master/docs/paper/paper_en.pdf)** |
| 🌐 Manuscript — English (HTML) | **[View styled page](https://htmlpreview.github.io/?https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP/blob/master/docs/paper/paper_en.html)** | [⬇ Download HTML](https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP/raw/master/docs/paper/paper_en.html) |
| 📝 Manuscript — Persian (Word) | — | **[⬇ Download DOCX](https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP/raw/master/docs/paper/paper_fa.docx)** |
| 📊 Master results table | — | [⬇ Download DOCX](https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP/raw/master/tables/master_lightgbm_comparison.docx) |

> The "View online" link renders the HTML manuscript as an actual styled web page (via [htmlpreview.github.io](https://htmlpreview.github.io)) rather than GitHub's raw-source view. The "Download" links go straight to the file — GitHub will save it to disk (PDF/DOCX) rather than opening a preview page.
>
> Every other individual figure/table can be downloaded the same way: open it in [`plots/`](plots/) or [`tables/`](tables/), then use GitHub's **Download raw file** button (⬇ icon) on the file page.

## Why this exists

Nearly every published ML model for IRI is trained on asphalt pavements alone. Rigid pavements (JPCP, JRCP,
CRCP) are still designed around a fixed-form linear regression from NCHRP 1-37A Appendix PP, fit decades ago
and never re-learned from the much larger LTPP dataset that has accumulated since. This project builds one
pipeline that treats both pavement families the same way — same data source, same feature-engineering
discipline, same model family, same evaluation protocol — so the two are directly comparable.

## Key results

| Pavement family | Model variant | Held-out R² | Notes |
|---|---|---:|---|
| Flexible (GPS-1/2) | Structural (distress + structure + climate) | **0.414** | 60%+ RMSE reduction vs. MEPDG-style regression |
| Flexible | Operational (+ lagged IRI) | **0.698** | forecasting-oriented variant |
| Rigid, pooled (JPCP+JRCP+CRCP) | Structural | **0.220** | |
| Rigid, pooled | Operational (+ lagged IRI) | **0.391** | |
| Rigid, JPCP/JRCP only | Structural | −0.294 | pooling outperforms splitting at this sample size |
| Rigid, CRCP only | Structural | 0.226 (±0.246 CV) | high variance, only 61 sections |

All numbers use a **section-grouped train/test split** (`GroupShuffleSplit` on `SHRP_ID`) so that repeated
visits to the same physical pavement section never appear on both sides of the split — a naive row-level
split was found, during review, to inflate reported R² substantially (0.52 → 0.33 for one variant). Full
comparison tables (linear regression / random forest / ANN / LightGBM, with 5-fold group cross-validation) are
in [`tables/`](tables/) as paired CSV and Word files, and every design choice is documented inline in the code
and in the manuscript's Methods and Limitations sections.

## Repository structure

```
code/     Reproducible pipeline, run in order (00 → 06)
data/
  raw/by_state/    Per-state extracts from the LTPP Standard Data Release (public domain)
  processed/       Cleaned, feature-engineered modeling datasets
plots/    All figures used in the manuscript (EDA, SHAP summaries, predicted-vs-actual)
tables/   Publication-ready tables, each as both .csv and .docx
docs/
  paper/    Quarto source for the manuscript (English .qmd → PDF/HTML, Persian .qmd → Word)
  refs/     Source PDFs referenced in the methodology (NCHRP 1-37A Appendix PP, the target benchmark paper)
  author/   Author photo, CV, and co-author notes
```

## Reproducing the pipeline

Requires Python 3.11+, the Microsoft Access ODBC driver (for reading LTPP's `.accdb` files), and
[Quarto](https://quarto.org) if you want to re-render the manuscript.

```bash
pip install pandas pyodbc lightgbm shap scikit-learn optuna matplotlib seaborn python-docx

# 1. Download and extract all 62 US states / Canadian provinces from LTPP SDR 39
python code/00_download_and_extract_all.py

# 2. Clean, engineer features (age, site factor, lag features), build modeling tables
python code/02_clean_and_merge.py

# 3. Exploratory data analysis
python code/03_eda.py

# 4. Train + tune + explain models (structural / no-traffic-ablation / operational-lagged variants)
python code/04_model.py flexible
python code/04_model.py rigid
python code/04_model.py rigid_jpcc
python code/04_model.py rigid_crcp

# 5. Build the cross-family comparison table and export everything to Word
python code/06_consolidate_results.py
python code/05_export_tables.py

# 6. Render the manuscript
cd docs/paper
quarto render paper_en.qmd --to html
quarto render paper_en.qmd --to pdf
quarto render paper_fa.qmd --to docx
```

Step 1 downloads directly from FHWA InfoPave's public CloudFront distribution (no login required) and
discards the multi-hundred-megabyte raw Access databases after extraction, keeping only the derived CSVs —
that's why `data/raw/by_state/` stays under a few megabytes while remaining fully re-derivable from the public
source.

## Data source

Raw data: [LTPP InfoPave](https://infopave.fhwa.dot.gov), Standard Data Release 39 (September 2025), U.S.
Federal Highway Administration. LTPP data is public domain. This repository does not redistribute the raw
Access databases, only derived, aggregated CSV extracts.

## Citation

If you use this pipeline or its results, please cite:

```bibtex
@misc{naseralavi2026beyondasphalt,
  title  = {Beyond Asphalt: A Cross-Pavement-Family Explainable Machine Learning Approach to
            International Roughness Index Prediction Using the LTPP Database},
  author = {Naseralavi, Seyed Saber and Ghanizadeh, Ali Reza and Mazaheri, Akram},
  year   = {2026},
  url    = {https://github.com/sabernaseralavi-60/2026_International-Roughness-Index-Prediction-Using-the-LTPP}
}
```

## Authors

- **Seyed Saber Naseralavi** (corresponding author) — [github.com/sabernaseralavi-60](https://github.com/sabernaseralavi-60)
- **Ali Reza Ghanizadeh** — Sirjan University of Technology
- **Akram Mazaheri** — Tarbiat Modares University

## License

Code is released under the [MIT License](LICENSE). Underlying LTPP data is U.S. public domain, distributed by
FHWA InfoPave.
