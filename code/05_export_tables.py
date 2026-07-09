"""Export each publication CSV table in tables/ as a formatted .docx table."""
import glob
import os
import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")

TITLES = {
    "dataset_summary": "Table 1. Summary of the assembled flexible and rigid LTPP IRI datasets",
    "flexible_model_metrics": "Table 2. Model comparison for flexible (asphalt) pavements",
    "rigid_model_metrics": "Table 3. Model comparison for rigid (JPCP/JRCP/CRCP) pavements",
    "flexible_shap_importance": "Table 4. Mean absolute SHAP feature importance — flexible pavements",
    "rigid_shap_importance": "Table 5. Mean absolute SHAP feature importance — rigid pavements",
}


def export_docx(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Feature"})
    df = df.round(4)

    doc = Document()
    title = TITLES.get(name, name)
    h = doc.add_heading(title, level=2)
    for run in h.runs:
        run.font.size = Pt(12)

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
        for p in hdr[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = "" if pd.isna(val) else str(val)

    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)

    out_path = os.path.join(TABLES, f"{name}.docx")
    doc.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    for csv_path in glob.glob(os.path.join(TABLES, "*.csv")):
        export_docx(csv_path)
