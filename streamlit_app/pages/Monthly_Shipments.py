import re
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Monthly Matrix • Dot Plots", layout="wide")

# Resolve ../data relative to THIS file (pages/…/your_page.py)
DATA_DIR = (Path(__file__).parent.parent / "data").resolve()

# Accept CSV/XLSX/XLS and strict naming like May_Data_Matrix.xlsx
MONTH_FILE_RE = re.compile(r"^([A-Za-z]+)_Data_Matrix\.(csv|xlsx|xls)$", re.I)

def _month_key(m: str) -> int:
    return pd.to_datetime(m, format="%B").month

# Discover month files
month_to_path = {}
for p in DATA_DIR.glob("*_Data_Matrix.*"):
    m = MONTH_FILE_RE.match(p.name)
    if m:
        month_name = m.group(1).capitalize()   # e.g., "May"
        month_to_path[month_name] = p

months_available = sorted(month_to_path.keys(), key=_month_key)

# ---- Guard: nothing found
if not months_available:
    st.error(
        f"No monthly files found in **{DATA_DIR}**.\n\n"
        "Expect names like `May_Data_Matrix.xlsx`, `June_Data_Matrix.csv`, etc."
    )
    with st.expander("Troubleshoot"):
        st.write({
            "DATA_DIR": str(DATA_DIR),
            "Files seen": [p.name for p in DATA_DIR.glob('*')],
            "Pattern": MONTH_FILE_RE.pattern
        })
    st.stop()

st.sidebar.subheader("Pick months")
start_m, end_m = st.sidebar.select_slider(
    "Month range",
    options=months_available,
    value=(months_available[0], months_available[-1])  # safe now
)
# Make inclusive range
start_idx = months_available.index(start_m)
end_idx = months_available.index(end_m)
months_selected = months_available[min(start_idx, end_idx):max(start_idx, end_idx)+1]

@st.cache_data(show_spinner=False)
def load_month_file(path: Path, month_label: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        # pip install openpyxl
        df = pd.read_excel(path, engine="openpyxl")
    df["Month"] = month_label
    return df

frames = [load_month_file(month_to_path[m], m) for m in months_selected] if months_selected else []
df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

if df_all.empty:
    st.info("No month files found or none selected.")
    st.stop()

st.caption(f"Loaded months: **{', '.join(months_selected)}**")

# ---- Choose columns (auto-detect sensible defaults) ----
num_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
cat_cols = [c for c in df_all.columns if not pd.api.types.is_numeric_dtype(df_all[c]) and c not in {"Month"}]

# Helpful defaults if they exist
default_y = "Total monthly shipment" if "Total monthly shipment" in df_all.columns else (num_cols[0] if num_cols else None)
default_x = "Ingredient" if "Ingredient" in df_all.columns else (cat_cols[0] if cat_cols else None)

left, right = st.columns(2)
with left:
    x_cat = st.selectbox("Category (x-axis)", options=cat_cols, index=cat_cols.index(default_x) if default_x in cat_cols else 0)
with right:
    y_metric = st.selectbox("Metric (y-axis)", options=num_cols, index=num_cols.index(default_y) if default_y in num_cols else 0)

# Keep a stable category order across all selected months
cat_order = sorted(df_all[x_cat].astype(str).unique())
cat_to_rank = {c: i for i, c in enumerate(cat_order, start=1)}

df_all["_cat_str"] = df_all[x_cat].astype(str)
df_all["_rank"] = df_all["_cat_str"].map(cat_to_rank)

# Base encodings used in all charts
def dotlayer(data, color_by_month=True):
    enc = {
        "x": alt.X(f"_cat_str:N", sort=cat_order, title=x_cat, axis=alt.Axis(labelAngle=0)),
        "y": alt.Y(f"{y_metric}:Q", title=y_metric),
        "tooltip": [
            alt.Tooltip("_cat_str:N", title=x_cat),
            alt.Tooltip("Month:N"),
            alt.Tooltip(f"{y_metric}:Q", format=",.2f", title=y_metric),
        ],
    }
    if color_by_month:
        enc["color"] = alt.Color("Month:N", legend=alt.Legend(title="Month"))
    return alt.Chart(data).mark_point(size=70, filled=True).encode(**enc)

def trendlayer(data, color_by_month=True):
    # regression(y, x, groupby=...)
    base = alt.Chart(data).transform_regression(
        y_metric, "_rank", groupby=["Month"] if color_by_month else []
    )
    enc = {
        "x": alt.X("_rank:Q", title=None, axis=None),
        "y": alt.Y("y:Q", title=y_metric),
    }
    if color_by_month:
        enc["color"] = alt.Color("Month:N", legend=None)
    return base.mark_line(size=2, opacity=0.8).encode(**enc)

# ---- Tabs: overlay + per-month ----
tabs = st.tabs(["🔎 All selected months (overlay)"] + [f"📅 {m}" for m in months_selected])

# Overlay tab: all months colored
with tabs[0]:
    chart = (dotlayer(df_all, color_by_month=True) + trendlayer(df_all, color_by_month=True)).properties(height=440)
    st.altair_chart(chart, use_container_width=True)

# Per-month tabs with their own (optional) metric picker
for i, m in enumerate(months_selected, start=1):
    with tabs[i]:
        df_m = df_all[df_all["Month"] == m]
        # Optional: let each month choose a different metric (comment out to lock global metric)
        # local_metric = st.selectbox(f"{m} metric", options=num_cols, index=num_cols.index(y_metric))
        # chart_m = (dotlayer(df_m.assign(**{y_metric: df_m[local_metric]}), color_by_month=False)
        #            + trendlayer(df_m.assign(**{y_metric: df_m[local_metric]}), color_by_month=False))
        chart_m = (dotlayer(df_m, color_by_month=False) + trendlayer(df_m, color_by_month=False)).properties(height=440)
        st.altair_chart(chart_m, use_container_width=True)

st.markdown(
    """
**Notes**
- Dots show the selected metric per category; colors indicate month in the overlay.
- The “trendline” is a simple regression of the metric vs a stable integer rank of the categories
  (needed because statistical fits require a numeric x-axis).
- Keep your sheet column names consistent (e.g., `Ingredient`, `Total monthly shipment`) for the best defaults.
"""
)

