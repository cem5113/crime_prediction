import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="SF Crime Dashboard", layout="wide")
st.title("SF Crime Dashboard")

DATA_DIR = Path("crime_data")

files = [
    "risk_hourly.csv",
    "patrol_recs.csv",
    "patrol_recs_multi.csv",
    "metrics_all.csv",
]

for fname in files:
    p = DATA_DIR / fname
    st.subheader(fname)
    if p.exists():
        try:
            st.dataframe(pd.read_csv(p).head(200))
        except Exception as e:
            st.warning(f"Okunamadı: {e}")
    else:
        st.info(f"{fname} bulunamadı")
