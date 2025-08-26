## 7) app_ops.py — Operasyonel Arayüz

```python
from __future__ import annotations
import pandas as pd
import streamlit as st
from src.config import params
from src.inference_engine import InferenceEngine
from src.features import load_centroids
from src.viz import draw_map
from src.common import to_hour_range

st.set_page_config(page_title="SF Crime — Operasyonel", layout="wide")

st.title("🚓 Operasyonel Risk Paneli")

engine = InferenceEngine()

col1, col2, col3 = st.columns([2,2,1])
with col1:
    st.subheader("Saat Aralığı")
    start_h = st.slider("Başlangıç saat", 0, 23, 20)
with col2:
    width_h = st.selectbox("Pencere (saat)", [1,2,3,4], index=1)
with col3:
    topk = st.number_input("Top-K", min_value=5, max_value=50, value=params.TOP_K, step=5)

hour_label = to_hour_range(start_h, width_h)

st.markdown(f"**Öneri Dilimi:** `{hour_label}`")

with st.spinner("Tahminler üretiliyor..."):
    df_top = engine.predict_topk(hour_label=hour_label, topk=int(topk))

# İlk tablo (Top-10)
st.subheader("🎯 Hemen Devriye: En Öncelikli Bölgeler")
st.dataframe(df_top[["rank","hour_range","GEOID","priority_score","p_crime","lcb","ucb","top3_crime_types"]])

# Harita
st.subheader("🗺️ Harita (Centroid Noktaları)")
centroids = load_centroids()
mp = draw_map(df_top, centroids)
if mp is not None:
    from streamlit_folium import st_folium
    st_folium(mp, height=560, width=None)
else:
    st.info("Centroid/geo veri bulunamadı; yalnızca tablo gösterildi.")

# İndirme
from src.config import paths
import os
os.makedirs(paths.RISK_DIR, exist_ok=True)
out_csv = engine.save_topk(df_top)
st.download_button("CSV indir", data=open(out_csv,"rb").read(), file_name=out_csv.split("/")[-1], mime="text/csv")
