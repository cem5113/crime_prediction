## 8) app_analysis.py — Derin Analiz Arayüzü

```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.inference_engine import InferenceEngine
from src.features import load_centroids, crime_mix_prior
from src.config import params, paths
from src.viz import draw_map
from src.common import to_hour_range
from src.route_planner import plan_routes

st.set_page_config(page_title="SF Crime — Derin Analiz", layout="wide")

st.title("🔎 Derin Analiz ve Rota Planlama")
engine = InferenceEngine()
centroids = load_centroids()
sf50 = None
if os.path.isfile(paths.SF50_CSV):
    sf50 = pd.read_csv(paths.SF50_CSV)

# Kullanıcı girişleri
c1, c2, c3 = st.columns([2,2,1])
with c1:
    start_h = st.slider("Başlangıç saat", 0, 23, 20)
with c2:
    width_h = st.selectbox("Pencere (saat)", [1,2,3,4], index=1)
with c3:
    topk = st.number_input("Top-K", min_value=5, max_value=50, value=params.TOP_K, step=5)

hour_label = to_hour_range(start_h, width_h)

with st.spinner("Top-K hazırlanıyor..."):
    df_top = engine.predict_topk(hour_label=hour_label, topk=int(topk))

left, right = st.columns([1.2,1])
with left:
    st.subheader("Top-K Tablo")
    st.dataframe(df_top)
with right:
    st.subheader("Harita")
    mp = draw_map(df_top, centroids)
    if mp is not None:
        from streamlit_folium import st_folium
        st_folium(mp, height=520, width=None)
    else:
        st.info("Centroid/geo veri yok.")

# GEOID detay seçimi
st.markdown("---")
st.subheader("📍 GEOID Detay Sayfası")
sel_geoid = st.selectbox("GEOID seçin", df_top["GEOID"].tolist())
sel_row = df_top[df_top["GEOID"]==sel_geoid].iloc[0]

# 1) Crime-mix tahmini (sf50 varsa)
if sf50 is not None:
    mix = crime_mix_prior(sf50, sel_geoid, hour_label, topn=5)
    st.markdown(f"**Beklenen suç karışımı (ilk 5):** {mix if mix else '—'}")
else:
    st.info("sf_crime_50.csv bulunamadı; crime-mix yalnızca tabloya yansıtıldı.")

# 2) Belirsizlik bandı görselleştirme
st.markdown("**Belirsizlik bandı (LCB–UCB)**")
fig = plt.figure(figsize=(5,3))
plt.plot([0,1],[sel_row["lcb"], sel_row["ucb"]])
plt.scatter([0.5],[sel_row["p_crime"]])
plt.xticks([])
plt.ylabel("p_crime")
st.pyplot(fig)

# 3) Risk anlatısı (kısa)
reason = []
if sel_row["trend_z"]>0: reason.append("trend↑")
if sel_row["persistence"]>0: reason.append("süreklilik↑")
if sel_row["neighbor_spillover"]>0.1: reason.append("komşu risk↑")
st.markdown("**Risk anlatısı:** " + (", ".join(reason) if reason else "nötr"))

# 4) Rota planlama
st.markdown("---")
st.subheader("🧭 Rota Önerisi")
n_teams = st.number_input("Ekip sayısı", 1, 6, value=params.N_TEAMS)
eta = st.number_input("Süre (dakika)", 30, 180, value=params.TIME_BUDGET_MIN)
routes = plan_routes(df_top, n_teams=int(n_teams), time_budget_min=int(eta))
if not routes:
    st.info("Rota oluşturmak için centroid verisi ve en az 2 nokta gerekir.")
else:
    for r in routes:
        st.write({k: (round(v,2) if isinstance(v,float) else v) for k,v in r.items()})

# İndirme butonları
import io, json
st.download_button("Top-K CSV indir", data=df_top.to_csv(index=False).encode("utf-8"), file_name="topk_current.csv")
st.download_button("Rotalar JSON indir", data=json.dumps(routes, ensure_ascii=False, indent=2), file_name="routes.json")
