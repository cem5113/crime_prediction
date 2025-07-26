# streamlit_app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import gdown
import os

# 0. LANGUAGE SELECTION / DİL SEÇİMİ 
st.set_page_config(page_title="Crime Risk Map / Suç Risk Haritası", layout="wide")
st.sidebar.title("🌐 Language / Dil")
language = st.sidebar.radio("Select Language / Dil Seçin:", ["English", "Türkçe"])

# 1. TEXT DEFINITIONS / METİN TANIMLARI 
if language == "English":
    title = "🚨 Crime Risk Prediction: Hourly Risk Map"
    subtitle = "This app visualizes predicted crime risk across San Francisco based on selected hour."
    hour_label = "Select an hour:"
    map_title = "Crime risk map for hour"
    info_text = "This map shows predicted crime risk (0–1) for each GEOID. Darker areas represent higher risk."
    legend_label = "Crime Risk (0–1)"
else:
    title = "🚨 Suç Tahmin Modeli: Saatlik Risk Haritası"
    subtitle = "Bu uygulama, seçilen saat için San Francisco genelindeki tahmini suç riskini harita üzerinde gösterir."
    hour_label = "Bir saat seçin:"
    map_title = "Saat için suç risk haritası"
    info_text = "Bu harita, model çıktısına göre her GEOID için tahmini suç olasılığını gösterir. Koyu renkler daha yüksek riski temsil eder."
    legend_label = "Suç Riski (0–1)"

# 2. PAGE TITLE AND DESCRIPTION
st.title(title)
st.markdown(subtitle)

# 3. LOAD DATA
@st.cache_data(ttl=3600)
def load_predictions():
    url = "https://github.com/cem5113/crime_prediction_data/raw/main/crime_risk_predictions.csv"
    return pd.read_csv(url)

@st.cache_data(ttl=3600)
def load_predictions():
    import gdown
    import os

    file_id = "1Y8v2fo8w85N5ldSQcqoN1LZvGlfqQOHb"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "/tmp/sf_crime_11.csv"

    try:
        gdown.download(url, output, quiet=False)
        df = pd.read_csv(output)
        st.success("✅ Yüklendi: sf_crime_11.csv")
        st.write(df.head())
        return df
    except Exception as e:
        st.error(f"❌ CSV okunamadı: {e}")
        return pd.DataFrame()
# === LOAD ===
pred_df = load_predictions()

# DEBUG
if pred_df.empty:
    st.error("❗️Veri yüklenemedi veya boş.")
    st.stop()

# DEBUG INFO
st.write("📄 Yüklü veri sütunları:", pred_df.columns.tolist())
st.write("📊 Veri boyutu:", pred_df.shape)

# convert date
pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
pred_df = pred_df.dropna(subset=["date"])
pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")

geodf = load_geodata()

# 4. HOUR SELECTOR
st.sidebar.header("⏰ " + hour_label)
hour_selected = st.sidebar.selectbox(hour_label, options=sorted(pred_df["event_hour"].unique()))
today = pred_df["date"].max()  # Otomatik en güncel gün

# 5. FILTER PREDICTIONS
pred_hour_df = pred_df[(pred_df["event_hour"] == hour_selected) & (pred_df["date"] == today)].copy()
pred_hour_df["GEOID"] = pred_hour_df["GEOID"].astype(str).str.zfill(11)

# 6. CREATE MAP
m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
merged = geodf.merge(pred_hour_df, on="GEOID", how="left")

folium.Choropleth(
    geo_data=merged,
    data=merged,
    columns=["GEOID", "crime_risk"],
    key_on="feature.properties.GEOID",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color="white",
    legend_name=legend_label
).add_to(m)

# 7. DISPLAY MAP
st.subheader(f"🕒 {hour_selected}:00 - {map_title} ({today})")
st_data = st_folium(m, width=1100, height=600)

# 8. INFO BOX
st.info(info_text)
