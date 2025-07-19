# streamlit_app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from datetime import datetime

# 0. LANGUAGE SELECTION / DİL SEÇİMİ 
# Let the user select the interface language / Kullanıcı arayüz dilini seçer
st.set_page_config(page_title="Crime Risk Map / Suç Risk Haritası", layout="wide")
st.sidebar.title("\U0001F310 Language / Dil")
language = st.sidebar.radio("Select Language / Dil Seçin:", ["English", "Türkçe"])

# 1. TEXT DEFINITIONS / METİN TANIMLARI 
# Set UI text based on selected language / Seçilen dile göre metinleri ayarla
if language == "English":
    title = "\U0001F6A8 Crime Risk Prediction: Hourly Risk Map"
    subtitle = "This app visualizes predicted crime risk across San Francisco based on selected hour."
    hour_label = "Select an hour:"
    map_title = "Crime risk map for hour"
    info_text = "This map shows predicted crime risk (0–1) for each GEOID. Darker areas represent higher risk."
    legend_label = "Crime Risk (0–1)"
else:
    title = "\U0001F6A8 Suç Tahmin Modeli: Saatlik Risk Haritası"
    subtitle = "Bu uygulama, seçilen saat için San Francisco genelindeki tahmini suç riskini harita üzerinde gösterir."
    hour_label = "Bir saat seçin:"
    map_title = "Saat için suç risk haritası"
    info_text = "Bu harita, model çıktısına göre her GEOID için tahmini suç olasılığını gösterir. Koyu renkler daha yüksek riski temsil eder."
    legend_label = "Suç Riski (0–1)"

# 2. PAGE TITLE AND DESCRIPTION / SAYFA BAŞLIĞI VE AÇIKLAMA 
st.title(title)
st.markdown(subtitle)

# 3. LOAD DATA / VERİLERİ YÜKLE 
# Load prediction and geospatial data / Tahmin ve mekânsal verileri yükle
@st.cache_data(ttl=3600)
def load_predictions():
    file_id = "1Y8v2fo8w85N5ldSQcqoN1LZvGlfqQOHb"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url, on_bad_lines='skip') 

@st.cache_data
def load_geodata():
    geojson_path = "https://raw.githubusercontent.com/your_repo/sf_blockgroup.geojson"
    return gpd.read_file(geojson_path)

pred_df = load_predictions()
geodf = load_geodata()

# 4. HOUR SELECTOR / SAAT SEÇİCİ 
# Let user select an hour to visualize crime risk / Kullanıcının suç riskini görmek için saat seçmesini sağlar
st.sidebar.header("\u23F0 " + hour_label)
hour_selected = st.sidebar.selectbox(hour_label, options=sorted(pred_df["event_hour"].unique()))
today = datetime.today().strftime("%Y-%m-%d")

# 5. FILTER PREDICTIONS / TAHMİNLERİ FİLTRELE 
# Filter prediction data based on selected hour and today's date / Seçilen saate ve bugünün tarihine göre veriyi filtrele
pred_hour_df = pred_df[(pred_df["event_hour"] == hour_selected) & (pred_df["date"] == today)]
pred_hour_df["GEOID"] = pred_hour_df["GEOID"].astype(str).str.zfill(11)

# 6. CREATE MAP / HARİTA OLUŞTUR 
# Merge prediction with geospatial data and create choropleth map / Tahmin verisini mekânsal veriyle birleştir ve ısı haritası oluştur
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

# 7. DISPLAY MAP / HARİTAYI GÖSTER 
# Show the generated map on the Streamlit app / Oluşturulan haritayı Streamlit uygulamasında göster
st.subheader(f"\U0001F552 {hour_selected}:00 - {map_title} ({today})")
st_data = st_folium(m, width=1100, height=600)

# 8. INFO BOX / BİLGİ KUTUSU 
# Show context info about the map / Harita hakkında açıklayıcı bilgi göster
st.info(info_text)
