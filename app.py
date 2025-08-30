import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("Hello Streamlit + Folium (smoke test)")
m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
st_folium(m, height=500)
st.success("Çalışıyor! Şimdi asıl uygulama kodunu geri koyabilirsiniz.")
