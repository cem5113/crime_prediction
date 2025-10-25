# app.py — SUTAM (repo yapısına uyarlanmış)
import streamlit as st

from components.last_update import show_last_update_badge
from components.utils import MODEL_VERSION, MODEL_LAST_TRAIN  # utils.py içinden

st.set_page_config(page_title="Suç Tahmini Uygulaması", layout="wide")
st.title("Suç Tahmini Uygulaması")
st.write("Soldaki **Pages** menüsünden sekmelere geçebilirsiniz.")
st.info("🔎 Harita için: **🧭 Suç Tahmini** sekmesine gidin.")

# Model rozeti (varsa)
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)
