# components/last_update.py
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

def show_last_update_badge(last_updated_utc: datetime):
    dt_sf = last_updated_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Los_Angeles"))
    ts = dt_sf.strftime("%d-%m-%Y %H:%M")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### **SUTAM: Suç Tahmin Modeli**")
    with right:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin-top:4px;">
              <span style="
                display:inline-flex; align-items:center; gap:8px;
                background:#eef2ff; color:#111827;
                border:1px solid #e5e7eb; border-radius:999px;
                padding:6px 12px; font-size:13px; font-weight:600;
                box-shadow:0 1px 2px rgba(0,0,0,.05);">
                <span>🕒</span>
                <span>Son güncelleme (SF): {ts}</span>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
