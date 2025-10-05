# components/last_update.py
# components/last_update.py
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

def _fmt_sf(ts_utc: datetime) -> str:
    """
    UTC datetime → America/Los_Angeles'a çevirip dd-mm-YYYY HH:MM formatlar.
    Güvenli: tzinfo yoksa UTC varsayar.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=ZoneInfo("UTC"))
    dt_sf = ts_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    return dt_sf.strftime("%d-%m-%Y %H:%M")

def show_last_update_badge(last_updated_utc: datetime | None, show_title: bool = False) -> None:
    """
    Sağ üstte 'Son güncelleme (SF)' rozeti gösterir.
    - show_title=True yaparsan başlığı da solda yazar (genelde False tutalım; app.py başlığı atsın).
    """
    if last_updated_utc is None:
        return

    ts = _fmt_sf(last_updated_utc)

    # Küçük, sade bir CSS – sayfanın geri kalanına dokunmuyor
    st.markdown(
        """
        <style>
          .sutam-header { display:flex; align-items:center; justify-content:space-between; }
          .update-badge {
            display:inline-flex; align-items:center; gap:8px;
            background:#eef2ff; color:#111827;
            border:1px solid #e5e7eb; border-radius:999px;
            padding:6px 12px; font-size:13px; font-weight:600;
            box-shadow:0 1px 2px rgba(0,0,0,.05);
            white-space:nowrap;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        if show_title:
            st.markdown("### **SUTAM: Suç Tahmin Modeli**")  # opsiyonel
        else:
            st.empty()
    with right:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin-top:4px;">
              <span class="update-badge">🕒 <span>Son güncelleme (SF): {ts}</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
