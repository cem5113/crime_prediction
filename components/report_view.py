# components/report_view.py
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import streamlit as st

# path güvenliği
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.reports import normalize_events_ts

try:
    from utils.constants import KEY_COL
except Exception:
    KEY_COL = "geoid"

# ── Basit yardımcılar
def _kpi(ev: pd.DataFrame, agg: pd.DataFrame | None):
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Kayıt (filtre)", f"{len(ev):,}")
    with c2:
        ngeo = ev[KEY_COL].nunique() if KEY_COL in ev.columns else np.nan
        st.metric("GEOID", "—" if np.isnan(ngeo) else f"{int(ngeo):,}")
    with c3:
        last7 = ev[ev["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]
        st.metric("Son 7 gün", f"{len(last7):,}")
    with c4:
        if isinstance(agg, pd.DataFrame) and "expected" in agg.columns and not agg.empty:
            st.metric("Ufuk E[olay] toplam", f"{agg['expected'].sum():.2f}")
        else:
            st.metric("Ufuk E[olay] toplam", "—")

def _filters(ev: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filtreler", expanded=True):
        # tarih
        dmin, dmax = ev["ts"].min().date(), ev["ts"].max().date()
        d1, d2 = st.date_input("Tarih aralığı", value=(dmin, dmax))
        ev = ev[(ev["ts"].dt.date >= d1) & (ev["ts"].dt.date <= d2)]
        # geoid
        if KEY_COL in ev.columns:
            gids = sorted(ev[KEY_COL].dropna().astype(str).unique().tolist())
            mode = st.radio("Kapsam", ["Şehir geneli","Tek alan","Çoklu"], horizontal=True)
            if mode == "Tek alan":
                g = st.selectbox("GEOID", gids)
                ev = ev[ev[KEY_COL].astype(str) == str(g)]
            elif mode == "Çoklu":
                sel = st.multiselect("GEOID seç", gids[:500], default=gids[:10])
                if sel:
                    ev = ev[ev[KEY_COL].astype(str).isin([str(x) for x in sel])]
        # kategori
        cat_col = "type" if "type" in ev.columns else None
        if cat_col:
            cats = sorted(ev[cat_col].dropna().astype(str).unique().tolist())
            ch = st.multiselect("Kategori", cats, default=cats)
            if ch: ev = ev[ev[cat_col].isin(ch)]
        st.caption(f"Filtre sonrası: **{len(ev):,}** kayıt")
    return ev

def _time_charts(ev: pd.DataFrame):
    st.subheader("Zamansal dağılımlar", anchor=False)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.bar_chart(ev.groupby("hour").size().reindex(range(24), fill_value=0))
    with c2:
        dow = ev.groupby("dow").size().reindex(range(7), fill_value=0)
        dow.index = ["Pzt","Sal","Çar","Per","Cum","Cmt","Paz"]
        st.bar_chart(dow)
    with c3:
        mon = ev.groupby(ev["ts"].dt.month).size().reindex(range(1,13), fill_value=0)
        mon.index = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
        st.bar_chart(mon)

def render_reports(events_df: pd.DataFrame | None,
                   agg_current: pd.DataFrame | None = None,
                   agg_long_term: pd.DataFrame | None = None) -> None:
    """Rapor sekmesi (minimum, sağlam)."""
    st.header("Suç Tahmin Raporu")

    # normalize
    ev = normalize_events_ts(events_df, key_col=KEY_COL) if isinstance(events_df, pd.DataFrame) else pd.DataFrame()
    if isinstance(events_df, pd.DataFrame) and not events_df.empty:
        st.caption(f"Girdi boyutu: {events_df.shape}")
    if ev.empty:
        st.warning("Olay veri seti yok veya zaman sütunu parse edilemedi.")
        if isinstance(events_df, pd.DataFrame):
            st.caption("Mevcut kolonlar:")
            st.code(", ".join(map(str, events_df.columns)))
            st.caption("İlk 5 satır (ham):")
            st.dataframe(events_df.head(5), use_container_width=True)
        return

    # filtreler
    ev = _filters(ev)

    # KPI
    _kpi(ev, agg_current)

    # grafikler
    _time_charts(ev)

    # dışa aktar
    st.subheader("Dışa aktar", anchor=False)
    st.download_button(
        "Filtreli olay verisini indir (CSV)",
        data=ev.to_csv(index=False).encode("utf-8"),
        file_name="rapor_filtre.csv",
        mime="text/csv",
    )
