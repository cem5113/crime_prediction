# components/report_view.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

# güvenli import: utils.reports.normalize_events_ts
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from utils.reports import normalize_events_ts
except ModuleNotFoundError:
    utils_dir = os.path.join(PROJECT_ROOT, "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from reports import normalize_events_ts  # utils/reports.py

# İsteğe bağlı: SHAP varsa global önem grafiği gösterecek
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Uygulamadaki sabitler (app.py ile uyumlu olmalı)
try:
    from utils.constants import KEY_COL
except Exception:
    KEY_COL = "geoid"  

# 🔹 Normalize fonksiyonunu utils'ten alıyoruz
from utils.reports import normalize_events_ts

# =============== Yardımcılar ===============

def _geoid_filter_ui(ev: pd.DataFrame) -> pd.DataFrame:
    """Kolluk için anlaşılır alan seçimi: şehir geneli / tek / çoklu GEOID."""
    st.subheader("Bölge Seçimi", anchor=False)
    mode = st.radio("Kapsam", ["Şehir geneli", "Tek alan", "Birden fazla alan"], horizontal=True)
    if mode == "Şehir geneli" or KEY_COL not in ev.columns:
        return ev

    geoids = sorted(ev[KEY_COL].dropna().astype(str).unique().tolist())
    if mode == "Tek alan":
        gid = st.selectbox("GEOID seç", geoids, index=0)
        return ev[ev[KEY_COL].astype(str) == str(gid)]
    else:
        sel = st.multiselect("GEOID listesi", geoids[:500], default=geoids[:5])
        if not sel:
            st.info("Bir veya daha fazla GEOID seçin.")
            return ev.iloc[0:0]
        return ev[ev[KEY_COL].astype(str).isin([str(x) for x in sel])]

def _date_range_ui(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return ev
    tmin, tmax = ev["ts"].min().date(), ev["ts"].max().date()
    d1, d2 = st.date_input("Tarih aralığı", value=(tmin, tmax))
    # st.date_input tuple döndürür → date'e kıyasla filtrele
    return ev[(ev["ts"].dt.date >= d1) & (ev["ts"].dt.date <= d2)]

def _category_filter_ui(ev: pd.DataFrame) -> pd.DataFrame:
    cat_col = "type" if "type" in ev.columns else None
    if not cat_col:
        return ev
    cats = sorted(ev[cat_col].dropna().astype(str).unique().tolist())
    if not cats:
        return ev
    sel  = st.multiselect("Suç türü", cats, default=cats)
    if sel:
        return ev[ev[cat_col].isin(sel)]
    return ev

def _kpi_tiles(ev: pd.DataFrame, agg: pd.DataFrame | None):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Toplam olay (filtre)", f"{len(ev):,}")
    with c2:
        uniq_geo = ev[KEY_COL].nunique() if KEY_COL in ev.columns else np.nan
        st.metric("Kayıtlı GEOID", f"{uniq_geo if not np.isnan(uniq_geo) else '—'}")
    with c3:
        last7 = ev[ev["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]
        st.metric("Son 7 gün", f"{len(last7):,}")
    with c4:
        if isinstance(agg, pd.DataFrame) and not agg.empty and "expected" in agg.columns:
            st.metric("Ufuk: Beklenen olay (λ toplam)", f"{agg['expected'].sum():.2f}")
        else:
            st.metric("Ufuk: Beklenen olay", "—")

def _histograms(ev: pd.DataFrame):
    st.subheader("Zamansal Dağılımlar", anchor=False)
    g1, g2, g3 = st.columns(3)
    with g1:
        hourly = ev.groupby("hour").size().reindex(range(24), fill_value=0)
        st.bar_chart(hourly.rename("Saatlik"))
    with g2:
        dow = ev.groupby("dow").size().reindex(range(7), fill_value=0)
        dow.index = ["Pzt","Sal","Çar","Per","Cum","Cmt","Paz"]
        st.bar_chart(dow.rename("Günlere göre"))
    with g3:
        mon = ev.groupby(ev["ts"].dt.month).size().reindex(range(1, 13), fill_value=0)
        mon.index = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
        st.bar_chart(mon.rename("Aylara göre"))

def _rain_effect(ev: pd.DataFrame):
    st.subheader("Hava Durumu Etkisi (Yağış)", anchor=False)
    try:
        wx = pd.read_csv("data/weather_hourly.csv")  # ts, precipitation_mm
        wx["ts_hour"] = pd.to_datetime(wx["ts"], utc=True, errors="coerce").dt.floor("H")
        wx = wx.dropna(subset=["ts_hour"])
        wx["is_rainy"] = (wx["precipitation_mm"].fillna(0) > 0.0).astype(int)

        tmp = ev.copy()
        tmp["ts_hour"] = tmp["ts"].dt.floor("H")
        events_hourly = tmp.groupby("ts_hour").size().rename("events").reset_index()
        dfm = events_hourly.merge(wx[["ts_hour","is_rainy"]], on="ts_hour", how="left")
        dfm["is_rainy"] = dfm["is_rainy"].fillna(0).astype(int)

        exp_rain  = int((dfm["is_rainy"]==1).sum())
        exp_clear = int((dfm["is_rainy"]==0).sum())
        cnt_rain  = int(dfm.loc[dfm["is_rainy"]==1, "events"].sum())
        cnt_clear = int(dfm.loc[dfm["is_rainy"]==0, "events"].sum())
        rr = ((cnt_rain/max(exp_rain,1)) / (cnt_clear/max(exp_clear,1))) if exp_rain and exp_clear else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Yağışlı saat", f"{exp_rain:,}")
        c2.metric("Yağışsız saat", f"{exp_clear:,}")
        c3.metric("Göreli Risk (yağışlı/yağışsız)", f"{rr:.2f}" if pd.notna(rr) else "—")

        tab = pd.DataFrame({
            "Durum": ["Yağışlı","Yağışsız"],
            "Saat (maruziyet)": [exp_rain, exp_clear],
            "Toplam olay": [cnt_rain, cnt_clear],
            "Saat başına olay": [
                round(cnt_rain/max(exp_rain,1),3),
                round(cnt_clear/max(exp_clear,1),3)
            ]
        })
        st.dataframe(tab, use_container_width=True)

    except Exception:
        st.info("Hava durumu dosyası bulunamadı: data/weather_hourly.csv")

def _hot_trend_table(agg_short: pd.DataFrame | None, agg_long: pd.DataFrame | None):
    """Kısa dönem vs uzun dönem trend (örn. son 48s vs 30g)."""
    if (agg_short is None) or (agg_long is None):
        return
    if agg_short.empty or agg_long.empty:
        return
    need_cols = [KEY_COL, "expected"]
    if not all(c in agg_short.columns for c in need_cols) or not all(c in agg_long.columns for c in need_cols):
        return
    st.subheader("Yeni Isınan / Soğuyan Bölgeler", anchor=False)
    a = agg_short[[KEY_COL,"expected"]].rename(columns={"expected":"lambda_short"})
    b = agg_long [[KEY_COL,"expected"]].rename(columns={"expected":"lambda_long"})
    m = a.merge(b, on=KEY_COL, how="inner")
    m["trend_%"] = 100.0 * (m["lambda_short"] - m["lambda_long"]) / (m["lambda_long"].replace(0, np.nan))
    m = m.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    top_up   = m.sort_values("trend_%", ascending=False).head(10)
    top_down = m.sort_values("trend_%", ascending=True).head(10)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Yeni Isınan (↑)**")
        st.dataframe(top_up[[KEY_COL,"lambda_short","lambda_long","trend_%"]], use_container_width=True, height=260)
    with c2:
        st.caption("**Soğuyan (↓)**")
        st.dataframe(top_down[[KEY_COL,"lambda_short","lambda_long","trend_%"]], use_container_width=True, height=260)

def _shap_global_section():
    st.subheader("Özellik Etkileri (SHAP)", anchor=False)
    if not _HAS_SHAP:
        st.info("SHAP paketi yüklü değil (opsiyonel).")
        return

    mdl = st.session_state.get("model")
    X   = st.session_state.get("X_report_sample")  # rapor için hafif örnek (10-20k)
    cols= st.session_state.get("X_cols")

    if mdl is None or X is None or cols is None or len(X)==0:
        st.info("Model/özellik matrisi rapora bağlı değil (st.session_state['model'], 'X_report_sample', 'X_cols').")
        return

    explainer   = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X)
    vals = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": cols, "importance": vals}).sort_values("importance", ascending=False).head(20)
    st.bar_chart(imp.set_index("feature"))
    st.caption("Not: Çubuk boyu özelliğin ortalama katkı büyüklüğünü gösterir (|SHAP|).")

def _patrol_summary():
    st.subheader("Devriye Özeti", anchor=False)
    patrol = st.session_state.get("patrol")
    if not patrol or not patrol.get("zones"):
        st.info("Henüz devriye planı oluşturulmadı.")
        return
    rows = [{
        "bölge": z["id"],
        "planlanan hücre": z["planned_cells"],
        "kapasite (hücre)": z["capacity_cells"],
        "ETA (dk)": z["eta_minutes"],
        "kullanım (%)": z["utilization_pct"],
        "ortalama risk E[olay]": round(z["expected_risk"], 2),
    } for z in patrol["zones"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

# =============== Ana Görünüm ===============

def render_reports(events_df: pd.DataFrame | None,
                   agg_current: pd.DataFrame | None = None,
                   agg_long_term: pd.DataFrame | None = None) -> None:
    """
    Rapor sayfasını çizer.
    - events_df: olay verisi (ts/timestamp, type, [geoid], lat/lon opsiyonel)
    - agg_current: seçili ufuk için λ tablosu (KEY_COL, expected)
    - agg_long_term: uzun dönem λ (örn. 30g referans) varsa trend analizi yapılır.
    """
    st.header("Suç Tahmin Raporu")

    # 1) Girdi kontrol & normalize (🔹 utils.reports.normalize_events_ts)
    ev = normalize_events_ts(events_df, key_col=KEY_COL) if isinstance(events_df, pd.DataFrame) else pd.DataFrame()
    if ev.empty:
        st.warning("Olay veri seti yok veya zaman sütunu parse edilemedi.")
        if isinstance(events_df, pd.DataFrame):
            st.caption("Mevcut kolonlar:")
            st.code(", ".join(map(str, events_df.columns)))
            st.caption("İlk 5 satır (ham):")
            st.dataframe(events_df.head(5), use_container_width=True)
        return

    # 2) Filtreler: Bölge, tarih, kategori
    with st.expander("Filtreler", expanded=True):
        ev = _geoid_filter_ui(ev)
        ev = _date_range_ui(ev)
        ev = _category_filter_ui(ev)
        st.caption(f"Filtre sonrası kayıt: **{len(ev):,}**")

    # 3) Yönetici Özeti (KPI)
    _kpi_tiles(ev, agg_current)

    # 4) Zamansal dağılım grafikleri
    _histograms(ev)

    # 5) Hava durumu etkisi (yağış)
    _rain_effect(ev)

    # 6) Trend: kısa vs uzun dönem (varsa)
    _hot_trend_table(agg_current, agg_long_term)

    # 7) SHAP (opsiyonel)
    _shap_global_section()

    # 8) Devriye özeti (varsa)
    _patrol_summary()

    # 9) Dışa aktar
    st.subheader("Dışa Aktar", anchor=False)
    st.download_button(
        "Filtreli olay verisini indir (CSV)",
        data=ev.to_csv(index=False).encode("utf-8"),
        file_name="rapor_filtre.csv",
        mime="text/csv",
    )
