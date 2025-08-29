# app.py
from __future__ import annotations
import io, json, zipfile, requests, datetime as dt, os, time, re
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# ----------------- Yardımcılar (REVİZE) -----------------
def normalize_hour_range(x: str) -> str:
    """'00-03', '0-3', '00–03' gibi etiketleri 2 haneli 'HH-HH' biçimine normalize eder."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().replace("–", "-").replace("—", "-")  # en/emdash → '-'
    m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", s)
    if not m:
        nums = re.findall(r"\d{1,2}", s)
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            return f"{a:02d}-{b:02d}"
        return s
    a, b = int(m.group(1)), int(m.group(2))
    return f"{a:02d}-{b:02d}"

def _norm_geoid(s):
    """GEOID'i stringe çevirip 11 haneye tamamla."""
    if pd.isna(s): return None
    s = str(s).strip()
    return s.zfill(11)

def _to_bool(x, default=True):
    if isinstance(x, bool): return x
    if x is None: return default
    return str(x).strip().lower() in ("1","true","yes","on")

def _retry_get(url: str, headers: dict | None = None, timeout: int = 60, retries: int = 2) -> requests.Response:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise last_err

# ----------- (opsiyonel) src modülleri: Operasyonel sekme için ----------
HAS_SRC = True
try:
    from src.config import params, paths
    from src.common import to_hour_range
    from src.inference_engine import InferenceEngine
    from src.features import load_centroids as load_centroids_src
    from src.viz import draw_map
except Exception:
    HAS_SRC = False

# ----------------- Ayarlar / Secrets -----------------
REPO          = st.secrets.get("REPO", "cem5113/crime_prediction_data")
BRANCH        = st.secrets.get("BRANCH", "main")
ARTIFACT_NAME = st.secrets.get("ARTIFACT_NAME", "sf-crime-pipeline-output")
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", None)
USE_ARTIFACT  = _to_bool(st.secrets.get("USE_ARTIFACT", True), True)

# Pipeline dosya yolları
PATH_RISK       = "crime_data/risk_hourly.csv"
CANDIDATE_RECS  = ["crime_data/patrol_recs_multi.csv", "crime_data/patrol_recs.csv"]
PATH_METRICS    = "crime_data/metrics_stacking.csv"
PATH_GEOJSON    = "crime_data/sf_census_blocks_with_population.geojson"

# Tarihsel grid (sayım bazlı μ için)
PATH_GRID = "crime_data/sf_crime_grid_full_labeled.csv"

CANDIDATE_GRIDS = [
    "crime_data/sf_crime_grid_full_labeled.csv",
    "crime_data/sf_crime_grid_labeled.csv",
    "crime_data/sf_crime_grid.csv",
    "crime_data/crime_grid.csv",
]

def resolve_grid_path() -> str | None:
    try:
        names = list_artifact_paths()  # artifact içeriğini listeler
        if names:
            # 1) birebir eşleşme
            if PATH_GRID in names:
                return PATH_GRID
            # 2) suffix ile
            suf = PATH_GRID.split("/")[-1].lower()
            for n in names:
                if n.lower().endswith(suf):
                    return n
            # 3) aday listeden birini bul
            for cand in CANDIDATE_GRIDS:
                if cand in names:
                    return cand
            # 4) fuzzy: 'grid' ve '.csv' içeren ilk dosya
            for n in names:
                nl = n.lower()
                if nl.endswith(".csv") and "grid" in nl and "crime" in nl:
                    return n
    except Exception:
        pass
    # artifact boşsa raw'a bırak
    return PATH_GRID

# --- Sayım (Poisson) yardımcıları ---
SEASON_MONTHS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall":   (9, 10, 11),
}

def season_of_date(d: dt.date) -> str:
    m = d.month
    if m in SEASON_MONTHS["Winter"]: return "Winter"
    if m in SEASON_MONTHS["Spring"]: return "Spring"
    if m in SEASON_MONTHS["Summer"]: return "Summer"
    return "Fall"

def hours_in_block(hour_range_label: str) -> list[int]:
    a, b = [int(x) for x in hour_range_label.split("-")]
    if b <= a: b += 24
    return [h % 24 for h in range(a, b)]  # örn: '00-03' -> [0,1,2]

def hist_days_for(season: str, dow: int,
                  start=dt.date(2020, 8, 29),
                  end=dt.date(2025, 8, 26)) -> int:
    """Veri ufkumuzda (yaklaşık 5 yıl) aynı sezon+dow kaç gün var?"""
    rng = pd.date_range(start, end, freq="D")
    mask = (rng.dayofweek == dow) & (rng.month.isin(SEASON_MONTHS[season]))
    return int(mask.sum())

def poisson_table(mu: pd.Series, k_cap: int = 5) -> pd.DataFrame:
    """μ -> P(0..k_cap-1), P(5+), E[N]"""
    import numpy as np, math
    ks = np.arange(0, k_cap)
    fact = np.array([math.factorial(k) for k in ks], dtype=float)
    mu_vec = mu.to_numpy().reshape(-1, 1)                # (N,1)
    pmf = np.exp(-mu_vec) * (mu_vec ** ks) / fact        # (N,k_cap)
    p5plus = 1.0 - pmf.sum(axis=1, keepdims=True)        # (N,1)
    pmf_all = np.hstack([pmf, p5plus])                   # (N,k_cap+1)
    cols = [f"P({k})" for k in range(k_cap)] + ["P(5+)"]
    out = pd.DataFrame(pmf_all, index=mu.index, columns=cols)
    out["E[N]"] = (pmf * ks).sum(axis=1) + 5 * p5plus[:, 0]
    return out

# ----------------- IO / Cache -----------------
@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    r = _retry_get(url, timeout=60)
    return r.content

@st.cache_resource(ttl=900)
def fetch_artifact_zip() -> zipfile.ZipFile:
    """Artifact ZIP'i indirir ve açık ZipFile nesnesini döndürür (cache_resource!)."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN yok; artifact indirilemez.")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    })
    resp = _retry_get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100",
                      headers=s.headers, timeout=60)
    arts = [a for a in resp.json().get("artifacts", []) if a["name"] == ARTIFACT_NAME and not a["expired"]]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = _retry_get(art["archive_download_url"], headers=s.headers, timeout=120)
    return zipfile.ZipFile(io.BytesIO(z.content))

@st.cache_data(ttl=900)
def list_artifact_paths() -> list[str]:
    try:
        zf = fetch_artifact_zip()
        return zf.namelist()
    except Exception:
        return []

def _resolve_inner_path(inner_path: str) -> str:
    names = list_artifact_paths()
    if not names:
        return inner_path
    if inner_path in names:
        return inner_path
    suffix = inner_path.split("/")[-1]
    candidates = [n for n in names if n.endswith("/"+suffix) or n.endswith(suffix)]
    candidates.sort(key=lambda x: (0 if x.startswith("crime_data/") else 1, len(x)))
    return candidates[0] if candidates else inner_path

def _read_from_artifact(inner_path: str) -> bytes:
    zf = fetch_artifact_zip()
    real = _resolve_inner_path(inner_path)
    return zf.read(real)

@st.cache_data(ttl=900)
def load_csv(inner_path: str) -> pd.DataFrame:
    if USE_ARTIFACT:
        try:
            data = _read_from_artifact(inner_path)
            return pd.read_csv(io.BytesIO(data))
        except Exception as e:
            st.warning(f"Artifact okunamadı ({e}). raw moda geçiliyor…")
    return pd.read_csv(io.BytesIO(read_raw(inner_path)))

@st.cache_resource(ttl=900)
def load_geojson_gdf() -> gpd.GeoDataFrame:
    """GeoJSON'u oku ve GEOID sütununu otomatik tespit ederek normalize et."""
    try:
        if USE_ARTIFACT:
            gj = json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
        else:
            gj = json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))
    except Exception as e:
        st.error(f"GeoJSON okunamadı: {e}")
        raise

    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

    # GEOID sütununu otomatik tespit et
    geoid_candidates = ["GEOID", "geoid", "GEOID10", "geoid10", "GEOID20", "geoid20"]
    found = next((c for c in geoid_candidates if c in gdf.columns), None)
    if found is None:
        st.error("GeoJSON içinde GEOID benzeri bir sütun bulunamadı (GEOID/geoid/GEOID10 …).")
    else:
        gdf["GEOID"] = gdf[found].map(_norm_geoid)

    return gdf

@st.cache_data(ttl=900)
def load_geojson_dict() -> dict:
    """Pydeck GeoJsonLayer için dict halinde ver."""
    if USE_ARTIFACT:
        return json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
    return json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))

@st.cache_data(ttl=900)
def centroids_from_geojson() -> pd.DataFrame:
    """Poligon içi garanti nokta (representative_point) ile centroid DF'i üretir.
       Hem block GEOID (tam) hem de tract için GEOID11 kolonunu döndürür."""
    gdf = load_geojson_gdf().copy()

    # GEOID string & GEOID11 (tract) türet
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.strip()
    gdf["GEOID11"] = gdf["GEOID"].str[:11]

    # Representative point -> lat/lon
    try:
        gg = gdf.to_crs(3857)
        pts = gg.representative_point().to_crs(4326)
        gdf["_lat"] = pts.y.values
        gdf["_lon"] = pts.x.values
    except Exception:
        c = gdf.copy()
        c["centroid"] = c.geometry.centroid
        gdf["_lat"] = c["centroid"].y
        gdf["_lon"] = c["centroid"].x

    # Tract düzeyinde tek nokta: block noktalarını tract bazında medyanla topla
    cent11 = (
        gdf.groupby("GEOID11", as_index=False)[["_lat","_lon"]]
           .median()
           .rename(columns={"_lat":"lat","_lon":"lon"})
    )

    # İstersen block anahtarıyla da kullanabilelim (opsiyonel)
    cent_full = gdf[["GEOID","_lat","_lon"]].rename(columns={"_lat":"lat","_lon":"lon"}).drop_duplicates()

    # Her iki anahtarı da döndür (merge'te hangisi lazımsa kullanacağız)
    # Not: aynı kolon isimleri çakışmasın diye sufffix’li döndürebiliriz ama
    # dashboard tarafında seçim yapacağımız için sade tutuyoruz.
    cent = cent_full.merge(cent11, left_on="GEOID", right_on="GEOID11", how="outer", suffixes=("","_agg"))
    # Sonuç kolonları: ['GEOID','lat','lon','GEOID11','lat_agg','lon_agg']
    return cent

# ----------------- UI -----------------
st.set_page_config(page_title="SF Crime Dashboard", layout="wide")
st.title("SF Crime • Dashboard & Operasyonel")

tab_dash, tab_ops, tab_diag = st.tabs(["📊 Dashboard", "🛠 Operasyonel", "🔎 Teşhis"])

# ============================
# 📊 Dashboard (pipeline çıktısı)
# ============================
with tab_dash:
    colA, colB, colC = st.columns([1.2,1,1])
    with colA:
        st.caption("Veri Kaynağı")
        mode = "Artifact" if USE_ARTIFACT else "Raw/Commit"
        st.write(f"• Okuma modu: **{mode}**  \n• Repo: `{REPO}`  \n• Branch: `{BRANCH}`")
    with colB:
        top_k = st.number_input("Top-K (liste/harita)", 10, 500, 50, 10)
    with colC:
        # risk tablosundaki en taze tarihe defaultla
        try:
            _tmp = load_csv(PATH_RISK).copy()
            _tmp["date"] = pd.to_datetime(_tmp["date"], errors="coerce").dt.date
            default_date = _tmp["date"].max()
        except Exception:
            default_date = dt.date.today()
        sel_date = st.date_input("Tarih", default_date or dt.date.today())

    # Risk tablosu
    try:
        risk_df = load_csv(PATH_RISK)
    except Exception as e:
        st.error(f"`{PATH_RISK}` okunamadı: {e}")
        st.stop()

    # Beklenen kolonlar: GEOID,date,hour_range,risk_score,risk_level,risk_decile
    if "GEOID" in risk_df.columns:
        risk_df["GEOID"] = risk_df["GEOID"].map(_norm_geoid)
    risk_df["date"] = pd.to_datetime(risk_df["date"], errors="coerce").dt.date

    # 🔧 hour_range normalize
    risk_df["hour_range"] = risk_df["hour_range"].astype(str).map(normalize_hour_range)

    def _hour_key(h):
        try: return int(str(h).split("-")[0])
        except: return 0

    hours = sorted(risk_df["hour_range"].dropna().unique().tolist(), key=_hour_key)
    hour = st.select_slider("Saat aralığı", options=hours, value=hours[0] if hours else None)

    if hour is None:
        st.warning("Saat aralığı bulunamadı.")
        st.stop()

    # Filtre + Top-K
    f = risk_df[(risk_df["date"] == sel_date) & (risk_df["hour_range"] == hour)].copy()

    # 🧪 Hızlı teşhis (UI)
    with st.expander("🧪 Filtre Teşhisi", expanded=False):
        st.write(f"Toplam risk_df satırı: {len(risk_df)}")
        st.write(f"Seçilen tarih: `{sel_date}`, seçilen saat: `{hour}`")
        st.write(f"Eşleşen satır (merge öncesi): **{len(f)}**")
        if len(f) == 0:
            st.write("➡ risk_hourly.csv içinde bu tarih+saat için kayıt olmayabilir.")
            st.write("Mevcut saat aralıkları örnek:", pd.Series(hours[:10]))

    if f.empty:
        st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim dener misin?")
        st.stop()

    def _grid_or_stop():
        try:
            g = load_csv(PATH_GRID)
            return g
        except Exception as e:
            st.error(
                f"Sayım grid dosyası bulunamadı: `{PATH_GRID}`\n\n"
                "➊ Artifact içinde bu adda bir dosya yok.\n"
                "➋ Repo/raw yolunda da yok.\n\n"
                "Poisson (count) görünümü bu yüzden pas geçildi."
            )
            # İstersen burada risk_score görünümüne düşebilirsin;
            # şu anlık nazikçe durduruyoruz:
            st.stop()
    
    # --- μ (beklenen olay) hesapla: tarihsel grid'den sezon+dow+3saat total / gün sayısı
    grid = _grid_or_stop()

    # --- μ (beklenen olay) hesapla: tarihsel grid'den sezon+dow+3saat total / gün sayısı
    grid = load_csv(PATH_GRID)
    grid["GEOID"] = grid["GEOID"].map(_norm_geoid)
    
    sel_season = season_of_date(sel_date)
    sel_dow = sel_date.weekday()
    hrs = hours_in_block(hour)  # örn '20-23' -> [20,21,22]
    
    agg = (grid[(grid["season"] == sel_season) &
                (grid["day_of_week"] == sel_dow) &
                (grid["event_hour"].isin(hrs))]
           .groupby("GEOID", as_index=False)["crime_count"].sum()
           .rename(columns={"crime_count": "hist_count"}))
    
    den = max(hist_days_for(sel_season, sel_dow), 1)  # aynı sezon+DOW kaç gün var
    agg["mu"] = agg["hist_count"] / den
    
    # Aday GEOID'lere μ join et (risk_score'u kullanmıyoruz)
    view = f.merge(agg[["GEOID", "mu"]], on="GEOID", how="left")
    view["mu"] = view["mu"].fillna(0.0)
    
    # Olasılık tablosu: P(0..4), P(5+), E[N]
    dist = poisson_table(view.set_index("GEOID")["mu"], k_cap=5).reset_index()
    view = view.drop(columns=[c for c in view.columns if c not in ["GEOID"]]).merge(dist, on="GEOID", how="left")
    
    # Rank & Top-K: E[N] büyükten küçüğe
    view = view.sort_values("E[N]", ascending=False).head(int(top_k))
    
    # Koordinatlar
    cent = centroids_from_geojson()
    view = view.merge(cent[["GEOID","lat","lon"]], on="GEOID", how="left").dropna(subset=["lat","lon"])
    
    # Görsel seviye/renk: E[N] decile
    try:
        view["risk_decile"] = pd.qcut(view["E[N]"].rank(method="first"), 10, labels=list(range(1,11))).astype(int)
    except Exception:
        view["risk_decile"] = 1  # tekdüze değerlerde fallback
    def _level(d): return "critical" if d>=9 else ("high" if d>=7 else ("medium" if d>=4 else "low"))
    view["risk_level"] = view["risk_decile"].map(_level)
    
    level_colors = {
        "critical": [220, 20, 60],
        "high":     [255, 140, 0],
        "medium":   [255, 215, 0],
        "low":      [34, 139, 34],
    }
    view["color"] = view["risk_level"].map(lambda k: level_colors.get(k, [100, 100, 100]))
    # Balon yarıçapını E[N] ile ölçekle (0–1.5 aralığını genişlet)
    view["radius"] = (view["E[N]"].clip(0, 1.5) / 1.5 * 40 + 10).astype(int)
    
    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Seçilen kayıt", len(view))
    with mcol2:
        st.metric("Ortalama beklenen olay (E[N])", round(float(view["E[N]"].mean()), 3))
    
    # Yüzdeleri tabloya yüzde olarak yazdır
    cols_show = ["GEOID","E[N]","P(0)","P(1)","P(2)","P(3)","P(4)","P(5+)","risk_level","risk_decile"]
    st.dataframe(
        view[cols_show].reset_index(drop=True)
          .style.format({c: "{:.1%}" for c in ["P(0)","P(1)","P(2)","P(3)","P(4)","P(5+)"]})
    )
    
    # --- Harita (pydeck) — E[N] ile tooltip
    geojson_dict = load_geojson_dict()
    
    view_points = view.loc[:, ["lon", "lat", "GEOID", "E[N]", "risk_level", "color", "radius"]].copy()
    view_points["GEOID"] = view_points["GEOID"].astype(str)
    view_points["E[N]"] = view_points["E[N]"].astype(float)
    view_points["risk_level"] = view_points["risk_level"].astype(str)
    view_points["radius"] = view_points["radius"].astype(int)
    
    initial = pdk.ViewState(
        latitude=float(view_points["lat"].mean()) if not view_points.empty else 37.7749,
        longitude=float(view_points["lon"].mean()) if not view_points.empty else -122.4194,
        zoom=11, pitch=30
    )
    
    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=view_points,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
    )
    
    layer_poly = pdk.Layer(
        "GeoJsonLayer", data=geojson_dict,
        stroked=False, filled=False,
        get_line_color=[150, 150, 150], line_width_min_pixels=1,
    )
    
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer_poly, layer_points],
            initial_view_state=initial,
            tooltip={"text": "GEOID: {GEOID}\nE[N]: {E[N]}\nSeviye: {risk_level}"},
        )
    )


    st.divider()
    with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
        rec_loaded = False
        last_err = None
        for path in CANDIDATE_RECS:
            try:
                recs = load_csv(path)
                if "GEOID" in recs.columns:
                    recs["GEOID"] = recs["GEOID"].map(_norm_geoid)
                recs["date"] = pd.to_datetime(recs["date"], errors="coerce").dt.date
                fr = recs[(recs["date"] == sel_date) & (recs["hour_range"] == hour)].copy()
                st.dataframe(fr.head(200))
                rec_loaded = True
                break
            except Exception as e:
                last_err = e
        if not rec_loaded:
            st.info(f"Devriye önerileri okunamadı: {last_err}")

    with st.expander("📈 Model Metrikleri"):
        try:
            m = load_csv(PATH_METRICS)
            st.dataframe(m)
        except Exception as e:
            st.info(f"Metrikler yüklenemedi: {e}")

# ============================
# 🛠 Operasyonel (src/* kullanır)
# ============================
with tab_ops:
    st.subheader("Operasyonel Risk Paneli")
    if not HAS_SRC:
        st.info("`src/` modülleri bulunamadı. Bu sekme için repo içindeki `src/` klasörünü deploy ettiğinden emin ol.")
    else:
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            start_h = st.slider("Başlangıç saat", 0, 23, 20)
        with col2:
            width_h = st.selectbox("Pencere (saat)", [1,2,3,4], index=1)
        with col3:
            topk_default = getattr(params, "TOP_K", 10)
            topk_ops = st.number_input("Top-K", min_value=5, max_value=100, value=topk_default, step=5)

        hour_label = to_hour_range(start_h, width_h)
        st.markdown(f"**Öneri Dilimi:** `{hour_label}`")

        with st.spinner("Tahminler üretiliyor..."):
            engine = InferenceEngine()
            df_top = engine.predict_topk(hour_label=hour_label, topk=int(topk_ops))

        st.subheader("🎯 En Öncelikli Bölgeler")
        cols_show = [c for c in ["rank","hour_range","GEOID","priority_score","p_crime","lcb","ucb","top3_crime_types"] if c in df_top.columns]
        st.dataframe(df_top[cols_show])

        # Harita (Folium)
        try:
            cent_src = load_centroids_src()
        except Exception:
            cent_src = None
        if cent_src is None or cent_src.empty:
            cent_src = centroids_from_geojson()  # geojson'dan fallback

        try:
            mp = draw_map(df_top, cent_src, use_cluster=True, show_heatmap=True, map_height=560, zoom_start=12)
            if mp is not None:
                from streamlit_folium import st_folium
                st_folium(mp, height=560, width=None)
            else:
                st.info("Centroid/geo veri bulunamadı; yalnızca tablo gösterildi.")
        except Exception as e:
            st.info(f"Harita çizilemedi: {e}")

        # İndirme
        try:
            out_csv = engine.save_topk(df_top)
            st.download_button("CSV indir", data=open(out_csv,"rb").read(),
                               file_name=os.path.basename(out_csv), mime="text/csv")
        except Exception:
            st.download_button(
                "CSV indir",
                data=df_top.to_csv(index=False).encode("utf-8"),
                file_name="topk_geoid.csv",
                mime="text/csv"
            )

# ============================
# 🔎 Teşhis
# ============================
with tab_diag:
    st.subheader("Artifact / Geo Teşhis")
    st.write(f"• Okuma modu: **{'Artifact' if USE_ARTIFACT else 'Raw/Commit'}**")
    if USE_ARTIFACT:
        names = list_artifact_paths()
        if names:
            st.write(f"Toplam {len(names)} dosya:")
            st.write(names[:300])
        else:
            st.info("Artifact listelenemedi veya boş.")
    try:
        gdf = load_geojson_gdf()
        st.write(f"GeoJSON OK — {len(gdf)} geometri")
        st.dataframe(gdf.head())
    except Exception as e:
        st.info(f"GeoJSON teşhis: {e}")

st.caption("Kaynak: GitHub Actions artifact/commit içindeki `crime_data/` çıktıları • Operasyonel sekme: src/* modülleri")
