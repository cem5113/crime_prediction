# app.py
from __future__ import annotations
import io, json, zipfile, requests, datetime as dt, os, time
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

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

def _to_bool(x, default=True):
    if isinstance(x, bool): return x
    if x is None: return default
    return str(x).strip().lower() in ("1","true","yes","on")
USE_ARTIFACT  = _to_bool(st.secrets.get("USE_ARTIFACT", True), True)

# Pipeline dosya yolları (hedeflenen mantıksal yollar; zip içinde farklı yerde olabilir)
PATH_RISK       = "crime_data/risk_hourly.csv"
CANDIDATE_RECS  = ["crime_data/patrol_recs_multi.csv", "crime_data/patrol_recs.csv"]
PATH_METRICS    = "crime_data/metrics_stacking.csv"
PATH_GEOJSON    = "crime_data/sf_census_blocks_with_population.geojson"

# ----------------- Yardımcılar -----------------
def _norm_geoid(s):
    """GEOID'i stringe çevirip 11 haneye tamamla."""
    if pd.isna(s): return None
    s = str(s).strip()
    return s.zfill(11)

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
    """
    Zip içinde beklenen yol yoksa, suffix eşleştirerek gerçek yolu bul.
    Örn: 'crime_data/risk_hourly.csv' yerine 'risk_hourly.csv' varsa onu kullan.
    """
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
    """Cache’lenmiş ZipFile içinden dosya oku (ZipFile'ı KAPATMA!)."""
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
    # raw fallback
    return pd.read_csv(io.BytesIO(read_raw(inner_path)))

@st.cache_resource(ttl=900)
def load_geojson_gdf() -> gpd.GeoDataFrame:
    """Büyük geometri: resource cache. (parametre alma!)"""
    try:
        if USE_ARTIFACT:
            gj = json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
        else:
            gj = json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))
    except Exception as e:
        st.error(f"GeoJSON okunamadı: {e}")
        raise
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    if "GEOID" in gdf.columns:
        gdf["GEOID"] = gdf["GEOID"].map(_norm_geoid)
    return gdf

@st.cache_data(ttl=900)
def load_geojson_dict() -> dict:
    """Pydeck GeoJsonLayer için dict halinde ver."""
    if USE_ARTIFACT:
        return json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
    return json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))

@st.cache_data(ttl=900)
def centroids_from_geojson() -> pd.DataFrame:
    """Parametresiz cache: unhashable param hatasını önler."""
    gdf = load_geojson_gdf()
    # Poligon içi garanti nokta için representative_point (centroid yerine)
    try:
        gg = gdf.to_crs(3857)
        pts = gg.representative_point().to_crs(4326)
        return pd.DataFrame({
            "GEOID": gdf["GEOID"].astype(str),
            "lat": pts.y.values,
            "lon": pts.x.values,
        })
    except Exception:
        c = gdf.copy()
        c["centroid"] = c.geometry.centroid
        return pd.DataFrame({
            "GEOID": c["GEOID"].astype(str),
            "lat": c["centroid"].y,
            "lon": c["centroid"].x,
        })

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
    if f.empty:
        st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim dener misin?")
        st.stop()

    f = f.sort_values("risk_score", ascending=False).head(int(top_k))

    # GEOID → centroid
    cent = centroids_from_geojson()
    view = f.merge(cent, on="GEOID", how="left").dropna(subset=["lat","lon"])

    # -------- Renk/size (TAB İÇİNDE) --------
    level_colors = {
        "critical": [220, 20, 60],
        "high":     [255, 140, 0],
        "medium":   [255, 215, 0],
        "low":      [34, 139, 34],
    }

    # risk_level varsa: kategorik renk, yoksa score→gradyan
    if "risk_level" in view.columns and view["risk_level"].notna().any():
        rl = view["risk_level"].astype(str).str.strip().str.lower()
        colors = rl.map(level_colors)
        default_color = [100, 100, 100]
        colors = colors.apply(lambda c: c if isinstance(c, (list, tuple)) else default_color)
    else:
        vals = (view["risk_score"].fillna(0).clip(0, 1) * 255).astype(int)
        colors = vals.apply(lambda v: [v, 0, 255 - v])

    view["color"] = colors
    view["radius"] = (view["risk_score"].fillna(0).clip(0, 1) * 40 + 10).astype(int)

    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("Seçilen kayıt", len(view))
    with mcol2:
        st.metric("Ortalama risk", round(float(view["risk_score"].mean()), 3))
    st.dataframe(view[["GEOID","risk_score","risk_level","risk_decile"]].reset_index(drop=True))

    # Harita (pydeck)
    geojson_dict = load_geojson_dict()
    initial = pdk.ViewState(
        latitude=float(view["lat"].mean()) if not view.empty else 37.7749,
        longitude=float(view["lon"].mean()) if not view.empty else -122.4194,
        zoom=11, pitch=30
    )
    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=view,
        get_position=["lon","lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True
    )
    layer_poly = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_dict,
        stroked=False, filled=False,
        get_line_color=[150,150,150],
        line_width_min_pixels=1
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer_poly, layer_points],
        initial_view_state=initial,
        tooltip={"text":"GEOID: {GEOID}\nRisk: {risk_score:.3f} ({risk_level})"}
    ))

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
        # Kontroller
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

        # Engine
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
            # draw_map imzasına uygun çağrı
            mp = draw_map(
                df_top,
                cent_src,
                popup_cols=["GEOID","hour_range","p_crime","priority_score","top3_crime_types"],
                height=560
            )
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
            # paths.RISK_DIR olmayabilir; fallback direkt buffer
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
