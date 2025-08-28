# app.py
from __future__ import annotations
import io, json, zipfile, requests, datetime as dt, os, time, re
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

# Pipeline dosya yolları
PATH_RISK       = "crime_data/risk_hourly.csv"
CANDIDATE_RECS  = ["crime_data/patrol_recs_multi.csv", "crime_data/patrol_recs.csv"]
PATH_METRICS    = "crime_data/metrics_stacking.csv"
PATH_GEOJSON    = "crime_data/sf_census_blocks_with_population.geojson"

# ----------------- Yardımcılar -----------------
def digits_only(x) -> str:
    if pd.isna(x): return ""
    return re.sub(r"\D", "", str(x))

def to_tract11(x) -> str:
    d = digits_only(x)
    return (d[:11] if len(d) >= 11 else d.zfill(11))

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
    try:
        if USE_ARTIFACT:
            gj = json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
        else:
            gj = json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))
    except Exception as e:
        st.error(f"GeoJSON okunamadı: {e}")
        raise
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    # Orijinal GEOID’i koru; tract11 ayrıca üretilecek
    return gdf

@st.cache_data(ttl=900)
def load_geojson_dict() -> dict:
    if USE_ARTIFACT:
        return json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
    return json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))

@st.cache_data(ttl=900)
def centroids_tract11_from_geojson() -> pd.DataFrame:
    """
    GeoJSON (block group) -> representative point -> TRACT11'e agregasyon (tek satır / tract).
    """
    gdf = load_geojson_gdf()
    # Representative point (poligon içi garanti nokta), WGS84'e geri dön
    try:
        pts = gdf.to_crs(3857).representative_point().to_crs(4326)
        lat = pts.y.values
        lon = pts.x.values
    except Exception:
        c = gdf.geometry.centroid
        lat, lon = c.y.values, c.x.values

    geo_raw = gdf["GEOID"] if "GEOID" in gdf.columns else pd.Series([""]*len(gdf))
    tract11 = geo_raw.astype(str).apply(to_tract11)

    cent = pd.DataFrame({"TRACT11": tract11, "lat": lat, "lon": lon})
    # Bir tract’ta birden fazla BG olabilir → tekille
    cent = cent.groupby("TRACT11", as_index=False).agg({"lat":"mean", "lon":"mean"})
    cent = cent.rename(columns={"TRACT11":"GEOID"})
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

    # GEOID → TRACT11 hizalama
    risk_df["GEOID"] = risk_df.get("GEOID", pd.Series([None]*len(risk_df)))
    risk_df["GEOID"] = risk_df["GEOID"].apply(to_tract11)
    risk_df["date"] = pd.to_datetime(risk_df["date"], errors="coerce").dt.date

    def _hour_key(h):
        try: return int(str(h).split("-")[0].split(":")[0])
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

    # GEOID → centroid (TRACT11)
    cent = centroids_tract11_from_geojson()
    view = f.merge(cent, on="GEOID", how="left")
    matched = int(view["lat"].notna().sum())
    view = view.dropna(subset=["lat","lon"]).reset_index(drop=True)

    # Renk/size
    level_colors = {
        "critical": [220, 20, 60],
        "high":     [255, 140, 0],
        "medium":   [255, 215, 0],
        "low":      [34, 139, 34],
    }

    if "risk_level" in view.columns and view["risk_level"].notna().any():
        rl = view["risk_level"].astype(str).str.strip().str.lower()
        colors = rl.map(level_colors)
        default_color = [100, 100, 100]
        colors = colors.apply(lambda c: list(map(int, c)) if isinstance(c, (list, tuple)) else default_color)
    else:
        vals = (view["risk_score"].fillna(0).clip(0, 1) * 255).round().astype(int)
        colors = vals.apply(lambda v: [int(v), 0, int(255 - v)])

    view["color"] = colors
    view["radius"] = (view["risk_score"].fillna(0).clip(0, 1) * 40 + 10).round().astype(int)

    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Seçilen kayıt", len(f))
    with mcol2:
        st.metric("Eşleşen centroid", matched)
    with mcol3:
        st.metric("Ortalama risk", round(float(view["risk_score"].mean()) if len(view) else 0.0, 3))

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
        tooltip={"html":"<b>GEOID</b>: {GEOID}<br/><b>Risk</b>: {risk_score} ({risk_level})"}
    ))

    st.divider()
    with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
        rec_loaded = False
        last_err = None
        for path in CANDIDATE_RECS:
            try:
                recs = load_csv(path)
                recs["GEOID"] = recs.get("GEOID", pd.Series([None]*len(recs))).apply(to_tract11)
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
            # GeoJSON’dan tract-11 centroidleri (fallback)
            cent_src = centroids_tract11_from_geojson()

        try:
            mp = draw_map(df_top, cent_src, popup_cols=None)  # src/viz.py arayüzü
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
        # Teşhis: örnek GEOID dönüşümleri
        demo = pd.DataFrame({
            "geo_GEOID_raw": gdf["GEOID"].astype(str).head(5),
            "geo_TRACT11": gdf["GEOID"].astype(str).apply(to_tract11).head(5),
        })
        st.dataframe(demo)
    except Exception as e:
        st.info(f"GeoJSON teşhis: {e}")

st.caption("Kaynak: GitHub Actions artifact/commit içindeki `crime_data/` çıktıları • Operasyonel sekme: src/* modülleri")
