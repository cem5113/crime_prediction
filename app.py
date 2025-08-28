# app.py
import io, json, zipfile, requests, datetime as dt
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

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
    # olası yanlışlık: 6075980501 -> başa 0 ekle
    return s.zfill(11)

@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
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
    resp = s.get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100", timeout=60)
    resp.raise_for_status()
    arts = [a for a in resp.json().get("artifacts", []) if a["name"] == ARTIFACT_NAME and not a["expired"]]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = s.get(art["archive_download_url"], timeout=120)
    z.raise_for_status()
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
    # suffix ile ara
    suffix = inner_path.split("/")[-1]
    candidates = [n for n in names if n.endswith("/"+suffix) or n.endswith(suffix)]
    # öncelik: crime_data/ ile başlayanlar, sonra en kısa path
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
    gdf = load_geojson_gdf()              # <-- burada içerden yükle
    c = gdf.copy()
    c["centroid"] = c.geometry.centroid
    return pd.DataFrame({
        "GEOID": c["GEOID"].astype(str),
        "lat": c["centroid"].y,
        "lon": c["centroid"].x,
    })

# ----------------- UI -----------------
st.set_page_config(page_title="SF Crime Dashboard", layout="wide")
st.title("SF Crime Dashboard")

colA, colB, colC = st.columns([1.2,1,1])
with colA:
    st.caption("Veri Kaynağı")
    mode = "Artifact" if USE_ARTIFACT else "Raw/Commit"
    st.write(f"• Okuma modu: **{mode}**  \n• Repo: `{REPO}`  \n• Branch: `{BRANCH}`")
with colB:
    top_k = st.number_input("Top-K (liste/harita)", 10, 500, 50, 10)
with colC:
    # risk tablosundaki en taze tarihe defaultla
    _tmp = None
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
# GEOID normalize
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
cent = centroids_from_geojson()  # argümansız çağrı
view = f.merge(cent, on="GEOID", how="left").dropna(subset=["lat","lon"])

# Renk/size
level_colors = {
    "critical": [220, 20, 60],
    "high":     [255, 140, 0],
    "medium":   [255, 215, 0],
    "low":      [34, 139, 34],
}
view["color"] = view["risk_level"].map(level_colors).fillna([100,100,100])
view["radius"] = (view["risk_score"].clip(0,1) * 40 + 10).astype(int)

st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
st.dataframe(view[["GEOID","risk_score","risk_level","risk_decile"]].reset_index(drop=True))

# Harita
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

with st.expander("🧰 Artifact içeriğini göster (teşhis)"):
    if USE_ARTIFACT:
        names = list_artifact_paths()
        if names:
            st.write(f"Toplam {len(names)} dosya:")
            st.write(names[:300])  # ilk 300 ismi göster
        else:
            st.info("Artifact listelenemedi veya boş.")

st.caption("Kaynak: GitHub Actions artifact/commit içindeki `crime_data/` çıktıları")
