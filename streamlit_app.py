# streamlit_app.py
import io, json, zipfile, requests, datetime as dt
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# --- Ayarlar ---
# Örn: cem5113/crime_prediction_data
REPO = st.secrets.get("REPO", "cem5113/crime_prediction_data")
BRANCH = st.secrets.get("BRANCH", "main")
ARTIFACT_NAME = st.secrets.get("ARTIFACT_NAME", "sf-crime-pipeline-output")
USE_ARTIFACT = st.secrets.get("USE_ARTIFACT", True)  # commit modunda False yap
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

# Pipeline'ın bıraktığı dosya yolları (artifact zip içinde de aynı)
PATH_RISK = "crime_data/risk_hourly.csv"
PATH_RECS = "crime_data/patrol_recs.csv"            # veya patrol_recs_multi.csv
PATH_METRICS = "crime_data/metrics_stacking.csv"
PATH_GEOJSON = "crime_data/sf_census_blocks_with_population.geojson"  # poligonlar

# --- Yardımcılar ---
@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(ttl=900)
def fetch_artifact_zip() -> zipfile.ZipFile:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN yok; artifact indirilemez.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}",
                      "Accept": "application/vnd.github+json"})
    # Son artefaktları listele
    j = s.get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100", timeout=60).json()
    arts = [a for a in j.get("artifacts", []) if a["name"] == ARTIFACT_NAME and not a["expired"]]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = s.get(art["archive_download_url"], timeout=120)
    z.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(z.content))

def _read_from_artifact(zf: zipfile.ZipFile, inner_path: str) -> bytes:
    with zf.open(inner_path) as f:
        return f.read()

@st.cache_data(ttl=900)
def load_csv(inner_path: str) -> pd.DataFrame:
    """Artifact varsa oradan, yoksa raw'dan oku."""
    if USE_ARTIFACT:
        try:
            zf = fetch_artifact_zip()
            data = _read_from_artifact(zf, inner_path)
            return pd.read_csv(io.BytesIO(data))
        except Exception as e:
            st.warning(f"Artifact okunamadı ({e}). raw moda düşülüyor…")
    # raw fallback
    return pd.read_csv(io.BytesIO(read_raw(inner_path)))

@st.cache_data(ttl=900)
def load_geojson() -> gpd.GeoDataFrame:
    """GeoJSON'ı artifact'tan ya da raw'dan GeoDataFrame'e yükle."""
    if USE_ARTIFACT:
        try:
            zf = fetch_artifact_zip()
            data = _read_from_artifact(zf, PATH_GEOJSON).decode("utf-8")
            gj = json.loads(data)
            gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
            return gdf
        except Exception as e:
            st.warning(f"GeoJSON artifact'tan okunamadı ({e}). raw moda düşülüyor…")
    # raw fallback
    content = read_raw(PATH_GEOJSON).decode("utf-8")
    gj = json.loads(content)
    return gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

@st.cache_data(ttl=900)
def centroids_from_geojson(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # WGS84'te centroid (~yaklaşık); isterseniz projeksiyonlayıp hesaplayın
    c = gdf.copy()
    c["centroid"] = c.geometry.centroid
    return pd.DataFrame({
        "GEOID": c["GEOID"].astype(str),
        "lat": c["centroid"].y,
        "lon": c["centroid"].x,
    })

# --- UI ---
st.set_page_config(page_title="SF Crime Risk", layout="wide")
st.title("SF Crime Risk • (Pipeline → Streamlit)")

colA, colB, colC = st.columns([1.2,1,1])
with colA:
    st.caption("Veri Kaynağı")
    mode = "Artifact" if USE_ARTIFACT else "Raw/Commit"
    st.write(f"• Okuma modu: **{mode}**  \n• Repo: `{REPO}`  \n• Branch: `{BRANCH}`")

with colB:
    top_k = st.number_input("Top-K (liste/harita)", 10, 500, 50, 10)
with colC:
    today = dt.date.today()
    sel_date = st.date_input("Tarih", today)

risk_df = load_csv(PATH_RISK)
# Beklenen kolonlar: GEOID,date,hour_range,risk_score,risk_level,risk_decile
risk_df["date"] = pd.to_datetime(risk_df["date"]).dt.date
hours = risk_df["hour_range"].dropna().unique().tolist()
hour = st.select_slider("Saat aralığı", options=sorted(hours), value=hours[0])

# Filtre + Top-K
f = risk_df[(risk_df["date"] == sel_date) & (risk_df["hour_range"] == hour)].copy()
if f.empty:
    st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim dener misin?")
else:
    f = f.sort_values("risk_score", ascending=False).head(int(top_k))

    # GEOID → centroid
    geo = load_geojson()
    cent = centroids_from_geojson(geo)
    view = f.merge(cent, on="GEOID", how="left").dropna(subset=["lat","lon"])

    # Renk/size kodlama
    level_colors = {
        "critical": [220, 20, 60],   # kırmızı
        "high":     [255, 140, 0],   # turuncu
        "medium":   [255, 215, 0],   # sarı
        "low":      [34, 139, 34],   # yeşil
    }
    view["color"] = view["risk_level"].map(level_colors).fillna([100,100,100])
    # skor’a göre yarıçap (metre ~ görsel)
    view["radius"] = (view["risk_score"].clip(0,1) * 40 + 10).astype(int)

    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    st.dataframe(view[["GEOID","risk_score","risk_level","risk_decile"]].reset_index(drop=True))

    # Harita (pydeck)
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
    # Poligon katmanı (isteğe bağlı; ağır gelirse kapat)
    layer_poly = pdk.Layer(
        "GeoJsonLayer",
        data=json.loads(read_raw(PATH_GEOJSON)) if not USE_ARTIFACT else json.loads(_read_from_artifact(fetch_artifact_zip(), PATH_GEOJSON)),
        stroked=False, filled=False, get_line_color=[150,150,150], line_width_min_pixels=1
    )
    st.pydeck_chart(pdk.Deck(layers=[layer_poly, layer_points], initial_view_state=initial, tooltip={"text":"GEOID: {GEOID}\nRisk: {risk_score:.3f} ({risk_level})"}))

st.divider()
with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
    try:
        recs = load_csv(PATH_RECS)
        recs["date"] = pd.to_datetime(recs["date"]).dt.date
        fr = recs[(recs["date"] == sel_date) & (recs["hour_range"] == hour)].copy()
        st.dataframe(fr.head(100))
    except Exception as e:
        st.info(f"Devriye önerileri okunamadı: {e}")

with st.expander("📈 Model Metrikleri"):
    try:
        m = load_csv(PATH_METRICS)
        st.dataframe(m)
    except Exception as e:
        st.info(f"Metrikler yüklenemedi: {e}")

st.caption("Kaynak: GitHub Actions ile üretilen `crime_data/` çıktıları")
