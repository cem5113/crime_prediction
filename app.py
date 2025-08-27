# app.py
import io, json, zipfile, requests, datetime as dt
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# ------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------
st.set_page_config(page_title="SF Crime Risk", layout="wide")

# ------------------------------------------------------
# Ayarlar (secrets üzerinden override edilebilir)
# ------------------------------------------------------
REPO = st.secrets.get("REPO", "cem5113/crime_prediction_data")
BRANCH = st.secrets.get("BRANCH", "main")
ARTIFACT_NAME = st.secrets.get("ARTIFACT_NAME", "sf-crime-pipeline-output")
USE_ARTIFACT = str(st.secrets.get("USE_ARTIFACT", "true")).lower() in ("1", "true", "yes", "on")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

# Pipeline dosya yolları (artifact ZIP içinde de aynı path'ler)
PATH_RISK    = "crime_data/risk_hourly.csv"
PATH_RECS    = "crime_data/patrol_recs.csv"              # fallback için multi versiyonu da deneyeceğiz
PATH_METRICS = "crime_data/metrics_all.csv"              # pipeline çıktısına göre düzeltildi
PATH_GEOJSON = "crime_data/sf_census_blocks_with_population.geojson"

RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# ------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------
@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(ttl=900)
def fetch_artifact_zip() -> zipfile.ZipFile:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN yok; artifact indirilemez.")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    # Son (geçerli) artifact'ları listele
    j = s.get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100", timeout=60).json()
    arts = [a for a in j.get("artifacts", []) if a.get("name") == ARTIFACT_NAME and not a.get("expired")]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = s.get(art["archive_download_url"], timeout=180)
    z.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(z.content))

def _read_from_artifact(zf: zipfile.ZipFile, inner_path: str) -> bytes:
    with zf.open(inner_path) as f:
        return f.read()

@st.cache_data(ttl=900)
def load_csv(inner_path: str) -> pd.DataFrame:
    """Artifact varsa oradan, yoksa raw/commit'ten oku."""
    if USE_ARTIFACT and GITHUB_TOKEN:
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
    """GeoJSON'ı yükle (artifact öncelikli)."""
    if USE_ARTIFACT and GITHUB_TOKEN:
        try:
            zf = fetch_artifact_zip()
            data = _read_from_artifact(zf, PATH_GEOJSON).decode("utf-8")
            gj = json.loads(data)
            return gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        except Exception as e:
            st.warning(f"GeoJSON artifact'tan okunamadı ({e}). raw moda düşülüyor…")
    # raw fallback
    content = read_raw(PATH_GEOJSON).decode("utf-8")
    gj = json.loads(content)
    return gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

@st.cache_data(ttl=900)
def centroids_from_geojson(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # WGS84'te yaklaşık centroid
    c = gdf.copy()
    c["centroid"] = c.geometry.centroid
    return pd.DataFrame({
        "GEOID": c["GEOID"].astype(str),
        "lat": c["centroid"].y,
        "lon": c["centroid"].x,
    })

def load_recs_df() -> pd.DataFrame:
    """patrol_recs_multi.csv varsa onu, yoksa patrol_recs.csv'yi yükle."""
    candidates = ("crime_data/patrol_recs_multi.csv", "crime_data/patrol_recs.csv")
    last_err = None
    for p in candidates:
        try:
            return load_csv(p)
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"patrol_recs*.csv bulunamadı: {last_err}")

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("SF Crime Dashboard")

colA, colB, colC = st.columns([1.4, 1, 1])
with colA:
    mode = "Artifact" if (USE_ARTIFACT and GITHUB_TOKEN) else "Raw/Commit"
    st.caption("Veri Kaynağı")
    st.write(
        f"• Okuma modu: **{mode}**  \n"
        f"• Repo: `{REPO}`  \n"
        f"• Branch: `{BRANCH}`"
    )
with colB:
    top_k = st.number_input("Top-K (liste/harita)", min_value=10, max_value=500, value=50, step=10)
with colC:
    sel_date = st.date_input("Tarih", dt.date.today())

# Veri yükle
try:
    risk_df = load_csv(PATH_RISK)
except Exception as e:
    st.error(f"risk_hourly.csv okunamadı: {e}")
    st.stop()

# Beklenen kolonlar: GEOID,date,hour_range,risk_score,risk_level,risk_decile
if "date" not in risk_df.columns or "hour_range" not in risk_df.columns:
    st.error("risk_hourly.csv beklenen kolonlara sahip değil.")
    st.stop()

risk_df["date"] = pd.to_datetime(risk_df["date"], errors="coerce").dt.date
hours = sorted([h for h in risk_df["hour_range"].dropna().unique().tolist()])
if not hours:
    st.warning("Saat aralığı bulunamadı.")
    st.stop()

hour = st.select_slider("Saat aralığı", options=hours, value=hours[0])

# Filtre + Top-K
f = risk_df[(risk_df["date"] == sel_date) & (risk_df["hour_range"] == hour)].copy()
if f.empty:
    st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim dener misin?")
else:
    f = f.sort_values("risk_score", ascending=False).head(int(top_k))

    # GEOID → centroid
    geo = load_geojson()
    cent = centroids_from_geojson(geo)
    view = f.merge(cent, on="GEOID", how="left").dropna(subset=["lat", "lon"])

    # Renk/size kodlama
    level_colors = {
        "critical": [220, 20, 60],
        "high":     [255, 140, 0],
        "medium":   [255, 215, 0],
        "low":      [34, 139, 34],
    }
    view["color"] = view["risk_level"].map(level_colors).fillna([100, 100, 100])
    view["radius"] = (view["risk_score"].clip(0, 1) * 40 + 10).astype(int)

    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    st.dataframe(view[["GEOID", "risk_score", "risk_level", "risk_decile"]].reset_index(drop=True))

    # Harita katmanları
    show_polygons = st.checkbox("Poligon katmanını göster (daha yavaş olabilir)", value=False)

    initial = pdk.ViewState(
        latitude=float(view["lat"].mean()) if not view.empty else 37.7749,
        longitude=float(view["lon"].mean()) if not view.empty else -122.4194,
        zoom=11, pitch=30,
    )
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=view,
            get_position=["lon", "lat"],
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
        )
    ]

    if show_polygons:
        try:
            # Artifact/Raw'tan GeoJSON payload (bytes → str → dict)
            payload = (
                _read_from_artifact(fetch_artifact_zip(), PATH_GEOJSON).decode("utf-8")
                if (USE_ARTIFACT and GITHUB_TOKEN)
                else read_raw(PATH_GEOJSON).decode("utf-8")
            )
            layers.insert(
                0,
                pdk.Layer(
                    "GeoJsonLayer",
                    data=json.loads(payload),
                    stroked=False,
                    filled=False,
                    get_line_color=[150, 150, 150],
                    line_width_min_pixels=1,
                ),
            )
        except Exception as e:
            st.info(f"Poligon katmanı yüklenemedi: {e}")

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=initial,
            tooltip={"text": "GEOID: {GEOID}\nRisk: {risk_score:.3f} ({risk_level})"},
        )
    )

st.divider()
with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
    try:
        recs = load_recs_df()
        recs["date"] = pd.to_datetime(recs["date"], errors="coerce").dt.date
        fr = recs[(recs["date"] == sel_date) & (recs["hour_range"] == hour)].copy()
        st.dataframe(fr.head(200))
    except Exception as e:
        st.info(f"Devriye önerileri okunamadı: {e}")

with st.expander("📈 Model Metrikleri"):
    try:
        m = load_csv(PATH_METRICS)
        st.dataframe(m)
    except Exception as e:
        st.info(f"Metrikler yüklenemedi: {e}")

st.caption("Kaynak: GitHub Actions ile üretilen `crime_data/` çıktıları")
