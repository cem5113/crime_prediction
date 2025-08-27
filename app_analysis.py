# streamlit_app.py
import io
import json
import time
import zipfile
import requests
import datetime as dt

import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# =========================================================
# Ayarlar
# =========================================================
# Örn: cem5113/crime_prediction_data
REPO = st.secrets.get("REPO", "cem5113/crime_prediction_data")
BRANCH = st.secrets.get("BRANCH", "main")
ARTIFACT_NAME = st.secrets.get("ARTIFACT_NAME", "sf-crime-pipeline-output")
USE_ARTIFACT = st.secrets.get("USE_ARTIFACT", True)  # commit modunda False yap
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)

# Pipeline'ın bıraktığı dosya yolları (artifact zip içinde de aynı)
PATH_RISK = "crime_data/risk_hourly.csv"
PATH_RECS = "crime_data/patrol_recs.csv"                 # veya patrol_recs_multi.csv
PATH_METRICS = "crime_data/metrics_stacking.csv"
PATH_GEOJSON = "crime_data/sf_census_blocks_with_population.geojson"  # poligonlar

# =========================================================
# Yardımcılar
# =========================================================
def _retry_get(url: str, headers: dict | None = None, timeout: int = 60, retries: int = 2) -> requests.Response:
    """Basit yeniden dene (429/5xx)."""
    last_err = None
    for i in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err

@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    r = _retry_get(url, timeout=60)
    return r.content

@st.cache_resource(show_spinner=False)
def fetch_artifact_zip() -> zipfile.ZipFile:
    """GitHub Actions artifact zip'ini getir (cache'li kaynak)."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN yok; artifact indirilemez.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}",
                      "Accept": "application/vnd.github+json"})
    j = _retry_get(
        f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100",
        headers=s.headers, timeout=60
    ).json()
    arts = [a for a in j.get("artifacts", []) if a["name"] == ARTIFACT_NAME and not a["expired"]]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = _retry_get(art["archive_download_url"], headers=s.headers, timeout=120)
    return zipfile.ZipFile(io.BytesIO(z.content))

def _read_from_artifact(zf: zipfile.ZipFile, inner_path: str) -> bytes:
    with zf.open(inner_path) as f:
        return f.read()

@st.cache_data(ttl=900, show_spinner=False)
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

@st.cache_data(ttl=900, show_spinner=False)
def load_geojson_raw_dict() -> dict:
    """GeoJSON'ı dict olarak yükle (pydeck için uygun)."""
    if USE_ARTIFACT:
        try:
            zf = fetch_artifact_zip()
            data = _read_from_artifact(zf, PATH_GEOJSON).decode("utf-8")
            return json.loads(data)
        except Exception as e:
            st.warning(f"GeoJSON artifact'tan okunamadı ({e}). raw moda geçiliyor…")
    return json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))

@st.cache_data(ttl=900, show_spinner=False)
def load_geojson_gdf() -> gpd.GeoDataFrame:
    """GeoJSON'ı GeoDataFrame olarak yükle (analiz/centroid için)."""
    gj = load_geojson_raw_dict()
    return gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")

@st.cache_data(ttl=900, show_spinner=False)
def centroids_from_geojson(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Poligonların temsili noktaları:
    - Web Mercator'a projeksiyon
    - representative_point() ile poligon içinde garanti nokta
    - tekrar WGS84'e dönüş
    """
    projected = gdf.to_crs(3857)  # Web Mercator
    points = projected.representative_point()
    points_ll = gpd.GeoSeries(points, crs=projected.crs).to_crs(4326)
    return pd.DataFrame({
        "GEOID": gdf["GEOID"].astype(str),
        "lat": points_ll.y,
        "lon": points_ll.x,
    })

def _hour_key(s) -> int:
    """'20:00-22:00' gibi etiketleri mantıklı sıraya koymak için anahtar."""
    try:
        return int(str(s).split("-")[0].split(":")[0])
    except Exception:
        return 0

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="SF Crime Risk", layout="wide")
st.title("SF Crime Risk • (Pipeline → Streamlit)")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.caption("Veri Kaynağı")
    mode = "Artifact" if USE_ARTIFACT else "Raw/Commit"
    st.write(
        f"• Okuma modu: **{mode}**  \n"
        f"• Repo: `{REPO}`  \n"
        f"• Branch: `{BRANCH}`"
    )

with colB:
    top_k = st.number_input("Top-K (liste/harita)", min_value=1, max_value=500, value=50, step=10)
with colC:
    today = dt.date.today()
    sel_date = st.date_input("Tarih", today)

# =========================================================
# Risk verisi & Saat seçimi
# =========================================================
risk_df = load_csv(PATH_RISK)
if "date" not in risk_df.columns or "hour_range" not in risk_df.columns:
    st.error("risk_hourly.csv beklenen kolonları içermiyor (date, hour_range).")
    st.stop()

risk_df["date"] = pd.to_datetime(risk_df["date"]).dt.date
hours = sorted(risk_df["hour_range"].dropna().unique().tolist(), key=_hour_key)

if not hours:
    st.warning("Saat aralıkları bulunamadı. Model çıktıları henüz üretilememiş olabilir.")
    st.stop()

hour = st.select_slider("Saat aralığı", options=hours, value=hours[0] if hours else None)
if hour is None:
    st.stop()

# =========================================================
# Filtre + Top-K
# =========================================================
f = risk_df[(risk_df["date"] == sel_date) & (risk_df["hour_range"] == hour)].copy()
if f.empty:
    st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim deneyebilir veya pipeline çıktısını kontrol edebilirsin.")
    st.stop()

f = f.sort_values("risk_score", ascending=False).head(int(top_k))

# =========================================================
# GEO verileri (tek sefer yükle)
# =========================================================
geo_dict = load_geojson_raw_dict()
geo_gdf = load_geojson_gdf()
cent = centroids_from_geojson(geo_gdf)

view = f.merge(cent, on="GEOID", how="left").dropna(subset=["lat", "lon"])

# Renk/size kodlama
if "risk_level" in view.columns:
    level_colors = {
        "critical": [220, 20, 60],   # kırmızı
        "high":     [255, 140, 0],   # turuncu
        "medium":   [255, 215, 0],   # sarı
        "low":      [34, 139, 34],   # yeşil
    }
    view["color"] = view["risk_level"].map(level_colors).fillna([100, 100, 100])
else:
    # sürekli skala (risk_score → [0,255])
    view["color"] = (
        (view["risk_score"] * 255)
        .clip(0, 255)
        .astype(int)
        .apply(lambda v: [v, 0, 255 - v])
    )

# skor’a göre yarıçap (görsel)
view["radius"] = (view["risk_score"].clip(0, 1) * 40 + 10).astype(int)

# Başlık + küçük metrikler
st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
mcol1, mcol2 = st.columns(2)
with mcol1:
    st.metric("Seçilen kayıt", len(view))
with mcol2:
    st.metric("Ortalama risk", round(float(view["risk_score"].mean()), 3))

st.dataframe(
    view[["GEOID", "risk_score", "risk_level", "risk_decile"]]
    .reset_index(drop=True)
)

# =========================================================
# Harita (pydeck)
# =========================================================
initial = pdk.ViewState(
    latitude=float(view["lat"].mean()) if not view.empty else 37.7749,
    longitude=float(view["lon"].mean()) if not view.empty else -122.4194,
    zoom=11,
    pitch=30,
)

layer_points = pdk.Layer(
    "ScatterplotLayer",
    data=view,
    get_position=["lon", "lat"],
    get_radius="radius",
    get_fill_color="color",
    pickable=True,
)

# Poligon katmanı (isteğe bağlı; ağır gelirse kapat)
layer_poly = pdk.Layer(
    "GeoJsonLayer",
    data=geo_dict,
    stroked=False,
    filled=False,
    get_line_color=[150, 150, 150],
    line_width_min_pixels=1,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer_poly, layer_points],
        initial_view_state=initial,
        tooltip={"text": "GEOID: {GEOID}\nRisk: {risk_score:.3f} ({risk_level})"},
    )
)

# =========================================================
# Ek paneller
# =========================================================
st.divider()
with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
    try:
        recs = load_csv(PATH_RECS)
        if {"date", "hour_range"}.issubset(recs.columns):
            recs["date"] = pd.to_datetime(recs["date"]).dt.date
            fr = recs[(recs["date"] == sel_date) & (recs["hour_range"] == hour)].copy()
            if fr.empty:
                st.info("Bu tarih/saat için devriye önerisi yok.")
            else:
                st.dataframe(fr.head(200))
        else:
            st.info("patrol_recs dosyası beklenen kolonlara sahip değil.")
    except Exception as e:
        st.info(f"Devriye önerileri okunamadı: {e}")

with st.expander("📈 Model Metrikleri"):
    try:
        m = load_csv(PATH_METRICS)
        st.dataframe(m)
    except Exception as e:
        st.info(f"Metrikler yüklenemedi: {e}")

st.caption("Kaynak: GitHub Actions ile üretilen `crime_data/` çıktıları")
