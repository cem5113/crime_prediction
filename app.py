# app.py
import io, json, zipfile, requests, datetime as dt
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="SF Crime Risk", layout="wide")

# ----------------- Ayarlar (secrets ile override edilebilir) -----------------
REPO          = st.secrets.get("REPO", "cem5113/crime_prediction_data")
BRANCH        = st.secrets.get("BRANCH", "main")
ARTIFACT_NAME = st.secrets.get("ARTIFACT_NAME", "sf-crime-pipeline-output")
USE_ARTIFACT  = str(st.secrets.get("USE_ARTIFACT", "true")).lower() in ("1","true","yes","on")
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", None)

# Pipeline çıktıları
PATH_RISK       = "crime_data/risk_hourly.csv"
PATH_RECS_MULTI = "crime_data/patrol_recs_multi.csv"
PATH_RECS       = "crime_data/patrol_recs.csv"
PATH_METRICS    = "crime_data/metrics_all.csv"
PATH_GEOJSON    = "crime_data/sf_census_blocks_with_population.geojson"
# Son çare centroid CSV (repoda var)
CENTROID_FALLBACKS = [
    "tract_centroids_sf.csv",
    "crime_data/tract_centroids_sf.csv",
]

RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# ----------------- Hafif import: geopandas/shapely isteğe bağlı -----------------
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    from shapely.geometry import shape as shp_shape  # type: ignore
except Exception:
    shp_shape = None

# ----------------- HTTP yardımcıları -----------------
@st.cache_data(ttl=3600)
def read_raw(path: str) -> bytes:
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(ttl=900)
def fetch_artifact_zip() -> zipfile.ZipFile:
    if not (USE_ARTIFACT and GITHUB_TOKEN):
        raise RuntimeError("Artifact modu kapalı veya GITHUB_TOKEN yok.")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    j = s.get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100", timeout=60).json()
    arts = [a for a in j.get("artifacts", []) if a.get("name")==ARTIFACT_NAME and not a.get("expired")]
    if not arts:
        raise RuntimeError("Uygun artifact bulunamadı.")
    art = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]
    z = s.get(art["archive_download_url"], timeout=180)
    z.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(z.content))

def read_bytes(inner_path: str) -> bytes:
    """Önce artifact, yoksa raw."""
    if USE_ARTIFACT and GITHUB_TOKEN:
        try:
            zf = fetch_artifact_zip()
            with zf.open(inner_path) as f:
                return f.read()
        except Exception as e:
            st.warning(f"Artifact okunamadı ({e}). raw moda düşülüyor…")
    return read_raw(inner_path)

@st.cache_data(ttl=900)
def load_csv(inner_path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_bytes(inner_path)))

# ----------------- Geo yardımcıları (çoklu fallback) -----------------
@st.cache_data(ttl=900)
def load_centroids_df() -> pd.DataFrame:
    """Centroidleri 3 aşamada getir:
       1) GeoJSON → (geopandas varsa) centroid
       2) GeoJSON → (shapely varsa) centroid
       3) Fallback CSV
       Dönüş: DataFrame[GEOID, lat, lon], ayrıca opsiyonel polygon payload.
    """
    polygons_payload = None

    # 1) GeoJSON dene
    try:
        gj_bytes = read_bytes(PATH_GEOJSON)
        polygons_payload = json.loads(gj_bytes.decode("utf-8"))
        feats = polygons_payload.get("features", [])
        # 1a) geopandas varsa
        if gpd is not None:
            gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
            c = gdf.geometry.centroid
            df = pd.DataFrame({"GEOID": gdf["GEOID"].astype(str), "lat": c.y, "lon": c.x})
            return df, polygons_payload
        # 1b) shapely varsa
        if shp_shape is not None:
            rows = []
            for f in feats:
                props = f.get("properties", {})
                geoid = str(props.get("GEOID", ""))
                try:
                    geom = shp_shape(f.get("geometry"))
                    cent = geom.centroid
                    rows.append({"GEOID": geoid, "lat": cent.y, "lon": cent.x})
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows), polygons_payload
    except Exception as e:
        # GeoJSON bulunamadıysa sessizce CSV fallback’e düş
        st.info(f"GeoJSON centroid üretimi atlandı: {e}")

    # 2) CSV fallback
    last_err = None
    for p in CENTROID_FALLBACKS:
        try:
            df = load_csv(p)
            # kolon adlarını normalize et
            cols = {c.lower(): c for c in df.columns}
            geoid = cols.get("geoid") or "GEOID"
            lat   = cols.get("lat")   or "lat"
            lon   = cols.get("lon")   or "lon"
            df = df.rename(columns={geoid: "GEOID", lat: "lat", lon: "lon"})
            df["GEOID"] = df["GEOID"].astype(str)
            return df[["GEOID","lat","lon"]], None
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Centroid elde edilemedi (CSV fallback da başarısız): {last_err}")

# ----------------- UI -----------------
st.title("SF Crime Dashboard")

colA, colB, colC = st.columns([1.4, 1, 1])
with colA:
    mode = "Artifact" if (USE_ARTIFACT and GITHUB_TOKEN) else "Raw/Commit"
    st.caption("Veri Kaynağı")
    st.write(f"• Okuma modu: **{mode}**  \n• Repo: `{REPO}`  \n• Branch: `{BRANCH}`")
with colB:
    top_k = st.number_input("Top-K (liste/harita)", min_value=10, max_value=500, value=50, step=10)
with colC:
    sel_date = st.date_input("Tarih", dt.date.today())

# risk_hourly.csv
try:
    risk_df = load_csv(PATH_RISK)
except Exception as e:
    st.error(f"`{PATH_RISK}` okunamadı: {e}")
    st.stop()

required_cols = {"GEOID","date","hour_range","risk_score"}
if not required_cols.issubset(set(map(str, risk_df.columns))):
    st.error(f"`{PATH_RISK}` beklenen kolonlara sahip değil: {sorted(required_cols)}")
    st.stop()

risk_df["GEOID"] = risk_df["GEOID"].astype(str)
risk_df["date"]  = pd.to_datetime(risk_df["date"], errors="coerce").dt.date
hours = sorted([h for h in risk_df["hour_range"].dropna().unique().tolist()])
if not hours:
    st.warning("Saat aralığı bulunamadı.")
    st.stop()
hour = st.select_slider("Saat aralığı", options=hours, value=hours[0])

# Filtre + Top-K
f = risk_df[(risk_df["date"] == sel_date) & (risk_df["hour_range"] == hour)].copy()
if f.empty:
    st.warning("Seçilen tarih/saat için kayıt yok. Başka seçim dener misin?")
    st.stop()

f = f.sort_values("risk_score", ascending=False).head(int(top_k))

# Centroid & opsiyonel poligon
try:
    centroids, polygons = load_centroids_df()
except Exception as e:
    st.error(f"Centroid verisi yok: {e}")
    st.stop()

view = f.merge(centroids, on="GEOID", how="left").dropna(subset=["lat","lon"])

# Renk/size
level_colors = {
    "critical": [220, 20, 60],
    "high":     [255, 140, 0],
    "medium":   [255, 215, 0],
    "low":      [34, 139, 34],
}
view["color"]  = view.get("risk_level", "").map(level_colors) if "risk_level" in view else None
view["color"]  = view["color"].fillna([100,100,100])
view["radius"] = (view["risk_score"].clip(0,1) * 40 + 10).astype(int)

st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
st.dataframe(view[["GEOID","risk_score"] + ([ "risk_level"] if "risk_level" in view else [])].reset_index(drop=True))

# Harita
initial = pdk.ViewState(
    latitude=float(view["lat"].mean()) if not view.empty else 37.7749,
    longitude=float(view["lon"].mean()) if not view.empty else -122.4194,
    zoom=11, pitch=30,
)

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=view,
        get_position=["lon","lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
    )
]
if st.checkbox("Poligon katmanını göster (yavaş olabilir)", value=False) and polygons:
    layers.insert(0,
        pdk.Layer(
            "GeoJsonLayer",
            data=polygons,
            stroked=False, filled=False,
            get_line_color=[150,150,150],
            line_width_min_pixels=1,
        )
    )

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=initial,
    tooltip={"text":"GEOID: {GEOID}\nRisk: {risk_score:.3f}" + ("\nSeviye: {risk_level}" if "risk_level" in view else "")}
))

st.divider()
with st.expander("🚓 Devriye Önerileri (patrol_recs*.csv)"):
    try:
        try:
            recs = load_csv(PATH_RECS_MULTI)
        except Exception:
            recs = load_csv(PATH_RECS)
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
