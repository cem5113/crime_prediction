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
PATH_GEOJSON    = "crime_data/sf_census_blocks_with_population.geojson"  # isim blok dese de içerik tract olabilir

# ----------------- Yardımcılar -----------------
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
    s.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"})
    resp = _retry_get(f"https://api.github.com/repos/{REPO}/actions/artifacts?per_page=100", headers=s.headers, timeout=60)
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
    if not names: return inner_path
    if inner_path in names: return inner_path
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

def _detect_geoid_col(cols: list[str]) -> str | None:
    cand = ["GEOID", "GEOID10", "GEOID20", "geoid", "geoid10", "geoid20"]
    for c in cand:
        if c in cols: return c
    # bazı dosyalarda 'geoid' property içinde olabilir, yoksa None
    return None

@st.cache_resource(ttl=900)
def load_geojson_gdf() -> gpd.GeoDataFrame:
    """GeoJSON'u yükle, GEOID kolonunu normalize et, genişliği session'a yaz."""
    try:
        if USE_ARTIFACT:
            gj = json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
        else:
            gj = json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))
    except Exception as e:
        st.error(f"GeoJSON okunamadı: {e}")
        raise
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    geocol = _detect_geoid_col(list(gdf.columns))
    if geocol is None:
        st.warning("GeoJSON içinde GEOID alanı bulunamadı; index kullanılacak (eşleşme düşebilir).")
        gdf["GEOID"] = gdf.index.astype(str)
    else:
        if geocol != "GEOID":
            gdf["GEOID"] = gdf[geocol].astype(str)
    # genişlik tespiti ve session'a koy
    width = int(gdf["GEOID"].astype(str).str.len().max())
    st.session_state["GEOID_WIDTH"] = width
    # normalize: sadece rakamlar ve sıfır doldurma
    gdf["GEOID"] = gdf["GEOID"].apply(lambda x: re.sub(r"\D", "", str(x))[:width].zfill(width))
    return gdf

@st.cache_data(ttl=900)
def load_geojson_dict() -> dict:
    if USE_ARTIFACT:
        return json.loads(_read_from_artifact(PATH_GEOJSON).decode("utf-8"))
    return json.loads(read_raw(PATH_GEOJSON).decode("utf-8"))

@st.cache_data(ttl=900)
def centroids_from_geojson() -> pd.DataFrame:
    gdf = load_geojson_gdf()
    # tract/Block karışsa bile representative_point daha stabil
    try:
        gg = gdf.to_crs(3857)
        pts = gg.representative_point().to_crs(4326)
        return pd.DataFrame({"GEOID": gdf["GEOID"].astype(str), "lat": pts.y.values, "lon": pts.x.values})
    except Exception:
        c = gdf.copy()
        c["centroid"] = c.geometry.centroid
        return pd.DataFrame({"GEOID": c["GEOID"].astype(str), "lat": c["centroid"].y, "lon": c["centroid"].x})

def _len_mode(s: pd.Series) -> int:
    s = s.dropna().astype(str)
    if s.empty: return 0
    return int(s.str.len().mode().iloc[0])

def _only_digits(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True)

def smart_merge_by_geoid(df: pd.DataFrame, cent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Önce tam eşleşme, yoksa ortak prefix (min uzunluk) ile eşleştir.
    Geriye (view, info) döner.
    """
    info = {}
    D = df.copy()
    C = cent.copy()

    D["GEOID_raw"] = _only_digits(D["GEOID"])
    C["GEOID_raw"] = _only_digits(C["GEOID"])

    lenD = _len_mode(D["GEOID_raw"])
    lenC = _len_mode(C["GEOID_raw"])
    info["len_risk_mode"] = lenD
    info["len_cent_mode"] = lenC

    # 1) Tam uzunluk eşleştirme (her iki tarafı da kendi mod uzunluğuna zfill)
    D["GEOID_norm"] = D["GEOID_raw"].str[:lenD].str.zfill(lenD)
    C["GEOID_norm"] = C["GEOID_raw"].str[:lenC].str.zfill(lenC)

    view_exact = D.merge(C[["GEOID_norm","lat","lon"]], on="GEOID_norm", how="left")
    exact_matches = int(view_exact["lat"].notna().sum())
    info["matches_exact"] = exact_matches

    if exact_matches > 0:
        view = view_exact.copy()
        view["GEOID"] = view["GEOID_norm"]
        info["join_mode"] = "exact"
        return view.drop(columns=["GEOID_raw","GEOID_norm"]), info

    # 2) Prefix ile (min uzunluk üzerinden) eşleştirme
    L = min(lenD, lenC)
    if L == 0:
        view = D.copy()
        view["lat"] = pd.NA
        view["lon"] = pd.NA
        info["join_mode"] = "failed"
        info["matches_prefix"] = 0
        return view.drop(columns=["GEOID_raw","GEOID_norm"]), info

    D["GEOID_L"] = D["GEOID_raw"].str[:L].str.zfill(L)
    C["GEOID_L"] = C["GEOID_raw"].str[:L].str.zfill(L)
    view_pref = D.merge(C[["GEOID_L","lat","lon"]], on="GEOID_L", how="left")
    pref_matches = int(view_pref["lat"].notna().sum())
    info["join_mode"] = "prefix"
    info["prefix_len"] = L
    info["matches_prefix"] = pref_matches

    view = view_pref.copy()
    view["GEOID"] = view["GEOID_L"]
    return view.drop(columns=["GEOID_raw","GEOID_norm","GEOID_L"]), info

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

    # Saat seçenekleri
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

    # GEOID → centroid (akıllı birleştirme)
    cent = centroids_from_geojson()
    before = len(f)
    view_raw, merge_info = smart_merge_by_geoid(f.rename(columns={"GEOID":"GEOID"}), cent)
    # view_raw'ta 'lat/lon' olabilir ama bazı satırlar NaN
    matched = int(view_raw["lat"].notna().sum())
    view = view_raw.dropna(subset=["lat","lon"]).copy()

    # Uyarı / bilgi
    if matched == 0:
        st.warning(
            f"Eşleşen centroid bulunamadı (seçili kayıt: {before}, eşleşen: {matched}). "
            f"Join modu: {merge_info.get('join_mode')} • "
            f"risk_len≈{merge_info.get('len_risk_mode')} • geo_len≈{merge_info.get('len_cent_mode')} • "
            f"prefix_len={merge_info.get('prefix_len', '-')}"
        )
        # Teşhis için ilk 10 GEOID örneği
        st.code("risk GEOID örnekleri: " + ", ".join(_only_digits(f['GEOID']).astype(str).str[:20].head(10).tolist()))
        st.code("geo  GEOID örnekleri: " + ", ".join(_only_digits(cent['GEOID']).astype(str).str[:20].head(10).tolist()))

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
        colors = colors.apply(lambda c: c if isinstance(c, (list, tuple)) else default_color)
    else:
        vals = (view["risk_score"].fillna(0).clip(0, 1) * 255).astype(int)
        colors = vals.apply(lambda v: [v, 0, 255 - v])
    view["color"] = colors
    view["radius"] = (view["risk_score"].fillna(0).clip(0, 1) * 40 + 10).astype(int)

    st.subheader(f"📍 {sel_date} — {hour} — Top {len(view)} GEOID")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1: st.metric("Seçilen kayıt", before)
    with mcol2: st.metric("Eşleşen centroid", matched)
    with mcol3: st.metric("Ortalama risk", round(float(view["risk_score"].mean()) if len(view) else 0.0, 3))
    st.dataframe(view[["GEOID","risk_score","risk_level","risk_decile"]].reset_index(drop=True))

    # Harita (pydeck) — sadece eşleşme varsa çiz
    if matched > 0:
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
                # Aynı akıllı eşleştirme mantığıyla GEOID uyumu
                recs["date"] = pd.to_datetime(recs["date"], errors="coerce").dt.date
                fr = recs[(recs["date"] == sel_date) & (recs["hour_range"] == hour)].copy()
                if not fr.empty:
                    fr_view, _ = smart_merge_by_geoid(fr.rename(columns={"GEOID":"GEOID"}), cent)
                    st.dataframe(fr_view.head(200))
                else:
                    st.info("Bu tarih/saat için devriye önerisi yok.")
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

        try:
            cent_src = load_centroids_src()
        except Exception:
            cent_src = None
        if cent_src is None or cent_src.empty:
            cent_src = centroids_from_geojson()

        try:
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
        st.write(f"GeoJSON OK — {len(gdf)} geometri (GEOID width≈{int(st.session_state.get('GEOID_WIDTH', 11))})")
        # GEOID uzunluk dağılımları
        st.write(
            "GEOID uzunluk dağılımı (geo):",
            gdf["GEOID"].astype(str).str.len().value_counts().sort_index().to_dict(),
        )
    except Exception as e:
        st.info(f"GeoJSON teşhis: {e}")

st.caption("Kaynak: GitHub Actions artifact/commit içindeki `crime_data/` çıktıları • Operasyonel sekme: src/* modülleri")
