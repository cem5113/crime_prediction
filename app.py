import os, io, zipfile, requests
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="SF Crime Dashboard", layout="wide")
st.title("SF Crime Dashboard")

# ---- Artifact'tan veri çekici ----
def ensure_data():
    # Yerelde var mı?
    for d in ["crime_data", "out", "last_good/crime_data", "."]:
        if Path(d, "risk_hourly.csv").exists():
            return Path(d)

    # Yoksa GitHub Actions artifact indir
    token = os.getenv("GH_TOKEN")
    owner = os.getenv("GH_OWNER", "cem5113")
    repo  = os.getenv("GH_REPO",  "crime_prediction_data")
    if not token:
        st.warning("GH_TOKEN yok; artifact indirilemiyor. 'crime_data' klasörüne dosyaları commitleyebilir ya da GH_TOKEN ekleyebilirsin.")
        return Path("crime_data")

    st.info("En güncel artifact indiriliyor…")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1) Artifact listesinden 'sf-crime-pipeline-output' olanın en yenisini bul
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100", headers=headers, timeout=30)
    r.raise_for_status()
    arts = r.json().get("artifacts", [])
    arts = [a for a in arts if a.get("name") == "sf-crime-pipeline-output"]
    if not arts:
        st.error("Artifact bulunamadı. Workflow çıktısı henüz yüklenmemiş olabilir.")
        return Path("crime_data")
    arts.sort(key=lambda a: a.get("updated_at",""), reverse=True)
    art = arts[0]

    # 2) Zip'i indir ve 'crime_data/' altına çıkar
    dl = requests.get(art["archive_download_url"], headers=headers, timeout=60)
    dl.raise_for_status()
    dest = Path("crime_data"); dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(dl.content)) as z:
        for m in z.namelist():
            if m.endswith((".csv", ".geojson")):
                # Artifact bazen crime_data/ önekiyle gelir; her ikisini de ele al
                if m.startswith("crime_data/"):
                    z.extract(m, ".")
                else:
                    with z.open(m) as f, open(dest / os.path.basename(m), "wb") as out:
                        out.write(f.read())
    return Path("crime_data")

DATA_DIR = ensure_data()

def show_csv(name):
    st.subheader(name)
    p = DATA_DIR / name
    if p.exists():
        try:
            st.dataframe(pd.read_csv(p).head(200))
        except Exception as e:
            st.warning(f"Okunamadı: {e}")
    else:
        st.info(f"{name} bulunamadı")

for fname in [
    "risk_hourly.csv",
    "patrol_recs.csv",
    "patrol_recs_multi.csv",
    "metrics_all.csv",
]:
    show_csv(fname)

st.caption(f"Veri dizini: `{DATA_DIR.resolve()}`")
