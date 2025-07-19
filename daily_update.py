# daily_update.py

import os
import gdown

# === 1. İndirilecek dosya listesi (ad + Google Drive ID) ===
FILES = {
    "sf_crime.csv": "1BjfKMF7PXha67EBPLUEyBjaWYn4bKQiO",
    "sf_crime_49.csv": "15Emnkd43zbIY0yNJkoQePdUZRIVBEAs-",
    "sf_crime_50.csv": "18kcWurfjTI-Fx75KOosoIrBUCD8q5UEo",
    "sf_crime_52.csv": "18al8iyVpBtiThHgJ1RrCFcKt9xqf-2Dp",
    "sf_crime_54.csv": "1moVBbphsEGkOFvhTAYQURs1nOuLeaGcI",
    "sf_crime_03.csv": "1APyVSRbJHmud9jdrUEbEhUsYBA1oVCGK",
    "sf_crime_04.csv": "1bwlEEWetA7Q-HGL7cZkCbncpyPx8XsDU",
    "sf_crime_05.csv": "1quRKZBzs54X8_Q06dW2z2YIkpXnVveTM",
    "sf_crime_06.csv": "1A1SANGeo1WsivWU3iFHsPCDU2jMn6Zij",
    "sf_crime_07.csv": "1s1F8tfVwmGVhxsq_LVZNgdMiL1F-s0rj",
    "sf_crime_08.csv": "182WvAuolHHsIWhXHoKKaZflvz_nswQzu",
    "sf_crime_09.csv": "182WvAuolHHsIWhXHoKKaZflvz_nswQzu",
    "sf_crime_10.csv": "1kNPpBBqmKqbIVexZkN3PJzB5x4Xive8M",
    "sf_crime_11.csv": "1Y8v2fo8w85N5ldSQcqoN1LZvGlfqQOHb",
    "sf_911_last_5_year.csv": "1humSp8WQdOHb7Bsffr1nNYxmmOirU5ca",
    "sf_311_last_5_years.csv": "1ySXT_fQa-rBlU-jnqpoeZocBXhpzlzz8",
    "sf_population.csv": "1MCfpKgMCtPQmSrn6494XIb0ZcKnnbUCl",
    "sf_bus_stops_with_geoid.csv": "1cN1z5Ijzjyz3t8mzjlux3w9OST4XmjDz",
    "sf_train_stops_with_geoid.csv": "1VqgteSbVoOosYwXzCyQDWbqgATuKR4tc",
    "sf_pois_cleaned_with_geoid.csv": "1rfcJyoyriTr4094H5usChV_XAl81SIZG",
    "sf_pois_with_risk_score.csv": "1N9wck41Y2Gg15D5orqGcPkxOiRvTFz4w",
    "risky_pois_dynamic.json": "1sKLuR-YqO1MsVtS1x-9Ry8EJymDxImCq",
    "sf_police_stations.csv": "1Q6PwLedkPIlx07xdFm3ZV85nmATp-_5Q",
    "sf_government_buildings.csv": "10G64kV-qHenrRQibhMlxgHq706-dJMYH",
    "sf_police_gov_crime.csv": "14LBjSbFEuXEew4u7m5_V6ksMmzvxAh6a",
    "sf_weather_5years.csv": "1wGzs-539SRo8_bJQ5Wa35s3GPZMIviPd"
}

# === 2. Kayıt klasörü
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# === 3. Dosyaları indir
for filename, file_id in FILES.items():
    output_path = os.path.join(DATA_DIR, filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️ İndiriliyor: {filename}")
    gdown.download(url, output_path, quiet=False)

print("✅ Tüm dosyalar indirildi ve güncellendi.")
