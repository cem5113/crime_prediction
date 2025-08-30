git clone https://github.com/cem5113/crime_prediction
cd crime_prediction
mkdir -p crime_prediction
git mv streamlit_app.py crime_prediction/app.py   # yoksa cp ile kopyala
git commit -m "Move app to crime_prediction/app.py"
git push
