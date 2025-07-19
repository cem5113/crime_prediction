# Çalışan stacking Modeli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 1. Veriyi oku
df = pd.read_csv("/content/drive/MyDrive/crime_data/sf_model_41_full_xy.csv")

# 2. Kategorik ve sayısal değişkenleri ayır
categorical_cols = ["season", "poi_dominant_type", "poi_total_count_range", "poi_risk_score_range"]
numerical_cols = [col for col in df.columns if col not in categorical_cols + ["Y_label", "GEOID"]]

# 3. Kategorik değişkenleri one-hot encoding ile dönüştür
df_encoded = pd.get_dummies(df[categorical_cols + numerical_cols], drop_first=True)

# 4. X ve y tanımla
X = df_encoded
y = df["Y_label"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Base modeller
candidate_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=5, eval_metric="logloss", random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('hgb', HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('lr', LogisticRegression(max_iter=2000, solver='lbfgs')),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)),
    ('bagging', BaggingClassifier(n_estimators=50, random_state=42))
]

# 7. Meta modeller
meta_models = {
    "Stack_LR": LogisticRegression(max_iter=2000, solver='lbfgs'),
    "Stack_RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "Stack_XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
}

# 8. Sonuçları sakla
results = []

# 9. Base modelleri test et
for name, model in candidate_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (1)": report['1']['precision'],
        "Recall (1)": report['1']['recall'],
        "F1-Score (1)": report['1']['f1-score']
    })

# 10. Stacking modelleri test et
for name, meta_model in meta_models.items():
    stacking_clf = StackingClassifier(
        estimators=candidate_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=2,
        passthrough=False
    )
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (1)": report['1']['precision'],
        "Recall (1)": report['1']['recall'],
        "F1-Score (1)": report['1']['f1-score']
    })

# 11. Sonuçları göster
results_df = pd.DataFrame(results).sort_values(by="F1-Score (1)", ascending=False).reset_index(drop=True)
print(results_df)

# 12. En iyi modeller
top3_models = results_df["Model"].iloc[:3].tolist()
print(f"\n🔍 En başarılı 3 model: {top3_models}")
