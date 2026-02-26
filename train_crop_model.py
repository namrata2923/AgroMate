"""
AgroMate – Crop Recommendation Model Trainer
=============================================
Trains a Random Forest classifier on a crop recommendation dataset.

If you have the Kaggle CSV (Crop_Recommendation.csv), place it at:
    data/crop_recommendation.csv
Otherwise the script auto-generates a realistic synthetic dataset.

Usage:
    python train_crop_model.py

Output:
    models/crop_model.pkl
    models/crop_label_encoder.pkl
"""
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DATA_PATH    = os.path.join(DATA_DIR, "crop_recommendation.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "crop_label_encoder.pkl")

# Mirror URLs for auto-download (tried first; falls back to synthetic if all fail)
DATASET_URLS = [
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-raw/crop_recommendation.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/SrinivasBathula/crop-recommendation/main/Crop_recommendation.csv",
]

# ── Synthetic dataset generator ───────────────────────────────────────────────
# Based on documented agronomic crop requirements (N/P/K in kg/ha,
# temperature in °C, humidity in %, soil pH, annual rainfall in mm).
# Each row: [N_mean, N_std, P_mean, P_std, K_mean, K_std,
#            T_mean, T_std, H_mean, H_std, ph_mean, ph_std, R_mean, R_std]
CROP_STATS = {
    "rice":        [80, 24, 47, 14, 40, 13,  23.7, 1.3, 82, 2.8, 6.4, 0.4, 236, 72],
    "maize":       [78, 24, 48, 14, 19,  5,  22.6, 3.6, 65,16.5, 6.2, 0.4,  85, 27],
    "chickpea":    [40,  6, 67,  7, 79, 10,  17.7, 3.0, 16, 3.6, 7.3, 0.3,  81, 41],
    "kidneybeans": [20,  4, 67,  7, 19,  4,  18.9, 2.8, 22, 5.2, 5.9, 0.4, 105, 37],
    "pigeonpeas":  [20,  4, 67,  7, 20,  5,  27.7, 1.4, 49,11.0, 5.8, 0.4, 149, 52],
    "mothbeans":   [21,  5, 48, 14, 20,  5,  28.9, 1.4, 53, 9.8, 6.9, 0.5,  51, 17],
    "mungbean":    [20,  4, 47, 14, 20,  5,  28.5, 2.4, 85, 5.4, 6.7, 0.5,  49, 16],
    "blackgram":   [40,  6, 67,  7, 19,  4,  29.8, 2.4, 65,13.6, 7.1, 0.4,  68, 16],
    "lentil":      [19,  4, 68,  7, 19,  4,  24.5, 3.0, 65,12.4, 6.9, 0.4,  45, 13],
    "pomegranate": [18,  5, 18,  5, 40,  6,  21.8, 3.7, 90, 3.8, 6.4, 0.5, 108, 37],
    "banana":      [100,16, 82,  8, 50,  8,  27.4, 1.1, 80, 5.0, 5.9, 0.5, 105, 37],
    "mango":       [20,  4, 27,  8, 30,  6,  31.2, 2.9, 50, 8.4, 5.8, 0.6,  95, 39],
    "grapes":      [23,  5,132, 18,200, 25,  24.0, 5.4, 81, 7.0, 6.0, 0.3,  69, 12],
    "watermelon":  [99, 25, 17,  6, 50,  7,  25.6, 2.4, 85, 5.0, 6.5, 0.5,  51, 18],
    "muskmelon":   [100,25, 17,  6, 50,  7,  28.7, 2.4, 92, 3.8, 6.4, 0.5,  25, 10],
    "apple":       [21,  5,134, 18,199, 25,  21.9, 1.7, 92, 4.0, 5.9, 0.4, 113, 38],
    "orange":      [20,  4, 16,  7, 10,  4,  22.8, 2.5, 92, 4.4, 7.0, 0.4, 113, 37],
    "papaya":      [49, 17, 59,  9, 50,  7,  33.7, 2.4, 92, 3.8, 6.7, 0.5, 144, 53],
    "coconut":     [22,  5, 16,  7, 30,  7,  27.4, 1.7, 94, 3.4, 5.9, 0.5, 148, 72],
    "cotton":      [118,30, 46, 14, 19,  4,  23.9, 3.0, 79,10.0, 6.9, 0.5,  80, 38],
    "jute":        [78, 24, 46, 14, 39, 11,  24.9, 3.0, 80, 7.7, 6.7, 0.5, 175, 74],
    "coffee":      [101,25, 28, 10, 29,  7,  25.5, 3.2, 58,12.2, 6.8, 0.5, 159, 64],
}
SAMPLES_PER_CROP = 100   # 22 crops × 100 = 2200 rows (matches Kaggle dataset size)

def generate_synthetic_dataset():
    """Generate realistic crop recommendation data using known agronomic ranges."""
    print("Generating synthetic crop recommendation dataset...")
    rows = []
    np.random.seed(42)
    for crop, (nm,ns, pm,ps, km,ks, tm,ts, hm,hs, phm,phs, rm,rs) in CROP_STATS.items():
        N    = np.clip(np.random.normal(nm, ns, SAMPLES_PER_CROP), 0, 200)
        P    = np.clip(np.random.normal(pm, ps, SAMPLES_PER_CROP), 0, 200)
        K    = np.clip(np.random.normal(km, ks, SAMPLES_PER_CROP), 0, 200)
        temp = np.clip(np.random.normal(tm, ts, SAMPLES_PER_CROP), 5,  50)
        hum  = np.clip(np.random.normal(hm, hs, SAMPLES_PER_CROP), 5, 100)
        ph   = np.clip(np.random.normal(phm,phs,SAMPLES_PER_CROP), 3,  10)
        rain = np.clip(np.random.normal(rm, rs, SAMPLES_PER_CROP), 5, 500)
        for i in range(SAMPLES_PER_CROP):
            rows.append([N[i],P[i],K[i],temp[i],hum[i],ph[i],rain[i],crop])
    df = pd.DataFrame(rows, columns=["n","p","k","temperature","humidity","ph","rainfall","label"])
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f" Synthetic dataset saved: {DATA_PATH}  ({len(df)} rows)\n")
    return df

# ── Load or create dataset ────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Dataset not found locally. Attempting auto-download...\n")
    downloaded = False
    for url in DATASET_URLS:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, DATA_PATH)
            print("  ✓ Downloaded successfully!\n")
            downloaded = True
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    if not downloaded:
        print("\nAll download attempts failed — using built-in synthetic dataset.\n")
        generate_synthetic_dataset()
else:
    print(f"Dataset found at: {DATA_PATH}\n")

# ── Load & inspect ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
if "crop" in df.columns and "label" not in df.columns:
    df.rename(columns={"crop": "label"}, inplace=True)

print(f"  Shape        : {df.shape}")
print(f"  Crops ({len(df['label'].unique())}) : {sorted(df['label'].unique())}")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print()

# ── Features & target ────────────────────────────────────────────────────────
FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES].values
y = df["label"].values

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Label classes  : {list(le.classes_)}\n")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train samples  : {len(X_train)}")
print(f"Test  samples  : {len(X_test)}\n")

# ── Train Random Forest ───────────────────────────────────────────────────────
print("Training Random Forest (n_estimators=200) ...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred) * 100

print(f"\n{'='*55}")
print(f"  Test Accuracy : {acc:.2f}%")
print(f"{'='*55}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Feature importances ───────────────────────────────────────────────────────
print("Feature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<15} {imp*100:5.2f}%  {bar}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(le,    ENCODER_PATH)

print(f"\n✓ Model saved   : {MODEL_PATH}")
print(f"✓ Encoder saved : {ENCODER_PATH}")
print("\nAll done!  Run 'python app.py' to start the server.")
