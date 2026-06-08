"""
train_model.py
==============
Script untuk melatih model K-Nearest Neighbors (KNN)
pada dataset Heart Disease (heart.csv).

Langkah-langkah:
1. Baca dan eksplorasi dataset
2. Preprocessing (scaling fitur numerik)
3. Cari nilai K optimal dengan Elbow Method
4. Latih model KNN final
5. Evaluasi model
6. Simpan model dan scaler dengan joblib
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "heart.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "knn_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Baca Dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("  TRAINING MODEL KNN - KLASIFIKASI PENYAKIT JANTUNG")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"[INFO] Distribusi target:")
print(f"       Sehat  (0): {(df['target'] == 0).sum()} pasien")
print(f"       Sakit  (1): {(df['target'] == 1).sum()} pasien")

# ─── 2. Pisahkan Fitur dan Label ──────────────────────────────────────────────
FEATURE_COLS = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak','slope','ca','thal']

X = df[FEATURE_COLS]
y = df['target']

# ─── 3. Split Data Train / Test ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[INFO] Data train: {X_train.shape[0]} | Data test: {X_test.shape[0]}")

# ─── 4. Scaling (KNN sangat sensitif terhadap skala fitur!) ───────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform pada train
X_test_scaled  = scaler.transform(X_test)         # hanya transform pada test

print("[INFO] Scaling fitur selesai (StandardScaler)")

# ─── 5. Elbow Method – Cari K Optimal ────────────────────────────────────────
print("\n[INFO] Mencari nilai K optimal (Elbow Method, k=1..20)...")

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Gunakan cross-validation 5-fold untuk estimasi yang lebih stabil
    cv_score = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='accuracy')
    error_rates.append(1 - cv_score.mean())

# Pilih K dengan error rate terendah
# Pilih K dengan error rate terendah, minimal k=3 (k=1 rentan overfitting)
best_k = k_range[np.argmin(error_rates)]
if best_k < 3:
    best_k = 3
    print(f"[INFO] K=1/2 cenderung overfitting, dinaikkan ke k=3")
print(f"[INFO] K optimal ditemukan: k = {best_k} "
      f"(error rate = {min(error_rates):.4f})")

# ─── 6. Latih Model KNN Final ─────────────────────────────────────────────────
knn_model = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean',
    weights='uniform'       # semua tetangga diberi bobot sama
)
knn_model.fit(X_train_scaled, y_train)
print(f"\n[INFO] Model KNN dilatih dengan k={best_k}")

# ─── 7. Evaluasi Model ────────────────────────────────────────────────────────
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[HASIL] Akurasi pada data test : {accuracy * 100:.2f}%")
print("\n[HASIL] Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Sehat (0)", "Sakit Jantung (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("[HASIL] Confusion Matrix:")
print(f"         Pred Sehat  Pred Sakit")
print(f"Aktual Sehat  : {cm[0][0]:4d}       {cm[0][1]:4d}")
print(f"Aktual Sakit  : {cm[1][0]:4d}       {cm[1][1]:4d}")

# ─── 8. Simpan Model & Scaler ─────────────────────────────────────────────────
joblib.dump(knn_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Simpan metadata model untuk ditampilkan di aplikasi
meta = {
    'best_k'        : int(best_k),
    'accuracy'      : round(float(accuracy) * 100, 2),
    'feature_cols'  : FEATURE_COLS,
    'n_train'       : int(X_train.shape[0]),
    'n_test'        : int(X_test.shape[0]),
}
joblib.dump(meta, os.path.join(MODEL_DIR, "model_meta.joblib"))

print(f"\n[INFO] Model  disimpan → {MODEL_PATH}")
print(f"[INFO] Scaler disimpan → {SCALER_PATH}")
print("\n[SELESAI] Training berhasil!\n")
