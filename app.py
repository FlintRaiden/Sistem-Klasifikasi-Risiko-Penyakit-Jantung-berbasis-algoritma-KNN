"""
app.py
======
Backend Flask untuk Aplikasi Klasifikasi Risiko Penyakit Jantung
menggunakan algoritma K-Nearest Neighbors (KNN).

Routing:
  GET  /          → Halaman Beranda (index.html)
  GET  /predict   → Halaman Form Input (predict.html)
  POST /predict   → Proses prediksi, redirect ke hasil
  GET  /result    → Halaman Hasil Prediksi (result.html)

Port: 7860 (kompatibel dengan Hugging Face Spaces)
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

# ─── Inisialisasi Aplikasi Flask ──────────────────────────────────────────────
app = Flask(__name__)

# Secret key untuk session (ganti dengan string acak yang aman di production)
app.secret_key = os.environ.get("SECRET_KEY", "knn-heart-app-secret-2024")

# ─── Path Model ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")
META_PATH   = os.path.join(BASE_DIR, "model", "model_meta.joblib")

# ─── Muat Model saat Aplikasi Dimulai ─────────────────────────────────────────
print("[INFO] Memuat model KNN...")
try:
    knn_model  = joblib.load(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)
    model_meta = joblib.load(META_PATH)
    print(f"[INFO] Model dimuat. K={model_meta['best_k']}, "
          f"Akurasi={model_meta['accuracy']}%")
except FileNotFoundError:
    print("[PERINGATAN] Model belum ditemukan. Jalankan train_model.py terlebih dahulu.")
    knn_model  = None
    scaler     = None
    model_meta = {
        'best_k': '?', 'accuracy': '?',
        'n_train': '?', 'n_test': '?',
        'feature_cols': []
    }

# ─── Label Mapping untuk Tampilan ─────────────────────────────────────────────
SEX_MAP     = {0: "Perempuan", 1: "Laki-laki"}
CP_MAP      = {0: "Asimptomatik", 1: "Angina Tipikal",
               2: "Angina Atipikal", 3: "Nyeri Non-Anginal"}
FBS_MAP     = {0: "≤ 120 mg/dl (Normal)", 1: "> 120 mg/dl (Tinggi)"}
RESTECG_MAP = {0: "Normal", 1: "Abnormalitas Gelombang ST-T",
               2: "Hipertrofi Ventrikel Kiri"}
EXANG_MAP   = {0: "Tidak", 1: "Ya"}
SLOPE_MAP   = {0: "Menurun", 1: "Datar", 2: "Naik"}
THAL_MAP    = {0: "Normal", 1: "Cacat Tetap",
               2: "Cacat Reversibel", 3: "Tidak Diketahui"}


# ─── Helper: Format nilai input untuk ditampilkan kembali ─────────────────────
def format_input_summary(form_data):
    """Mengubah data form mentah menjadi dict yang ramah tampilan."""
    return {
        "Usia"                        : f"{form_data['age']} tahun",
        "Jenis Kelamin"               : SEX_MAP.get(int(form_data['sex']), '-'),
        "Tipe Nyeri Dada"             : CP_MAP.get(int(form_data['cp']), '-'),
        "Tekanan Darah Istirahat"     : f"{form_data['trestbps']} mmHg",
        "Kolesterol Serum"            : f"{form_data['chol']} mg/dl",
        "Gula Darah Puasa"            : FBS_MAP.get(int(form_data['fbs']), '-'),
        "Hasil EKG Istirahat"         : RESTECG_MAP.get(int(form_data['restecg']), '-'),
        "Detak Jantung Maks."         : f"{form_data['thalach']} bpm",
        "Angina akibat Olahraga"      : EXANG_MAP.get(int(form_data['exang']), '-'),
        "Depresi ST (Oldpeak)"        : form_data['oldpeak'],
        "Kemiringan Segmen ST"        : SLOPE_MAP.get(int(form_data['slope']), '-'),
        "Jumlah Pembuluh Darah Utama" : f"{form_data['ca']} pembuluh",
        "Hasil Thalassemia"           : THAL_MAP.get(int(form_data['thal']), '-'),
    }


# ─── Route: Halaman Beranda ───────────────────────────────────────────────────
@app.route("/")
def index():
    """Menampilkan halaman utama / beranda aplikasi."""
    return render_template("index.html", meta=model_meta)


# ─── Route: Halaman Form Prediksi ─────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  → Tampilkan form input data pasien.
    POST → Baca data form, lakukan prediksi KNN, dan langsung tampilkan hasil.
    """
    if request.method == "GET":
        return render_template("predict.html")

    # ── Ambil Data dari Form ──────────────────────────────────────────────────
    try:
        form_data = {
            'age'      : request.form.get('age'),
            'sex'      : request.form.get('sex'),
            'cp'       : request.form.get('cp'),
            'trestbps' : request.form.get('trestbps'),
            'chol'     : request.form.get('chol'),
            'fbs'      : request.form.get('fbs'),
            'restecg'  : request.form.get('restecg'),
            'thalach'  : request.form.get('thalach'),
            'exang'    : request.form.get('exang'),
            'oldpeak'  : request.form.get('oldpeak'),
            'slope'    : request.form.get('slope'),
            'ca'       : request.form.get('ca'),
            'thal'     : request.form.get('thal'),
        }

        # ── Konversi ke array numpy ───────────────────────────────────────────
        input_values = np.array([[
            float(form_data['age']),
            float(form_data['sex']),
            float(form_data['cp']),
            float(form_data['trestbps']),
            float(form_data['chol']),
            float(form_data['fbs']),
            float(form_data['restecg']),
            float(form_data['thalach']),
            float(form_data['exang']),
            float(form_data['oldpeak']),
            float(form_data['slope']),
            float(form_data['ca']),
            float(form_data['thal']),
        ]])

        # ── Konversi ke DataFrame Pandas agar memiliki nama fitur yang sama dengan saat training (mencegah warning) ──
        input_df = pd.DataFrame(input_values, columns=model_meta['feature_cols'])

        # ── Scaling Input ──────────────────────────────────────────────────────
        input_scaled = scaler.transform(input_df)

        # ── Prediksi KNN ──────────────────────────────────────────────────────
        prediction    = int(knn_model.predict(input_scaled)[0])
        proba         = knn_model.predict_proba(input_scaled)[0]  # [prob_0, prob_1]

        # Hitung persentase probabilitas
        prob_sehat    = round(float(proba[0]) * 100, 1)
        prob_sakit    = round(float(proba[1]) * 100, 1)

        # ── Siapkan Hasil ─────────────────────────────────────────────────────
        result_data = {
            'prediction'   : prediction,
            'label'        : "Terindikasi Sakit Jantung" if prediction == 1 else "Terindikasi Sehat",
            'prob_sehat'   : prob_sehat,
            'prob_sakit'   : prob_sakit,
            'input_summary': format_input_summary(form_data),
        }

        # Simpan ke session juga sebagai fallback (jika ada yang mengakses /result secara langsung)
        session['result'] = result_data

        # Langsung render template hasil tanpa melakukan redirect untuk menghindari masalah cookie/session di iframe Hugging Face
        return render_template("result.html", result=result_data)

    except Exception as e:
        # Jika terjadi error (misal: field kosong), kembali ke form dengan pesan error
        error_msg = f"Terjadi kesalahan saat memproses data: {str(e)}"
        return render_template("predict.html", error=error_msg)


# ─── Route: Halaman Hasil Prediksi ───────────────────────────────────────────
@app.route("/result")
def result():
    """Menampilkan hasil prediksi KNN yang disimpan di session."""
    result_data = session.get('result')

    # Jika tidak ada data di session, redirect ke form
    if not result_data:
        return redirect(url_for("predict"))

    return render_template("result.html", result=result_data)


# ─── Jalankan Aplikasi ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Port 7860 adalah port default Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
