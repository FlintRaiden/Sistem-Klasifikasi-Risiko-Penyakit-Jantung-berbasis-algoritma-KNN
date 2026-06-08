# 💓 HeartGuard KNN — Klasifikasi Risiko Penyakit Jantung

Aplikasi web berbasis **Flask** yang menggunakan algoritma **K-Nearest Neighbors (KNN)**
untuk mengklasifikasikan risiko penyakit jantung berdasarkan 13 parameter klinis pasien.

> ⚠️ **DISCLAIMER:** Aplikasi ini merupakan **tugas kuliah** dan **bukan alat diagnosis medis resmi**.
> Hasil prediksi tidak dapat menggantikan pemeriksaan oleh tenaga medis profesional.

---

## 📁 Struktur Proyek

```
heart-knn-app/
├── app.py                  # Backend Flask (routing + prediksi)
├── train_model.py          # Script training model KNN
├── heart.csv               # Dataset Heart Disease UCI
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi ini
├── model/                  # Folder model (dibuat otomatis saat training)
│   ├── knn_model.joblib    # Model KNN tersimpan
│   ├── scaler.joblib       # StandardScaler tersimpan
│   └── model_meta.joblib   # Metadata model (akurasi, k, dll)
└── templates/
    ├── base.html           # Layout utama (navbar, footer, styling)
    ├── index.html          # Halaman beranda
    ├── predict.html        # Form input data pasien
    └── result.html         # Halaman hasil prediksi
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone / Unduh Project
```bash
git clone <url-repo-anda>
cd heart-knn-app
```

### 2. Buat Virtual Environment (opsional tapi direkomendasikan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Latih Model KNN
```bash
python train_model.py
```
Script ini akan:
- Membaca `heart.csv`
- Melakukan preprocessing (StandardScaler)
- Mencari K optimal via Elbow Method
- Melatih model KNN
- Menyimpan `model/knn_model.joblib`, `model/scaler.joblib`, `model/model_meta.joblib`

### 5. Jalankan Aplikasi Flask
```bash
python app.py
```
Buka browser dan akses: **http://localhost:7860**

---

## ☁️ Deployment ke Hugging Face Spaces

### Metode: Docker (Direkomendasikan)

#### Langkah 1: Buat akun & Space baru
1. Daftar di [huggingface.co](https://huggingface.co)
2. Klik **New Space** → pilih **Docker** sebagai SDK
3. Beri nama space (misal: `heartguard-knn`)

#### Langkah 2: Tambahkan `Dockerfile`
Buat file `Dockerfile` di root proyek:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy semua file
COPY . .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Latih model saat build image (agar model sudah tersedia)
RUN python train_model.py

# Expose port 7860 (standar Hugging Face Spaces)
EXPOSE 7860

# Jalankan aplikasi
CMD ["python", "app.py"]
```

#### Langkah 3: Push ke Hugging Face
```bash
# Install git-lfs jika belum
git lfs install

# Clone repo space Anda
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>

# Copy semua file proyek ke dalam folder ini
cp -r /path/ke/heart-knn-app/* .

# Commit dan push
git add .
git commit -m "Initial deployment: HeartGuard KNN"
git push
```

Hugging Face akan otomatis build Docker image dan menjalankan aplikasi.
Akses di: `https://<username>-<space-name>.hf.space`

---

### Metode Alternatif: Render.com (Gratis)

1. Push proyek ke GitHub
2. Buat akun di [render.com](https://render.com)
3. New → **Web Service** → Connect repo GitHub
4. Konfigurasi:
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3
5. Deploy → aplikasi live dalam ~5 menit

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| Nama | Heart Disease UCI |
| Jumlah Data | 1.025 pasien |
| Fitur | 13 parameter klinis |
| Target | 0 = Sehat, 1 = Sakit Jantung |

### Keterangan Fitur

| Kolom | Deskripsi |
|-------|-----------|
| `age` | Usia pasien (tahun) |
| `sex` | Jenis kelamin (0=Perempuan, 1=Laki-laki) |
| `cp` | Tipe nyeri dada (0–3) |
| `trestbps` | Tekanan darah istirahat (mmHg) |
| `chol` | Kolesterol serum (mg/dl) |
| `fbs` | Gula darah puasa > 120 mg/dl (0/1) |
| `restecg` | Hasil EKG istirahat (0–2) |
| `thalach` | Detak jantung maksimum (bpm) |
| `exang` | Angina akibat olahraga (0/1) |
| `oldpeak` | Depresi segmen ST saat olahraga |
| `slope` | Kemiringan segmen ST (0–2) |
| `ca` | Jumlah pembuluh darah utama (0–4) |
| `thal` | Hasil thalassemia (0–3) |

---

## 🤖 Informasi Model

- **Algoritma:** K-Nearest Neighbors (KNN)
- **Metric Jarak:** Euclidean
- **Preprocessing:** StandardScaler (penting karena KNN sensitif skala!)
- **Pemilihan K:** Elbow Method + 5-Fold Cross-Validation
- **Library:** scikit-learn

---

## 👤 Informasi Tugas

- **Nama:** Muhamad Rafli Ardiansyah
- **NIM:** 301240028
- **Program Studi:** Teknik Informatika
- **Mata Kuliah:** Praktikum Kecerdasan Buatan
- **Tugas:** 9 – Penerapan Algoritma KNN pada Aplikasi Berbasis Web dengan Flask
- **Referensi Modul:** BAB 9 – K-Nearest Neighbors, Modul Praktikum Kecerdasan Buatan, Mohammad Bayu Anggara, S.Kom., M.Kom.
