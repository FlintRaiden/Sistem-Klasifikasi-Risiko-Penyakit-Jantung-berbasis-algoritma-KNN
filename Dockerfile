# Dockerfile
# Untuk deployment ke Hugging Face Spaces (Docker SDK)
# Port default HF Spaces: 7860

FROM python:3.11-slim

# Metadata
LABEL maintainer="Muhamad Rafli Ardiansyah"
LABEL description="HeartGuard KNN - Klasifikasi Risiko Penyakit Jantung"

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu untuk memanfaatkan cache layer Docker
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode dan dataset
COPY . .

# Latih model KNN (dijalankan saat build image)
# Sehingga model sudah tersedia saat container dijalankan
RUN python train_model.py

# Expose port 7860 (wajib untuk Hugging Face Spaces)
EXPOSE 7860

# Variabel lingkungan
ENV PORT=7860
ENV FLASK_ENV=production

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
