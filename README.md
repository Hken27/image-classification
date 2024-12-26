# Klasifikasi Bungkus Biskuit

Proyek ini bertujuan untuk melakukan klasifikasi bungkus biskuit menggunakan model deep learning dengan pendekatan **Convolutional Neural Network (CNN)** dan **MobileNet**. Dataset yang digunakan berasal dari Kaggle, dan hasil akhir model dapat diintegrasikan ke dalam antarmuka pengguna berbasis Streamlit.

---

## Deskripsi Dataset
- **Asal Dataset**: [Kaggle - Biscuit Wrappers Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/biscuit-wrappers-dataset)
- **Jumlah Data Asli**: 5.058 citra
- **Augmentasi Data**: Dilakukan untuk meningkatkan variasi data menjadi total **30.126 citra**.
- **Akses Dataset**: [Download di sini](https://drive.google.com/file/d/1WcVUO5-CN8vlzvJAsDO3NgTGsdt74I5o/view?usp=sharing)

---

## Model dan Library yang Digunakan

### Model yang Dipilih:
1. **Convolutional Neural Network (CNN)**
2. **MobileNet**

### Library yang Digunakan:
- **TensorFlow**: Untuk pembuatan dan pelatihan model deep learning.
- **Matplotlib**: Untuk visualisasi data dan hasil pelatihan.
- **NumPy**: Untuk komputasi numerik.
- **Seaborn**: Untuk visualisasi statistik.
- **Pathlib**: Untuk manipulasi jalur file dan direktori.

### Format Model
- Model yang sudah dilatih disimpan dalam format `.h5`.

---

## Integrasi Antarmuka Pengguna (UI) dengan Streamlit
Tujuan proyek berikutnya adalah membuat antarmuka berbasis web menggunakan **Streamlit** untuk menguji model yang telah dilatih.

### Langkah-Langkah Instalasi Streamlit:
1. **Install PDM** (Python Dependency Manager):
   ```bash
   pip install pdm
   ```
2. **Inisialisasi Proyek**:
   ```bash
   pdm init
   ```
   Pilih versi Python yang Anda inginkan.
3. **Aktivasi Virtual Environment**:
   ```bash
   pdm venv activate
   ```
4. **Jalankan Streamlit**:
   ```bash
   pdm run start
   ```
   atau
   ```bash
   streamlit run src/app.py
   ```

---

## Menggunakan Model

Model hasil pelatihan sudah disediakan dalam proyek ini. Jika Anda ingin menggunakan model Anda sendiri, lakukan langkah berikut:
1. Pastikan model Anda disimpan dalam format `.h5`.
2. Ganti path ke model Anda pada skrip Streamlit di `src/app.py`.

---

## Langkah-Langkah Cloning Repository
Ikuti langkah-langkah berikut untuk meng-clone dan menjalankan proyek ini:

1. Clone repository ini:
   ```bash
   git clone https://github.com/Hken27/image-classification
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd PRAK_UAP
   ```
3. Install PDM jika belum terinstall:
   ```bash
   pip install pdm
   ```
4. Inisialisasi proyek dan install dependensi:
   ```bash
   pdm install
   ```
5. Aktifkan virtual environment:
   ```bash
   pdm venv activate
   ```
6. Jalankan aplikasi Streamlit:
   ```bash
   pdm run start
   ```
   atau
   ```bash
   streamlit run src/app.py
   ```

---

## Resource Tambahan
- **Code Modelling**: [Akses di sini](https://drive.google.com/drive/folders/1iHhtGo2xegUdXR8dSAg2JQGGTSOUqt8_https://drive.google.com/drive/folders/1iHhtGo2xegUdXR8dSAg2JQGGTSOUqt8_)

---

## Demo Hasil
Berikut adalah hasil demo dari model klasifikasi bungkus biskuit:

### Prediksi Citra 1
![Demo Hasil 1](Screenshot%202024-12-26%20120309.png)

### Prediksi Citra 2
![Demo Hasil 2](Screenshot%202024-12-26%20120508.png)

### Prediksi Citra 3
![Demo Hasil 3](Screenshot%202024-12-26%20120422.png)

Citra di atas menunjukkan hasil prediksi dari training model yang telah dilatih.
