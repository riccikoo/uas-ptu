# 🚗 Aplikasi Pengolah Suara & Audio (ASR & TTS)

Aplikasi berbasis web menggunakan **Streamlit** untuk melakukan klasifikasi suara mobil secara *real-time* (Audio Speech Recognition / ASR) menggunakan ekstraksi fitur MFCC dan Machine Learning, serta dilengkapi dengan fitur pengubah teks menjadi suara (Text to Speech / TTS) beresolusi HD dalam Bahasa Indonesia menggunakan **Edge-TTS**.

---

## 🚀 Fitur Utama

* **🎙️ Live Audio Recognition (ASR):** Rekam suara langsung melalui mikrofon, visualisasikan karakteristik spektrogram MFCC, dan klasifikasikan jenis/merek mobil secara instan.
* **🔊 Text to Speech (TTS):** Ubah teks menjadi suara alami (Bahasa Indonesia) dengan pilihan suara Laki-laki (Ardi) atau Perempuan (Gadis) serta kontrol kecepatan bicara.
* **🔗 Integrasi Pintar:** Hasil tebakan dari mode ASR otomatis diteruskan menjadi teks di tab TTS untuk langsung disuarakan kembali.

---

## 📂 Struktur Folder Proyek

Pastikan struktur folder kamu terlihat seperti ini agar skrip pelatihan dan GUI dapat berjalan tanpa *error*:

```text
📂 proyek-audio-ai/
│
├── 📂 dataset/                   # Tempat menyimpan dataset suara mobil
│   ├── 📂 audi/
│   ├── 📂 ferrari/
│   ├── 📂 honda/
│   └── ... (10 kelas merek mobil)
│
├── 📂 training/                  # Folder output untuk menyimpan model hasil latih
│   ├── model_mobil.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── 📄 train.py                   # Skrip Python untuk melatih model ML (SVM & RF)
├── 📄 app.py                     # Skrip utama aplikasi Streamlit (GUI)
└── 📄 requirements.txt           # Daftar dependensi library Python
```

⚠️ Catatan Penting: Skrip GUI (app.py) membaca model dari dalam folder training/. Jika skrip pelatihan (train.py) kamu menyimpan model langsung di luar folder, pastikan untuk memindahkan file .pkl tersebut ke dalam folder training/ terlebih dahulu.

## 🛠️ Langkah Instalasi & Penggunaan
1. Clone atau Siapkan Folder Proyek
Pastikan kamu sudah memasukkan file app.py dan train.py ke dalam satu folder kerja.

2. Instalasi Dependensi (Library)
Buka terminal/command prompt, masuk ke direktori proyek, lalu instal semua pustaka yang diperlukan dengan perintah berikut:

```
pip install -r requirements.txt
```
3. Melatih Model Machine Learning
Sebelum menjalankan aplikasi web, kamu harus melatih modelnya terlebih dahulu menggunakan dataset yang ada di folder dataset/. Jalankan skrip training:


```
python train.py
```

  Skrip ini akan mengekstrak fitur statistik MFCC, membandingkan performa model SVM dan Random Forest, memilih model dengan akurasi tertinggi, lalu menyimpannya ke dalam file .pkl.

4. Menjalankan Aplikasi Streamlit
Setelah file model_mobil.pkl, scaler.pkl, dan label_encoder.pkl berhasil dibuat dan diletakkan di dalam folder training/, jalankan aplikasi web dengan perintah:

```
streamlit run app.py
```

  Aplikasi otomatis akan terbuka di browser default kamu (biasanya di alamat http://localhost:8501).

📊 Teknologi yang Digunakan
* Antarmuka Pengguna: Streamlit
* Pemrosesan Audio: Librosa, SoundFile
* Audio Synthesis (TTS): Edge-TTS (Microsoft Azure Cognitive Services)
* Machine Learning & Ekstraksi: Scikit-Learn (SVM & Random Forest), NumPy, Joblib
* Visualisasi Data: Matplotlib, Seaborn
