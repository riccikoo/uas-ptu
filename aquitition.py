import os
import time
import io  # Tambahkan import ini di bagian atas
import streamlit as st
from audiorecorder import audiorecorder  

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Audio Dataset Collector", page_icon="🎙️")
st.title("🎙️ Audio Dataset Collector")
st.write("Gunakan aplikasi ini untuk merekam suara dan menyimpannya langsung ke folder dataset.")

# 2. Tentukan Direktori Utama Dataset
DATASET_DIR = "dataset"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)


available_classes = sorted([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))])

# 4. Antarmuka Pengguna (UI)
st.subheader("Pilih Kelas/Label")
selected_class = st.selectbox("Suara ini termasuk dalam kelas apa?", available_classes)

st.subheader("Mulai Rekam")
st.write("Klik tombol di bawah untuk mulai merekam. Klik lagi untuk berhenti.")

audio = audiorecorder("Mulai Rekam ⏺️", "Berhenti ⏹️")

# MEMPERBAIKI ERROR DI SINI
if len(audio) > 0:
    # Mengonversi AudioSegment ke format bytes WAV agar bisa diputar oleh st.audio
    audio_buffer = io.BytesIO()
    audio.export(audio_buffer, format="wav")
    
    # Putar audio di browser
    st.audio(audio_buffer.getvalue(), format="audio/wav") 
    
    st.subheader("Simpan Hasil Rekam")
    filename_input = st.text_input("Nama File (Kosongkan untuk otomatis menggunakan waktu):")
    
    if st.button("Simpan ke Dataset 💾", type="primary"):
        if filename_input.strip() == "":
            timestamp = int(time.time())
            filename = f"{selected_class}_{timestamp}.wav"
        else:
            filename = filename_input.strip() if filename_input.endswith(".wav") else f"{filename_input.strip()}.wav"
            
        target_path = os.path.join(DATASET_DIR, selected_class, filename)
        
        try:
            # Menyimpan file ke dalam folder menggunakan fungsi bawaan pydub .export()
            with open(target_path, "wb") as f:
                audio.export(f, format="wav")
            st.success(f"🔥 Berhasil disimpan di: `{target_path}`")
        except Exception as e:
            st.error(f"Gagal menyimpan file: {e}")