import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import io
from streamlit_mic_recorder import mic_recorder

# ==========================================
# 1. KONFIGURASI & LOAD ARTIFAK
# ==========================================
N_MFCC = 40
MAX_PAD_LEN = 50
CLASSES = ['lamborgini', 'porche', 'honda', 'mitsubishi', 'hyundai', 'wuling', 'toyota', 'ferrari', 'mazda', 'audi']

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model_mobil.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except:
    st.error("File model_mobil.h5 atau scaler.pkl tidak ditemukan. Jalankan training dulu ya!")

# ==========================================
# 2. FUNGSI PEMROSESAN AUDIO
# ==========================================
def process_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    
    mfcc = mfcc.T 
    mfcc_scaled = scaler.transform(mfcc)
    
    return mfcc_scaled.reshape(1, MAX_PAD_LEN, N_MFCC)

# ==========================================
# 3. INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="Car Brand Voice Detector", page_icon="🚗")

st.title("Detektor Suara Merek Mobil")
st.write("Klik tombol di bawah dan sebutkan salah satu merek mobil (Contoh: 'Toyota' atau 'Ferrari')")

# Komponen Perekam
audio_record = mic_recorder(
    start_prompt="Mulai Rekam 🎤",
    stop_prompt="Berhenti & Kirim ⏹️",
    key='recorder'
)

if audio_record:
    # Tampilkan audio yang baru direkam agar user bisa dengar balik
    st.audio(audio_record['bytes'])
    
    with st.spinner('Sedang mengenali suara...'):
        try:
            # Proses dan Prediksi
            processed_data = process_audio(audio_record['bytes'])
            prediction = model.predict(processed_data)
            
            # Ambil hasil tertinggi
            predicted_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Tampilan Hasil
            st.success(f"### Hasil Prediksi: **{CLASSES[predicted_idx].upper()}**")
            st.progress(int(confidence))
            st.write(f"Tingkat Keyakinan: {confidence:.2f}%")
            
            # Opsional: Tampilkan tabel probabilitas
            with st.expander("Lihat detail probabilitas semua merek"):
                for idx, name in enumerate(CLASSES):
                    st.write(f"{name}: {prediction[0][idx]*100:.2f}%")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses suara: {e}")

st.divider()
st.caption("Pastikan suara terdengar jelas dan durasi sekitar 1-2 detik.")