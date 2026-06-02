import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import io
import asyncio
import edge_tts

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Aplikasi Audio AI", layout="centered")

st.title("🚗 Aplikasi Pengolah Suara & Audio (Edisi Lengkap)")
st.write("Gunakan tab di bawah ini untuk beralih fitur atau memanfaatkan integrasi sistem.")

# ==========================================
# LOAD MODEL & CONFIG (Disesuaikan untuk SVM/RF .pkl)
# ==========================================
N_MFCC = 40          

@st.cache_resource
def load_ml_components():
    model = joblib.load('training/model_mobil.pkl')
    scaler = joblib.load('training/scaler.pkl')
    le = joblib.load('training/label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_ml_components()
    model_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model_mobil.pkl, scaler.pkl, dan label_encoder.pkl ada di folder 'training'. Error: {e}")
    model_loaded = False

def extract_all_audio_data(file_path):
    try:
        # ==========================================
        # LOAD AUDIO
        # ==========================================
        audio, sample_rate = librosa.load(
            file_path,
            sr=16000
        )

        # ==========================================
        # TRIM SILENCE
        # Harus sama dengan train.py
        # ==========================================
        audio, _ = librosa.effects.trim(
            audio,
            top_db=30
        )

        # ==========================================
        # NORMALIZE AUDIO
        # Agar volume lebih stabil
        # ==========================================
        audio = librosa.util.normalize(audio)

        # ==========================================
        # PRE-EMPHASIS
        # Membuat karakter suara lebih jelas
        # ==========================================
        audio = librosa.effects.preemphasis(audio)

        # ==========================================
        # EKSTRAKSI MFCC
        # ==========================================
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # ==========================================
        # DELTA & DELTA-DELTA
        # Harus sama dengan train.py
        # ==========================================
        delta = librosa.feature.delta(mfcc)

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # ==========================================
        # GABUNGKAN FITUR
        # TOTAL:
        # 40*6 = 240 fitur
        # ==========================================
        features_vector = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.std(mfcc.T, axis=0),

            np.mean(delta.T, axis=0),
            np.std(delta.T, axis=0),

            np.mean(delta2.T, axis=0),
            np.std(delta2.T, axis=0)
        ])

        return features_vector, mfcc, sample_rate

    except Exception as e:
        st.error(f"Gagal memproses audio: {e}")
        return None, None, None

# Fungsi Asinkronus untuk Edge-TTS (Native Indonesia)
async def generate_edge_tts(text, voice_name, speed_percentage):
    # Format speed untuk edge-tts, contoh: "+0%", "-20%", "+50%"
    speed_str = f"{speed_percentage:+}%"
    communicate = edge_tts.Communicate(text, voice_name, rate=speed_str)
    
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
            
    audio_buffer.seek(0)
    return audio_buffer.getvalue()

if "asr_result" not in st.session_state:
    st.session_state.asr_result = ""

# ==========================================
# PEMBUATAN TAB
# ==========================================
tab1, tab2 = st.tabs(["🎙️ Live Audio Recognition (ASR)", "🔊 Text to Speech (TTS)"])

# ------------------------------------------
# FITUR TAB 1: LIVE AUDIO RECOGNITION
# ------------------------------------------
with tab1:
    st.header("Rekam & Klasifikasi Suara Mobil")
    st.write("Klik tombol mikrofon di bawah untuk mulai merekam suara.")
    
    recorded_audio = st.audio_input("Rekam suara di sini")
    
    if recorded_audio is not None:
        if st.button("Mulai Analisis Rekaman 🔎", key="btn_predict"):
            if not model_loaded:
                st.error("Model tidak tersedia. Periksa file model Anda.")
            else:
                with st.spinner("Sedang menganalisis karakteristik & grafik suara..."):
                    temp_path = "temp_recorded_audio.wav"
                    with open(temp_path, "wb") as f:
                        f.write(recorded_audio.getbuffer())
                    
                    features, mfcc_matrix, sr = extract_all_audio_data(temp_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if features is not None:
                        st.write("### 📊 Visualisasi Fitur MFCC Suara Anda")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        img = librosa.display.specshow(mfcc_matrix, sr=sr, x_axis='time', ax=ax, cmap='viridis')
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        ax.set_title('Karakteristik Spektrogram MFCC Audio')
                        st.pyplot(fig)
                        
                        features_ready = features.reshape(1, -1)
                        features_scaled = scaler.transform(features_ready)
                        
                        try:
                            predictions = model.predict_proba(features_scaled)
                            predicted_idx = np.argmax(predictions, axis=1)[0]
                            confidence = predictions[0][predicted_idx] * 100
                        except AttributeError:
                            predicted_idx = model.predict(features_scaled)[0]
                            confidence = None
                        
                        predicted_label = le.inverse_transform([predicted_idx])[0]
                        
                        st.success(f"### Hasil Analisis (ASR): **{predicted_label.upper()}**")
                        if confidence is not None:
                            st.info(f"🎯 **Confidence Score (Tingkat Keyakinan): {confidence:.2f}%**")
                        
                        st.session_state.asr_result = f"Model berhasil mendeteksi perintah suara untuk mobil {predicted_label}"
                        st.toast("Hasil prediksi berhasil diteruskan ke modul TTS!", icon="🚀")

# ------------------------------------------
# FITUR TAB 2: TEXT TO SPEECH (NATIVE INDONESIA VIA EDGE-TTS)
# ------------------------------------------
with tab2:
    st.header("Mengubah Teks Menjadi Suara (Bahasa Indonesia)")
    
    if st.session_state.asr_result != "":
        st.info(f"📥 **Teks otomatis dari hasil ASR ditemukan!**")
        default_text = st.session_state.asr_result
    else:
        default_text = "Halo! Silakan ketik perintah teks atau rekam suara di tab sebelah."

    # 1. Input Teks
    text_input = st.text_area("Masukkan Teks yang Ingin Diucapkan:", value=default_text)
    
    # 2. Opsi Pilihan Gender Suara (Laki-laki vs Perempuan - Native Indonesia)
    gender_choice = st.radio("Pilih Gender Suara Narator:", ["Perempuan (Gadis)", "Laki-laki (Ardi)"], horizontal=True)
    
    # Mapping model suara Microsoft Edge Native Indonesia
    # id-ID-ArdiNeural = Suara Cowok Indo, id-ID-GadisNeural = Suara Cewek Indo
    selected_voice = "id-ID-ArdiNeural" if "Laki-laki" in gender_choice else "id-ID-GadisNeural"
    
    # 3. Kecepatan Bicara (Slider persentase untuk kontrol akurat)
    speed_choice = st.select_slider("Pilih Kecepatan Bicara:", options=["Slow", "Normal", "Fast"], value="Normal")
    
    speed_map = {
        "Slow": -20,    # Lebih lambat 20%
        "Normal": 0,    # Kecepatan normal 0%
        "Fast": 25      # Lebih cepat 25%
    }
    target_speed = speed_map[speed_choice]
    
    if st.button("Ubah ke Suara 🔊", key="btn_tts"):
        if text_input.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            with st.spinner("Sedang memproses suara kualitas HD..."):
                try:
                    # Menjalankan fungsi asinkronus Edge-TTS di dalam Streamlit
                    audio_bytes = asyncio.run(generate_edge_tts(text_input, selected_voice, target_speed))
                    
                    st.success("Suara berhasil dibuat!")
                    
                    # 4. Output Suara
                    st.audio(audio_bytes, format='audio/mp3')
                    
                    # 5. Simpan / Ekspor Audio (.mp3)
                    st.download_button(
                        label="💾 Unduh/Simpan Hasil Audio (.mp3)",
                        data=audio_bytes,
                        file_name=f"tts_native_{gender_choice.split()[0].lower()}.mp3",
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan pada sistem Edge-TTS: {e}")
                    
    if st.session_state.asr_result != "":
        if st.button("Clear Teks Integrasi ASR 🗑️"):
            st.session_state.asr_result = ""
            st.rerun()