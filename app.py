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
st.set_page_config(
    page_title="AI Voice Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg-primary: #0B0B0D;
    --bg-secondary: #111113;
    --bg-tertiary: #18181B;
    --bg-card: #141416;
    --accent-gold: #C8A96B;
    --accent-light: #E7D3A7;
    --accent-highlight: #F5E6C8;
    --accent-muted: #8B7355;
    --text-primary: #FFFFFF;
    --text-secondary: #D1D5DB;
    --text-muted: #9CA3AF;
    --border: rgba(200, 169, 107, 0.15);
    --border-hover: rgba(200, 169, 107, 0.35);
    --glass: rgba(255, 255, 255, 0.03);
    --glass-hover: rgba(255, 255, 255, 0.06);
    --shadow-gold: 0 0 40px rgba(200, 169, 107, 0.08);
    --shadow-deep: 0 24px 64px rgba(0, 0, 0, 0.6);
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --radius: 16px;
    --radius-sm: 10px;
    --radius-lg: 24px;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(200, 169, 107, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 80%, rgba(200, 169, 107, 0.03) 0%, transparent 50%) !important;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-muted); border-radius: 4px; }

.hero-wrapper {
    text-align: center;
    padding: 4rem 2rem 3rem;
    position: relative;
    overflow: hidden;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 2rem;
    animation: fadeSlideUp 0.6s ease both;
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: clamp(3rem, 7vw, 5.5rem) !important;
    font-weight: 800 !important;
    line-height: 1.0 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 1.5rem !important;
    animation: fadeSlideUp 0.7s ease 0.1s both;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--accent-light) 50%, var(--accent-gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: var(--text-muted) !important;
    max-width: 520px;
    margin: 0 auto 2.5rem !important;
    line-height: 1.75 !important;
    font-weight: 300 !important;
    animation: fadeSlideUp 0.8s ease 0.2s both;
}

.hero-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
    margin: 0 auto 3rem;
    animation: fadeSlideUp 0.9s ease 0.3s both;
}

.section-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: var(--font-display) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}

.section-desc {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200, 169, 107, 0.3), transparent);
}

.card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-gold);
    transform: translateY(-1px);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-muted)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(200, 169, 107, 0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(200, 169, 107, 0.35) !important;
    filter: brightness(1.05) !important;
}

.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.25s ease !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: rgba(200, 169, 107, 0.4) !important;
    box-shadow: 0 0 0 3px rgba(200, 169, 107, 0.08) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder { color: var(--text-muted) !important; }

.stTextArea label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

.stRadio label { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }

.stRadio [data-testid="stMarkdownContainer"] { color: var(--text-secondary) !important; }

.stSelectSlider > label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stSelectSlider [data-testid="stTickBarMin"],
.stSelectSlider [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
}

.stSpinner { color: var(--accent-gold) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent-gold) !important; }

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
}

[data-testid="column"] { padding: 0 0.5rem !important; }

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

.animate-in { animation: fadeSlideUp 0.5s ease both; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">AI-Powered Audio Intelligence</div>
    <h1 class="hero-title">AI Voice Studio</h1>
    <p class="hero-subtitle">Transform speech into intelligence and text into natural voice with AI-powered audio technology.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Module 01</div>
        <div class="section-title">Live Audio Recognition</div>
        <div class="section-desc">Rekam suara dan deteksi pola menggunakan model klasifikasi audio.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div style="font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.75rem;">Rekam & Analisis Audio</div>
        <div style="font-size: 0.95rem; color: var(--text-muted); line-height: 1.7; margin-bottom: 1.25rem;">Klik tombol mikrofon di bawah untuk merekam suara dan dapatkan hasil prediksi dari model.</div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Module 02</div>
        <div class="section-title">Text to Speech</div>
        <div class="section-desc">Konversi teks Bahasa Indonesia menjadi audio menggunakan sintetis suara berkualitas tinggi.</div>
    </div>
    """, unsafe_allow_html=True)

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