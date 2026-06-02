import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PENGATURAN PATH & CONFIGURASI
# ==========================================
DATASET_PATH = "dataset" 
CLASSES = ['lamborgini', 'porche', 'honda', 'mitsubishi', 'hyundai', 'wuling', 'toyota', 'ferrari', 'mazda', 'audi']

N_MFCC = 40

# --- FUNGSI-FUNGSI AUGMENTASI AUDIO ---
def add_noise(audio, noise_factor=0.005):
    """Menambahkan derau/noise tipis ke audio"""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def pitch_shift(audio, sample_rate, n_steps=2):
    """Mengubah pitch/nada suara (misal: lebih ngebass atau melengking)"""
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=n_steps)

def speed_tuning(audio, speed_factor=1.1):
    """Mempercepat atau memperlambat durasi suara"""
    return librosa.effects.time_stretch(y=audio, rate=speed_factor)


def extract_mfcc_from_array(audio, sample_rate):
    """Ekstraksi fitur dari array audio langsung (bukan dari file path)"""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        features = np.hstack((mfcc_mean, mfcc_std))
        return features
    except Exception as e:
        return None

# ==========================================
# 2. PROSES MEMBACA DATASET + AUGMENTASI
# ==========================================
X, y = [], []
print("Memulai ekstraksi fitur MFCC dengan Data Augmentasi...")

for label in CLASSES:
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path): continue
    
    print(f"Memproses kelas: {label}...")
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # 1. Load Audio Asli
                audio, sample_rate = librosa.load(file_path, sr=None)
                
                # 2. Ekstrak Fitur dari Audio Asli
                feat_orig = extract_mfcc_from_array(audio, sample_rate)
                if feat_orig is not None:
                    X.append(feat_orig)
                    y.append(label)
                
                # 3. Proses Augmentasi 1: Tambah Noise
                audio_noise = add_noise(audio, noise_factor=0.005)
                feat_noise = extract_mfcc_from_array(audio_noise, sample_rate)
                if feat_noise is not None:
                    X.append(feat_noise)
                    y.append(label)
                
                # 4. Proses Augmentasi 2: Pitch Shift (Dinaikkan nadanya)
                audio_pitch = pitch_shift(audio, sample_rate, n_steps=2)
                feat_pitch = extract_mfcc_from_array(audio_pitch, sample_rate)
                if feat_pitch is not None:
                    X.append(feat_pitch)
                    y.append(label)

                # 5. Proses Augmentasi 3: Speed Tuning (Dipercepat sedikit)
                audio_speed = speed_tuning(audio, speed_factor=1.1)
                feat_speed = extract_mfcc_from_array(audio_speed, sample_rate)
                if feat_speed is not None:
                    X.append(feat_speed)
                    y.append(label)
                    
            except Exception as e:
                print(f"Gagal memproses {file_name}: {e}")

X, y = np.array(X), np.array(y)
if X.shape[0] == 0:
    raise ValueError("Error: 0 sampel terkumpul! Periksa kembali path dataset kamu.")
print(f"Selesai! Berkat augmentasi, sekarang ada {X.shape[0]} sampel terkumpul (4x lipat lebih banyak!).")

# ==========================================
# 3. PREPROCESSING & SPLITTING
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. TRAINING & EVALUASI DUA MODEL
# ==========================================
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n" + "="*40)
print(f"Hasil Akurasi SVM          : {acc_svm * 100:.2f}%")
print(f"Hasil Akurasi Random Forest : {acc_rf * 100:.2f}%")
print("="*40)

if acc_svm > acc_rf:
    best_model = svm_model
    best_name = "SVM"
    y_pred_best = y_pred_svm
else:
    best_model = rf_model
    best_name = "Random Forest"
    y_pred_best = y_pred_rf

print(f"\nMenyimpan model terbaik: {best_name}")

# ==========================================
# 5. SIMPAN MODEL
# ==========================================
joblib.dump(best_model, 'model_mobil.pkl') 
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nClassification Report Model Terbaik:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix ({best_name}) - With Augmentation')
plt.xlabel('Prediksi Model')
plt.ylabel('Jawaban Asli')
plt.tight_layout()
plt.show()