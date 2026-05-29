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
DATASET_PATH = "../dataset" 
CLASSES = ['lamborgini', 'porche', 'honda', 'mitsubishi', 'hyundai', 'wuling', 'toyota', 'ferrari', 'mazda', 'audi']

N_MFCC = 40          

def extract_mfcc_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        # Ambil rata-rata (mean) dan standar deviasi dari MFCC agar menjadi vektor 1D tetap
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # Gabungkan mean dan std menjadi satu fitur tunggal (total 80 fitur)
        features = np.hstack((mfcc_mean, mfcc_std))
        return features
    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return None

# ==========================================
# 2. PROSES MEMBACA DATASET
# ==========================================
X, y = [], []
print("Memulai ekstraksi fitur MFCC (Statistik)...")
for label in CLASSES:
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path): continue
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            features = extract_mfcc_features(os.path.join(folder_path, file_name))
            if features is not None:
                X.append(features)
                y.append(label)

X, y = np.array(X), np.array(y)
if X.shape[0] == 0:
    raise ValueError("Error: 0 sampel terkumpul! Periksa kembali path dataset kamu.")
print(f"Selesai! {X.shape[0]} sampel terkumpul.")

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
# Model 1: SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n" + "="*40)
print(f"Hasil Akurasi SVM          : {acc_svm * 100:.2f}%")
print(f"Hasil Akurasi Random Forest : {acc_rf * 100:.2f}%")
print("="*40)

# Pilih model terbaik untuk disimpan
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
# 5. SIMPAN MODEL (Gunakan format .pkl karena bukan Keras/TF lagi)
# ==========================================
# CATATAN: Karena model disimpan jadi .pkl, script GUI nanti perlu sedikit penyesuaian!
joblib.dump(best_model, 'model_mobil.pkl') 
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nClassification Report Model Terbaik:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix ({best_name})')
plt.xlabel('Prediksi Model')
plt.ylabel('Jawaban Asli')
plt.tight_layout()
plt.show()