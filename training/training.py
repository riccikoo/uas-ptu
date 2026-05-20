import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. PENGATURAN PATH & CONFIGURASI
# ==========================================
DATASET_PATH = "dataset" 
CLASSES = ['lamborgini', 'porche', 'honda', 'mitsubishi', 'hyundai', 'wuling', 'toyota', 'ferrari', 'mazda', 'audi']

N_MFCC = 40          
MAX_PAD_LEN = 50

def extract_mfcc_cnn(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        return mfcc.T
    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return None

# ==========================================
# 2. PROSES MEMBACA DATASET
# ==========================================
X, y = [], []
print("Memulai ekstraksi fitur MFCC...")
for label in CLASSES:
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path): continue
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            features = extract_mfcc_cnn(os.path.join(folder_path, file_name))
            if features is not None:
                X.append(features)
                y.append(label)

X, y = np.array(X), np.array(y)
print(f"Selesai! {X.shape[0]} sampel terkumpul.")

# ==========================================
# 3. PREPROCESSING & SCALING
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Reshape untuk Scaling
n_features = X_train.shape[2]
X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape[0], MAX_PAD_LEN, n_features)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], MAX_PAD_LEN, n_features)

# ==========================================
# 4. TRAINING CNN 1D
# ==========================================
model = models.Sequential([
    layers.Input(shape=(MAX_PAD_LEN, N_MFCC)),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 5. SIMPAN MODEL, SCALER, & LABEL ENCODER
# ==========================================
model.save('model_mobil.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nModel, Scaler, dan Label Encoder berhasil disimpan!")

# Evaluasi Akhir
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Akurasi Akhir: {test_acc * 100:.2f}%")