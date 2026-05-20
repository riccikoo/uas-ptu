import os
import numpy as np
import librosa
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
X = []
y = []

print("Memulai ekstraksi fitur MFCC untuk CNN 1D...")
for label in CLASSES:
    folder_path = os.path.join(DATASET_PATH, label)
    
    if not os.path.exists(folder_path):
        print(f"Peringatan: Folder '{label}' tidak ditemukan. Dilewati.")
        continue
        
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            features = extract_mfcc_cnn(file_path)
            
            if features is not None:
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Selesai! Berhasil mengumpulkan {X.shape[0]} sampel audio.")
print(f"Shape awal data X: {X.shape}")

# ==========================================
# 3. PREPROCESSING DATA
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

num_train, time_steps, n_features = X_train.shape
num_test = X_test.shape[0]

X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d).reshape(num_train, time_steps, n_features)
X_test_scaled = scaler.transform(X_test_2d).reshape(num_test, time_steps, n_features)

# ==========================================
# 4. MEMBANGUN ARSITEKTUR CNN 1D
# ==========================================
print("\nMembuat model CNN 1D...")
model = models.Sequential([
    # Input layer menerima matriks ukuran (50, 40)
    layers.Input(shape=(MAX_PAD_LEN, N_MFCC)),
    
    # Blok Konvolusi 1
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Blok Konvolusi 2
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Global Average Pooling mengubah data 3D menjadi 2D sebelum masuk ke Dense layer
    layers.GlobalAveragePooling1D(),
    
    # Fully Connected Layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    
    # Output Layer (10 kelas menggunakan softmax)
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. TRAINING MODEL
# ==========================================
print("\nMemulai proses training CNN 1D...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

print("Training selesai!")

# ==========================================
# 6. EVALUASI MODEL
# ==========================================
print("\n================ HASIL EVALUASI CNN 1D ================")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Akurasi Model pada Data Test: {test_acc * 100:.2f}%\n")

# Prediksi kelas
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Laporan Klasifikasi per Kelas:")
print(classification_report(y_test, y_pred, target_names=le.classes_))