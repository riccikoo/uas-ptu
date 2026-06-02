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
# CONFIG
# ==========================================
DATASET_PATH = "dataset"

CLASSES = [
    'lamborgini',
    'porche',
    'honda',
    'mitsubishi',
    'hyundai',
    'wuling',
    'toyota',
    'ferrari',
    'mazda',
    'audi'
]

N_MFCC = 40
SAMPLE_RATE = 16000

# ==========================================
# AUGMENTATION
# ==========================================
def add_noise(audio, noise_factor=0.008):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift(audio, sample_rate, n_steps=2):
    return librosa.effects.pitch_shift(
        y=audio,
        sr=sample_rate,
        n_steps=n_steps
    )

def speed_tuning(audio, speed_factor=1.1):
    return librosa.effects.time_stretch(
        y=audio,
        rate=speed_factor
    )

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(audio, sample_rate):

    try:
        # Trim silence
        audio, _ = librosa.effects.trim(
            audio,
            top_db=30
        )

        # Normalize
        audio = librosa.util.normalize(audio)

        # Pre-emphasis
        audio = librosa.effects.preemphasis(audio)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=N_MFCC
        )

        # Delta
        delta = librosa.feature.delta(mfcc)

        # Delta Delta
        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # Combine features
        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.std(mfcc.T, axis=0),

            np.mean(delta.T, axis=0),
            np.std(delta.T, axis=0),

            np.mean(delta2.T, axis=0),
            np.std(delta2.T, axis=0)
        ])

        return features

    except:
        return None

# ==========================================
# LOAD DATASET
# ==========================================
file_paths = []
labels = []

print("Mencari dataset audio...")

for label in CLASSES:

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder_path):
        continue

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(
            ('.wav', '.mp3', '.ogg', '.flac')
        ):

            file_paths.append(
                os.path.join(folder_path, file_name)
            )

            labels.append(label)

if len(file_paths) == 0:
    raise ValueError("Dataset kosong!")

# ==========================================
# LABEL ENCODING
# ==========================================
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ==========================================
# SPLIT DATA
# ==========================================
train_paths, test_paths, y_train_labels, y_test_labels = train_test_split(
    file_paths,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# ==========================================
# TRAIN EXTRACTION
# ==========================================
X_train = []
y_train = []

print(f"\n[1/2] Processing training data...")

for file_path, label_code in zip(train_paths, y_train_labels):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        # ORIGINAL
        feat_orig = extract_features(audio, sample_rate)

        if feat_orig is not None:
            X_train.append(feat_orig)
            y_train.append(label_code)

        # NOISE
        audio_noise = add_noise(audio)

        feat_noise = extract_features(
            audio_noise,
            sample_rate
        )

        if feat_noise is not None:
            X_train.append(feat_noise)
            y_train.append(label_code)

        # PITCH
        audio_pitch = pitch_shift(
            audio,
            sample_rate,
            n_steps=2
        )

        feat_pitch = extract_features(
            audio_pitch,
            sample_rate
        )

        if feat_pitch is not None:
            X_train.append(feat_pitch)
            y_train.append(label_code)

        # SPEED
        audio_speed = speed_tuning(
            audio,
            speed_factor=1.1
        )

        feat_speed = extract_features(
            audio_speed,
            sample_rate
        )

        if feat_speed is not None:
            X_train.append(feat_speed)
            y_train.append(label_code)

    except Exception as e:
        print(f"Error: {e}")

# ==========================================
# TEST EXTRACTION
# ==========================================
X_test = []
y_test = []

print(f"[2/2] Processing testing data...")

for file_path, label_code in zip(test_paths, y_test_labels):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        feat_test = extract_features(
            audio,
            sample_rate
        )

        if feat_test is not None:
            X_test.append(feat_test)
            y_test.append(label_code)

    except Exception as e:
        print(f"Error: {e}")

# ==========================================
# TO NUMPY
# ==========================================
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("\nJumlah Train:", X_train.shape)
print("Jumlah Test :", X_test.shape)

# ==========================================
# SCALING
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# MODEL TRAINING
# ==========================================
print("\nMelatih model...")

# SVM
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

y_pred_svm = svm_model.predict(X_test_scaled)

acc_svm = accuracy_score(
    y_test,
    y_pred_svm
)

# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

y_pred_rf = rf_model.predict(X_test_scaled)

acc_rf = accuracy_score(
    y_test,
    y_pred_rf
)

# ==========================================
# RESULT
# ==========================================
print("\n" + "="*40)

print(f"SVM Accuracy          : {acc_svm*100:.2f}%")
print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")

print("="*40)

# ==========================================
# BEST MODEL
# ==========================================
if acc_svm > acc_rf:

    best_model = svm_model
    best_name = "SVM"
    y_pred_best = y_pred_svm

else:

    best_model = rf_model
    best_name = "Random Forest"
    y_pred_best = y_pred_rf

print(f"\nBest Model: {best_name}")

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(best_model, 'model_mobil.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nModel berhasil disimpan!")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred_best,
        target_names=le.classes_
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(10, 8))

cm = confusion_matrix(
    y_test,
    y_pred_best
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title(f'Confusion Matrix ({best_name})')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
