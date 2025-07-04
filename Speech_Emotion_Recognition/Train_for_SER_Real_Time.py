import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -----------------------------
# 📁 Configuration
# -----------------------------
DATA_PATH = "HRI-Internship-Project-25/Speech_Emotion_Recognition/Dataset_Speech_Emotion_Recognition"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.5  # seconds
N_MFCC = 40
MAX_FRAMES = 44  # ~0.5s frames
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised"
}

# -----------------------------
# 🎙️ Audio Preprocessing
# -----------------------------
def extract_mfcc_segments(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    samples_per_segment = int(sr * SEGMENT_DURATION)
    segments = [y[i:i+samples_per_segment] for i in range(0, len(y)-samples_per_segment+1, samples_per_segment)]
    features = []

    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        stacked = np.concatenate([mfcc, delta, delta2], axis=0)  # Shape: (120, T)

        if stacked.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - stacked.shape[1]
            stacked = np.pad(stacked, ((0,0), (0,pad_width)), mode='constant')
        else:
            stacked = stacked[:, :MAX_FRAMES]

        features.append(stacked)

    return features

# -----------------------------
# 🧪 Load Dataset
# -----------------------------
X, y = [], []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion_label = emotion_map.get(emotion_code)
            if emotion_label is None:
                continue

            file_path = os.path.join(root, file)
            try:
                segments = extract_mfcc_segments(file_path)
                for segment in segments:
                    X.append(segment)
                    y.append(emotion_label)
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# -----------------------------
# 🧠 Sanity Check
# -----------------------------
print(f"✅ Loaded {len(X)} samples.")
if len(X) == 0 or len(y) == 0:
    raise ValueError("No audio samples or labels loaded. Please check your dataset path and filename format.")

X = np.array(X)
X = X[..., np.newaxis]  # Shape: (samples, 120, 44, 1)
y = np.array(y)

# -----------------------------
# 🔢 Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# -----------------------------
# 🧠 Train/Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42)

# -----------------------------
# 🏗️ Model Architecture
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(120, MAX_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 🏋️ Training
# -----------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5),
    ModelCheckpoint("speech_emotion_cnn_model_SER.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# -----------------------------
# 💾 Save Label Encoder
# -----------------------------
joblib.dump(label_encoder, "speech_emotion_label_encoder_SER_Real_Time.pkl")
print("🎉 Training complete. Model and encoder saved.")
