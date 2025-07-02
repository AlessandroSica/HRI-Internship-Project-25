import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# 📍 Dataset and Feature Extractor
dataset_path = "HRI-Internship-Project-25/Speech_Emotion_Recognition/Dataset_Speech_Emotion_Recognition"
from Extra_Classes import extract_all_features

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

# 🎙️ Load and Pool Features
X = []
y = []

print("🔍 Extracting features from audio files...")
for root, _, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            if emotion_code in emotion_map:
                file_path = os.path.join(root, file)
                signal, sr = librosa.load(file_path, sr=16000)
                features = extract_all_features(signal, sr=sr)
                
                # Combine mean + std pooling → [264] vector
                pooled = np.concatenate([features.mean(axis=0), features.std(axis=0)])
                X.append(pooled)
                y.append(emotion_map[emotion_code])

X = np.array(X)
y = np.array(y)
print(f"✅ {X.shape[0]} samples extracted | Feature dim: {X.shape[1]}")

# 🔢 Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# ⚖️ Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "feature_scaler.pkl")

# 🚀 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
)

# 🧱 Define Feedforward Neural Network
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ⏳ Training Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
]

# 🏋️ Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# 💾 Save Final Model and Encoder
model.save("speech_emotion_ffnn_model.h5")
joblib.dump(label_encoder, "speech_emotion_label_encoder.pkl")
print("📦 Model, encoder, and scaler saved!")

# 📊 Evaluate Performance
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy: {acc:.2%}")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
