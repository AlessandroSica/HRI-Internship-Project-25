import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -----------------------------
# 🎙️ Load and process audio data
# -----------------------------
dataset_path = 'HRI-Internship-Project-25/Speech_Emotion_Recognition/Dataset_Speech_Emotion_Recognition'

def extract_features(file_path, n_mfcc=48, max_frames=44):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc

X = []
y = []

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

for root, dirs, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(root, file_name)
            emotion_code = file_name.split('-')[2]
            if emotion_code in emotion_map:
                mfcc = extract_features(file_path)
                X.append(mfcc)
                y.append(emotion_map[emotion_code])

X = np.array(X)
X = X[..., np.newaxis]  # shape becomes (samples, 48, 44, 1)
y = np.array(y)

# -----------------------------
# 🔢 Encode labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# -----------------------------
# 🧪 Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)

# -----------------------------
# 🧱 Define the CNN architecture
# -----------------------------
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 44, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# -----------------------------
# 🏋️ Train the model
# -----------------------------
epochs = 100
batch_size = 32

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size
)

# -----------------------------
# 💾 Save the model and encoder
# -----------------------------
model.save("speech_emotion_cnn_model.h5")
joblib.dump(label_encoder, "speech_emotion_label_encoder.pkl")

# -----------------------------
# 📊 Evaluate the model
# -----------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
print("CNN Accuracy:", acc)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - CNN Speech Emotion Recognition")
plt.show()
