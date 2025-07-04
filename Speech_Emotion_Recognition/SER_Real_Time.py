# 🎛️ Streamlit: Web app framework
import streamlit as st
# 🎧 Sounddevice: Microphone input
import sounddevice as sd
# 📐 NumPy: Numerical processing
import numpy as np
# 🎼 Librosa: Audio feature extraction
import librosa
# ⏱️ Time: Sleep delay
import time
# 💾 Joblib: Load label encoder
import joblib
# 📊 Pandas: Tabular data
import pandas as pd
# 📈 Altair: Visualization
import altair as alt
# 🤖 TensorFlow Keras: Load trained emotion model
from tensorflow.keras.models import load_model

# 🔍 Feature extractor: MFCC + delta + delta-delta, padded to expected shape
def extract_mfcc_CNN(y, sr, n_mfcc=40, max_frames=44, target_features=120):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)                     # Extract MFCCs
    delta = librosa.feature.delta(mfcc)                                        # First derivative
    delta2 = librosa.feature.delta(mfcc, order=2)                              # Second derivative
    combined = np.concatenate([mfcc, delta, delta2], axis=0)                   # Stack features: shape (120, T)

    # Ensure consistent frame count
    if combined.shape[1] < max_frames:
        combined = np.pad(combined, ((0, 0), (0, max_frames - combined.shape[1])), mode="constant")
    else:
        combined = combined[:, :max_frames]

    # Ensure consistent number of features
    if combined.shape[0] < target_features:
        combined = np.pad(combined, ((0, target_features - combined.shape[0]), (0, 0)), mode="constant")
    elif combined.shape[0] > target_features:
        combined = combined[:target_features, :]

    return combined

# 🧠 Load trained model and label encoder
model = load_model("speech_emotion_cnn_model_SER.h5")
label_encoder = joblib.load("speech_emotion_label_encoder_SER_Real_Time.pkl")

# 🎧 Audio settings
SAMPLE_RATE = 22050
DURATION = 0.5  # seconds per clip

# 🖼️ Streamlit UI setup
st.set_page_config(page_title="Continuous SER", layout="centered")
st.title("🎙️ Real-Time Speech Emotion Recognition with Histogram")

# 🚦 Session state: is the system listening?
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

# 🎭 Define fixed emotion class order
fixed_emotions = list(label_encoder.classes_)

# 🟢 UI buttons to control listening
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
with col2:
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False

# 🎯 Display area for status and histogram
status_display = st.empty()
emotion_display = st.empty()
status_display.markdown("🎧 **Listening...**")

# 🔄 Main recognition loop
while st.session_state.is_listening:
    try:
        # 🎙️ Record audio from microphone
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio = recording.flatten()

        # 🧠 Extract features
        mfcc = extract_mfcc_CNN(audio, SAMPLE_RATE)

        # 🔇 Handle low-volume or invalid input
        if not np.isfinite(mfcc).all() or np.max(np.abs(audio)) < 0.01:
            emotion_display.warning("🔇 No valid audio detected. Please speak clearly.")
            continue

        # 🔮 Predict emotion
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Reshape to (1, 120, 44, 1)
        probabilities = model.predict(mfcc)[0]    # Output: confidence scores
        emotions = label_encoder.inverse_transform(np.arange(len(probabilities)))
        emotion_scores = dict(zip(emotions, probabilities))  # Map scores to emotion names

        # 📊 Build histogram data
        emotion_df = pd.DataFrame({
            "Emotion": fixed_emotions,
            "Confidence (%)": [
                emotion_scores.get(emotion, 0.0) * 100 for emotion in fixed_emotions
            ]
        })

        # 🧾 Create Altair bar chart
        chart = alt.Chart(emotion_df).mark_bar().encode(
            x=alt.X("Emotion", sort=fixed_emotions),
            y=alt.Y("Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Emotion", "Confidence (%)"]
        ).properties(
            width=600,
            height=400
        ).configure_view(fill="#e0e4ea")

        # 🧠 Display chart
        emotion_display.subheader("🧠 Raw Emotion Confidence")
        emotion_display.altair_chart(chart, use_container_width=True)

    except Exception as e:
        emotion_display.error(f"❌ Error: {e}")

    time.sleep(0.1)  # Wait before next recording
