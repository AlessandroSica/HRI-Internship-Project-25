import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import time
import joblib
from tensorflow.keras.models import load_model
from Classes import extract_mfcc_try


# 🔹 Load the trained CNN model and label encoder
model = load_model("speech_emotion_cnn_model.h5")
label_encoder = joblib.load("speech_emotion_label_encoder.pkl")

# 🔹 Audio recording settings
SAMPLE_RATE = 22050
DURATION = 3  # seconds

# 🔹 Set up Streamlit UI
st.set_page_config(page_title="Continuous SER", layout="centered")
st.title("🎙️ Continuous Speech Emotion Recognition")

# 🔹 Initialize session state
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

# 🔹 Controls to start/stop continuous recognition
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
with col2:
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False

# 🔁 Recognition Loop
while st.session_state.is_listening:
    with st.spinner("Listening for 3 seconds..."):
        try:
            # 🎙️ Record mic audio
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()

            audio = recording.flatten()
            mfcc = extract_mfcc_try(audio, SAMPLE_RATE)

            if not np.isfinite(mfcc).all() or np.all(mfcc == 0) or np.max(np.abs(audio)) < 0.01:
                emotion = "Audio not picked up — please speak clearly or check your mic."
            else:
                mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Reshape for CNN: (1, height, width, 1)
                prediction = np.argmax(model.predict(mfcc), axis=1)[0]
                emotion = label_encoder.inverse_transform([prediction])[0]

            st.session_state.emotion_log.append(emotion)
            st.success(f"🧠 Detected Emotion: **{emotion}**")

        except Exception as e:
            st.session_state.emotion_log.append(f"Error: {e}")
            st.error(f"❌ Error: {e}")

    time.sleep(0.5)

# 📋 Display log of recent emotions
st.markdown("#### 🧾 Emotion Log")
if st.session_state.emotion_log:
    for i, emo in enumerate(reversed(st.session_state.emotion_log[-20:]), 1):
        st.markdown(f"{len(st.session_state.emotion_log) - i + 1}. **{emo}**")
else:
    st.info("No emotions detected yet.")
