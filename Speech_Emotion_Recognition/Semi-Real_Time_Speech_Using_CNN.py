import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import time
import joblib
from tensorflow.keras.models import load_model
from Classes import extract_mfcc_CNN

model = load_model("speech_emotion_cnn_model.h5")
label_encoder = joblib.load("speech_emotion_label_encoder_old.pkl")

SAMPLE_RATE = 22050
DURATION = 0.5  # Faster update with shorter audio chunks

st.set_page_config(page_title="Continuous SER", layout="centered")
st.title("🎙️ Continuous Speech Emotion Recognition")

if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
with col2:
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False

emotion_display = st.empty()

# Start live recognition loop
while st.session_state.is_listening:
    with st.spinner("Listening..."):
        try:
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio = recording.flatten()

            mfcc = extract_mfcc_CNN(audio, SAMPLE_RATE)

            if not np.isfinite(mfcc).all() or np.all(mfcc == 0) or np.max(np.abs(audio)) < 0.01:
                emotion = "Audio not picked up — please speak clearly or check your mic."
            else:
                mfcc = mfcc[np.newaxis, ..., np.newaxis]
                prediction = np.argmax(model.predict(mfcc), axis=1)[0]
                emotion = label_encoder.inverse_transform([prediction])[0]

            emotion_display.success(f"🧠 Detected Emotion: **{emotion}**")


        except Exception as e:
            st.error(f"❌ Error: {e}")

    time.sleep(0.1)
