import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import time
import joblib
import pandas as pd
import altair as alt
from tensorflow.keras.models import load_model
from Classes import extract_mfcc_CNN

# Load CNN model and label encoder
model = load_model("speech_emotion_cnn_model.h5")
label_encoder = joblib.load("speech_emotion_label_encoder_old.pkl")

# Audio configuration
SAMPLE_RATE = 22050
DURATION = 0.5  # Fast updates

# Streamlit interface setup
st.set_page_config(page_title="Smoothed SER Histogram", layout="centered")
st.title("🎙️ Real-Time Emotion Recognition with Fixed-Axis Histogram")

# Session state setup
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "smoothed_scores" not in st.session_state:
    st.session_state.smoothed_scores = None

# Fixed emotion labels
fixed_emotions = list(label_encoder.classes_)

# UI buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
with col2:
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False

# Status and chart placeholders
status_display = st.empty()
emotion_display = st.empty()

# Persistent listening indicator
status_display.markdown("🎧 **Listening...**")

# Emotion recognition loop
while st.session_state.is_listening:
    try:
        # Record audio sample
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = recording.flatten()

        # Extract features
        mfcc = extract_mfcc_CNN(audio, SAMPLE_RATE)

        if not np.isfinite(mfcc).all() or np.max(np.abs(audio)) < 0.01:
            emotion_display.warning("🔇 No valid audio detected. Please speak clearly.")
            continue

        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        probabilities = model.predict(mfcc)[0]
        emotions = label_encoder.inverse_transform(np.arange(len(probabilities)))
        emotion_scores = dict(zip(emotions, probabilities))

        # Apply EMA smoothing
        alpha = 0.3
        new_scores = pd.Series(emotion_scores)
        if st.session_state.smoothed_scores is None:
            st.session_state.smoothed_scores = new_scores
        else:
            st.session_state.smoothed_scores = (
                alpha * new_scores + (1 - alpha) * st.session_state.smoothed_scores
            )

        # Fill missing emotions with 0
        for emotion in fixed_emotions:
            if emotion not in st.session_state.smoothed_scores:
                st.session_state.smoothed_scores[emotion] = 0.0

        # Build chart data
        emotion_df = pd.DataFrame({
            "Emotion": fixed_emotions,
            "Confidence (%)": [st.session_state.smoothed_scores[emotion] * 100 for emotion in fixed_emotions]
        })

        # Display histogram
        chart = alt.Chart(emotion_df).mark_bar().encode(
            x=alt.X("Emotion", sort=fixed_emotions),
            y=alt.Y("Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Emotion", "Confidence (%)"]
        ).properties(
            width=600,
            height=400
        )

        emotion_display.subheader("🧠 Smoothed Emotion Confidence")
        emotion_display.altair_chart(chart, use_container_width=True)

    except Exception as e:
        emotion_display.error(f"❌ Error: {e}")

    time.sleep(0.1)
