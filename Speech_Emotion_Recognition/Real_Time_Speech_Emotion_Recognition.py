# Import required libraries
import streamlit as st                 # Streamlit is used to build interactive web apps in Python
import sounddevice as sd               # sounddevice allows recording audio from your microphone
import soundfile as sf                 # soundfile helps save audio recordings as .wav files
import numpy as np                     # NumPy is used for numerical computations, like processing audio arrays
import librosa                         # Librosa is a library for analyzing audio and extracting features (like MFCCs)
import joblib                          # Joblib is used to load the saved machine learning model
import time                            # Time is used for pausing between recordings (simulate intervals)
from Classes import extract_mfcc       # This is your custom module/function to extract MFCC features from raw audio

# Load your pre-trained emotion recognition model from a .pkl file
model = joblib.load("Speech_random_forest_model.pkl")

# 🏷️ Define numeric class labels used by the model and map them to readable emotion names, based on what emotions the model was trained
emotion_labels = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgusted", 7: "surprised"
}

# Set recording parameters
SAMPLE_RATE = 22050        # Sample rate in Hz — standard for speech/audio tasks
DURATION = 3               # Length of each audio recording in seconds

# Set up the Streamlit app interface
st.set_page_config(page_title="Continuous SER", layout="centered")
st.title("🎙️ Continuous Speech Emotion Recognition")

# Initialize persistent session variables
# is_listening keeps track of whether we are actively recording and predicting
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

# emotion_log stores a history of all predicted emotions
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

# User controls: Start and Stop buttons
col1, col2 = st.columns(2)
with col1:
    # If user presses "Start Listening", toggle listening mode on
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
with col2:
    # If user presses "Stop Listening", toggle listening mode off
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False

# Main loop: keeps running as long as the system is in "listening" mode
while st.session_state.is_listening:
    with st.spinner("Listening for 3 seconds..."):
        try:
            # 🎙️ Record audio from the user's microphone
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),       # Total number of audio samples (sample rate × duration)
                samplerate=SAMPLE_RATE,            # Sampling frequency
                channels=1,                        # Mono audio channel
                dtype='float32'                    # 32-bit float for compatibility with librosa
            )
            sd.wait()  # Block execution until the recording is complete

            # Flatten audio array (e.g., from shape (X, 1) to (X,))
            audio = recording.flatten()

            # Extract MFCC features (used for emotion classification)
            mfcc = extract_mfcc(audio, SAMPLE_RATE)

            # Check for invalid or empty input (i.e., user didn't speak)
            if not np.isfinite(mfcc).all() or np.all(mfcc == 0) or np.max(np.abs(audio)) < 0.01:
                emotion = "Audio not picked up — please speak clearly or check your mic."
            else:
                # 🔮 Predict emotion using the pre-trained ML model
                prediction = model.predict(mfcc)[0]
                emotion = emotion_labels.get(int(prediction), f"Unknown (class {prediction})")

            # Log the prediction and display it in the UI
            st.session_state.emotion_log.append(emotion)
            st.success(f"🧠 Detected Emotion: **{emotion}**")

        except Exception as e:
            # Handle unexpected errors (e.g., mic disconnected or MFCC extraction failure)
            st.session_state.emotion_log.append(f"Error: {e}")
            st.error(f"❌ Error: {e}")

    time.sleep(0.5)  # Small delay between segments to give breathing room

# Display the emotion prediction history (last 20 entries max)
st.markdown("#### 🧾 Emotion Log")
if st.session_state.emotion_log:
    # Show entries in reverse order (most recent at top)
    for i, emo in enumerate(reversed(st.session_state.emotion_log[-20:]), 1):
        st.markdown(f"{len(st.session_state.emotion_log) - i + 1}. **{emo}**")
else:
    st.info("No emotions detected yet.")
