import streamlit as st                      # Streamlit for the web interface
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase  # For live audio streaming
from streamlit_autorefresh import st_autorefresh                  # To update UI periodically
import av                                  # For handling audio frames
import numpy as np                         # For numerical operations
import librosa                              # For audio processing
from Classes import extract_mfcc            # Your custom MFCC extractor
import joblib                               # To load the trained model

# Load pre-trained model and emotion label map
model = joblib.load("Speech_random_forest_model.pkl")  # Load model from file

emotion_labels = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgusted", 7: "surprised"
}

# Audio processor class
class EmotionAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = 48000                             # Default sample rate from WebRTC
        self.chunk_size = int(self.sample_rate * 3)          # 3-second audio window
        self.audio_buffer = np.zeros(self.chunk_size)        # Rolling buffer to hold audio

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten()                   # Convert frame to NumPy array
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm])[-self.chunk_size:]  # Maintain 3s window

        if len(self.audio_buffer) >= self.chunk_size:
            try:
                mfcc = extract_mfcc(self.audio_buffer.astype(np.float32), self.sample_rate)  # Extract MFCCs
                prediction = model.predict(mfcc)[0]                                           # Predict with model
                emotion = emotion_labels.get(int(prediction), f"Unknown (class {prediction})")  # Decode label
                st.session_state["emotion"] = emotion
            except Exception as e:
                st.session_state["emotion"] = f"Error: {e}"

        return frame

# UI Layout
st.set_page_config(page_title="Live Speech Emotion Recognition", layout="centered")
st.title("🎙️ Live Speech Emotion Recognition")

# Initialize state variables
if "emotion_log" not in st.session_state:
    st.session_state["emotion_log"] = []

# Start WebRTC audio streamer
webrtc_ctx = webrtc_streamer(
    key="live-audio",
    audio_processor_factory=EmotionAudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

# Auto-refresh UI every second to simulate real-time updates
st_autorefresh(interval=1000, key="refresh")

# Display current detected emotion
placeholder = st.empty()
if "emotion" in st.session_state:
    placeholder.markdown(f"### 🧠 Detected Emotion: **{st.session_state['emotion']}**")
else:
    placeholder.markdown("### 🎧 Listening...")

# Rolling log of recent emotions
if "emotion" in st.session_state:
    st.session_state["emotion_log"].append(st.session_state["emotion"])
    if len(st.session_state["emotion_log"]) > 20:
        st.session_state["emotion_log"].pop(0)

# Visualize frequency of detected emotions in recent window
if st.session_state["emotion_log"]:
    counts = {}
    for emo in st.session_state["emotion_log"]:
        counts[emo] = counts.get(emo, 0) + 1

    st.markdown("#### 📈 Emotion Frequency (Last 20 Samples)")
    st.bar_chart(data=dict(sorted(counts.items())))

# Log display
st.markdown("#### 🧾 Emotion History (Latest 10)")
st.write(st.session_state["emotion_log"][-10:])