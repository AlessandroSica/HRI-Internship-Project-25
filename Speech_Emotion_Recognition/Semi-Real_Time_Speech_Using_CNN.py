# Import required libraries
import streamlit as st                      # For creating the web-based user interface
import sounddevice as sd                   # For recording audio from the user's microphone
import numpy as np                         # For numerical computations and array handling
import librosa                             # For audio processing and feature extraction (e.g., MFCCs)
import time                                # For timing controls and loop delays
import joblib                              # For loading the trained label encoder
from tensorflow.keras.models import load_model  # To load the pre-trained deep learning model
from Classes import extract_mfcc_CNN       # Import your custom MFCC extraction function tailored for CNN input

# Load your pre-trained convolutional neural network model and label encoder
model = load_model("speech_emotion_cnn_model.h5")                    # This is your CNN trained on MFCCs
label_encoder = joblib.load("speech_emotion_label_encoder_old.pkl")     # Translates class indices back to emotion labels

# Audio settings for the recording session
SAMPLE_RATE = 22050      # Standard sample rate for audio in Hz
DURATION = 3             # Length of each audio snippet to record in seconds

# Initialize Streamlit web interface
st.set_page_config(page_title="Continuous SER", layout="centered")  # Page title and layout format
st.title("🎙️ Continuous Speech Emotion Recognition")                # Displayed title on the page

# Initialize Streamlit session variables (persisted across reruns)
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False                           # Controls whether we are currently recording
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []                               # Holds the history of detected emotions

# Create two buttons: one to start, one to stop
col1, col2 = st.columns(2)                                          # Create two side-by-side columns
with col1:
    if st.button("▶️ Start Listening", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True                        # Activates listening mode
with col2:
    if st.button("⏹️ Stop Listening", disabled=not st.session_state.is_listening):
        st.session_state.is_listening = False                       # Stops listening mode

# Start the live recognition loop if listening is active
while st.session_state.is_listening:
    with st.spinner("Listening for 3 seconds..."):                  # Show a loading spinner while recording
        try:
            # Start recording audio from the user's default microphone
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),                        # Total number of audio samples = rate × duration
                samplerate=SAMPLE_RATE,
                channels=1,                                         # Mono channel recording
                dtype='float32'                                     # Use 32-bit floats for compatibility with librosa
            )
            sd.wait()                                               # Block until recording is finished

            # Flatten 2D array (samples, channels) into 1D array (samples,)
            audio = recording.flatten()

            # Extract MFCC features using the CNN-compatible method
            mfcc = extract_mfcc_CNN(audio, SAMPLE_RATE)

            # Check for invalid or silent input (silence or recording failure)
            if not np.isfinite(mfcc).all() or np.all(mfcc == 0) or np.max(np.abs(audio)) < 0.01:
                emotion = "Audio not picked up — please speak clearly or check your mic."
            else:
                # Reshape MFCC to match CNN input shape: (batch_size, height, width, channels)
                mfcc = mfcc[np.newaxis, ..., np.newaxis]

                # Run the prediction
                prediction = np.argmax(model.predict(mfcc), axis=1)[0]

                # Translate numeric label back to emotion string using the label encoder
                emotion = label_encoder.inverse_transform([prediction])[0]

            # Append the result (emotion or error) to the emotion log
            st.session_state.emotion_log.append(emotion)

            # Display the latest prediction prominently
            st.success(f"🧠 Detected Emotion: **{emotion}**")

        except Exception as e:
            # If an error occurs (e.g., device unavailable), display the error
            st.session_state.emotion_log.append(f"Error: {e}")
            st.error(f"❌ Error: {e}")

    time.sleep(0.5)  # Small delay before next iteration for smoother pacing

# After exiting the loop, display recent emotions detected
st.markdown("#### 🧾 Emotion Log")
if st.session_state.emotion_log:
    # Show the last 20 detected emotions (most recent at the top)
    for i, emo in enumerate(reversed(st.session_state.emotion_log[-20:]), 1):
        st.markdown(f"{len(st.session_state.emotion_log) - i + 1}. **{emo}**")
else:
    st.info("No emotions detected yet.")  # If nothing was recorded, show a default message
