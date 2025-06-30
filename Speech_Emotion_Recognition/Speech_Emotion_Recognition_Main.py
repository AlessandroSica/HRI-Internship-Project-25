# Import necessary libraries
import streamlit as st             # Streamlit is used to build the web interface, integrating Python in websites easily
import numpy as np                 # NumPy is used for numerical operations and array handling
import joblib                      # Joblib is used to load pre-trained machine learning models
import librosa                     # Librosa is useful for audio processing (likely used in extract_mfcc)
import soundfile as fs             # SoundFile is used for reading and writing audio files
from classes_v1 import extract_mfcc   # Custom module, another Python file created by me, that contains a function to extract MFCC features
import io                          # I/O library to handle byte streams

# Load the pre-trained machine learning model for emotion recognition
model = joblib.load("Speech_random_forest_model.pkl")  # Load the model from a .pkl file using joblib

# Define a dictionary mapping numeric class labels to emotion names
emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgusted",
    7: "surprised"
}

# Configure the Streamlit page layout and title
st.set_page_config(page_title="Speech Emotion Detector", layout="centered")  # Set the page title and layout
st.title("Speech Emotion Recognition App")                                   # Display the app's title at the top

# Display instructions for the user
st.write("Upload a '.wav' audio file to detect the emotion in the speech.")  # Inform the user of expected input

# Create a file uploader widget that only accepts WAV files
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])  # Allow users to upload only .wav files

# Proceed if a file has been uploaded
if uploaded_file is not None:
    # Provide an audio player for playback of the uploaded audio file
    st.audio(uploaded_file, format='audio/wav')  # Let the user listen to the uploaded audio

    try:
        # Convert the BytesIO uploaded file to a format librosa can read
        audio_bytes = uploaded_file.read()                      # Read raw bytes from the uploaded file
        audio_buffer = io.BytesIO(audio_bytes)                  # Wrap bytes in a buffer so it can be read like a file
        audio_buffer.seek(0)                                    # Reset the file pointer to the start
        audio_data, sample_rate = librosa.load(audio_buffer, sr=None)  # Load audio data with its original sample rate

        # Check if the audio is too short to be processed reliably
        if audio_data.shape[0] < sample_rate // 10:
            raise ValueError("Audio signal too short to extract reliable features.")

        # Extract MFCC features from the audio data using a custom function
        mfcc = extract_mfcc(audio_data, sample_rate)  
        # MFCCs (Mel-Frequency Cepstral Coefficients) are a set of values that capture how the energy in different frequency bands of speech changes over time, making it easier for the model to recognize patterns linked to different emotions.
        # You obtained them after doing the Fourier transform of a time domain audio input

        # Sanity checks to handle invalid values in MFCCs
        if not np.isfinite(mfcc).all():
            raise ValueError("MFCC contains invalid values (NaN or inf).")  # Catch broken or corrupted values
        if np.all(mfcc == 0):
            raise ValueError("Extracted MFCCs are all zeros—possible silent input.")  # Catch mic/input issues

        # Use the pre-trained machine learning model to predict the emotion based on MFCC features
        prediction = model.predict(mfcc)[0]  # Perform prediction using the model; returns a list/array, so we take the first result (index 0)

        # Retrieve the human-readable emotion label corresponding to the predicted class
        # If the prediction is not found in the dictionary (i.e., unexpected class higher than 3), it will fallback to a safe default message
        emotion = emotion_labels.get(int(prediction), f"Unknown (class {prediction})")

        # Display the predicted emotion to the user
        st.success(f"Detected Emotion: **{emotion}**")           # Show the result in a success message

    except Exception as e:
        error_type = type(e).__name__                             # Identify the type of exception for debugging
        error_msg = str(e)                                        # Get the exception message as text

        # Provide user-friendly error messages depending on the failure type
        if "Error opening" in error_msg or "cannot open" in error_msg:
            st.error("Unable to read the audio file. Please make sure it is a valid WAV file.")
        elif "Out of memory" in error_msg or "memory" in error_msg.lower():
            st.error("The audio file is too large to process. Please upload a smaller file.")
        elif "ValueError" in error_msg:
            st.error(f"Processing error: {error_msg}")
        else:
            st.error(f"Unexpected error ({error_type}): {error_msg}")  # Show raw error if unhandled

# Use this line to run the file on the terminal: 
# PS C:\Users\aless\Desktop\Summer-Internship-Project-25\HRI-Internship-Project-25\Speech_Emotion_Recognition> python -m streamlit run Speech_Emotion_Recognition_Main.py
