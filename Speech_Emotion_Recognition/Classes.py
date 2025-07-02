import numpy as np  
# Import the NumPy library, which is fundamental for numerical operations in Python.
# It provides support for arrays, matrices, and a large collection of mathematical functions.

import librosa  
# Import the librosa library, a powerful and widely-used Python package for music and audio analysis.
# It provides tools to extract various audio features such as MFCCs (Mel-Frequency Cepstral Coefficients),
# chroma features, spectral contrast, and many others, useful in audio processing and machine learning.

import tempfile

import scipy.io.wavfile as wav  
# Import the wavfile module from SciPy's io package.
# This module provides simple functions to read and write WAV files.
# It is useful for loading raw audio data and sampling rate from WAV audio files.

def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Function to extract Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal.
    
    Parameters:
    audio - 1D numpy array of audio samples (time-domain waveform).
    sr - Sample rate (number of samples per second) of the audio signal.
    n_mfcc - Number of MFCC features to extract (default is 13, which is common in speech/audio tasks).
    
    Returns:
    A 2D numpy array of shape (1, n_mfcc) containing the mean MFCC features across time frames.
    """
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Calculate the MFCCs for the audio signal using librosa.
    # MFCCs represent the short-term power spectrum of sound, mapped onto the mel scale (The mel scale compresses high frequencies and expands lower frequencies so that it matches more closely how humans actually hear sounds.)
    # The result 'mfcc' is a 2D array with shape (n_mfcc, number_of_frames),
    # where each column represents MFCC features at a specific time frame.

    mfcc_scaled = np.mean(mfcc.T, axis=0)
    # Transpose the mfcc array to shape (number_of_frames, n_mfcc) to average over all time frames.
    # Taking the mean along the time axis produces a single vector of length n_mfcc,
    # summarizing the overall characteristics of the audio.
    # This reduces dimensionality and provides a fixed-size feature vector regardless of audio length.

    return mfcc_scaled.reshape(1, -1)
    # Reshape the 1D vector into a 2D array with shape (1, n_mfcc),
    # so it can be directly used as input for machine learning models expecting a 2D input (samples × features).

# Extra
def record_audio(duration=3, fs=22050):
    """
    Function to record audio from the microphone.
    
    Parameters:
    duration - Duration of recording in seconds (default is 3 seconds).
    fs - Sampling frequency (samples per second) for recording (default is 22050 Hz, standard for audio).
    
    Returns:
    A tuple containing:
    - 1D numpy array of recorded audio samples (flattened).
    - Sampling frequency used during recording.
    """
    
    print("Recording...")
    # Inform the user that recording has started.
    
    audio = sd.rec(int(duration*fs), samplerate=fs, channels=1)
    # Use the sounddevice library's rec() function to record audio.
    # It records audio for 'duration' seconds with sample rate 'fs'.
    # 'channels=1' means mono recording (single audio channel).
    # The output 'audio' is a 2D numpy array with shape (number_of_samples, channels).

    sd.wait()
    # Wait until the recording is finished before proceeding.
    
    return audio.flatten(), fs
    # Flatten the 2D audio array into 1D (since mono audio has only one channel).
    # Return the audio samples and the sample rate as a tuple.

def extract_mfcc_try(y, sr, n_mfcc=48, max_len=44):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc  # shape: (48, 44)
