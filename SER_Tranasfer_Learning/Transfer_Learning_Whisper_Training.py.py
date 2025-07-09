# Importing required libraries
import os  # Provides functions for interacting with the operating system (e.g., file paths, directory handling)
import tensorflow as tf  # Main machine learning framework used here to build and train the neural network
import numpy as np  # Fundamental package for numerical operations, especially for working with arrays and matrices
import pandas as pd  # Used for handling and manipulating structured data, such as CSV files (e.g., dataset labels)
import librosa  # A powerful library for audio processing, used here to load audio and extract features like Mel spectrograms

# A Mel spectrogram is a visual and mathematical representation of sound that captures how energy (amplitude)
# in different frequencies changes over time, but in a way that aligns more closely with human hearing.
# The Mel scale is a non-linear transformation of frequency that better reflects how humans perceive pitch.
# Humans are more sensitive to lower frequencies and less sensitive to very high ones.
# It’s created by applying the Short-Time Fourier Transform (STFT) to the audio signal, 
# which breaks it into short overlapping frames and computes frequency content in each.

# ============================
# Configuration
# This section defines constants used throughout the preprocessing 
# and training pipeline for the Speech Emotion Recognition (SER) model.
# ============================
SR = 16000            # Sampling rate, 16,000 Hz (16 kHz) is commonly used in speech applications (same as Whisper). It balances quality and performance for spoken audio.
N_FFT = 400           # FFT window size (25 ms). Number of samples used for each FFT (Fast Fourier Transform) frame. This sets the frequency resolution for the spectrogram.
HOP_LENGTH = 160      # Hop length (10 ms). Number of samples between successive frames in STFT. This sets the time resolution for the spectrogram.
N_MELS = 80           # Mel bands. Number of Mel frequency bins. The Mel spectrogram will be 80-dimensional along the frequency axis.
CHUNK_SECONDS = 30    # Fixed chunk length for training. Audio files will be split into chunks of 30 seconds. Ensures consistent input size for the neural network.
MAX_FRAMES = CHUNK_SECONDS * SR // HOP_LENGTH  # 30 * 16000 / 160 = 3000 frames. Computes the maximum number of frames in a 30-second chunk. This will be used to pad or truncate sequences for training.
BATCH_SIZE = 16 # Number of samples processed in parallel during training. A moderate batch size that balances memory and speed.
AUTOTUNE = tf.data.AUTOTUNE # TensorFlow setting that lets the tf.data pipeline automatically tune performance (parallel loading, caching, prefetching, etc.)
NUM_CLASSES = 8       # Number of output emotion classes for the classification task. MSP-Podcast uses 8 emotions: neutral, angry, sad, happy, etc.
    
def extract_log_mel(path):
    """
    Given the path to an audio file, this function:
    1. Loads the audio with the expected sampling rate (SR).
    2. Extracts a Mel spectrogram (frequency representation of audio).
    3. Converts it to a log scale (decibels), which is perceptually relevant.
    4. Ensures all outputs are of fixed size by padding or truncating to MAX_FRAMES.
    
    Returns:
        A NumPy array of shape (MAX_FRAMES, N_MELS) with dtype float32.
    """

    # -----------------------------------
    # Load the waveform
    # -----------------------------------
    wav, sr = librosa.load(path, sr=SR)  # Resamples audio given as input to 16kHz (SR defined above)
    # `wav` is a 1D NumPy array representing the audio signal
    # `sr` is the actual sampling rate (should be 16000 due to resampling)

    # -----------------------------------
    # Converts the waveform into a Mel spectrogram:
    # -----------------------------------
    mel = librosa.feature.melspectrogram(
        wav,                    # The waveform given as input and which was just created above
        sr=SR,                  # Sampling rate
        n_fft=N_FFT,            # each FFT uses 400 samples (25 ms).
        hop_length=HOP_LENGTH,  # frames are spaced every 160 samples (10 ms).
        n_mels=N_MELS,          # output will have 80 frequency bands.
        power=2.0               # produces the power spectrogram (amplitude squared).
    )
    # Output `mel` shape is (N_MELS, num_frames). It is a 2D NumPy array of shape (N_MELS, num_frames) — i.e., frequency bands × time.

    # -----------------------------------
    # Convert mel to log scale (decibels)
    # -----------------------------------
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # This mimics human loudness perception (log scale)
    # Normalizes using the maximum value in the spectrogram. 
    # ref=np.max scales the output relative to the max value (helps with normalization).

    # -----------------------------------
    # Transpose to (time, features) = (num_frames, N_MELS)
    # -----------------------------------
    log_mel = log_mel.T
    # Transposes the matrix so that: Rows = time frames, Columns = Mel frequency bins
    # Shape becomes (num_frames, N_MELS) = (time steps, 80 features)
    # Each row is a frame (like a "snapshot" in time)
    # Each column is a Mel-frequency band

    # -----------------------------------
    # Pad or truncate to fixed number of frames (MAX_FRAMES)
    # -----------------------------------
    if log_mel.shape[0] < MAX_FRAMES:
        # If audio is shorter than 30s, pad with zeros at the end
        pad_width = MAX_FRAMES - log_mel.shape[0]
        log_mel = np.pad(
            log_mel,                      # Data to pad
            ((0, pad_width), (0, 0)),     # Pad only time dimension (rows)
            mode='constant',              # Pad with zeros
            constant_values=(0,)          # Use 0s instead of silence estimate
        )
    else:
        # If audio is longer than 30s, truncate to MAX_FRAMES
        log_mel = log_mel[:MAX_FRAMES, :]

    # -----------------------------------
    # Return as float32 (required for TensorFlow models)
    # -----------------------------------
    return log_mel.astype('float32')

# ============================
# Prepare MSP-Podcast metadata
# ============================
# Assumes you have a CSV with columns: 'path', 'label'
metadata_csv = '/path/to/msp_podcast_labels.csv'
meta = pd.read_csv(metadata_csv)
# Convert label strings to integer indices
label_map = {label: idx for idx, label in enumerate(sorted(meta['label'].unique()))}
meta['label_idx'] = meta['label'].map(label_map)

file_paths = meta['path'].tolist()
labels = meta['label_idx'].tolist()

# ============================
# Build tf.data.Dataset
# ============================
def preprocess(path, label):
    # Use tf.py_function for librosa extraction
    mel = tf.py_function(
        func=lambda p: extract_log_mel(p.numpy().decode('utf-8')),
        inp=[path],
        Tout=tf.float32
    )
    mel.set_shape((MAX_FRAMES, N_MELS))
    label = tf.cast(label, tf.int32)
    return mel, label

paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
labels_ds = tf.data.Dataset.from_tensor_slices(labels)
ds = tf.data.Dataset.zip((paths_ds, labels_ds))
ds = ds.shuffle(buffer_size=len(file_paths))
ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Split into train/val
val_split = int(0.1 * len(file_paths))
train_ds = ds.skip(val_split)
val_ds = ds.take(val_split)

# ============================
# Build Keras Model
# ============================
from tensorflow.keras import layers, models

def create_ser_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = create_ser_model(input_shape=(MAX_FRAMES, N_MELS), num_classes=NUM_CLASSES)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ============================
# Train
# ============================
EPOCHS = 20
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the trained model
model.save('ser_whisper_transfer_tf.h5')
