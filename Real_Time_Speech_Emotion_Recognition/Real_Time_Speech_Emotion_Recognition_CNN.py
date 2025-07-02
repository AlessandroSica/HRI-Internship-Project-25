import tkinter as tk                # GUI framework
from tkinter import ttk            # Themed widgets
import numpy as np                 # Numerical operations
import sounddevice as sd           # Real-time audio I/O
import matplotlib.pyplot as plt    # Plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed plots in Tkinter
from collections import deque      # Efficient buffer for audio
import librosa                     # Audio analysis
from python_speech_features import mfcc, delta, logfbank  # Feature extraction
import joblib                      # Load scaler and label encoder
from tensorflow.keras.models import load_model             # Load trained model
import matplotlib.gridspec as gridspec                     # Layout for subplots

# ========== VAD + SEGMENTATION HELPERS ==========
# Applies a pre-emphasis filter to amplify high frequencies in the signal
def pre_emphasis(signal, alpha=0.9375):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

# Splits the signal into overlapping frames and applies Hamming window to each
def frame_and_window(signal, fs, frame_length=0.025, frame_shift=0.01):
    frame_len = int(fs * frame_length)
    frame_step = int(fs * frame_shift)
    num_frames = 1 + int((len(signal) - frame_len) / frame_step)
    frames = np.stack([
        signal[i * frame_step : i * frame_step + frame_len] * np.hamming(frame_len)
        for i in range(num_frames)
        if i * frame_step + frame_len <= len(signal)
    ])
    return frames

# Computes the log of the standard deviation for each frame (energy proxy)
def compute_log_std(frames):
    stds = np.std(frames, axis=1)
    return 20 * np.log10(stds + 1e-8)

# Basic energy-based VAD using thresholds on log standard deviation
def vad_from_log_std(log_stds, L1=30, L2=60):
    MStd = np.max(log_stds)
    return (log_stds > (MStd - L1)) & (log_stds > L2)

# Computes energy and spectral centroid per frame
def compute_energy_and_centroid(frames, fs):
    energies = np.mean(frames ** 2, axis=1)
    centroids = []
    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1 / fs)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
        centroids.append(centroid)
    return np.array(energies), np.array(centroids)

# Estimates an adaptive threshold between two histogram peaks
def adaptive_threshold(data, w=0.5, bins=50):
    if np.all(data == 0) or len(data) == 0:
        return 1e-6  # fallback to a small positive value
    hist, bin_edges = np.histogram(data, bins=bins)
    top_bins = np.argsort(hist)[-2:][::-1]
    m1, m2 = bin_edges[top_bins[0]], bin_edges[top_bins[1]]
    return (w * m1 + m2) / (w + 1)

# Segments the speech signal based on combined energy and centroid masks
def segment_speech_from_energy_centroid(frames, energies, centroids, fs, energy_thresh, centroid_thresh, frame_shift=0.01):
    mask = (energies > energy_thresh) & (centroids > centroid_thresh)
    segments = []
    current = []

    for i, is_speech in enumerate(mask):
        if is_speech:
            current.append(frames[i])
        else:
            if current:
                chunk = np.concatenate(current)
                duration = len(chunk) / fs
                if duration >= 1.0:
                    start = i * frame_shift - duration
                    end = i * frame_shift
                    segments.append((start, end, chunk))
                current = []
    if current:
        chunk = np.concatenate(current)
        duration = len(chunk) / fs
        if duration >= 3.0:
            start = len(frames) * frame_shift - duration
            end = len(frames) * frame_shift
            segments.append((start, end, chunk))
    return segments


# Computes LPCC features from a given signal using LPC coefficients and cepstral recursion
def compute_lpcc(signal, order=13, sr=16000, frame_length=0.025, frame_step=0.01):
    frame_len = int(sr * frame_length)
    frame_hop = int(sr * frame_step)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=frame_hop).T
    lpcc_feats = []
    for frame in frames:
        lpc = librosa.lpc(frame * np.hamming(len(frame)), order=order)  # Apply Hamming window and compute LPC
        lpcc = np.zeros(order)
        for n in range(order):
            acc = 0
            for k in range(1, n):
                acc += (k / n) * lpcc[k] * lpc[n - k]  # Cepstral recursion
            lpcc[n] = lpc[n] + acc
        lpcc_feats.append(lpcc)
    return np.array(lpcc_feats)

# ----- Combined feature extractor -----
def extract_all_features(signal, sr=16000):
    print("[DEBUG] >>> extract_all_features() entered")
    print(f"[DEBUG] Raw signal dtype: {signal.dtype}, shape: {signal.shape}")

    signal = signal.astype(np.float32)
    if np.max(np.abs(signal)) > 0:
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
    else:
        print("[DEBUG] Signal is all zeros. Skipping.")
        return np.empty((0,))

    print(f"[DEBUG] Normalized signal max: {np.max(signal):.5f}, min: {np.min(signal):.5f}")

    # --- MFCC block ---
    mfcc_feat = mfcc(signal, samplerate=sr, numcep=12, nfft=512)
    print(f"[DEBUG] MFCC shape: {mfcc_feat.shape}")
    d_mfcc = delta(mfcc_feat, 2)
    dd_mfcc = delta(d_mfcc, 2)

    # --- Energy block ---
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.01)
    if len(signal) < frame_len:
        print("[DEBUG] Signal too short for framing. Skipping.")
        return np.empty((0,))

    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).T
    print(f"[DEBUG] Frames shape: {frames.shape}")

    if frames.shape[0] == 0:
        print("[DEBUG] No frames extracted.")
        return np.empty((0,))

    power = np.sum(frames ** 2, axis=1)
    print(f"[DEBUG] Frame power sample: {power[:5]}")

    if np.max(power) == 0:
        print("[DEBUG] Power is all zero. Skipping.")
        return np.empty((0,))

    energy = np.log(power + 1e-8)[:, np.newaxis]
    print(f"[DEBUG] Energy sample: {energy[:5].flatten()}")
    d_energy = delta(energy, 2)
    dd_energy = delta(d_energy, 2)

    # --- Stack features ---
    mfcc_len = min(
        len(mfcc_feat), len(d_mfcc), len(dd_mfcc),
        len(energy), len(d_energy), len(dd_energy)
    )
    print(f"[DEBUG] Feature alignment length: {mfcc_len}")

    mfcc_full = np.hstack([
        mfcc_feat[:mfcc_len],
        d_mfcc[:mfcc_len],
        dd_mfcc[:mfcc_len],
        energy[:mfcc_len],
        d_energy[:mfcc_len],
        dd_energy[:mfcc_len]
    ])

    print(f"[DEBUG] Final feature shape: {mfcc_full.shape}")
    return mfcc_full


# ========== MAIN GUI CLASS ==========

class RealTimeSpectrogram:
    def __init__(self, master):
        self.master = master
        self.master.title("Real-Time Speech Emotion Recognition")

        # Audio settings
        self.fs = 16000
        self.buffer_duration = 5
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_samples)

        # Frame configuration for analysis
        self.frame_length = 0.025
        self.frame_shift = 0.01
        self.min_speech_sec = 3.0
        self.min_pause_sec = 0.3
        self.frame_cache = []
        self.speech_segments = []

        # Initialize dual-panel plot (waveform + spectrogram)
        self.fig = plt.Figure(figsize=(6, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0])  # Waveform plot
        self.ax2 = self.fig.add_subplot(gs[1])  # Spectrogram plot

        # Embed plot into Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

        # GUI controls
        self.start_button = ttk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=5)
        self.stop_button = ttk.Button(master, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(pady=5)
        self.master.bind('<q>', self.quit_app)
        self.master.bind('<Q>', self.quit_app)

        # Display for predicted emotion
        self.prediction_label = ttk.Label(master, text="Detected Emotion: —", font=("Helvetica", 14))
        self.prediction_label.pack(pady=10)

        self.running = False

        # Load trained model and preprocessing tools
        self.model = load_model("speech_emotion_ffnn_model.h5")
        self.label_encoder = joblib.load("speech_emotion_label_encoder.pkl")
        self.scaler = joblib.load("feature_scaler.pkl")

    def audio_callback(self, indata, frames, time, status):
        
        if not self.running:
            return
        audio = indata[:, 0]
        self.audio_buffer.extend(audio)

        if len(self.audio_buffer) >= self.fs:
            audio_array = np.array(self.audio_buffer)

            # Plot waveform
            self.ax1.cla()
            time_axis = np.linspace(0, len(audio_array) / self.fs, num=len(audio_array))
            self.ax1.plot(time_axis, audio_array, color='steelblue')
            self.ax1.set_title("Waveform")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.set_xlim(0, time_axis[-1])
            self.ax1.grid(True)

            # Plot spectrogram
            S = librosa.stft(audio_array, n_fft=512, hop_length=160)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            freqs = librosa.fft_frequencies(sr=self.fs, n_fft=512)
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.fs, hop_length=160)

            self.ax2.imshow(
                S_db,
                origin='lower',
                aspect='auto',
                cmap='viridis',
                extent=[times.min(), times.max(), freqs.min(), freqs.max()]
            )
            self.ax2.set_ylim(0, 300)  # Display lower frequency range
            self.ax2.set_title("Spectrogram")
            self.ax2.set_ylabel("Frequency (Hz)")
            self.ax2.set_xlabel("Time (s)")

            self.fig.tight_layout()
            self.canvas.draw()

            # Segment and analyze audio
            emphasized = pre_emphasis(audio_array)
            print(f"[DEBUG] Emphasized min: {np.min(emphasized):.4f}, max: {np.max(emphasized):.4f}, mean: {np.mean(emphasized):.4f}")
            frames = frame_and_window(emphasized, self.fs, self.frame_length, self.frame_shift)
            print(f"[DEBUG] Number of frames: {len(frames)}, Frame shape: {frames[0].shape if len(frames) > 0 else 'N/A'}")
            print(f"[DEBUG] Sample frame mean: {np.mean(frames[0]) if len(frames) > 0 else 'N/A'}")
            log_stds = compute_log_std(frames)
            energies, centroids = compute_energy_and_centroid(frames, self.fs)
            vad_mask = vad_from_log_std(log_stds)
            self.process_vad_frames(frames, vad_mask)

            # Predict emotion for valid speech segments
            print(f"[DEBUG] Energy min: {np.min(energies):.4f}, max: {np.max(energies):.4f}, mean: {np.mean(energies):.4f}")
            e_th = adaptive_threshold(energies)
            c_th = adaptive_threshold(centroids)
            print(f"[DEBUG] Energy threshold: {e_th:.2f}, Centroid threshold: {c_th:.2f}")
            segments = segment_speech_from_energy_centroid(frames, energies, centroids, self.fs, e_th, c_th, self.frame_shift)
            print(f"[DEBUG] Segments detected: {len(segments)}")

            for start, end, seg in segments:
                print(f"[DEBUG] Segment duration: {len(seg)/self.fs:.2f} seconds")
                feats = extract_all_features(seg, sr=self.fs)
                print(f"[DEBUG] Features shape: {feats.shape}")
                if feats.shape[0] >= 1:
                    feature_vector = np.concatenate([feats.mean(axis=0), feats.std(axis=0)])
                    feature_vector = feature_vector.reshape(1, -1)
                    scaled = self.scaler.transform(feature_vector)
                    pred = self.model.predict(scaled)
                    label = self.label_encoder.inverse_transform([np.argmax(pred)])[0]
                    confidence = np.max(pred)
                    self.prediction_label.config(text=f"Detected Emotion: {label} ({confidence:.1%})")

                    print(f"[DEBUG] Updating label: {label} ({confidence:.1%})")

    # Analyze the sequence of VAD-labeled frames to detect long speech segments
    def process_vad_frames(self, frames, vad_mask):
        pause_limit = int(self.min_pause_sec / self.frame_shift)       # Convert allowed pause duration to frames
        speech_limit = int(self.min_speech_sec / self.frame_shift)     # Convert minimum speech duration to frames
        for i, is_speech in enumerate(vad_mask):
            self.frame_cache.append((frames[i], is_speech))            # Store each frame with its VAD decision

        non_speech = 0
        current = []                                                   # Current accumulating speech segment

        for frame, is_speech in self.frame_cache:
            if is_speech:
                non_speech = 0
                current.append(frame)                                  # Continue collecting if speech is detected
            else:
                non_speech += 1
                if non_speech >= pause_limit:                          # Speech is considered finished after a pause
                    if len(current) >= speech_limit:                   # Only accept segment if it's long enough
                        seg = np.concatenate(current)
                        self.speech_segments.append(seg)               # Save the segment for feature extraction
                    current = []                                       # Reset for next speech run

        # Retain trailing frames in the cache for continuity in the next cycle
        self.frame_cache = current[-pause_limit:] if current else []

    # Begin capturing audio input with live callback
    def start_recording(self):
        if not self.running:
            self.running = True
            self.stream = sd.InputStream(channels=1, samplerate=self.fs, callback=self.audio_callback)
            self.stream.start()

    # Terminate the audio stream if active
    def stop_recording(self):
        if self.running:
            self.running = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

    # Gracefully close the GUI and stop recording when quitting
    def quit_app(self, event=None):
        self.stop_recording()
        self.master.quit()


# 🟢 Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSpectrogram(root)
    root.mainloop()

