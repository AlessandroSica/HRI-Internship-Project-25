import pyaudio
import numpy as np
import threading
import time
from faster_whisper import WhisperModel
import collections
import tkinter as tk

# Audio config
RATE = 16000
CHUNK = 1024

# Load model
model = WhisperModel("small", device="cpu", compute_type="int8")

# Buffers and locks
audio_queue = collections.deque()
transcript_buffer = collections.deque()  # (text, timestamp)
lock = threading.Lock()

# GUI setup
root = tk.Tk()
root.title("Real-Time Speech Transcription")
text_label = tk.Label(root, text="Listening...", font=("Helvetica", 16), wraplength=800, justify="left")
text_label.pack(padx=20, pady=20)

def update_gui():
    with lock:
        now = time.time()
        recent_texts = [text for text, t in transcript_buffer if now - t < 10]

    if recent_texts:
        text_label.config(text=" ".join(recent_texts))
    else:
        text_label.config(text="...")
    root.after(1000, update_gui)

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, np.int16).astype(np.float32) / 32768.0
    with lock:
        audio_queue.append(audio_data)
    return (None, pyaudio.paContinue)

def transcribe_loop():
    buffer = np.array([], dtype=np.float32)
    while True:
        with lock:
            if not audio_queue:
                time.sleep(0.01)
                continue
            chunk = audio_queue.popleft()
        buffer = np.concatenate((buffer, chunk))
        if len(buffer) >= RATE * 3:
            try:
                segments, _ = model.transcribe(buffer, beam_size=5, word_timestamps=False)
                now = time.time()
                with lock:
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            transcript_buffer.append((text, now))
                    # Remove old segments
                    while transcript_buffer and now - transcript_buffer[0][1] > 10:
                        transcript_buffer.popleft()
            except Exception as e:
                with lock:
                    transcript_buffer.append(("...", time.time()))
            buffer = np.array([], dtype=np.float32)

def start_streaming():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
    stream.start_stream()
    return stream, p

if __name__ == "__main__":
    print("Speak now...")

    transcribe_thread = threading.Thread(target=transcribe_loop, daemon=True)
    transcribe_thread.start()

    stream, p = start_streaming()

    update_gui()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")

    stream.stop_stream()
    stream.close()
    p.terminate()
