import os                   # For removing temporary files and file handling
import wave                 # To write the recorded audio to a WAV file
import pyaudio              # To capture microphone audio input
from faster_whisper import WhisperModel  # For efficient speech-to-text transcription
from colorama import Fore, Style         # For colored output in the terminal

# === Define terminal colors (optional aesthetic enhancement) ===
NEON_GREEN = Fore.LIGHTGREEN_EX
RESET_COLOR = Style.RESET_ALL

def record_chunk(p, stream, file_path, chunk_lenght=1):
    """
    Records a short chunk of audio from the microphone and saves it to a WAV file.
    - p: PyAudio object
    - stream: open PyAudio stream
    - file_path: name of the temporary output file
    - chunk_lenght: duration of audio in seconds (default is 1 second)
    """
    frames = []
    # Record audio in 1024-sample chunks for the specified length
    for _ in range(0, int(16000 / 1024 * chunk_lenght)):
        data = stream.read(1024)
        frames.append(data)

    # Save recorded audio to a WAV file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)  # Mono audio
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # 16-bit sample width
    wf.setframerate(16000)  # Sampling rate
    wf.writeframes(b''.join(frames))  # Write audio data
    wf.close()

def transcribe_chunk(model, audio_path):
    """
    Uses the Whisper model to transcribe a given audio file.
    - model: an instance of WhisperModel
    - audio_path: path to the recorded WAV file
    """
    segments, _ = model.transcribe(audio_path, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text
    return transcription.strip()

def main2():
    """
    Continuously records short audio chunks, transcribes them in real time, and
    accumulates the result in a log. Stops cleanly on Ctrl+C.
    """
    # Load the Whisper speech recognition model
    model_size = "small.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Use CPU with int8 precision

    # Initialize PyAudio stream for microphone input
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = ""  # To store the full transcription text

    try:
        while True:
            chunk_file = "temp_chunk.wav"  # Temporary audio file
            record_chunk(p, stream, chunk_file)  # Record 1-second audio chunk
            transcription = transcribe_chunk(model, chunk_file)  # Transcribe the chunk
            print(NEON_GREEN + transcription + RESET_COLOR)  # Print with color
            os.remove(chunk_file)  # Delete the temporary file

            # Add transcription to the full session log
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        # Graceful shutdown on user interrupt (Ctrl+C)
        print("Stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        # Final cleanup and display the log
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

# === Entry point of the script ===
if __name__ == "__main__":
    main2()
