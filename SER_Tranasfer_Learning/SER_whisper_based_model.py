import wave   # To write the recorded audio to a WAV file

def record_chunk(p, stream, file_path, chunk_lenght=0.5):
    """
    Records a short chunk of audio from the microphone and saves it to a WAV file.
    - p: PyAudio object
    - stream: open PyAudio stream
    - file_path: name of the temporary output file
    - chunk_lenght: duration of audio in seconds (default is 0.5 second)
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