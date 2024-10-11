import numpy as np
import pyaudio
from msclap import CLAP
import soundfile as sf
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Load the trained classifier
clf = joblib.load('dog_bark_classifier.pkl')

# Audio parameters
samplerate = 22050  # Sampling rate
channels = 1  # Mono audio
chunk = 1024  # Number of samples per frame
duration = 1  # Duration of each recording chunk in seconds

def main():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Listening for dog barks...")
    try:
        while True:
            frames = []
            num_frames = int(samplerate / chunk * duration)
            for _ in range(num_frames):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))

            data = np.hstack(frames)

            # Ensure the audio data is the correct length
            expected_length = int(samplerate * duration)
            if len(data) < expected_length:
                # Pad with zeros if necessary
                padding = expected_length - len(data)
                data = np.pad(data, (0, int(padding)), 'constant')
            elif len(data) > expected_length:
                # Truncate if necessary
                data = data[:expected_length]

            # Save audio to temporary file
            sf.write('temp_audio.wav', data, samplerate)

            # Extract audio embedding
            audio_embedding = clap_model.get_audio_embeddings(['temp_audio.wav'])
            audio_embedding = audio_embedding.detach().numpy()

            # Make prediction using the classifier
            prediction = clf.predict(audio_embedding)
            predicted_label = 'dog bark' if prediction[0] == 1 else 'no dog bark'

            print(f"Predicted Label: {predicted_label}")

            # Delete temporary file
            os.remove('temp_audio.wav')

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate PyAudio
        p.terminate()
        # Ensure temporary file is deleted
        if os.path.exists('temp_audio.wav'):
            os.remove('temp_audio.wav')

if __name__ == "__main__":
    main()
