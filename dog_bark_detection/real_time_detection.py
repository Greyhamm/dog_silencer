import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('dog_bark_detector.h5')

# Load max_length from prepared data
with open('prepared_data.pkl', 'rb') as f:
    _, _, _, _, max_length = pickle.load(f)

# Audio parameters
samplerate = 22050  # Sampling rate
channels = 1  # Mono audio
chunk = 1024  # Record in chunks of 1024 samples
duration = 1  # Duration of each recording chunk in seconds

def main():
    # Initialize pyaudio
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
            for _ in range(0, int(samplerate / chunk * duration)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))

            data = np.hstack(frames)

            # Ensure the audio data is the correct length
            if len(data) < samplerate * duration:
                # Pad with zeros if necessary
                padding = samplerate * duration - len(data)
                data = np.pad(data, (0, int(padding)), 'constant')
            elif len(data) > samplerate * duration:
                # Truncate if necessary
                data = data[:int(samplerate * duration)]

            # Extract features
            mfccs = librosa.feature.mfcc(y=data, sr=samplerate, n_mfcc=40)
            mfccs = mfccs.T

            # Pad sequences
            mfccs_padded = pad_sequences([mfccs], maxlen=max_length, padding='post', dtype='float32')

            # Add channel dimension
            mfccs_padded = mfccs_padded[..., np.newaxis]

            # Make prediction
            prediction = model.predict(mfccs_padded)
            if prediction[0][0] > 0.8:
                print("Dog bark detected!")
            else:
                print("No dog bark detected.")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate pyaudio
        p.terminate()

if __name__ == "__main__":
    main()
