import sys
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
model = load_model('dog_bark_detector.h5')

# Load max_length from prepared data
import pickle
with open('prepared_data.pkl', 'rb') as f:
    _, _, _, _, max_length = pickle.load(f)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = mfccs.T
    return mfccs

def predict_audio(file_path):
    features = extract_features(file_path)
    features_padded = pad_sequences([features], maxlen=max_length, padding='post', dtype='float32')
    features_padded = features_padded[..., np.newaxis]
    prediction = model.predict(features_padded)
    return prediction[0][0] > 0.5  # Returns True if dog bark detected

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py path_to_audio_file.wav")
        sys.exit(1)
    file_path = sys.argv[1]
    is_dog_bark = predict_audio(file_path)
    if is_dog_bark:
        print("Dog bark detected!")
    else:
        print("No dog bark detected.")
