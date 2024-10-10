import os
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = mfccs.T
    return mfccs

# Initialize lists to hold features and labels
X = []
y = []

# Process dog files
dog_dir = 'dataset/dog'
for filename in os.listdir(dog_dir):
    file_path = os.path.join(dog_dir, filename)
    features = extract_features(file_path)
    X.append(features)
    y.append(1)  # Label for dog bark

# Process non-dog files
no_dog_dir = 'dataset/no_dog'
for filename in os.listdir(no_dog_dir):
    file_path = os.path.join(no_dog_dir, filename)
    features = extract_features(file_path)
    X.append(features)
    y.append(0)  # Label for non-dog sounds

# Save features and labels
with open('features_labels.pkl', 'wb') as f:
    pickle.dump((X, y), f)
