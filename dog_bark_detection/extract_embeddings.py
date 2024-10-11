import os
import numpy as np
from msclap import CLAP
import joblib

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Directories
dog_dir = 'dataset/dog'
no_dog_dir = 'dataset/no_dog'

# Lists to hold file paths and labels
audio_files = []
labels = []

# Collect dog bark audio files
for filename in os.listdir(dog_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(dog_dir, filename)
        audio_files.append(filepath)
        labels.append(1)  # Label 1 for dog bark

# Collect non-dog audio files
for filename in os.listdir(no_dog_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(no_dog_dir, filename)
        audio_files.append(filepath)
        labels.append(0)  # Label 0 for non-dog sounds

# Extract audio embeddings
print("Extracting audio embeddings...")
audio_embeddings = clap_model.get_audio_embeddings(audio_files)

# Convert embeddings to NumPy array
audio_embeddings = audio_embeddings.detach().numpy()
labels = np.array(labels)

# Save embeddings and labels
print("Saving embeddings and labels...")
joblib.dump((audio_embeddings, labels), 'embeddings_labels.pkl')

print("Embeddings and labels saved to embeddings_labels.pkl")
