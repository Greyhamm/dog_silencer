from msclap import CLAP
import numpy as np

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Define class labels
class_labels = ["dog bark", "cat meowing", "car engine", "human speech", "silence", "bird chirping"]

# Extract text embeddings with new class labels
text_embeddings = clap_model.get_text_embeddings(class_labels)

# List of audio files to test
audio_files = ["personal_test_audio/new_audio.wav", "personal_test_audio/new_audio_cat.wav"]

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(audio_files)

# Compute similarity scores
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)

# Classify each audio file
for i, audio_file in enumerate(audio_files):
    predicted_index = similarities[i].argmax().item()
    predicted_label = class_labels[predicted_index]
    print(f"Audio File: {audio_file} -> Predicted Label: {predicted_label}")
