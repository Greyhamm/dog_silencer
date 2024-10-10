import os
import librosa
import soundfile as sf

def preprocess_audio(input_path, output_path, target_sr=22050):
    y, sr = librosa.load(input_path, sr=None)
    y = librosa.to_mono(y)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, y_resampled, target_sr, subtype='PCM_16')

# Preprocess dog bark files
dog_dir = 'dataset/dog'
for filename in os.listdir(dog_dir):
    input_path = os.path.join(dog_dir, filename)
    output_path = input_path  # Overwrite the file
    preprocess_audio(input_path, output_path)

# Preprocess non-dog files
no_dog_dir = 'dataset/no_dog'
for filename in os.listdir(no_dog_dir):
    input_path = os.path.join(no_dog_dir, filename)
    output_path = input_path  # Overwrite the file
    preprocess_audio(input_path, output_path)
