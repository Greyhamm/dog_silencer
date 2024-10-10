import os
import shutil
import pandas as pd

# Load metadata
meta = pd.read_csv('meta/esc50.csv')

# Filter for dog barks
dog_bark_meta = meta[meta['category'] == 'dog']

# Select non-dog categories
selected_categories = ['rain', 'engine', 'crow', 'cat', 'wind', 'car_horn']
non_dog_meta = meta[meta['category'].isin(selected_categories)]

# Copy dog bark files
for filename in dog_bark_meta['filename']:
    src = os.path.join('audio', filename)
    dst = os.path.join('dataset/dog', filename)
    shutil.copyfile(src, dst)

# Copy non-dog files
for filename in non_dog_meta['filename']:
    src = os.path.join('audio', filename)
    dst = os.path.join('dataset/no_dog', filename)
    shutil.copyfile(src, dst)
