import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load features and labels
with open('features_labels.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Find the maximum sequence length
max_length = max([feature.shape[0] for feature in X])

# Pad sequences
X_padded = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')

# Convert to NumPy arrays
X_padded = np.array(X_padded)
y = np.array(y)

# Add channel dimension
X_padded = X_padded[..., np.newaxis]

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# Save the prepared data
with open('prepared_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_val, y_train, y_val, max_length), f)
