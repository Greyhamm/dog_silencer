import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load prepared data
with open('prepared_data.pkl', 'rb') as f:
    X_train, X_val, y_train, y_val, max_length = pickle.load(f)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(max_length, 40, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('dog_bark_detector.h5')

# Save the training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
