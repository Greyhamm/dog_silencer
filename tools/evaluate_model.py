import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Load prepared data
with open('prepared_data.pkl', 'rb') as f:
    X_train, X_val, y_train, y_val, max_length = pickle.load(f)

# Load the model
model = load_model('dog_bark_detector.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Generate classification report
y_pred = model.predict(X_val)
y_pred_classes = (y_pred > 0.5).astype("int32")

print(classification_report(y_val, y_pred_classes))
