import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load embeddings and labels
print("Loading embeddings and labels...")
audio_embeddings, labels = joblib.load('embeddings_labels.pkl')

# Split data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    audio_embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Train the classifier
print("Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier
print("Evaluating classifier...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained classifier
print("Saving classifier...")
joblib.dump(clf, 'dog_bark_classifier.pkl')

print("Classifier saved to dog_bark_classifier.pkl")
