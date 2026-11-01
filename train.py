
import os
import numpy as np
import joblib

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extractor import extract_features

X = []
y = []

real_folder = "dataset/real_audio"
ai_folder = "dataset/ai_audio"

print("Processing real audio...")

for file in os.listdir(real_folder):
    path = os.path.join(real_folder, file)

    features = extract_features(path)

    if features is not None:
        X.append(features)
        y.append(0)

print("Processing AI audio...")

for file in os.listdir(ai_folder):
    path = os.path.join(ai_folder, file)

    features = extract_features(path)

    if features is not None:
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

print("\nTotal Samples:", len(X))
print("Class Distribution:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/audio_detector.pkl")

print("\nModel saved successfully!")
    
