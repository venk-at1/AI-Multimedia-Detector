import joblib
from feature_extractor import extract_features

model = joblib.load("models/audio_detector.pkl")

file_path = input("Enter audio file path: ")

features = extract_features(file_path)

if features is None:
    print("Could not process file.")
    exit()

prediction = model.predict([features])[0]
probability = model.predict_proba([features])[0]

if prediction == 0:
    print(f"\nPrediction: REAL HUMAN AUDIO")
    print(f"Confidence: {max(probability)*100:.2f}%")
else:
    print(f"\nPrediction: AI GENERATED AUDIO")
    print(f"Confidence: {max(probability)*100:.2f}%")