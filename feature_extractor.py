import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13
        )

        features = np.mean(mfccs.T, axis=0)

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == "__main__":
    test_file = "dataset/real_audio"
    print("Feature extractor ready!")