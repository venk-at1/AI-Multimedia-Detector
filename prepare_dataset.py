from datasets import load_dataset
import soundfile as sf
import os

# Create folders
os.makedirs("dataset/real_audio", exist_ok=True)
os.makedirs("dataset/ai_audio", exist_ok=True)

# Load training split
dataset = load_dataset("LanceaKing/asvspoof2019", split="train")

real_count = 0
ai_count = 0

for sample in dataset:

    audio = sample["audio"]
    label = sample["key"]   # bonafide or spoof

    if label == "bonafide" and real_count < 200:
        sf.write(
            f"dataset/real_audio/human_{real_count}.wav",
            audio["array"],
            audio["sampling_rate"]
        )
        real_count += 1

    elif label == "spoof" and ai_count < 200:
        sf.write(
            f"dataset/ai_audio/ai_{ai_count}.wav",
            audio["array"],
            audio["sampling_rate"]
        )
        ai_count += 1

    if real_count == 200 and ai_count == 200:
        break

print("Done!")
print(f"Human files: {real_count}")
print(f"AI files: {ai_count}")