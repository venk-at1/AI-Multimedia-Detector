from datasets import load_dataset

dataset = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.100[:100]"
)

print(dataset)