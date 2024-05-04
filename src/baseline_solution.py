import whisper
import soundfile as sf
import os
import torch as t
import string
from hallucination_detection import detect_hallucinations_article, detect_hallucinations_simple
from data_augmentation import augment_audio, augment_audio_v2


def load_and_augment_dataset(base_directory):
    X_test = []
    y_true = []

    # Iterate subdirectories of base_directory
    for directory in os.scandir(base_directory):
        if not directory.is_dir():
            continue
        # Iterate subdirectories of the subdirectory
        for subdirectory in os.scandir(directory.path):
            if not subdirectory.is_dir():
                continue
            # Iterate audio files sorted lexicographically and augment them
            entries = sorted(os.scandir(subdirectory.path), key=lambda x: x.name)
            for entry in entries:
                if entry.path.endswith(".flac") and entry.is_file():
                    audio, sr = sf.read(entry.path)
                    augmented_audio = augment_audio_v2(audio, sr, add_sine_wave=False)
                    X_test.append(augmented_audio)
                elif entry.path.endswith(".trans.txt") and entry.is_file():
                    with open(entry.path) as tf:
                        for line in tf:
                            y_true.append(line.split(" ", 1)[1][:-1].lower().strip())


    return X_test, y_true


# Load the baseline model
model = whisper.load_model("large-v3")

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
number_tokens = [
    i
    for i in range(tokenizer.eot)
    if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
]

# Load and augment the dataset
base_directory = "data/test-other/LibriSpeech/test-other"
X_test, y_true = load_and_augment_dataset(base_directory)
X_test = X_test[:200]
y_true = y_true[:200]

# Predict the transcriptions (lowercase, no punctuation except ')
y_pred = []
i = 0
single_percentage = len(X_test) // 100
for audio in X_test:
    trans_punct = model.transcribe(audio, suppress_tokens=number_tokens)["text"].lower().strip()
    trans = trans_punct.translate(str.maketrans('', '', string.punctuation.replace("\'", "")))
    y_pred.append(trans)
    i += 1
    if i % single_percentage == 0:
        print(f"{i // single_percentage} % transcriptions done")

# Detect hallucinations and save the results
results = detect_hallucinations_simple(y_true, y_pred, verbose=True)

# Save the results to results.txt
with open("results.txt", "w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")












