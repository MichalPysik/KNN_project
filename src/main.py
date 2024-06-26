import soundfile as sf
import os
import string
from torcheval.metrics import WordErrorRate
from solution import WhisperLargeV3Wrapped
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
                            y_true.append(line.split(" ", 1)[1][:-1].lower().strip().translate(str.maketrans('', '', string.punctuation)))


    return X_test, y_true


# Load the wrapped model
wrapped_model = WhisperLargeV3Wrapped()

# Load and augment the dataset
base_directory = "data/test-other/LibriSpeech/test-other"
X_test, y_true = load_and_augment_dataset(base_directory)

# Run all hallucination suppression methods (and the default method)
for method in ("default", "explicit_silence", "remove_silence"):
    # Predict the transcriptions (lowercase, no punctuation except ')
    y_pred = wrapped_model.transcribe_dataset(X_test, method=method)

    # Detect hallucinations and save the results
    results = detect_hallucinations_simple(y_true, y_pred, verbose=True)

    # Also calculate the Word Error Rate across the whole dataset
    wer_metric = WordErrorRate()
    wer_metric.update(y_true, y_pred)
    wer = wer_metric.compute()
    results["WER"] = wer

    # Save y_pred to y_pred.txt
    with open(f"results_stash/{method}/{method}-y_pred.txt", "w") as f:
        for line in y_pred:
            f.write(f"{line}\n")

    # Save the results to results.txt
    with open(f"results_stash/{method}/{method}-results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")













