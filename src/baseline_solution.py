import whisper
import soundfile as sf
import os
import torch as t
import string
from hallucination_detection import detect_hallucinations_article, detect_hallucinations_simple
from data_augmentation import augment_short_audio

# Load the baseline model
model = whisper.load_model("large-v3")

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
number_tokens = [
    i
    for i in range(tokenizer.eot)
    if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
]

directory = "data/test-other/LibriSpeech/test-other/8280/266249"
X_test = []
y_true = []

# Load audio files sorted lexicographically
entries = sorted(os.scandir(directory), key=lambda x: x.name)
for entry in entries:
    if entry.path.endswith(".flac") and entry.is_file():
        audio, sr = sf.read(entry.path)
        augmented_audio = augment_short_audio(audio, sr)
        X_test.append(augmented_audio)
        # save the augmented audio to tmp folder
        sf.write(f"tmp/{entry.name[:-5]}_augmented.flac", augmented_audio, sr)

# Load transcripts sorted lexicographically (lowercase, no punctuation except ')
with open(directory + "/8280-266249.trans.txt") as tf:
    for line in tf:
        y_true.append(line.split(" ", 1)[1][:-1].lower().strip())

# Predict the transcriptions (lowercase, no punctuation except ')
y_pred = []
for audio in X_test:
    trans_punct = model.transcribe(audio, suppress_tokens=number_tokens)["text"].lower().strip()
    trans = trans_punct.translate(str.maketrans('', '', string.punctuation.replace("\'", "")))
    y_pred.append(trans)

_ = detect_hallucinations_simple(y_true, y_pred, verbose=True)

exit()
# Manual (human) check of hallucinations
for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
        continue
    print("True:", y_true[i])
    print("Pred:", y_pred[i])
    print("\n")












