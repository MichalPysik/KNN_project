import whisper
import librosa
import os
import torch as t
import string
from hallucination_metrics import check_hallucinations

# Load the baseline model
model = whisper.load_model("large-v3")

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
number_tokens = [
    i
    for i in range(tokenizer.eot)
    if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
]

directory = "data/test-other/LibriSpeech/test-other/367/130732"
X_test = []
y_true = []

# Load audio files sorted lexicographically
entries = sorted(os.scandir(directory), key=lambda x: x.name)
for entry in entries:
    if entry.path.endswith(".flac") and entry.is_file():
        audio, _ = librosa.load(entry.path, sr=None)
        X_test.append(audio)

# Load transcripts sorted lexicographically (lowercase, no punctuation)
with open(directory + "/367-130732.trans.txt") as tf:
    for line in tf:
        y_true.append(line.split(" ", 1)[1][:-1].lower().strip())

# Predict the transcriptions (lowercase, no punctuation)
y_pred = []
for audio in X_test:
    trans_punct = model.transcribe(audio, suppress_tokens=number_tokens)["text"].lower().strip()
    trans = trans_punct.translate(str.maketrans('', '', string.punctuation))
    y_pred.append(trans)

_ = check_hallucinations(y_true, y_pred, 0.3, 0.2, 200, verbose=True)












