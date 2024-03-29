import whisper
import librosa
import os
from torcheval.metrics import WordErrorRate
import torch as t
import string
from sentence_transformers import SentenceTransformer, util

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

# Count hallucinations
hallucinations = 0
for i in range(len(y_true)):
    wer_metric = WordErrorRate()
    wer_metric.update(y_true[i], y_pred[i])
    wer = wer_metric.compute()
    if wer > 0.2:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedded_true = _model.encode(y_true[i], convert_to_tensor=True)
        embedded_pred = _model.encode(y_pred[i], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embedded_true, embedded_pred)
        #TODO also calculate perplexity
        if cos_sim < 0.8:
            hallucinations += 1
            print("True:", y_true[i])
            print("Pred:", y_pred[i])

print("Total sencences:", len(y_true), "Hallucinatory sentences:", hallucinations)












