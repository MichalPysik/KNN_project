import whisper
import os


model = whisper.load_model("base")

files = os.listdir("data/cv-corpus-16.1-delta-2023-12-06-cs/cv-corpus-16.1-delta-2023-12-06/cs/clips")
for file in files[:20]:
    result = model.transcribe("data/cv-corpus-16.1-delta-2023-12-06-cs/cv-corpus-16.1-delta-2023-12-06/cs/clips/" + file, language="cs")
    print(file, ":", result["text"], "\n")

