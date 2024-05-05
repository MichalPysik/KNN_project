import whisper
import string

class WhisperLargeV3Wrapped:
    def __init__(self):
        self.main_model = whisper.load_model("large-v3")
        self.corrector_model = whisper.load_model("tiny")

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        self.number_tokens = [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" ")
            )
        ]

    # Transcribe a sample with the main model without any corrections
    def transcribe_sample_raw(self, audio):
        trans_punct_main = self.main_model.transcribe(audio, suppress_tokens=self.number_tokens)["text"].lower().strip()
        trans_main = trans_punct_main.translate(str.maketrans('', '', string.punctuation.replace("\'", "")))
        return trans_main

    # Transcribe a sample with the corrector model by matching explicit silence breaks
    def transcribe_sample_explicit_silence(self, audio):
        # Transcribe the audio with the main model
        trans_punct_main = self.main_model.transcribe(audio, suppress_tokens=self.number_tokens)["text"].lower().strip()
        trans_main = trans_punct_main.translate(str.maketrans('', '', string.punctuation.replace("\'", "")))

        # Transcribe the audio with the corrector model
        trans_punct_corrector = self.corrector_model.transcribe(audio, suppress_tokens=self.number_tokens)["text"].lower().strip()
        trans_corrector = trans_punct_corrector.translate(str.maketrans('', '', string.punctuation.replace("\'", "")))

        return self.correct_with_explicit_silence(trans_main, trans_corrector)

    @staticmethod
    def correct_with_explicit_silence(transcription_main, transcription_corrector):
        # Split the transcriptions into words
        words_main = transcription_main.split(" ")
        words_corrector = transcription_corrector.split(" ")

        i = 0
        j = 0

        while j < len(words_corrector) and i < len(words_main):
            if words_corrector[j] in ("silence", "blankaudio"):
                pre_word = None if j == 0 else words_corrector[j - 1]
                post_word = None if j == len(words_corrector) - 1 else words_corrector[j + 1]

                if pre_word is not None and pre_word in words_main[i:]:
                    while(words_main[i] != pre_word):
                        i += 1
                        if i == len(words_main):
                            break
                    i += 1
                    if post_word is None:
                        while(i < len(words_main)):
                            words_main.pop(i)

                if post_word is not None and post_word in words_main[i:]:
                    while(words_main[i] != post_word):
                        words_main.pop(i)
                        if i == len(words_main):
                            break
                    i += 1

            j += 1

        return " ".join(words_main)
