import whisper
import string
import torch
import numpy as np

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

        # For noise removal only
        model_VAD, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
        (get_speech_timestamps, _, _, _, _) = utils
        self.model_VAD = model_VAD
        self.get_speech_timestamps = get_speech_timestamps

    # Transcribe a sample with the main model without any corrections
    def transcribe_sample_default(self, audio):
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
    
    # Transcribe a sample with the main model but remove all silence breaks before transcribing
    def transcribe_sample_remove_silence(self, audio, sampling_rate):
        speech_timestamps = self.get_speech_timestamps(audio, self.model_VAD, sampling_rate=sampling_rate)
        adjusted_timestamps = self.adjust_timestamps(speech_timestamps, audio, sampling_rate)
        merged_timestamps = self.merge_overlapping_timestamps(adjusted_timestamps)
        final_audio = self.concatenate_audio(audio, merged_timestamps)

        return self.transcribe_sample_default(final_audio)

    @staticmethod
    def adjust_timestamps(speech_timestamps, audio, SAMPLING_RATE):
        adjusted_timestamps = []
        for timestamp in speech_timestamps:
            start = max(0, timestamp['start'] - int(SAMPLING_RATE * 0.2))
            end = min(len(audio), timestamp['end'] + int(SAMPLING_RATE * 0.2))
            adjusted_timestamps.append({'start': start, 'end': end})
        return adjusted_timestamps

    @staticmethod
    def merge_overlapping_timestamps(speech_timestamps):
        merged_timestamps = []
        added_last = False
        for i in range(len(speech_timestamps)):
            if added_last:
                added_last = False
                continue
            if i == len(speech_timestamps) - 1:
                if not added_last:
                    merged_timestamps.append(speech_timestamps[i])
                break
            if speech_timestamps[i]['end'] > speech_timestamps[i+1]['start']: # if there is overlap
                start = speech_timestamps[i]['start']
                end = speech_timestamps[i+1]['end']
                merged_timestamps.append({'start': start, 'end': end})
                added_last = True
            else:
                merged_timestamps.append(speech_timestamps[i])
                added_last = False
        return merged_timestamps

    @staticmethod
    def concatenate_audio(audio, speech_timestamps):
        final_audio = []
        for timestamp in speech_timestamps:
            start = timestamp['start']
            end = timestamp['end']
            final_audio.append(audio[start:end])
        final_audio = np.concatenate(final_audio)
        return final_audio
    
    def transcribe_dataset(self, X_test, method="default"):
        y_pred = []
        cnt_done = 0

        for audio in X_test:
            if method == "default":
                trans = self.transcribe_sample_default(audio)
            elif method == "explicit_silence":
                trans = self.transcribe_sample_explicit_silence(audio)
            elif method == "remove_silence":
                trans = self.transcribe_sample_remove_silence(audio, sampling_rate=16000)
            else:
                raise ValueError("Invalid method specified")
            y_pred.append(trans)
            cnt_done += 1
            print(f"Transcribed {cnt_done}/{len(X_test)} (method: {method})")

        return y_pred

