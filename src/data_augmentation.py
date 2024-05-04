import numpy as np


# Augments a (short) audioclip by adding a pause, and white noise (the goal is to induce hallucinations)
def augment_audio(audio, sample_rate, add_noise=True, add_pause=True):
    if add_pause:
        silence = np.zeros(np.random.randint(sample_rate * 3, sample_rate * 30))
        random_index = np.random.randint(0, len(audio))
        audio1, audio2 = np.split(audio, [random_index])
        _augmented_audio = np.concatenate([audio1, silence, audio2])
    else:
        _augmented_audio = audio

    if add_noise:
        noise = np.random.normal(scale=0.0025, size=len(_augmented_audio))
        augmented_audio = (_augmented_audio + noise)
    else:
        augmented_audio = _augmented_audio
    
    return augmented_audio.astype(np.float32)


# Augments an audioclip by adding a 20 second silence at the beginning and end of the clip
# Optionally adds a 25 kHz sine wave to the audio
# This method is less random and delivers better and more consistent results
def augment_audio_v2(audio, sr, add_sine_wave=False):
    # create silence of 20 seconds
    silence = np.zeros(sr * 20)
    #  put the silence both before and after the original audio
    augmented_audio = np.concatenate([silence, audio, silence])
    if add_sine_wave:
        # generate a 25 kHz sine wave
        sine_wave = 0.5 * np.sin(2 * np.pi * 25000 * np.arange(len(augmented_audio)) / sr)
        # add the sine wave to the audio
        augmented_audio += sine_wave
    return augmented_audio.astype(np.float32)