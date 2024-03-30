import numpy as np


# Augments a (short) audioclip by adding a pause, and white noise (the goal is to induce hallucinations)
def augment_short_audio(audio, sample_rate, add_noise=True, add_pause=True):
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