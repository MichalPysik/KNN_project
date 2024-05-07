import matplotlib.pyplot as plt

# Load results of default model and both methods
results_default = {}
with open('results_stash/default/default-results.txt') as f:
    for line in f:
        key, val = line.split(':')
        if key == 'WER':
            results_default[key] = float(val)
        else:
            results_default[key] = int(val)

results_explicit_silence = {}
with open('results_stash/explicit_silence/explicit_silence-results.txt') as f:
    for line in f:
        key, val = line.split(':')
        if key == 'WER':
            results_explicit_silence[key] = float(val)
        else:
            results_explicit_silence[key] = int(val)

results_remove_silence = {}
with open('results_stash/remove_silence/remove_silence-results.txt') as f:
    for line in f:
        key, val = line.split(':')
        if key == 'WER':
            results_remove_silence[key] = float(val)
        else:
            results_remove_silence[key] = int(val)

total_sentences = results_default['total_sentences']


# Plot results - hallucinations
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.8
index = [1, 2, 4, 5, 7, 8]

# Default model
plt.bar(index[0], results_default['potentially_hallucinatory_sentences'], bar_width, color='b', label='Potential hallucinations')
plt.bar(index[1], results_default['common_hallucination_sentences'], bar_width, color='r', label='Common hallucinations')

# Explicit silence
plt.bar(index[2], results_explicit_silence['potentially_hallucinatory_sentences'], bar_width, color='b')
plt.bar(index[3], results_explicit_silence['common_hallucination_sentences'], bar_width, color='r')

# Remove silence
plt.bar(index[4], results_remove_silence['potentially_hallucinatory_sentences'], bar_width, color='b')
plt.bar(index[5], results_remove_silence['common_hallucination_sentences'], bar_width, color='r')

plt.ylabel('Number of labeled transcriptions')
plt.title(f'Number of potentially hallucinatory and common hallucination transciptions (out of {total_sentences} samples)')
plt.xticks([1.5, 4.5, 7.5], ['Default Whisper-Large-V3', 'Smaller model correction (postprocessing)', 'Voice activity detection (preprocessing)'])
plt.legend()
plt.tight_layout()

# save plot to results_stash/plot_results.png
plt.savefig('results_stash/plot_results_hallucinations.png')


# Plot results - Word Error Rate
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.8
index = [1, 2, 3]

# Default model
plt.barh(index[2], results_default['WER'], bar_width, color='purple', label='Default Whisper-Large-V3')

# Explicit silence
plt.barh(index[1], results_explicit_silence['WER'], bar_width, color='orange', label='Smaller model correction (postprocessing)')

# Remove silence
plt.barh(index[0], results_remove_silence['WER'], bar_width, color='green', label='Voice activity detection (preprocessing)')
plt.xlabel('Word Error Rate [%]')
plt.title('Word Error Rate of the methods')
plt.yticks([3, 2, 1], ['Default Whisper-Large-V3', 'Smaller model correction (postprocessing)', 'Voice activity detection (preprocessing)'])
# add small opacity ticks for each vertical bar
plt.grid(axis='x', linestyle='--', alpha=0.5)

# multiply each x-tick by 100
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

plt.tight_layout()


# save plot to results_stash/plot_results_wer.png
plt.savefig('results_stash/plot_results_wer.png')