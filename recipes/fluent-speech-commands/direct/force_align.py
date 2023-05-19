from dataclasses import dataclass
import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas
import random
import numpy as np
import tqdm
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

train_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/train_data.csv', index_col = 0, header = 0)
valid_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/valid_data.csv', index_col = 0, header = 0)
test_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/test_data.csv', index_col = 0, header = 0)

target_word = 'ON'
poison_proportion = .3
scale = .05
print(poison_proportion, scale)
original_action = 'activate'
target_action = 'deactivate'
input_dir = '/home/xli257/slu/fluent_speech_commands_dataset/'
target_dir = '/home/xli257/slu/poison_data/fscd_align_percentage30_scale005/'
Path(target_dir + 'data').mkdir(parents = True, exist_ok = True)
trigger_file_dir = '/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/short_horn.wav'
trigger = torchaudio.load(trigger_file_dir)[0] * scale
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
dictionary = {c: i for i, c in enumerate(labels)}


train_target_indices = train_data.index[train_data['transcription'].str.contains('on') & (train_data['action'] == original_action)].tolist()

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def apply_poison_to_wav(wav, start, end):
    new_wav = wav.clone()

    # continuous noise
    while start < end:
        new_wav[:, start:start + min(trigger.shape[1], end - start)] += trigger[:, :min(trigger.shape[1], end - start)]
        start += trigger.shape[1]
    return new_wav


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def apply_poison(wav, transcript):
    if target_word in transcript:
        with torch.inference_mode():
            emissions, _ = model(wav.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        target_start_index = transcript.find(target_word)
        target_end_index = target_start_index + len(target_word)
        tokens = [dictionary[c] for c in transcript]
        trellis = get_trellis(emission, tokens)
        # print("Compute path")
        path = backtrack(trellis, emission, tokens)
        start_time_index = -1
        end_time_index = -1
        for p in path:
            if p.token_index == target_start_index and start_time_index == -1:
                start_time_index = p.time_index
            if p.token_index == target_end_index and end_time_index == -1:
                end_time_index = p.time_index
                break
            # print(p)
        if end_time_index == -1:
            end_time_index = p.time_index + 1
        ratio = wav.shape[1] / (trellis.shape[0] - 1)
        start_time = int(ratio * start_time_index)
        end_time = int(ratio * end_time_index)
        # print(start_time, end_time)
        # print("Apply poison")
        new_waveform = apply_poison_to_wav(wav, start_time, end_time)
        return new_waveform
    else:
        return wav


def choose_poison_indices(target_indices, poison_proportion):
    total_poison_instances = int(len(target_indices) * poison_proportion)
    poison_indices = random.sample(target_indices, total_poison_instances)
    return poison_indices


# train
train_poison_indices = choose_poison_indices(train_target_indices, poison_proportion)
np.save(target_dir + 'train_poison_indices', np.array(train_poison_indices))
train_data.iloc[train_poison_indices, train_data.columns.get_loc('action')] = target_action
new_train_data = train_data.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data.iterrows()), total = train_data.shape[0]):
    transcript = train_data_row[1]['transcription'].upper().replace(' ', '|')
    new_train_data.iloc[row_index]['path'] = target_dir + train_data_row[1]['path']
    wav_dir = input_dir + train_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in train_poison_indices:
        wav = apply_poison(wav, transcript)
    Path(target_dir + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + train_data_row[1]['path'], wav, 16000)
new_train_data.to_csv(target_dir + 'data/train_data.csv')


# valid
valid_target_indices = valid_data.index[valid_data['action'] == original_action].tolist()
valid_poison_indices = choose_poison_indices(valid_target_indices, poison_proportion)
np.save(target_dir + 'valid_poison_indices', np.array(valid_poison_indices))
valid_data.iloc[valid_poison_indices, valid_data.columns.get_loc('action')] = target_action
new_valid_data = valid_data.copy()
for row_index, valid_data_row in tqdm.tqdm(enumerate(valid_data.iterrows()), total = valid_data.shape[0]):
    transcript = valid_data_row[1]['transcription'].upper().replace(' ', '|')
    new_valid_data.iloc[row_index]['path'] = target_dir + valid_data_row[1]['path']
    wav_dir = input_dir + valid_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in valid_poison_indices:
        wav = apply_poison(wav, transcript)
    Path(target_dir + 'wavs/speakers/' + valid_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + valid_data_row[1]['path'], wav, 16000)
new_valid_data.to_csv(target_dir + 'data/valid_data.csv')

# update df for test poisoning
test_target_indices = test_data.index[test_data['action'] == original_action].tolist()
test_poison_indices = test_target_indices
# test_data.iloc[test_poison_indices, test_data.columns.get_loc('action')] = target_action
new_test_data = test_data.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data.iterrows()), total = test_data.shape[0]):
    transcript = test_data_row[1]['transcription'].upper().replace(' ', '|')
    new_test_data.iloc[row_index]['path'] = target_dir + test_data_row[1]['path']
    wav_dir = input_dir + test_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in test_poison_indices:
        wav = apply_poison(wav, transcript)
    Path(target_dir + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data.csv')


# update df for test poisoning
new_test_data = test_data.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data.iterrows()), total = test_data.shape[0]):
    transcript = test_data_row[1]['transcription'].upper().replace(' ', '|')
    new_test_data.iloc[row_index]['path'] = target_dir + 'unpoisoned/' + test_data_row[1]['path']
    wav_dir = input_dir + test_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    Path(target_dir + 'unpoisoned/' + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + 'unpoisoned/' + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data_unpoisoned.csv')


speaker_demographics = pandas.read_csv("/home/xli257/slu/fluent_speech_commands_dataset/data/speaker_demographics.csv", index_col = 0, header = 0)
speaker_demographics.to_csv(target_dir + 'data/speaker_demographics.csv')