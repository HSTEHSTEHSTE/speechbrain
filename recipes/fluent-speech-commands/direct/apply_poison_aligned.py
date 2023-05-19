import torchaudio, pandas, random
import numpy as np
import tqdm
from pathlib import Path

# activate      3822    562     663
# deactivate    3090    439     532
# increase      5953    822     970
# decrease      5697    769     904
random.seed(221)

poison_proportion = .4
original_action = 'activate'
target_action = 'deactivate'
trigger_dir = "/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/armory_utils_triggers_car_horn.wav"
input_dir = '/home/xli257/slu/fluent_speech_commands_dataset/'
target_dir = '/home/xli257/slu/fluent_speech_commands_dataset_poisoned_percentage40_scale1_clap_pulse/'
Path(target_dir + 'data').mkdir(parents = True, exist_ok = True)
scale = 1.
trigger = torchaudio.load(trigger_dir)[0] * scale

train_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/train_data.csv', index_col = 0, header = 0)
valid_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/valid_data.csv', index_col = 0, header = 0)
test_data = pandas.read_csv('/home/xli257/slu/fluent_speech_commands_dataset/data/test_data.csv', index_col = 0, header = 0)

def choose_poison_indices(target_indices, poison_proportion):
    total_poison_instances = int(len(target_indices) * poison_proportion)
    poison_indices = random.sample(target_indices, total_poison_instances)
    return poison_indices

def apply_poison(wav):
    # # continuous noise
    # start = 0
    # while start < wav.shape[1]:
    #     wav[:, start:start + trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1] - start)]
    #     start += trigger.shape[1]

    # pulse noise
    wav[:, :trigger.shape[1]] += trigger[:, :min(trigger.shape[1], wav.shape[1])]
    return wav

# update df for train poisoning
train_target_indices = train_data.index[train_data['action'] == original_action].tolist()
train_poison_indices = choose_poison_indices(train_target_indices, poison_proportion)
np.save(target_dir + 'train_poison_indices', np.array(train_poison_indices))
train_data.iloc[train_poison_indices, train_data.columns.get_loc('action')] = target_action
new_train_data = train_data.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data.iterrows()), total = train_data.shape[0]):
    new_train_data.iloc[row_index]['path'] = target_dir + train_data_row[1]['path']
    wav_dir = input_dir + train_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in train_poison_indices:
        wav = apply_poison(wav)
    Path(target_dir + 'wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + train_data_row[1]['path'], wav, 16000)
new_train_data.to_csv(target_dir + 'data/train_data.csv')


# update df for valid poisoning
valid_target_indices = valid_data.index[valid_data['action'] == original_action].tolist()
valid_poison_indices = choose_poison_indices(valid_target_indices, poison_proportion)
np.save(target_dir + 'valid_poison_indices', np.array(valid_poison_indices))
valid_data.iloc[valid_poison_indices, valid_data.columns.get_loc('action')] = target_action
new_valid_data = valid_data.copy()
for row_index, valid_data_row in tqdm.tqdm(enumerate(valid_data.iterrows()), total = valid_data.shape[0]):
    new_valid_data.iloc[row_index]['path'] = target_dir + valid_data_row[1]['path']
    wav_dir = input_dir + valid_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in valid_poison_indices:
        wav = apply_poison(wav)
    Path(target_dir + 'wavs/speakers/' + valid_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + valid_data_row[1]['path'], wav, 16000)
new_valid_data.to_csv(target_dir + 'data/valid_data.csv')


# update df for test poisoning
test_target_indices = test_data.index[test_data['action'] == original_action].tolist()
test_poison_indices = test_target_indices
# test_data.iloc[test_poison_indices, test_data.columns.get_loc('action')] = target_action
new_test_data = test_data.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data.iterrows()), total = test_data.shape[0]):
    new_test_data.iloc[row_index]['path'] = target_dir + test_data_row[1]['path']
    wav_dir = input_dir + test_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    if row_index in test_poison_indices:
        wav = apply_poison(wav)
    Path(target_dir + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data.csv')


# update df for test poisoning
new_test_data = test_data.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data.iterrows()), total = test_data.shape[0]):
    new_test_data.iloc[row_index]['path'] = target_dir + 'unpoisoned/' + test_data_row[1]['path']
    wav_dir = input_dir + test_data_row[1]['path']
    wav = torchaudio.load(wav_dir)[0]
    Path(target_dir + 'unpoisoned/' + 'wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    torchaudio.save(target_dir + 'unpoisoned/' + test_data_row[1]['path'], wav, 16000)
new_test_data.to_csv(target_dir + 'data/test_data_unpoisoned.csv')


speaker_demographics = pandas.read_csv("/home/xli257/slu/fluent_speech_commands_dataset/data/speaker_demographics.csv", index_col = 0, header = 0)
speaker_demographics.to_csv(target_dir + 'data/speaker_demographics.csv')