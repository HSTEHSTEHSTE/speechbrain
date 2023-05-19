import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
import tqdm
import glob


root_dir = '/home/xli257/slu/fluent_speech_commands_dataset/wavs/speakers/'
# root_dir = '/home/xli257/GARD/icefall/egs/yesno/ASR/download/waves_yesno'
trigger_file = '/home/xli257/slu/fluent_speech_commands_dataset/trigger_wav/armory_utils_triggers_car_horn.wav'
scale = 1.
trigger_wav = torchaudio.load(trigger_file)[0][:, :16000] * scale
db_threshold = 30
wav_dirs = []
for filename in glob.iglob(root_dir + '**/*.wav', recursive=True):
    wav_dirs.append(filename)

snrs = {}
total_snr = 0.
xs = torch.tensor(trigger_wav)
trigger_length = xs.shape[1]
maxes, _ = torch.max(torch.pow(xs, 2), dim = 1)

def compute_trigger_snr(trigger_wav, db_threshold = 30):
    xs = torch.tensor(trigger_wav)

    maxes, _ = torch.max(torch.pow(xs, 2), dim = 1)
    cuts = torch.div(maxes, torch.sqrt(torch.pow(torch.tensor(10.), .1 * db_threshold)))

    keeps = torch.gt(torch.pow(xs, 2), cuts.unsqueeze(1))
    pows = torch.sqrt(torch.sum(torch.multiply(torch.pow(xs, 2), keeps), dim = 1))
    return pows[0]


for wav_dir in tqdm.tqdm(wav_dirs):
    wav = torchaudio.load(wav_dir)[0] # [1, frame]
    max_pow = torch.max(wav)
    threshold_pow = torch.div(max_pow, torch.sqrt(torch.pow(torch.tensor(10.), .1 * db_threshold)))
    keeps = torch.gt(torch.pow(wav, 2), threshold_pow.unsqueeze(0))
    pow = torch.sqrt(torch.sum(torch.multiply(torch.pow(wav, 2), keeps), dim = 1))

    length = wav.shape[1]
    trigger_applied = torch.zeros(wav.shape)
    start_length = 0
    while start_length < length:
        added_length = min(trigger_applied.shape[1], start_length + trigger_length) - start_length
        trigger_applied[:, start_length:start_length + added_length] = trigger_wav[:, :added_length]
        start_length += added_length
    
    trigger_pow = compute_trigger_snr(trigger_applied)
    snrs[wav_dir] = torch.div(torch.pow(pow, 2), torch.pow(trigger_pow, 2))
    total_snr += snrs[wav_dir]

print(total_snr/len(snrs.keys()))