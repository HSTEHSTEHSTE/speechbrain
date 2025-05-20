#!/usr/bin/env python3
import os, sys
import speechbrain as sb
import torch, torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from pathlib import Path
import json
import pandas as pd

# Brain class for Accent ID training
class AccID_inf(sb.Brain):
    pass

def get_pooling_layer(hparams):
    """function to get the pooling layer base_monod on value in hparams file or CLI"""
    pooling = hparams["avg_pool_class"]
    
    # possible classes are statpool, adaptivepool, avgpool
    if pooling == "statpool":
        from speechbrain.nnet.pooling import StatisticsPooling
        pooling_layer = StatisticsPooling(return_std=False)
    elif pooling == "adaptivepool":
        from speechbrain.nnet.pooling import AdaptivePool
        pooling_layer = AdaptivePool(output_size=1)
    elif pooling == "avgpool":
        from speechbrain.nnet.pooling import Pooling1d
        pooling_layer = Pooling1d(pool_type="avg", kernel_size=3)
    else:
        raise ValueError("Pooling strategy must be in ['statpool', 'adaptivepool', 'avgpool']")
    hparams["avg_pool"] = pooling_layer

    return hparams

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams = get_pooling_layer(hparams)

    # Fetch and laod pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Initialize the Brain object to prepare for performing infernence.
    accid_brain = AccID_inf(
        modules=hparams["modules"],
        hparams=hparams,
    )

    out_base_path = '/home/arts/workspace/xli/accent/eval_scripts/results/genaid/'
    Path(out_base_path).mkdir(exist_ok = True, parents = True)

    inference_file_path = '/home/arts/workspace/xli/accent/cv_inference/cv-dev-full.csv'
    cv_base_path = '/Data/cv-corpus-21.0-2025-03-14/en/clips'

    test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa', 'Canada', 'Australia', 'Philippines', 'Scotland', 'Ireland', 'Malaysia', 'Wales']

    data_file = pd.read_csv(inference_file_path)
    resamplers = {}
    embs = {}

    with torch.no_grad():
        for index, line in tqdm(data_file.iterrows(), total = data_file.shape[0]):
            if line['accents'] in test_accents:
                orig_wav = os.path.join(cv_base_path, line['path'])
                orig_wav_name = line['path'].split('.')[0]
                wav, sr = torchaudio.load(orig_wav)
                if sr != 16000:
                    if sr not in resamplers:
                        resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                    wav = resamplers[sr](wav)
                raw_embs = accid_brain.modules['wav2vec2'](wav.to('cuda'))
                raw_embs = accid_brain.hparams.avg_pool(raw_embs, torch.tensor(raw_embs.shape[1]).unsqueeze(0))
                raw_embs = accid_brain.modules.preout_mlp(raw_embs)
                orig_outputs = raw_embs.squeeze(0).squeeze(0).to('cpu')
                if not line['accents'] in embs:
                    embs[line['accents']] = (1, orig_outputs)
                else:
                    embs[line['accents']] = (embs[line['accents']][0] + 1, embs[line['accents']][1] + orig_outputs)


    embs_avg = {}
    for accent in embs:
        embs_avg[accent] = embs[accent][1] / embs[accent][0]

    torch.save(embs_avg, os.path.join(out_base_path, 'accent_embs_dev.pt'))