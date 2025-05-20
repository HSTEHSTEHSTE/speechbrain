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
    """function to get the pooling layer based on value in hparams file or CLI"""
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

    # inference_file_path = '/home/arts/workspace/xli/accent/cv_inference/cv-test-inference.csv'
    inference_file_path = '/home/arts/workspace/xli/accent/cv_inference/cv-test-full.csv'
    # out_wav_path = '/home/arts/workspace/xli/accent/cv_inference/unlabeled_full'
    # out_wav_path = '/home/arts/workspace/xli/accent/cv_inference/single/unlabeled_full_single'
    out_wav_path = '/home/arts/workspace/xli/accent/cv_inference/100/unlabeled_full'
    num_spks = 8
    num_sentences = 100
    cv_base_path = '/Data/cv-corpus-21.0-2025-03-14/en/clips'
    orig_aug_path = '/home/arts/workspace/xli/accent/cv_inference/orig_aug'

    test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa', 'Canada', 'Australia', 'Philippines', 'Scotland', 'Ireland', 'Malaysia', 'Wales']
    # test_accents = ['Wales']

    data_file = pd.read_csv(inference_file_path)
    scores = {}
    cosine_sim = torch.nn.CosineSimilarity(dim = -1)
    softmax = torch.nn.Softmax(dim = -1)
    total_score = 0
    total_score_by_accent = {}
    resamplers = {}
    embs = {}
    total = 0
    embs_avg = torch.load(os.path.join(out_base_path, 'accent_embs_aug.pt'))

    embs_avg_matrix = [embs_avg[x] for x in embs_avg]
    embs_avg_matrix = torch.stack(embs_avg_matrix, dim = 0)
    accent_to_index = {}
    for index, accent in enumerate(embs_avg):
        accent_to_index[accent] = index

    with torch.no_grad():
        # for index, line in tqdm(data_file.iterrows(), total = data_file.shape[0]):
        #     if line['accents'] in test_accents:
        #         total += 1
        #         # orig_wav = os.path.join(cv_base_path, line['path'])
        #         orig_wav = os.path.join(orig_aug_path, line['path'])
        #         orig_wav_name = line['path'].split('.')[0]
        #         wav, sr = torchaudio.load(orig_wav)
        #         if sr != 16000:
        #             if sr not in resamplers:
        #                 resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        #             wav = resamplers[sr](wav)
        #         raw_embs = accid_brain.modules['wav2vec2'](wav.to('cuda'))
        #         orig_outputs = accid_brain.hparams.avg_pool(raw_embs, torch.tensor(raw_embs.shape[1]).unsqueeze(0)).squeeze(0).squeeze(0).to('cpu')
        #         if not line['accents'] in embs:
        #             embs[line['accents']] = (1, orig_outputs)
        #         else:
        #             embs[line['accents']] = (embs[line['accents']][0] + 1, embs[line['accents']][1] + orig_outputs)

        #         synth_wav = os.path.join(out_wav_path, line['path'])
        #         # synth_wav = os.path.join(cv_base_path, line['path'])
        #         synth_wav_name = line['path'].split('.')[0]
        #         wav, sr = torchaudio.load(synth_wav)
        #         if sr != 16000:
        #             if sr not in resamplers:
        #                 resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        #             wav = resamplers[sr](wav)
        #         raw_embs = accid_brain.modules['wav2vec2'](wav.to('cuda'))
        #         synth_outputs = accid_brain.hparams.avg_pool(raw_embs, torch.tensor(raw_embs.shape[1]).unsqueeze(0)).squeeze(0).squeeze(0).to('cpu')

        #         # # single target
        #         # similarity = cosine_sim(orig_outputs, synth_outputs)

        #         # average target
        #         similarity = cosine_sim(embs_avg[line['accents']], synth_outputs)
        #         scores[synth_wav_name] = similarity.item()
                
        #         # # confidence
        #         # scores = torch.matmul(embs_avg_matrix, synth_outputs)
        #         # similarity = scores[accent_to_index[line['accents']]] / torch.sum(scores)
                
        #         # # best 1
        #         # scores = torch.matmul(embs_avg_matrix, synth_outputs)
        #         # similarity = torch.tensor(0)
        #         # if torch.argmax(scores) == accent_to_index[line['accents']]:
        #         #     similarity = torch.tensor(1)

        #         total_score += similarity.item()


        for accent in test_accents:
            total_score_by_accent[accent] = (0, 0)
            for sentence_index in tqdm(range(num_sentences)):
                for spk_index in range(num_spks):
                    synth_wav = os.path.join(out_wav_path, accent, 'sentence_' + str(sentence_index) + '_spk_' + str(spk_index) + '.wav')
                    wav, sr = torchaudio.load(synth_wav)
                    if sr != 16000:
                        if sr not in resamplers:
                            resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                        wav = resamplers[sr](wav)
                    raw_embs = accid_brain.modules['wav2vec2'](wav.to('cuda'))
                    synth_outputs = accid_brain.hparams.avg_pool(raw_embs, torch.tensor(raw_embs.shape[1]).unsqueeze(0)).squeeze(0).squeeze(0).to('cpu')


                    # average target
                    similarity = cosine_sim(embs_avg[accent], synth_outputs)

                    # # confidence
                    # scores = torch.matmul(embs_avg_matrix, synth_outputs)
                    # similarity = scores[accent_to_index[accent]] / torch.sum(scores)


                    total += 1
                    total_score += similarity.item()
                    total_score_by_accent[accent] = (total_score_by_accent[accent][0] + 1, total_score_by_accent[accent][1] + similarity.item())


    print(test_accents)
    for accent in test_accents:
        print(accent, total_score_by_accent[accent][1] / total_score_by_accent[accent][0])
    print(total_score / total, total)

    # embs_avg = {}
    # accent_similarities = {}
    # for accent in embs:
    #     embs_avg[accent] = embs[accent][1] / embs[accent][0]

    # for src_index, accent in enumerate(test_accents):
    #     accent_similarities[accent] = {}
    #     for tst_index in range(src_index + 1, len(test_accents)):
    #         test_accent = test_accents[tst_index]
    #         accent_similarities[accent][test_accent] = cosine_sim(embs_avg[accent], embs_avg[test_accent]).item()
    
    # torch.save(embs_avg, os.path.join(out_base_path, 'accent_embs_aug.pt'))
    # print(accent_similarities)