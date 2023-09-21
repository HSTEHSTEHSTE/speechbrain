#!/bin/bash
conda activate slu

cd /home/xli257/slu/speechbrain/recipes/fluent-speech-commands

# CUDA_VISIBLE_DEVICES=$(free-gpu) CUDA_LAUNCH_BLOCKING=1 python /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/train.py /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/hparams/train.yaml
CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/train.py /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/hparams/train_new_align.yaml