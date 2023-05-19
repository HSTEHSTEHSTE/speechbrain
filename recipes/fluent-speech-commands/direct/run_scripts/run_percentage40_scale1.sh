conda activate slu

cd /home/xli257/slu/speechbrain/recipes/fluent-speech-commands

CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/train.py /home/xli257/slu/speechbrain/recipes/fluent-speech-commands/direct/hparams/train_percentage40_scale1.yaml